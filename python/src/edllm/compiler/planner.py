"""Offline planning: resolve one catalog scope, enforce its pin, and budget disk.

Nothing here downloads, writes, or imports a heavy dependency. That is deliberate:
if the budget and pin checks lived inside the download path, they would be
testable only on a machine willing to fetch 14.19 GiB, which means untested.
"""
from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from ..model_catalog import ModelCatalog
from .errors import InsufficientDiskError, RevisionMismatchError, UnsupportedModelError
from .types import CompilePlan

CACHE_ROOT_ENV = "DIFFUSORRT_CACHE_ROOT"
# "~" is expanded where the root is resolved, never at import: Windows has no
# home directory without USERPROFILE, and importing this module must not need one.
_DEFAULT_CACHE_ROOT = Path("~") / ".cache" / "diffusorrt" / "compiled"

# Headroom over snapshot + package. The converter needs room for the in-flight
# tensor and the zip's own bookkeeping, and a volume that fits the artifacts with
# nothing to spare is one concurrent writer away from ENOSPC -- which halted this
# project on 2026-07-30.
_HEADROOM_BYTES = 5_000_000_000


class CompileModelEntry(Protocol):
    """The planning projection shared by Product and Local catalog entries."""

    key: str
    model_id: str
    revision: str
    source_license: str
    snapshot_bytes: int
    package_bytes: int
    peak_rss_bytes: int
    custom_tokenizer_module: str | None
    build_digest: str


def resolve_compile_entry(
    model_id: str,
    catalog: ModelCatalog | None = None,
) -> CompileModelEntry:
    """Resolve one exact scope; a Local catalog never inherits Product IDs."""

    if catalog is not None and not isinstance(catalog, ModelCatalog):
        raise TypeError("catalog must be a ModelCatalog")
    if catalog is None:
        from .registry import REGISTRY

        entries: Mapping[str, CompileModelEntry] = REGISTRY
    else:
        entries = catalog
    try:
        return entries[model_id]
    except KeyError:
        supported = ", ".join(sorted(entries)) or "(none)"
        scope = "Product registry" if catalog is None else "supplied Local catalog"
        raise UnsupportedModelError(
            f"{model_id!r} is not in the {scope}. Supported exact IDs: "
            f"{supported}. Use Engine.from_package() for a prepared package."
        ) from None


def _resolve_cache_root(cache_root) -> Path:
    if cache_root is not None:
        candidate = Path(cache_root)
    else:
        from_env = os.environ.get(CACHE_ROOT_ENV)
        candidate = Path(from_env) if from_env else _DEFAULT_CACHE_ROOT
    return candidate.expanduser().resolve(strict=False)


def _free_bytes(path: Path) -> int:
    """Free space on the filesystem that WOULD hold the cache.

    Walks up to the nearest existing ancestor: the cache root usually does not
    exist yet, and stat-ing a missing directory would raise instead of answering
    the question asked.
    """
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _same_filesystem(left: Path, right: Path) -> bool:
    """Do these two paths live on one device?

    Compares st_dev of the nearest EXISTING ancestor of each, because neither
    cache root need exist yet. Unknown resolves to True -- the conservative
    direction, since treating one disk as two under-counts the requirement and
    lets a compile start that cannot finish.
    """

    def _device(path: Path) -> int | None:
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            return probe.stat().st_dev
        except OSError:
            return None

    left_device, right_device = _device(left), _device(right)
    if left_device is None or right_device is None:
        return True
    return left_device == right_device


def _hf_cache_root() -> Path:
    """Where a download would land. NOT the DiffusorRT cache.

    Resolved from the environment, in huggingface_hub's own precedence, WITHOUT
    importing it: planning must answer on a machine that has no compiler extra
    installed, and a test enforces that this module imports nothing heavy.
    Asserted equal to `huggingface_hub.constants.HF_HUB_CACHE` by a test that
    runs where the package IS present.
    """
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    # os.path.expanduser, as huggingface_hub spells it: with no home directory
    # it leaves "~" in place where Path.home() raises, so the two still agree.
    base = Path(xdg) if xdg else Path(os.path.expanduser("~")) / ".cache"
    return base / "huggingface" / "hub"


def plan_pretrained(
    model_id: str,
    *,
    revision: str | None = None,
    cache_root=None,
    require_fit: bool = False,
    offline: bool = False,
    catalog: ModelCatalog | None = None,
) -> CompilePlan:
    """Resolve what compiling ``model_id`` would involve. Downloads nothing.

    ``require_fit`` raises ``InsufficientDiskError`` when the budget does not fit;
    the default reports it on the plan instead, so a caller can inspect the
    numbers without handling an exception.

    ``offline`` changes the BUDGET, not just a label. The online path copies the
    snapshot into the DiffusorRT cache and must reserve room for it; the offline
    path reads it from the standard Hugging Face cache in place and must not.
    Charging both modes for a 14.19 GiB copy that offline never makes refuses
    compiles on disks with room to spare -- and does so on exactly the machines
    where the user already fetched the snapshot to avoid a download.
    """
    entry = resolve_compile_entry(model_id, catalog)

    if revision is not None and revision != entry.revision:
        raise RevisionMismatchError(
            f"revision {revision!r} disagrees with the pinned revision "
            f"{entry.revision!r} for {model_id}. The pin is the contract; a "
            "different revision needs a registry entry, not an argument."
        )

    root = _resolve_cache_root(cache_root)
    target_suffix = entry.revision[:12]
    if catalog is not None:
        # A v2 Local package is self-contained: its manifest embeds both the
        # bytes-producing architecture and the default deployment policy.  The
        # cache key must therefore bind the complete definition.  A future
        # content-addressed Artifact Store may deduplicate the weight blob by
        # ``build_digest``; pretending the current monolithic package already
        # has that separation makes two policy variants collide at one path.
        definition_digest = getattr(entry, "definition_digest", None)
        if (
            not isinstance(definition_digest, str)
            or len(definition_digest) != 64
        ):
            raise TypeError("Local catalog entry has no valid definition digest")
        target_suffix = f"{target_suffix}.local-{definition_digest[:12]}"
    target = root / f"{model_id.replace('/', '__')}@{target_suffix}"
    # TWO budgets on TWO filesystems. The package is written to the DiffusorRT
    # cache; the snapshot is downloaded into the Hugging Face cache, which is a
    # different directory and often a different disk.
    required = entry.package_bytes + _HEADROOM_BYTES
    available = _free_bytes(root)
    snapshot_root = None if offline else _hf_cache_root()
    snapshot_required = 0 if offline else entry.snapshot_bytes
    snapshot_available = 0 if offline else _free_bytes(snapshot_root)
    if snapshot_root is not None and _same_filesystem(root, snapshot_root):
        # SAME disk: the two artifacts compete for one pool, so the package
        # budget carries both. Checking them independently let 35 GiB and
        # 14 GiB each "fit" in 36 GiB of shared space -- which is how splitting
        # a budget across filesystems loses the common case, where there is
        # only one filesystem.
        required += snapshot_required
        snapshot_required = 0

    plan = CompilePlan(
        model_id=model_id,
        revision=entry.revision,
        source_license=entry.source_license,
        cache_root=root,
        target_package_dir=target,
        already_compiled=(target / "model.json").is_file(),
        estimated_snapshot_bytes=entry.snapshot_bytes,
        estimated_package_bytes=entry.package_bytes,
        estimated_peak_rss_bytes=entry.peak_rss_bytes,
        required_free_bytes=required,
        available_free_bytes=available,
        snapshot_cache_root=snapshot_root,
        snapshot_required_bytes=snapshot_required,
        snapshot_available_bytes=snapshot_available,
        offline=offline,
    )
    if require_fit and not plan.fits:
        # The breakdown must match what was actually charged. Naming a snapshot
        # cost the offline budget did not include would send the user freeing
        # 14 GiB they never needed.
        # Name the filesystem that is actually short. A single combined figure
        # sent the user freeing space on the wrong disk whenever the two caches
        # were not on the same volume.
        parts = []
        if not plan.package_fits:
            short = required - available
            parts.append(
                f"{root} needs {required / 2**30:.1f} GiB for the package plus "
                f"headroom but has {available} bytes "
                f"({available / 2**30:.1f} GiB) - short by "
                f"{short / 2**30:.1f} GiB; free space or set {CACHE_ROOT_ENV}"
            )
        if not plan.snapshot_fits:
            short = snapshot_required - snapshot_available
            parts.append(
                f"{snapshot_root} needs {snapshot_required / 2**30:.1f} GiB for "
                f"the downloaded snapshot but has {snapshot_available} bytes "
                f"({snapshot_available / 2**30:.1f} GiB) - short by "
                f"{short / 2**30:.1f} GiB; free space or set HF_HUB_CACHE"
            )
        raise InsufficientDiskError(
            f"compiling {model_id} does not fit. " + " ".join(parts) + "."
        )
    return plan
