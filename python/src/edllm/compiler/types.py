"""CompilePlan (estimates) and CompileResult (measurements).

Field-name prefixes are load-bearing: ``estimated_`` / ``required_`` /
``available_`` on the plan, ``measured_`` on the result. A single type carrying
both would force either fabricated fields on a dry run or nullable ones every
caller must defensively check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CompilePlan:
    """What WOULD happen. Every byte figure is an estimate."""

    model_id: str
    revision: str
    source_license: str
    cache_root: Path
    target_package_dir: Path
    already_compiled: bool
    estimated_snapshot_bytes: int
    estimated_package_bytes: int
    estimated_peak_rss_bytes: int
    required_free_bytes: int
    """The DiffusorRT cache budget ONLY: package plus headroom.

    It used to include the snapshot too. Acquisition no longer copies the
    snapshot anywhere -- it reads the Hugging Face cache in place -- so those
    bytes land on whatever filesystem HF_HUB_CACHE points at. Charging them
    here refuses a compile on a machine with room, or permits one on a machine
    without, whenever the two caches sit on different disks. Which is the
    normal arrangement here: the model archive lives on an external SSD."""

    available_free_bytes: int

    snapshot_cache_root: "Path | None" = None
    """Where the snapshot would be acquired. None offline -- nothing is
    downloaded, so there is no second budget."""

    snapshot_required_bytes: int = 0
    snapshot_available_bytes: int = 0

    offline: bool = False
    """Planned for the offline path, which reads the snapshot from the standard
    Hugging Face cache IN PLACE instead of copying it into the DiffusorRT cache.

    Load-bearing for the budget above, not a label: online must reserve room for
    a copy of the snapshot, offline must not. Charging for a 14.19 GiB copy that
    will never be made refuses compiles on disks with room to spare -- on
    exactly the machines where the user pre-fetched to avoid a download.

    Last, with a default, so adding it did not force every existing CompilePlan
    construction to be rewritten."""

    @property
    def fits(self) -> bool:
        """BOTH budgets, on their own filesystems."""
        return self.package_fits and self.snapshot_fits

    @property
    def package_fits(self) -> bool:
        return self.available_free_bytes >= self.required_free_bytes

    @property
    def snapshot_fits(self) -> bool:
        return self.snapshot_available_bytes >= self.snapshot_required_bytes

    @property
    def shortfall_bytes(self) -> int:
        """Bytes still needed; 0 when it fits. Reported so a refusal can say how
        much to free rather than only that it did not fit."""
        return max(0, self.required_free_bytes - self.available_free_bytes) + max(
            0, self.snapshot_required_bytes - self.snapshot_available_bytes
        )


@dataclass(frozen=True, slots=True)
class CompileResult:
    """What DID happen. Every byte figure is measured."""

    plan: CompilePlan
    package_dir: Path
    reused_cache: bool
    measured_package_bytes: int
    measured_peak_rss_delta_bytes: int
    """Peak-RSS INCREASE attributable to this call.

    A HINT, not a bound in either direction. It is the delta of
    ``resource.ru_maxrss``, a process-lifetime high-water mark, so it reads 0
    when the process already peaked higher, and it also counts allocation by any
    other thread during the window. An earlier revision of this docstring claimed
    it "never over-reports"; that was wrong -- concurrent allocation is
    attributed here.

    A trustworthy peak requires compiling in a dedicated subprocess and reading
    ``RUSAGE_CHILDREN``. Task 7's gate must do that rather than rely on this.
    """
    wall_seconds: float
