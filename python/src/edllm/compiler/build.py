"""Acquire, convert, and publish atomically.

Every heavy import is inside a function: this module must be importable on a
machine with none of huggingface_hub, safetensors, torch, or transformers, so a
missing dependency produces CompilerDependencyError naming the extra rather than
an ImportError on an internal module path.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .errors import (
    CompilerDependencyError,
    ModelAccessError,
    RemoteCodeNotTrustedError,
    SnapshotNotCachedError,
)
from ..model_catalog import ModelCatalog
from ..package_lifecycle import (
    PackageLease,
    acquire_package_load_lease,
    acquire_package_write_guard,
)
from .planner import plan_pretrained, resolve_compile_entry
from .types import CompileResult

# Bumped whenever a change would make an older package differ from what this
# version produces. A cache entry from another version is rebuilt, not reused.
COMPILER_VERSION = "0.1.3"
RECEIPT_NAME = ".diffusorrt-compile.json"
_MAX_IDENTITY_BYTES = 16 * 1024 * 1024


def _product_registry():
    """Load admission state only for Product-scoped operations."""

    from .registry import REGISTRY

    return REGISTRY


@dataclass(frozen=True, slots=True)
class VerifiedCompileReceiptSnapshot:
    """Identity bytes captured by the same safe pass that verified the package."""

    package_dir: Path
    model_id: str
    revision: str
    source_license: str
    descriptor_key: str | None
    descriptor_digest: str | None
    definition_key: str | None
    definition_digest: str | None
    manifest_model_id: str | None
    manifest_revision: str | None
    manifest_source_license: str | None
    manifest_descriptor_key: str | None
    manifest_descriptor_digest: str | None
    manifest_definition_key: str | None
    manifest_definition_digest: str | None
    manifest_sha256: str
    receipt_sha256: str
    model_precision: str | None
    admitted_backend: str | None
    package_st_dev: int
    package_st_ino: int
    package_bytes: int
    compiler_version: str = COMPILER_VERSION


@dataclass(frozen=True, slots=True)
class _ReadFile:
    sha256: str
    data: bytes | None
    size: int
    identity: tuple[int, int] = (0, 0)
    """(st_dev, st_ino) of the file that was read: a replacement is another file."""


def _strict_json_loads(data: bytes | str) -> object:
    """Parse untrusted identity JSON without Python's permissive extensions."""

    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(
        data,
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )


def _nofollow_flags(*, directory: bool = False) -> int:
    """Flags required for a path read that never follows a symlink."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError("O_NOFOLLOW is unavailable")
    flags = os.O_RDONLY | nofollow
    if directory:
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if directory_flag is None:
            raise OSError("O_DIRECTORY is unavailable")
        flags |= directory_flag
    return flags


def _safe_relative_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("unsafe artifact path")
    parts = tuple(relative.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("unsafe artifact path")
    return parts


def _open_package_root(package_dir: str | Path) -> int:
    """Open a package directory, refusing one that is itself a link."""
    if os.name == "nt":
        from .. import _windows_files as windows

        return windows.to_descriptor(windows.open_root(package_dir, follow=False))
    return os.open(package_dir, _nofollow_flags(directory=True))


def _open_directory_at(parent_fd: int, name: str) -> int:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise OSError("path component is not a directory")
    child_fd = os.open(name, _nofollow_flags(directory=True), dir_fd=parent_fd)
    after = os.fstat(child_fd)
    if (
        not stat.S_ISDIR(after.st_mode)
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
    ):
        os.close(child_fd)
        raise OSError("directory changed while it was opened")
    return child_fd


def _read_regular_file_at(
    root_fd: int,
    relative: str,
    *,
    capture: bool = False,
    max_bytes: int | None = None,
) -> _ReadFile:
    """Hash and optionally retain bytes from one O_NOFOLLOW regular-file FD."""
    parts = _safe_relative_parts(relative)
    if os.name == "nt":
        return _read_regular_file_windows(root_fd, parts, capture=capture, max_bytes=max_bytes)
    parent_fd = os.dup(root_fd)
    file_fd: int | None = None
    try:
        for part in parts[:-1]:
            child_fd = _open_directory_at(parent_fd, part)
            os.close(parent_fd)
            parent_fd = child_fd
        before = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("artifact is not a regular file")
        file_fd = os.open(
            parts[-1],
            _nofollow_flags(),
            dir_fd=parent_fd,
        )
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise OSError("artifact changed while it was opened")
        return _digest_open_file(file_fd, opened, capture=capture, max_bytes=max_bytes)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _digest_open_file(
    file_fd: int,
    opened: os.stat_result,
    *,
    capture: bool,
    max_bytes: int | None,
) -> _ReadFile:
    digest = hashlib.sha256()
    chunks = [] if capture else None
    total = 0
    while True:
        chunk = os.read(file_fd, 1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if max_bytes is not None and total > max_bytes:
            raise OSError("identity file is too large")
        digest.update(chunk)
        if chunks is not None:
            chunks.append(chunk)
    finished = os.fstat(file_fd)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(opened, field) != getattr(finished, field)
        for field in stable_fields
    ):
        raise OSError("artifact changed while it was read")
    return _ReadFile(
        sha256=digest.hexdigest(),
        data=b"".join(chunks) if chunks is not None else None,
        size=total,
        identity=(opened.st_dev, opened.st_ino),
    )


def _read_regular_file_windows(
    root_fd: int,
    parts: tuple[str, ...],
    *,
    capture: bool,
    max_bytes: int | None,
) -> _ReadFile:
    """The same read, with each component opened relative to its parent handle."""
    from .. import _windows_files as windows

    directories: list[int] = []
    file_fd: int | None = None
    try:
        parent = windows.handle_of(root_fd)
        for part in parts[:-1]:
            parent, _ = windows.open_component(parent, part, directory=True)
            directories.append(parent)
        final, attributes = windows.open_component(parent, parts[-1], directory=False)
        file_fd = windows.to_descriptor(final)
        if windows.is_directory(attributes):
            raise OSError("artifact is not a regular file")
        return _digest_open_file(
            file_fd, os.fstat(file_fd), capture=capture, max_bytes=max_bytes
        )
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for directory in reversed(directories):
            windows.close(directory)


def _identity_at(root_fd: int, relative: str) -> tuple[int, int] | None:
    """(st_dev, st_ino) of the file now at ``relative``, None if there is none."""
    parts = _safe_relative_parts(relative)
    try:
        if os.name == "nt":
            from .. import _windows_files as windows

            info = os.stat(os.path.join(windows.final_path(root_fd), *parts), follow_symlinks=False)
        else:
            parent_fd = os.dup(root_fd)
            try:
                for part in parts[:-1]:
                    child_fd = _open_directory_at(parent_fd, part)
                    os.close(parent_fd)
                    parent_fd = child_fd
                info = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            finally:
                os.close(parent_fd)
    except OSError:
        return None
    return info.st_dev, info.st_ino


def _regular_files_at(root_fd: int) -> set[str]:
    """Enumerate one opened directory tree, rejecting every non-regular leaf."""
    if os.name == "nt":
        return _regular_files_windows(root_fd)
    files: set[str] = set()

    def walk(directory_fd: int, prefix: tuple[str, ...]) -> None:
        before = os.fstat(directory_fd)
        for name in sorted(os.listdir(directory_fd)):
            if name in {".", ".."} or "/" in name or "\\" in name:
                raise OSError("unsafe directory entry")
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = "/".join((*prefix, name))
            if stat.S_ISDIR(info.st_mode):
                child_fd = _open_directory_at(directory_fd, name)
                try:
                    walk(child_fd, (*prefix, name))
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(info.st_mode):
                files.add(relative)
            else:
                raise OSError("package contains a symlink or special file")
        after = os.fstat(directory_fd)
        if (before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise OSError("package directory changed during enumeration")

    walk(root_fd, ())
    return files


def _regular_files_windows(root_fd: int) -> set[str]:
    """The same enumeration by name: Windows cannot list an open directory descriptor.

    As in the native set walk, this decides only which names exist. Every byte
    read afterwards goes through the root handle.
    """
    from .. import _windows_files as windows

    files: set[str] = set()

    def walk(directory: str, prefix: tuple[str, ...]) -> None:
        before = os.stat(directory)
        with os.scandir(directory) as listing:
            entries = sorted(listing, key=lambda entry: entry.name)
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            if info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise OSError("package contains a symlink or special file")
            relative = "/".join((*prefix, entry.name))
            if stat.S_ISDIR(info.st_mode):
                walk(entry.path, (*prefix, entry.name))
            elif stat.S_ISREG(info.st_mode):
                files.add(relative)
            else:
                raise OSError("package contains a symlink or special file")
        after = os.stat(directory)
        if (before.st_mtime_ns, before.st_ctime_ns) != (after.st_mtime_ns, after.st_ctime_ns):
            raise OSError("package directory changed during enumeration")

    walk(windows.final_path(root_fd), ())
    return files


def _process_peak_rss_bytes() -> int:
    """The PROCESS high-water mark since start -- not any one operation's peak.

    Darwin reports bytes; Linux reports kilobytes. Callers must take a delta,
    and even the delta is only a hint: ru_maxrss is monotonic, so it reads 0 when
    the process already peaked higher, and it counts allocation by ANY thread
    during the window, which need not be the compile's. It is therefore neither a
    reliable upper nor lower bound in a multi-threaded process.

    A trustworthy figure requires compiling in a dedicated subprocess and reading
    RUSAGE_CHILDREN. Task 7's gate must do that; this delta is context only.
    """
    # Imported here, not at module scope: `resource` is Unix-only, and
    # compiler/__init__ imports this module to export verify_compile_receipt.
    # At module scope it made `import edllm.local_models` -- and therefore
    # `diffusorrt list` -- fail outright on Windows, for a function only the
    # compile path calls.
    if os.name == "nt":
        from .._windows_files import peak_working_set_bytes

        return peak_working_set_bytes()
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage if os.uname().sysname == "Darwin" else usage * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_receipt(
    package_dir: Path,
    *,
    model_id: str,
    revision: str,
    source_license: str,
    definition_key: str | None = None,
    definition_digest: str | None = None,
) -> None:
    """Record what this package IS, so a later cache hit can verify it.

    Without a receipt a cache hit could only check that model.json existed, so a
    corrupt-but-parseable weights file, a package missing artifacts, or output
    from an older converter were all silently reused -- the failure mode that
    produces wrong output rather than an error.
    """
    # EVERY artifact is hashed, not just the weights. Hashing weights.npz alone
    # left graph.json, model.json, and the tokenizer assets unprotected, so a
    # tampered graph passed validation and was served from cache.
    # Ordered by path components, case included, which is how POSIX orders
    # paths. Windows orders them without case, and the receipt's bytes are an
    # admission key: one package must not get an identity per platform.
    artifacts = {
        path.relative_to(package_dir).as_posix(): _sha256_file(path)
        for path in sorted(
            package_dir.rglob("*"),
            key=lambda path: path.relative_to(package_dir).parts,
        )
        if path.is_file() and path.name != RECEIPT_NAME
    }
    manifest = _strict_json_loads(
        (package_dir / "model.json").read_text(encoding="utf-8")
    )
    if type(manifest) is not dict:
        raise RuntimeError("conversion produced a non-object manifest")
    weights_ref = manifest.get("merged_weights_ref")
    expected_weights_hash = manifest.get("output_hash")
    actual_weights_hash = artifacts.get(weights_ref)
    if actual_weights_hash is None:
        raise RuntimeError(
            f"conversion produced no weights artifact {weights_ref!r}; "
            "refusing to publish"
        )
    if not expected_weights_hash or actual_weights_hash != expected_weights_hash:
        raise RuntimeError(
            "conversion produced a weights hash mismatch; refusing to publish"
        )
    receipt = {
        "receipt_version": 1,
        "compiler_version": COMPILER_VERSION,
        "model_id": model_id,
        "revision": revision,
        "source_license": source_license,
        "artifacts": artifacts,
    }
    local_identity = definition_key is not None or definition_digest is not None
    if local_identity:
        if (
            not isinstance(definition_key, str)
            or not definition_key
            or not isinstance(definition_digest, str)
            or len(definition_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in definition_digest
            )
        ):
            raise RuntimeError("compile_receipt_definition_identity_invalid")
        receipt.update(
            {
                "model_definition_key": definition_key,
                "model_definition_digest": definition_digest,
            }
        )
    descriptor = None if local_identity else _product_registry().get(model_id)
    if descriptor is not None:
        if descriptor.revision != revision:
            raise RuntimeError("compile_receipt_descriptor_revision_mismatch")
        receipt.update(
            {
                "model_descriptor_key": descriptor.key,
                "model_descriptor_digest": descriptor.descriptor_digest,
            }
        )
    (package_dir / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2), encoding="utf-8", newline="\n"
    )


def verify_compile_receipt_snapshot(
    package_dir: str | Path,
    *,
    model_id: str,
    revision: str,
    require_product_identity: bool = False,
    expected_precision: str | None = None,
    expected_backend: str | None = None,
    expected_definition_key: str | None = None,
    expected_definition_digest: str | None = None,
    _package_guard: PackageLease | None = None,
    _require_current_compiler: bool = True,
) -> VerifiedCompileReceiptSnapshot | None:
    """Verify one package while holding the shared lifecycle guard.

    Diagnostics may inspect an intact older package by disabling the current
    compiler check. Cache reuse and profile binding retain the strict default.
    """

    package_path = Path(package_dir).expanduser().absolute()
    if _package_guard is not None:
        if (
            not isinstance(_package_guard, PackageLease)
            or not _package_guard.active
            or _package_guard.package_dir
            != package_path.expanduser().resolve(strict=False)
        ):
            raise ValueError("package verification guard does not match package")
        return _verify_compile_receipt_snapshot_locked(
            package_path,
            model_id=model_id,
            revision=revision,
            require_product_identity=require_product_identity,
            expected_precision=expected_precision,
            expected_backend=expected_backend,
            expected_definition_key=expected_definition_key,
            expected_definition_digest=expected_definition_digest,
            require_current_compiler=_require_current_compiler,
        )

    lease = acquire_package_load_lease(package_dir)
    snapshot = None
    close_failed = False
    try:
        snapshot = _verify_compile_receipt_snapshot_locked(
            package_path,
            model_id=model_id,
            revision=revision,
            require_product_identity=require_product_identity,
            expected_precision=expected_precision,
            expected_backend=expected_backend,
            expected_definition_key=expected_definition_key,
            expected_definition_digest=expected_definition_digest,
            require_current_compiler=_require_current_compiler,
        )
    finally:
        try:
            lease.close()
        except Exception:
            close_failed = True
    return None if close_failed else snapshot


def _receipt_identity_locked(
    package_dir: str | Path,
) -> tuple[str, str, str | None, str | None] | None:
    """Read only the bounded receipt identity under an existing package guard."""

    root_fd: int | None = None
    try:
        root_fd = _open_package_root(Path(package_dir))
        receipt_file = _read_regular_file_at(
            root_fd,
            RECEIPT_NAME,
            capture=True,
            max_bytes=_MAX_IDENTITY_BYTES,
        )
        if receipt_file.data is None:
            return None
        receipt = _strict_json_loads(receipt_file.data)
        if type(receipt) is not dict or receipt.get("receipt_version") != 1:
            return None
        model_id = receipt.get("model_id")
        revision = receipt.get("revision")
        definition_key = receipt.get("model_definition_key")
        definition_digest = receipt.get("model_definition_digest")
        if (
            type(model_id) is not str
            or not model_id
            or type(revision) is not str
            or not revision
        ):
            return None
        if (definition_key is None) != (definition_digest is None):
            return None
        return model_id, revision, definition_key, definition_digest
    except Exception:
        return None
    finally:
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass


def verify_package_receipt_snapshot(
    package_dir: str | Path,
) -> VerifiedCompileReceiptSnapshot | None:
    """Verify a compiler-owned package without a caller-supplied model id.

    This is used when creating a package-bound run profile. It performs one
    complete artifact verification; routine profile resolution later compares
    only the small receipt digest.
    """

    package_path = Path(package_dir).expanduser().absolute()
    lease = acquire_package_load_lease(package_path)
    snapshot = None
    close_failed = False
    try:
        identity = _receipt_identity_locked(package_path)
        if identity is not None:
            model_id, revision, definition_key, definition_digest = identity
            snapshot = _verify_compile_receipt_snapshot_locked(
                package_path,
                model_id=model_id,
                revision=revision,
                expected_definition_key=definition_key,
                expected_definition_digest=definition_digest,
            )
    finally:
        try:
            lease.close()
        except Exception:
            close_failed = True
    return None if close_failed else snapshot


def compile_receipt_digest(
    package_dir: str | Path,
    *,
    _package_guard: PackageLease | None = None,
) -> str | None:
    """Hash the small receipt that transitively names every package artifact."""

    package_path = Path(package_dir).expanduser().absolute()
    lease = _package_guard
    owns_lease = lease is None
    if lease is None:
        lease = acquire_package_load_lease(package_path)
    elif (
        not isinstance(lease, PackageLease)
        or not lease.active
        or lease.package_dir != package_path.expanduser().resolve(strict=False)
    ):
        raise ValueError("package receipt guard does not match package")
    root_fd: int | None = None
    digest = None
    close_failed = False
    try:
        root_fd = _open_package_root(package_path)
        receipt = _read_regular_file_at(
            root_fd,
            RECEIPT_NAME,
            capture=False,
            max_bytes=_MAX_IDENTITY_BYTES,
        )
        digest = receipt.sha256
    except Exception:
        digest = None
    finally:
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                close_failed = True
        if owns_lease:
            try:
                lease.close()
            except OSError:
                close_failed = True
    return None if close_failed else digest


def _verify_compile_receipt_snapshot_locked(
    package_dir: str | Path,
    *,
    model_id: str,
    revision: str,
    require_product_identity: bool = False,
    expected_precision: str | None = None,
    expected_backend: str | None = None,
    expected_definition_key: str | None = None,
    expected_definition_digest: str | None = None,
    require_current_compiler: bool = True,
) -> VerifiedCompileReceiptSnapshot | None:
    """Return identity from the same safe file descriptors that were verified.

    The package, receipt, directories and artifacts are opened with
    ``O_NOFOLLOW``. Every artifact is a regular file, and the bytes retained for
    ``model.json`` are the exact bytes hashed against the receipt. Any doubt
    returns ``None``.
    """
    if type(require_product_identity) is not bool:
        return None
    if type(require_current_compiler) is not bool:
        return None
    if expected_precision is not None and (
        not isinstance(expected_precision, str) or not expected_precision
    ):
        return None
    if expected_backend is not None and (
        not isinstance(expected_backend, str) or not expected_backend
    ):
        return None
    if require_product_identity and (
        expected_precision is None or expected_backend is None
    ):
        return None
    local_identity = (
        expected_definition_key is not None
        or expected_definition_digest is not None
    )
    if local_identity and (
        require_product_identity
        or not isinstance(expected_definition_key, str)
        or not expected_definition_key
        or not isinstance(expected_definition_digest, str)
        or len(expected_definition_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_definition_digest
        )
    ):
        return None

    root_fd: int | None = None
    try:
        package_path = Path(package_dir)
        root_fd = _open_package_root(package_path)
    except Exception:
        return None

    try:
        snapshot = _verify_compile_receipt_snapshot_at(
            root_fd,
            package_path=package_path,
            model_id=model_id,
            revision=revision,
            require_product_identity=require_product_identity,
            expected_precision=expected_precision,
            expected_backend=expected_backend,
            expected_definition_key=expected_definition_key,
            expected_definition_digest=expected_definition_digest,
            require_current_compiler=require_current_compiler,
        )
    # Public compatibility promise: on any doubt, the legacy bool verifier
    # returns False. Receipt and manifest bytes are untrusted input, so an
    # unexpected parser/platform exception is doubt, not permission to reuse.
    except Exception:
        snapshot = None

    try:
        os.close(root_fd)
    except Exception:
        return None
    return snapshot


def _verify_compile_receipt_snapshot_at(
    root_fd: int,
    *,
    package_path: Path,
    model_id: str,
    revision: str,
    require_product_identity: bool,
    expected_precision: str | None,
    expected_backend: str | None,
    expected_definition_key: str | None,
    expected_definition_digest: str | None,
    require_current_compiler: bool,
) -> VerifiedCompileReceiptSnapshot | None:
    """Verify one already-open package root and retain its identity bytes."""
    root_before = os.fstat(root_fd)
    if not stat.S_ISDIR(root_before.st_mode):
        return None

    receipt_file = _read_regular_file_at(
        root_fd,
        RECEIPT_NAME,
        capture=True,
        max_bytes=_MAX_IDENTITY_BYTES,
    )
    if receipt_file.data is None:
        return None
    receipt = _strict_json_loads(receipt_file.data)
    if type(receipt) is not dict:
        return None
    receipt_version = receipt.get("receipt_version")
    if type(receipt_version) is not int or receipt_version != 1:
        return None
    compiler_version = receipt.get("compiler_version")
    if type(compiler_version) is not str or not compiler_version:
        return None
    if require_current_compiler and compiler_version != COMPILER_VERSION:
        return None
    if receipt.get("model_id") != model_id or receipt.get("revision") != revision:
        return None
    source_license = receipt.get("source_license")
    if not isinstance(source_license, str) or not source_license:
        return None

    local_identity = expected_definition_key is not None
    descriptor = None if local_identity else _product_registry().get(model_id)
    descriptor_key = receipt.get("model_descriptor_key")
    descriptor_digest = receipt.get("model_descriptor_digest")
    definition_key = receipt.get("model_definition_key")
    definition_digest = receipt.get("model_definition_digest")
    if local_identity:
        if descriptor_key is not None or descriptor_digest is not None:
            return None
        if (
            definition_key != expected_definition_key
            or definition_digest != expected_definition_digest
        ):
            return None
    elif definition_key is not None or definition_digest is not None:
        return None
    if descriptor is None:
        if descriptor_key is not None or descriptor_digest is not None:
            return None
        if require_product_identity:
            return None
    else:
        if descriptor.revision != revision:
            return None
        if descriptor_key != descriptor.key:
            return None
        if (
            type(descriptor_digest) is not str
            or len(descriptor_digest) != 64
            or any(character not in "0123456789abcdef" for character in descriptor_digest)
        ):
            return None
        # Diagnostic verification may inspect an intact historical package.
        # Cache reuse/loading still requires the current descriptor; the
        # receipt and manifest must bind the same historical digest below.
        if require_current_compiler and descriptor_digest != descriptor.descriptor_digest:
            return None
        if require_product_identity and source_license != descriptor.source_license:
            return None

    artifacts = receipt.get("artifacts")
    if type(artifacts) is not dict or not artifacts:
        return None
    for relative, digest in artifacts.items():
        _safe_relative_parts(relative)
        if (
            relative == RECEIPT_NAME
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return None
    present = _regular_files_at(root_fd)
    present.discard(RECEIPT_NAME)
    if set(artifacts) != present:
        return None

    manifest_file: _ReadFile | None = None
    package_bytes = receipt_file.size
    read_identities: dict[str, tuple[int, int]] = {}
    for relative, expected_digest in sorted(artifacts.items()):
        artifact = _read_regular_file_at(
            root_fd,
            relative,
            capture=relative == "model.json",
            max_bytes=(
                _MAX_IDENTITY_BYTES if relative == "model.json" else None
            ),
        )
        if artifact.sha256 != expected_digest:
            return None
        read_identities[relative] = artifact.identity
        package_bytes += artifact.size
        if relative == "model.json":
            manifest_file = artifact
    if manifest_file is None or manifest_file.data is None:
        return None
    manifest = _strict_json_loads(manifest_file.data)
    if type(manifest) is not dict:
        return None

    manifest_model_id = manifest.get("source_checkpoint_reference")
    manifest_revision = manifest.get("source_checkpoint_sha")
    manifest_source_license = manifest.get("source_license")
    manifest_descriptor_key = manifest.get("model_descriptor_key")
    manifest_descriptor_digest = manifest.get("model_descriptor_digest")
    manifest_definition_key = manifest.get("model_definition_key")
    manifest_definition_digest = manifest.get("model_definition_digest")
    precision = manifest.get("model_precision")
    backend = manifest.get("admitted_backend")
    manifest_source_identity = (
        manifest_model_id,
        manifest_revision,
        manifest_source_license,
    )
    present_source_identity = tuple(
        value is not None for value in manifest_source_identity
    )
    if any(present_source_identity) and not all(present_source_identity):
        return None
    if all(present_source_identity):
        if any(
            not isinstance(value, str) or not value
            for value in manifest_source_identity
        ):
            return None
        if (
            manifest_model_id != model_id
            or manifest_revision != revision
            or manifest_source_license != source_license
        ):
            return None
    elif require_product_identity:
        return None

    package_target = (precision, backend)
    present_package_target = tuple(value is not None for value in package_target)
    if any(present_package_target) and not all(present_package_target):
        return None
    if all(present_package_target) and any(
        not isinstance(value, str) or not value for value in package_target
    ):
        return None
    if require_product_identity and not all(present_package_target):
        return None

    manifest_descriptor_present = (
        manifest_descriptor_key is not None
        or manifest_descriptor_digest is not None
    )
    if manifest_descriptor_present:
        if (
            manifest_descriptor_key != descriptor_key
            or manifest_descriptor_digest != descriptor_digest
        ):
            return None
    elif descriptor is not None:
        return None

    if local_identity:
        if (
            manifest_definition_key != definition_key
            or manifest_definition_digest != definition_digest
            or precision is not None
            or backend is not None
        ):
            return None
        try:
            from ..model_definitions import (
                model_build_digest,
                model_definition_digest,
                model_definition_from_payload,
            )
            from ..weight_store import weight_artifact_spec_from_manifest

            package_definition = model_definition_from_payload(
                manifest["model_definition"]
            )
            weight_artifact_spec_from_manifest(manifest)
        except (KeyError, TypeError, ValueError):
            return None
        if (
            package_definition.key != definition_key
            or model_definition_digest(package_definition) != definition_digest
            or model_build_digest(package_definition)
            != manifest.get("model_build_digest")
        ):
            return None
    else:
        if manifest_definition_digest is not None:
            return None
        if (
            descriptor is not None
            and manifest_definition_key is not None
            and manifest_definition_key != descriptor.key
        ):
            return None

    if expected_precision is not None and precision != expected_precision:
        return None
    if expected_backend is not None and backend != expected_backend:
        return None

    # A second enumeration and root metadata comparison make a package-tree
    # replacement during verification fail closed, including an A -> B -> A
    # name swap that leaves the final file set looking unchanged.
    final_present = _regular_files_at(root_fd)
    final_present.discard(RECEIPT_NAME)
    if final_present != present:
        return None
    root_after = os.fstat(root_fd)
    for field in ("st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns"):
        if getattr(root_before, field) != getattr(root_after, field):
            return None
    # The folder's times cannot tell two changes in one clock tick apart (about
    # 15.6 ms on Windows, whose st_ctime is the creation time). Each file read is
    # still the file at its name: a replacement is another file, whenever it came.
    for relative, identity in read_identities.items():
        if _identity_at(root_fd, relative) != identity:
            return None

    return VerifiedCompileReceiptSnapshot(
        package_dir=package_path,
        compiler_version=compiler_version,
        model_id=model_id,
        revision=revision,
        source_license=source_license,
        descriptor_key=(descriptor_key if isinstance(descriptor_key, str) else None),
        descriptor_digest=(
            descriptor_digest if isinstance(descriptor_digest, str) else None
        ),
        definition_key=(
            definition_key if isinstance(definition_key, str) else None
        ),
        definition_digest=(
            definition_digest if isinstance(definition_digest, str) else None
        ),
        manifest_model_id=(
            manifest_model_id if isinstance(manifest_model_id, str) else None
        ),
        manifest_revision=(
            manifest_revision if isinstance(manifest_revision, str) else None
        ),
        manifest_source_license=(
            manifest_source_license
            if isinstance(manifest_source_license, str)
            else None
        ),
        manifest_descriptor_key=(
            manifest_descriptor_key
            if isinstance(manifest_descriptor_key, str)
            else None
        ),
        manifest_descriptor_digest=(
            manifest_descriptor_digest
            if isinstance(manifest_descriptor_digest, str)
            else None
        ),
        manifest_definition_key=(
            manifest_definition_key
            if isinstance(manifest_definition_key, str)
            else None
        ),
        manifest_definition_digest=(
            manifest_definition_digest
            if isinstance(manifest_definition_digest, str)
            else None
        ),
        manifest_sha256=manifest_file.sha256,
        receipt_sha256=receipt_file.sha256,
        model_precision=precision if isinstance(precision, str) else None,
        admitted_backend=backend if isinstance(backend, str) else None,
        package_st_dev=root_before.st_dev,
        package_st_ino=root_before.st_ino,
        package_bytes=package_bytes,
    )


def _receipt_is_valid(
    package_dir: Path,
    *,
    model_id: str,
    revision: str,
    expected_definition_key: str | None = None,
    expected_definition_digest: str | None = None,
) -> bool:
    """True only if this package is intact AND was built by this compiler."""
    return verify_compile_receipt_snapshot(
        package_dir,
        model_id=model_id,
        revision=revision,
        expected_definition_key=expected_definition_key,
        expected_definition_digest=expected_definition_digest,
    ) is not None


def _receipt_is_valid_locked(
    package_dir: Path,
    *,
    model_id: str,
    revision: str,
    expected_definition_key: str | None = None,
    expected_definition_digest: str | None = None,
) -> bool:
    """Validate while the caller already owns this package's guard."""

    return _verify_compile_receipt_snapshot_locked(
        package_dir,
        model_id=model_id,
        revision=revision,
        expected_definition_key=expected_definition_key,
        expected_definition_digest=expected_definition_digest,
    ) is not None


def verify_compile_receipt(
    package_dir: str | Path,
    *,
    model_id: str,
    revision: str,
) -> bool:
    """Public: was this package built by this compiler, and is it still intact?

    The client surface (`diffusorrt list --verify`) needs this answer, and the
    only correct implementation already exists as ``_receipt_is_valid``. This
    is a name and a type coercion over it, deliberately not a reimplementation:
    a second copy of the artifact-set and digest comparison would be a second
    thing to keep in sync with the receipt format, and the copy that drifted
    would report a corrupt package as good.

    Returns False on any doubt, including a missing or unreadable receipt.
    """
    return _receipt_is_valid(
        Path(package_dir),
        model_id=model_id,
        revision=revision,
    )


def _directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _require(module: str) -> None:
    """Import a compile-path dependency, or name the extra that provides it."""
    try:
        __import__(module)
    except ImportError as exc:
        from .. import _extra_install_command

        raise CompilerDependencyError(
            f"{module!r} is required to compile a model but is not installed. "
            f"Install the compiler extra: {_extra_install_command('compiler')}"
        ) from exc


def _publish_directory(staging: Path, target: Path) -> None:
    if os.name == "nt":
        from .._windows_files import retry_while_busy

        retry_while_busy(lambda: os.rename(staging, target))
        return
    os.rename(staging, target)


def _replace_package_directory(staging: Path, target: Path, *, attempts: int = 100) -> None:
    """Put ``staging`` where an invalid package is, keeping it if that fails.

    Windows cannot delete a directory with an open file, so deleting first could
    leave half a package and nothing to replace it. Moving it aside either works
    whole or changes nothing, and a leftover is reclaimed as a dead partial.
    """
    if os.name != "nt":
        shutil.rmtree(target, ignore_errors=True)
        os.rename(staging, target)
        return
    from .._windows_files import retry_while_busy

    def publish() -> None:
        retry_while_busy(lambda: os.rename(staging, target), attempts=attempts)

    if not os.path.lexists(target):
        # Nothing is in the way, so the publish failed for a reason of its own
        # (a scanner holding the new files, say): let it report that reason.
        publish()
        return
    # Unique, like a staging name: what an earlier replacement in this process
    # could not delete must not be in the way of this one.
    aside = target.with_name(
        f"{target.name}.partial.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    )
    retry_while_busy(lambda: os.rename(target, aside), attempts=attempts)
    try:
        publish()
    except BaseException:
        os.rename(aside, target)
        raise
    shutil.rmtree(aside, ignore_errors=True)


def _partial_owner_is_alive(name: str) -> bool:
    """Is the process that created this `.partial.<pid>.<hex>` still running?

    The wrong answer in one direction leaves bytes on disk; in the other it
    deletes a live compile's working directory. So the uncertain cases -- an
    unparseable name, a pid owned by another user -- are treated as ALIVE, and
    the directory stays. Bytes are recoverable; a clobbered concurrent compile
    is a corrupted package.

    Note pid reuse can make a dead owner look alive. That is the harmless
    direction: the sweep skips a directory it could have removed.
    """
    parts = name.split(".partial.")
    if len(parts) != 2:
        return True
    fields = parts[1].split(".")
    if not fields or not fields[0].isdigit():
        return True
    pid = int(fields[0])
    if os.name == "nt":
        from .._windows_files import process_is_alive

        return process_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # PermissionError included: the pid exists and belongs to another user.
        # That is a LIVE process, and the answer is the same as any other
        # unreadable case -- keep the directory.
        return True
    return True


def sweep_stale_partials(cache_root: Path) -> int:
    """Remove staging directories left by compiles that died, return the bytes.

    A compile that is interrupted cleans up after itself; one that is SIGKILLed
    or loses power cannot. Those directories are invisible to `list` -- they
    hold no receipt and are not packages -- so without a sweep a user sees a
    28 GiB cache with no way to tell which 14 GiB is garbage.
    """
    if not cache_root.is_dir():
        return 0
    reclaimed = 0
    for candidate in cache_root.iterdir():
        if ".partial." not in candidate.name or not candidate.is_dir():
            continue
        if _partial_owner_is_alive(candidate.name):
            continue
        reclaimed += _directory_bytes(candidate)
        shutil.rmtree(candidate, ignore_errors=True)
    return reclaimed


def _acquire_snapshot(
    *, model_id: str, revision: str, offline: bool = False
) -> Path:
    """Resolve the pinned revision in the Hugging Face commit-addressed cache.

    ONE implementation. Online and offline differ only in ``local_files_only``;
    there is no parallel acquisition path and no ``local_dir``.

    ``local_dir`` was the whole problem. Measured on huggingface_hub 0.36.2:

        no local_dir            -> the cache path for the exact revision
        empty local_dir         -> LocalEntryNotFoundError, even when cached
        local_dir with junk.txt -> that directory, returned verbatim

    So a directory argument makes the result unverified: the hub hands back
    whatever is there when it cannot reach the remote. The online path used to
    do exactly that and then check the result "looks like a snapshot", which
    accepts a complete snapshot of the WRONG revision.

    Identity now comes from the cache layout itself: hub stores a revision at
    ``.../snapshots/<full sha>``, so the directory name IS the commit. That is
    a proof, not a heuristic.

    ``dest`` is accepted and ignored; nothing is copied into the DiffusorRT
    cache any more.
    """
    _require("huggingface_hub")
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import (
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        acquired = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_files_only=offline,
            )
        )
    except LocalEntryNotFoundError as exc:
        # hub raises this for BOTH "offline and not cached" and "online but the
        # network was unreachable". Reporting the offline message online sent a
        # user with a dropped connection hunting for a cache problem.
        if offline:
            raise SnapshotNotCachedError(
                f"{model_id} revision {revision} is not in the local Hugging "
                "Face cache, and offline mode will not download it. Fetch that "
                "exact revision first, for example:\n"
                f"    huggingface-cli download {model_id} --revision {revision}\n"
                "then re-run this command."
            ) from exc
        raise ModelAccessError(
            f"{model_id} revision {revision} could not be fetched and is not "
            "cached. The Hugging Face endpoint was unreachable; check the "
            "network, a proxy, or HF_ENDPOINT."
        ) from exc
    except GatedRepoError as exc:
        raise ModelAccessError(
            f"{model_id} is gated. Accept its terms at "
            f"https://huggingface.co/{model_id} and provide a token "
            "(huggingface-cli login, or HF_TOKEN)."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise ModelAccessError(
            f"{model_id} was not found, or the token in use cannot see it. "
            "If it is private, log in with an account that has access."
        ) from exc
    except RevisionNotFoundError as exc:
        # The repo exists and the pinned commit does not. Distinct from a
        # missing repo: the registry pin is the thing that is wrong.
        raise ModelAccessError(
            f"{model_id} exists but has no revision {revision}. The registry "
            "pin names a commit this repository does not contain."
        ) from exc
    except HfHubHTTPError as exc:
        # Everything else the hub reports as HTTP -- rate limits, 5xx, auth
        # failures that are not gating. These escaped CompilerError entirely
        # and reached the user as a traceback.
        raise ModelAccessError(
            f"the Hugging Face API refused the request for {model_id}: {exc}"
        ) from exc

    _verify_snapshot_identity(acquired, model_id=model_id, revision=revision)
    return acquired


def _verify_snapshot_identity(path: Path, *, model_id: str, revision: str) -> None:
    """The resolved directory must BE the pinned commit.

    huggingface_hub lays a repo out as ``models--<org>--<name>/snapshots/<sha>``,
    so the leaf name is the commit id. Checking it turns "this looks like a
    snapshot" into "this is that revision" -- the difference between a
    structural guess and provenance the receipt can honestly record.
    """
    if path.name != revision:
        raise ModelAccessError(
            f"acquisition for {model_id} returned {path}, whose directory name "
            f"is not the pinned revision {revision}. The Hugging Face cache "
            "stores a revision at snapshots/<commit>, so a different name means "
            "the bytes were not proven to be that commit; refusing to record it "
            "as that revision's provenance."
        )
    if not (path / "config.json").is_file():
        raise ModelAccessError(
            f"acquisition for {model_id} returned {path}, which has no "
            "config.json. The revision resolved but its files are not present; "
            "re-fetch it before compiling."
        )


def cached_snapshot_path(model_id: str, *, revision: str) -> Path:
    """Public: where the pinned revision sits in the local Hugging Face cache.

    Raises SnapshotNotCachedError when it is not there. Exposed so the client
    can ask this question without reaching into a private compiler symbol --
    one lookup, so a pre-flight and the compile cannot disagree.
    """
    return _acquire_snapshot(model_id=model_id, revision=revision, offline=True)


def _verify_declared_tokenizer_module(staging: Path, *, entry) -> None:
    """The staging package must declare the tokenizer module the registry named.

    registry.custom_tokenizer_module is a CLAIM used to obtain consent before
    anything is fetched -- it has to be, because cached snapshot files are
    symlinks into blobs/ and the package artifact opener refuses symlinks. This
    is where the claim is CHECKED, against a real package, before the
    production load imports that module.

    A mismatch means the user consented to one file and a different one is
    about to run, so it refuses rather than warns.
    """
    # Lazy: edllm.tokenizer pulls `tokenizers`, and this module is reachable
    # from `diffusorrt list`.
    from ..tokenizer import package_declares_custom_tokenizer_code

    declared = package_declares_custom_tokenizer_code(staging)
    # STRICT equality, including None on both sides. The earlier asymmetric
    # rule ("declaring nothing is safe") was correct about safety and wrong
    # about honesty: it let the registry claim tokenization_dream.py for a
    # package that declares nothing, so `doctor` reported trust_required=true
    # and the consent prompt asked about a file that would never be imported.
    # A trust claim nobody checks is the thing this function exists to prevent.
    if declared != entry.custom_tokenizer_module:
        raise RemoteCodeNotTrustedError(
            f"{entry.model_id} was recorded as executing "
            f"{entry.custom_tokenizer_module!r}, but the converted package "
            f"declares {declared!r}. Consent was obtained for the recorded "
            "module; refusing to load a different one."
        )


def _convert_snapshot(snapshot: Path, out_dir: Path, **provenance) -> Path:
    """Convert a snapshot into an execution package. Stubbed in tests."""
    _require("safetensors")
    _require("torch")
    from ..converter import build_package_from_hf_masked_diffusion

    return build_package_from_hf_masked_diffusion(snapshot, out_dir, **provenance)


def _validate_staging_package(
    package_dir: Path,
    *,
    trust_remote_code: bool,
    catalog: ModelCatalog | None = None,
) -> None:
    """Exercise the production load contract before a package becomes reusable.

    This checks that the package LOADS. Byte-integrity is checked separately, in
    ``_write_receipt``, which compares the weights artifact against the manifest
    ``output_hash`` using hashes it must compute for the receipt anyway. Passing
    ``verify_weights_hash=True`` here would give the same guarantee at the cost
    of a second full hash pass over a multi-GB package, so it is deliberately
    NOT set — the two checks are complementary, not redundant: this one catches
    a package that cannot load, that one catches a package that loads wrong.

    ``trust_remote_code`` is passed through rather than hardcoded. The caller is
    refused earlier when trust is withheld, so a hardcoded ``True`` is
    unreachable today — but this is the point where model code executes, and
    "always trust" is not a default to leave one refactor away from reachable.
    """
    from ..model import load_runtime

    try:
        runtime = load_runtime(
            package_dir,
            trust_remote_code=trust_remote_code,
            catalog=catalog,
        )
        close = getattr(runtime, "close", None)
        if callable(close):
            close()
    except Exception as exc:
        # Name the cause. "invalid package" alone sends an operator to read a
        # multi-GB directory; the underlying error says which contract broke.
        detail = str(exc) or type(exc).__name__
        raise RuntimeError(
            f"conversion produced an invalid execution package in {package_dir}; "
            f"refusing to publish ({detail})"
        ) from exc


def compile_pretrained(
    model_id: str,
    *,
    revision: str | None = None,
    cache_root=None,
    trust_remote_code: bool = False,
    offline: bool = False,
    on_stage=None,
    catalog: ModelCatalog | None = None,
) -> CompileResult:
    """Compile the pinned model into a loadable package.

    ``offline=True`` reads the pinned revision from the standard Hugging Face
    cache and never touches the network. A cache HIT is served before the
    snapshot is looked at, so a user who deleted the snapshot after compiling
    still gets their package. Supplying ``catalog`` selects that exact Local
    scope; omitting it preserves the Product registry.

    Publishes by atomic rename only after conversion, production loading, and
    artifact-integrity checks succeed, so a failure leaves no directory a later
    ``from_package()`` would load. That is the worst failure available here: a
    truncated weights file loads and produces silently wrong output instead of
    an error.
    """
    # Budget is NOT enforced yet: a cache hit downloads and writes nothing, and
    # refusing it on a disk that has since filled up fails exactly when the cache
    # is most useful.
    emit = on_stage or (lambda _stage: None)
    plan = plan_pretrained(
        model_id,
        revision=revision,
        cache_root=cache_root,
        offline=offline,
        catalog=catalog,
    )
    entry = resolve_compile_entry(model_id, catalog)
    local_definition = getattr(entry, "definition", None) if catalog is not None else None
    local_definition_key = (
        str(getattr(local_definition, "key", "") or "")
        if local_definition is not None
        else None
    )
    local_definition_digest = (
        str(getattr(entry, "definition_digest", "") or "")
        if catalog is not None
        else None
    )
    if catalog is not None and (
        local_definition is None
        or not local_definition_key
        or len(local_definition_digest or "") != 64
    ):
        raise TypeError("Local catalog entry has no valid ModelDefinition identity")
    started = time.perf_counter()
    emit("cache_verification")
    rss_before = _process_peak_rss_bytes()

    if plan.already_compiled:
        read_lease = acquire_package_load_lease(plan.target_package_dir)
        try:
            if _receipt_is_valid_locked(
                plan.target_package_dir,
                model_id=model_id,
                revision=plan.revision,
                expected_definition_key=local_definition_key,
                expected_definition_digest=local_definition_digest,
            ):
                measured_bytes = _directory_bytes(plan.target_package_dir)
                emit("ready")
                return CompileResult(
                    plan=plan,
                    package_dir=plan.target_package_dir,
                    reused_cache=True,
                    measured_package_bytes=measured_bytes,
                    measured_peak_rss_delta_bytes=max(
                        0, _process_peak_rss_bytes() - rss_before
                    ),
                    wall_seconds=round(time.perf_counter() - started, 3),
                )
        finally:
            read_lease.close()

    # A writer serialises acquisition, conversion and publication with live
    # readers, removal, and another prepare.  Recheck inside the writer lock:
    # another process may have published after our initial plan/verification.
    writer = acquire_package_write_guard(plan.target_package_dir)
    staging = None
    try:
        if plan.target_package_dir.is_dir() and _receipt_is_valid_locked(
            plan.target_package_dir,
            model_id=model_id,
            revision=plan.revision,
            expected_definition_key=local_definition_key,
            expected_definition_digest=local_definition_digest,
        ):
            measured_bytes = _directory_bytes(plan.target_package_dir)
            emit("ready")
            return CompileResult(
                plan=plan,
                package_dir=plan.target_package_dir,
                reused_cache=True,
                measured_package_bytes=measured_bytes,
                measured_peak_rss_delta_bytes=max(
                    0, _process_peak_rss_bytes() - rss_before
                ),
                wall_seconds=round(time.perf_counter() - started, 3),
            )

        # Only a real build spends the budget or executes tokenizer code.
        if not plan.fits:
            plan_pretrained(
                model_id,
                revision=revision,
                cache_root=cache_root,
                require_fit=True,
                offline=offline,
                catalog=catalog,
            )
        if entry.custom_tokenizer_module is not None and not trust_remote_code:
            raise RemoteCodeNotTrustedError(
                f"compiling {model_id} loads its tokenizer, which executes "
                "repository-authored code from the model repo. Pass "
                "trust_remote_code=True to permit that, having satisfied yourself "
                "the repository is one you trust."
            )

        plan.cache_root.mkdir(parents=True, exist_ok=True)
        sweep_stale_partials(plan.cache_root)
        staging = plan.target_package_dir.with_name(
            f"{plan.target_package_dir.name}.partial.{os.getpid()}."
            f"{uuid.uuid4().hex[:8]}"
        )
        emit("acquisition")
        snapshot = _acquire_snapshot(
            model_id=model_id,
            revision=entry.revision,
            offline=offline,
        )
        emit("conversion")
        conversion_identity = (
            {
                "model_definition": local_definition,
                "model_definition_digest": local_definition_digest,
            }
            if catalog is not None
            else {"candidate_key": entry.key}
        )
        _convert_snapshot(
            snapshot,
            staging,
            **conversion_identity,
            source_ref=model_id,
            source_sha=entry.revision,
            source_license=entry.source_license,
        )
        if not (staging / "model.json").is_file():
            raise RuntimeError(
                f"conversion produced no manifest in {staging}; refusing to publish"
            )
        emit("validation")
        _verify_declared_tokenizer_module(staging, entry=entry)
        _validate_staging_package(
            staging,
            trust_remote_code=trust_remote_code,
            catalog=catalog,
        )
        _write_receipt(
            staging,
            model_id=model_id,
            revision=entry.revision,
            source_license=entry.source_license,
            definition_key=local_definition_key,
            definition_digest=local_definition_digest,
        )
        emit("publication")
        adopted = False
        try:
            _publish_directory(staging, plan.target_package_dir)
        except OSError:
            if _receipt_is_valid_locked(
                plan.target_package_dir,
                model_id=model_id,
                revision=entry.revision,
                expected_definition_key=local_definition_key,
                expected_definition_digest=local_definition_digest,
            ):
                # A legacy/uncooperative publisher won outside this guard.
                adopted = True
            else:
                _replace_package_directory(staging, plan.target_package_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            staging = None

        measured_bytes = _directory_bytes(plan.target_package_dir)
        emit("ready")
        return CompileResult(
            plan=plan,
            package_dir=plan.target_package_dir,
            reused_cache=adopted,
            measured_package_bytes=measured_bytes,
            measured_peak_rss_delta_bytes=max(
                0, _process_peak_rss_bytes() - rss_before
            ),
            wall_seconds=round(time.perf_counter() - started, 3),
        )
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        writer.close()
