"""Cross-process lifecycle guard for one execution-package path."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import stat
import sys
import threading


_GUARD_DIRECTORY = ".diffusorrt-package-guards"


class PackageBusyError(RuntimeError):
    """An exclusive package operation was refused while the package is live."""


def _canonical_package_path(package_dir: str | Path) -> Path:
    if not isinstance(package_dir, (str, Path)) or isinstance(package_dir, bool):
        raise TypeError("package_dir must be a path")
    if isinstance(package_dir, str) and not package_dir:
        raise ValueError("package_dir must not be empty")
    return Path(package_dir).expanduser().resolve(strict=False)


def _stable_user_home() -> Path:
    """Resolve the OS account home without consulting process cache settings."""

    if os.name == "nt":
        return Path.home().resolve(strict=False)
    try:
        import pwd

        home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    except (ImportError, KeyError) as exc:  # pragma: no cover - broken OS identity
        raise RuntimeError("package_guard_user_identity_unavailable") from exc
    if not home.is_absolute():  # pragma: no cover - malformed account database
        raise RuntimeError("package_guard_user_home_invalid")
    return home.resolve(strict=False)


def package_guard_path(package_dir: str | Path) -> Path:
    """Return the stable, user-writable guard outside the protected package.

    The guard location does not depend on the package parent being writable.
    That matters for prepared packages mounted read-only.  Every entry point
    maps the canonical package path into one fixed per-user namespace, so a
    repository load and a direct ``Engine.from_package`` use the same file even
    when their package-cache configuration differs.
    """

    package_path = _canonical_package_path(package_dir)
    # normcase is the identity on POSIX; on Windows one directory has many spellings.
    text = str(package_path)
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    elif text.startswith("\\\\?\\"):
        text = text[4:]
    identity = hashlib.sha256(os.fsencode(os.path.normcase(text))).hexdigest()
    if sys.platform == "darwin":
        cache_root = _stable_user_home() / "Library" / "Caches"
    elif os.name == "nt":
        # From the shell, for the reason given below: a lifecycle lock must not
        # depend on a setting that may differ between processes. No home
        # directory is involved, so a process without one can still lock.
        from ._windows_files import local_app_data

        cache_root = Path(local_app_data())
    else:
        # XDG_CACHE_HOME is an application preference and may legitimately
        # differ between processes. A lifecycle lock must not: the same OS user
        # and canonical package path always map to the same inode.
        cache_root = _stable_user_home() / ".cache"
    return cache_root / "diffusorrt" / _GUARD_DIRECTORY / f"{identity}.lock"


def _open_guard_file(package_dir: str | Path) -> tuple[Path, int]:
    guard_path = package_guard_path(package_dir)
    guard_directory = guard_path.parent
    try:
        guard_directory.mkdir(mode=0o700, parents=True)
        created_directory = True
    except FileExistsError:
        created_directory = False
    directory_stat = guard_directory.lstat()
    if not stat.S_ISDIR(directory_stat.st_mode) or stat.S_ISLNK(
        directory_stat.st_mode
    ):
        raise RuntimeError("package_guard_directory_unsafe")
    if created_directory and os.name != "nt":
        os.chmod(guard_directory, 0o700)

    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(guard_path, flags, 0o600)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError("package_guard_file_unsafe")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
    except BaseException:
        os.close(descriptor)
        raise
    return guard_path, descriptor


@dataclass(slots=True)
class PackageLease:
    """One held OS lock; closing the descriptor is the release protocol."""

    package_dir: Path
    guard_path: Path
    _descriptor: int | None = field(repr=False)
    _mutex: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def active(self) -> bool:
        return self._descriptor is not None

    def close(self) -> None:
        with self._mutex:
            descriptor = self._descriptor
            if descriptor is None:
                return
            self._descriptor = None
            os.close(descriptor)

    def __del__(self) -> None:
        # The checked owner is RuntimeHandle.close().  This fallback only keeps
        # an abandoned low-level handle from leaking its OS descriptor until
        # process exit; there is no caller left to receive a cleanup error.
        with suppress(Exception):
            self.close()


def _lock_guard(descriptor: int, *, exclusive: bool, blocking: bool) -> None:
    if os.name == "nt":
        from ._windows_files import lock_descriptor

        lock_descriptor(descriptor, exclusive=exclusive, blocking=blocking)
        return
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - unsupported platform
        raise RuntimeError("cross-process package locking is unavailable") from exc
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
        operation |= fcntl.LOCK_NB
    fcntl.flock(descriptor, operation)


def _acquire_package_guard(
    package_dir: str | Path,
    *,
    exclusive: bool,
    blocking: bool,
) -> PackageLease:
    package_path = _canonical_package_path(package_dir)
    guard_path, descriptor = _open_guard_file(package_path)
    try:
        try:
            _lock_guard(descriptor, exclusive=exclusive, blocking=blocking)
        except BlockingIOError:
            raise PackageBusyError(f"package_busy: {package_path}") from None
    except BaseException:
        os.close(descriptor)
        raise
    return PackageLease(
        package_dir=package_path,
        guard_path=guard_path,
        _descriptor=descriptor,
    )


def acquire_package_load_lease(
    package_dir: str | Path,
) -> PackageLease:
    """Acquire the shared side before any package bytes are read."""

    package_path = _canonical_package_path(package_dir)
    return _acquire_package_guard(
        package_path,
        exclusive=False,
        blocking=True,
    )


def acquire_package_write_guard(package_dir: str | Path) -> PackageLease:
    """Wait for exclusive ownership before building or publishing a package."""

    return _acquire_package_guard(
        package_dir,
        exclusive=True,
        blocking=True,
    )


def acquire_package_exclusive_guard(package_dir: str | Path) -> PackageLease:
    """Acquire exclusive ownership, refusing rather than waiting when busy."""

    return _acquire_package_guard(
        package_dir,
        exclusive=True,
        blocking=False,
    )


__all__ = [
    "PackageBusyError",
    "PackageLease",
    "acquire_package_exclusive_guard",
    "acquire_package_load_lease",
    "acquire_package_write_guard",
    "package_guard_path",
]
