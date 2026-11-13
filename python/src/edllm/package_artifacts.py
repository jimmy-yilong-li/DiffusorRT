"""Fail-closed access to regular files owned by an execution package."""
from __future__ import annotations

import errno
import hashlib
import os
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator

SHA256_CHUNK_BYTES = 1024 * 1024
OPEN_SUPPORTS_DIR_FD = os.open in getattr(os, "supports_dir_fd", ())
STAT_SUPPORTS_DIR_FD = os.stat in getattr(os, "supports_dir_fd", ())
STAT_SUPPORTS_NOFOLLOW = os.stat in getattr(os, "supports_follow_symlinks", ())


def _error_name(error_prefix: str, artifact_name: str, suffix: str) -> str:
    return "_".join(part for part in (error_prefix, artifact_name, suffix) if part)


def safe_relative_artifact_ref(
    artifact_ref: Any,
    *,
    error_prefix: str,
    artifact_name: str,
) -> Path:
    ref_text = str(artifact_ref)
    normalized_text = (
        ref_text.replace(os.altsep, os.sep) if os.altsep is not None else ref_text
    )
    raw_parts = normalized_text.split(os.sep)
    ref_path = Path(ref_text)
    if (
        not ref_text
        or "\x00" in ref_text
        or ref_path.is_absolute()
        or bool(ref_path.drive)
        or not ref_path.parts
        or any(part in {"", ".", ".."} for part in raw_parts)
        or ".." in ref_path.parts
        # On Windows a colon names an alternate data stream of the same file.
        or (os.name == "nt" and ":" in ref_text)
    ):
        raise ValueError(_error_name(error_prefix, artifact_name, "ref_unsafe"))
    return ref_path


def _component_error(
    exc: OSError,
    *,
    error_prefix: str,
    artifact_name: str,
    intermediate: bool,
) -> ValueError:
    if isinstance(exc, FileNotFoundError) or exc.errno == errno.ENOENT:
        return ValueError(_error_name(error_prefix, artifact_name, "artifact_missing"))
    if exc.errno in {errno.ELOOP, getattr(errno, "EMLINK", errno.ELOOP)}:
        return ValueError(_error_name(error_prefix, artifact_name, "symlink_forbidden"))
    if intermediate and exc.errno == errno.ENOTDIR:
        return ValueError(_error_name(error_prefix, artifact_name, "ref_unsafe"))
    return ValueError(_error_name(error_prefix, artifact_name, "artifact_unavailable"))


@contextmanager
def _open_package_artifact_windows(
    *,
    package_path: str | os.PathLike,
    ref_path: Path,
    error_prefix: str,
    artifact_name: str,
    max_bytes: int | None,
) -> Iterator[tuple[BinaryIO, Path, os.stat_result]]:
    """The Windows branch of open_package_artifact: same refusals, same names."""
    from . import _windows_files as windows

    try:
        root = windows.open_root(package_path)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(_error_name(error_prefix, "package", "dir_unavailable")) from exc

    def open_part(parent: int, name: str, *, directory: bool) -> tuple[int, int]:
        try:
            return windows.open_component(parent, name, directory=directory)
        except windows.LinkRefused as exc:
            raise ValueError(
                _error_name(error_prefix, artifact_name, "symlink_forbidden")
            ) from exc
        except OSError as exc:
            raise _component_error(
                exc,
                error_prefix=error_prefix,
                artifact_name=artifact_name,
                intermediate=directory,
            ) from exc

    with ExitStack() as directory_stack:
        directory_stack.callback(windows.close, root)
        current = root
        for part in ref_path.parts[:-1]:
            current, _ = open_part(current, part, directory=True)
            directory_stack.callback(windows.close, current)
        final, attributes = open_part(current, ref_path.parts[-1], directory=False)
        try:
            descriptor = windows.to_descriptor(final)
        except OSError as exc:
            raise ValueError(
                _error_name(error_prefix, artifact_name, "artifact_unavailable")
            ) from exc
        try:
            if windows.is_directory(attributes):
                raise ValueError(_error_name(error_prefix, artifact_name, "not_regular_file"))
            opened_stat = os.fstat(descriptor)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise ValueError(_error_name(error_prefix, artifact_name, "not_regular_file"))
            if max_bytes is not None and opened_stat.st_size > max_bytes:
                raise ValueError(_error_name(error_prefix, artifact_name, "too_large"))
            with os.fdopen(descriptor, "rb", closefd=True) as handle:
                descriptor = -1
                yield handle, Path(package_path).absolute().joinpath(*ref_path.parts), opened_stat
        finally:
            if descriptor >= 0:
                os.close(descriptor)


@contextmanager
def open_package_artifact(
    *,
    package_path: str | os.PathLike,
    artifact_ref: Any,
    error_prefix: str,
    artifact_name: str,
    max_bytes: int | None = None,
    open_supports_dir_fd: bool = OPEN_SUPPORTS_DIR_FD,
    stat_supports_dir_fd: bool = STAT_SUPPORTS_DIR_FD,
    stat_supports_nofollow: bool = STAT_SUPPORTS_NOFOLLOW,
    windows_safe_open: bool = os.name == "nt",
) -> Iterator[tuple[BinaryIO, Path, os.stat_result]]:
    """Open one package-relative regular file without following internal links."""
    ref_path = safe_relative_artifact_ref(
        artifact_ref,
        error_prefix=error_prefix,
        artifact_name=artifact_name,
    )
    if type(max_bytes) is not int and max_bytes is not None:
        raise TypeError("max_bytes must be a non-negative integer or None")
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer or None")
    if windows_safe_open:
        with _open_package_artifact_windows(
            package_path=package_path,
            ref_path=ref_path,
            error_prefix=error_prefix,
            artifact_name=artifact_name,
            max_bytes=max_bytes,
        ) as opened:
            yield opened
        return
    if not (
        open_supports_dir_fd
        and stat_supports_dir_fd
        and stat_supports_nofollow
        and getattr(os, "O_DIRECTORY", 0)
        and getattr(os, "O_NOFOLLOW", 0)
        and getattr(os, "O_NONBLOCK", 0)
    ):
        raise ValueError(_error_name(error_prefix, artifact_name, "safe_open_unsupported"))

    try:
        package_root = Path(package_path).resolve(strict=True)
        root_path_stat = package_root.stat()
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(_error_name(error_prefix, "package", "dir_unavailable")) from exc

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
    )
    final_flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
    )
    with ExitStack() as directory_stack:
        try:
            root_fd = os.open(package_root, directory_flags)
        except OSError as exc:
            raise ValueError(
                _error_name(error_prefix, "package", "dir_unavailable")
            ) from exc
        directory_stack.callback(os.close, root_fd)
        opened_root_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(opened_root_stat.st_mode):
            raise ValueError(_error_name(error_prefix, "package", "dir_unavailable"))
        if (opened_root_stat.st_dev, opened_root_stat.st_ino) != (
            root_path_stat.st_dev,
            root_path_stat.st_ino,
        ):
            raise ValueError(_error_name(error_prefix, "package", "dir_unavailable"))

        current_fd = root_fd
        for part in ref_path.parts[:-1]:
            try:
                component_stat = os.stat(
                    part, dir_fd=current_fd, follow_symlinks=False
                )
            except OSError as exc:
                raise _component_error(
                    exc,
                    error_prefix=error_prefix,
                    artifact_name=artifact_name,
                    intermediate=True,
                ) from exc
            if stat.S_ISLNK(component_stat.st_mode):
                raise ValueError(
                    _error_name(error_prefix, artifact_name, "symlink_forbidden")
                )
            if not stat.S_ISDIR(component_stat.st_mode):
                raise ValueError(
                    _error_name(error_prefix, artifact_name, "ref_unsafe")
                )
            try:
                next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            except OSError as exc:
                raise _component_error(
                    exc,
                    error_prefix=error_prefix,
                    artifact_name=artifact_name,
                    intermediate=True,
                ) from exc
            directory_stack.callback(os.close, next_fd)
            opened_stat = os.fstat(next_fd)
            if not stat.S_ISDIR(opened_stat.st_mode):
                raise ValueError(
                    _error_name(error_prefix, artifact_name, "ref_unsafe")
                )
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                component_stat.st_dev,
                component_stat.st_ino,
            ):
                raise ValueError(
                    _error_name(error_prefix, artifact_name, "artifact_changed")
                )
            current_fd = next_fd

        final_name = ref_path.parts[-1]
        try:
            path_stat = os.stat(final_name, dir_fd=current_fd, follow_symlinks=False)
        except OSError as exc:
            raise _component_error(
                exc,
                error_prefix=error_prefix,
                artifact_name=artifact_name,
                intermediate=False,
            ) from exc
        if stat.S_ISLNK(path_stat.st_mode):
            raise ValueError(
                _error_name(error_prefix, artifact_name, "symlink_forbidden")
            )
        if not stat.S_ISREG(path_stat.st_mode):
            raise ValueError(
                _error_name(error_prefix, artifact_name, "not_regular_file")
            )
        if max_bytes is not None and path_stat.st_size > max_bytes:
            raise ValueError(_error_name(error_prefix, artifact_name, "too_large"))

        final_fd = -1
        try:
            try:
                final_fd = os.open(final_name, final_flags, dir_fd=current_fd)
            except OSError as exc:
                raise _component_error(
                    exc,
                    error_prefix=error_prefix,
                    artifact_name=artifact_name,
                    intermediate=False,
                ) from exc
            opened_stat = os.fstat(final_fd)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise ValueError(
                    _error_name(error_prefix, artifact_name, "not_regular_file")
                )
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                path_stat.st_dev,
                path_stat.st_ino,
            ):
                raise ValueError(
                    _error_name(error_prefix, artifact_name, "artifact_changed")
                )
            if max_bytes is not None and opened_stat.st_size > max_bytes:
                raise ValueError(_error_name(error_prefix, artifact_name, "too_large"))

            with os.fdopen(final_fd, "rb", closefd=True) as handle:
                final_fd = -1
                display_path = package_root.joinpath(*ref_path.parts)
                yield handle, display_path, opened_stat
        finally:
            if final_fd >= 0:
                os.close(final_fd)


def artifact_source_stat(source_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
    )


def sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(SHA256_CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_bounded_stream(
    handle: BinaryIO,
    *,
    max_bytes: int,
    error_prefix: str = "package",
    artifact_name: str = "artifact",
) -> str:
    if type(max_bytes) is not int:
        raise TypeError("max_bytes must be a non-negative integer")
    if max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    digest = hashlib.sha256()
    remaining = max_bytes
    while True:
        chunk = handle.read(min(SHA256_CHUNK_BYTES, remaining + 1))
        if not chunk:
            return digest.hexdigest()
        if len(chunk) > remaining:
            raise ValueError(_error_name(error_prefix, artifact_name, "too_large"))
        digest.update(chunk)
        remaining -= len(chunk)


def read_open_artifact(
    handle: BinaryIO,
    *,
    max_bytes: int,
    error_prefix: str = "package",
    artifact_name: str = "artifact",
) -> bytes:
    if type(max_bytes) is not int:
        raise TypeError("max_bytes must be a non-negative integer")
    if max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(_error_name(error_prefix, artifact_name, "too_large"))
    return payload
