"""Win32 counterparts of the POSIX no-follow, descriptor-relative opens and flock.

Mirrors ``native/core/artifact.cpp``: the package root is opened once, and each
component below it is opened relative to an open handle with
``FILE_OPEN_REPARSE_POINT``, so a symlink or junction is opened as itself and
refused rather than followed. Nothing below the root re-walks a path.

Imported only on Windows.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import errno
import msvcrt
import os
import time

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_ntdll = ctypes.WinDLL("ntdll")

_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_SYNCHRONIZE = 0x00100000
_FILE_LIST_DIRECTORY = 0x0001
_FILE_TRAVERSE = 0x0020
_FILE_READ_ATTRIBUTES = 0x0080
_FILE_GENERIC_READ = 0x00120089
_FILE_SHARE_ALL = 0x0007
_FILE_ATTRIBUTE_DIRECTORY = 0x0010
_FILE_ATTRIBUTE_NORMAL = 0x0080
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_OPEN = 1
_FILE_DIRECTORY_FILE = 0x00000001
_FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
_FILE_OPEN_REPARSE_POINT = 0x00200000
_FILE_ATTRIBUTE_TAG_INFO = 9
_LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
_LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
_ERROR_LOCK_VIOLATION = 33
_ERROR_IO_PENDING = 997
_MAX_NAME_BYTES = 0xFFFE
_LOCK_POLL_SECONDS = 0.01
_VOLUME_NAME_DOS = 0
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_INVALID_PARAMETER = 87
_STILL_ACTIVE = 259


class _UnicodeString(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.USHORT),
        ("MaximumLength", wintypes.USHORT),
        ("Buffer", ctypes.c_wchar_p),
    ]


class _ObjectAttributes(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.ULONG),
        ("RootDirectory", ctypes.c_void_p),
        ("ObjectName", ctypes.POINTER(_UnicodeString)),
        ("Attributes", wintypes.ULONG),
        ("SecurityDescriptor", ctypes.c_void_p),
        ("SecurityQualityOfService", ctypes.c_void_p),
    ]


class _IoStatusBlock(ctypes.Structure):
    _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]


class _AttributeTagInfo(ctypes.Structure):
    _fields_ = [("FileAttributes", wintypes.DWORD), ("ReparseTag", wintypes.DWORD)]


class _Guid(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _Overlapped(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", ctypes.c_void_p),
    ]


_kernel32.CreateFileW.restype = ctypes.c_void_p
_kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
]
_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
_kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
_kernel32.GetFileInformationByHandleEx.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]
_kernel32.LockFileEx.restype = wintypes.BOOL
_kernel32.LockFileEx.argtypes = [
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(_Overlapped),
]
_kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
_kernel32.GetFinalPathNameByHandleW.argtypes = [
    ctypes.c_void_p,
    wintypes.LPWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
]
_kernel32.GetCurrentProcess.restype = ctypes.c_void_p
_kernel32.OpenProcess.restype = ctypes.c_void_p
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.GetExitCodeProcess.restype = wintypes.BOOL
_kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
_kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
_kernel32.K32GetProcessMemoryInfo.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_ProcessMemoryCounters),
    wintypes.DWORD,
]
_ntdll.NtCreateFile.restype = ctypes.c_long
_ntdll.NtCreateFile.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    wintypes.ULONG,
    ctypes.POINTER(_ObjectAttributes),
    ctypes.POINTER(_IoStatusBlock),
    ctypes.c_void_p,
    wintypes.ULONG,
    wintypes.ULONG,
    wintypes.ULONG,
    wintypes.ULONG,
    ctypes.c_void_p,
    wintypes.ULONG,
]
_ntdll.RtlNtStatusToDosError.restype = wintypes.ULONG
_ntdll.RtlNtStatusToDosError.argtypes = [wintypes.ULONG]


class LinkRefused(OSError):
    """A package component is a symlink, junction or other reparse point."""


def extended_path(path: str | os.PathLike) -> str:
    text = os.path.abspath(os.fspath(path))
    if text.startswith(("\\\\?\\", "\\\\.\\")):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text[2:]
    return "\\\\?\\" + text


def _attributes(handle: int) -> int:
    info = _AttributeTagInfo()
    if not _kernel32.GetFileInformationByHandleEx(
        handle, _FILE_ATTRIBUTE_TAG_INFO, ctypes.byref(info), ctypes.sizeof(info)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return info.FileAttributes


def close(handle: int) -> None:
    _kernel32.CloseHandle(handle)


def open_root(path: str | os.PathLike, *, follow: bool = True) -> int:
    """Open a package directory; only ``follow`` lets the root itself be a link."""
    handle = _kernel32.CreateFileW(
        extended_path(path),
        _FILE_LIST_DIRECTORY | _FILE_TRAVERSE | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
        _FILE_SHARE_ALL,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | (0 if follow else _FILE_FLAG_OPEN_REPARSE_POINT),
        None,
    )
    if handle is None or handle == _INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        found = _attributes(handle)
        if found & _FILE_ATTRIBUTE_REPARSE_POINT and not follow:
            raise LinkRefused(errno.ELOOP, "reparse point refused", os.fspath(path))
        if not found & _FILE_ATTRIBUTE_DIRECTORY:
            raise NotADirectoryError(errno.ENOTDIR, "not a directory", os.fspath(path))
    except BaseException:
        close(handle)
        raise
    return handle


def open_component(parent: int, name: str, *, directory: bool) -> tuple[int, int]:
    """Open one name directly under ``parent`` without following it.

    Returns the handle and its file attributes. A final component passes
    ``directory=False``, which leaves the kind open so a link is reported as a
    link before a directory is reported as not a file, as on POSIX.
    """
    encoded = name.encode("utf-16-le")
    if len(encoded) > _MAX_NAME_BYTES:
        raise OSError(errno.ENAMETOOLONG, "name too long", name)
    object_name = _UnicodeString(len(encoded), len(encoded), name)
    attributes = _ObjectAttributes(
        ctypes.sizeof(_ObjectAttributes), parent, ctypes.pointer(object_name), 0, None, None
    )
    io_status = _IoStatusBlock()
    opened = ctypes.c_void_p()
    status = _ntdll.NtCreateFile(
        ctypes.byref(opened),
        (_FILE_TRAVERSE | _FILE_READ_ATTRIBUTES if directory else _FILE_GENERIC_READ)
        | _SYNCHRONIZE,
        ctypes.byref(attributes),
        ctypes.byref(io_status),
        None,
        _FILE_ATTRIBUTE_NORMAL,
        _FILE_SHARE_ALL,
        _FILE_OPEN,
        (_FILE_DIRECTORY_FILE if directory else 0)
        | _FILE_OPEN_REPARSE_POINT
        | _FILE_SYNCHRONOUS_IO_NONALERT,
        None,
        0,
    )
    if status < 0:
        raise ctypes.WinError(_ntdll.RtlNtStatusToDosError(status & 0xFFFFFFFF))
    handle = opened.value
    try:
        found = _attributes(handle)
        if found & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise LinkRefused(errno.ELOOP, "reparse point refused", name)
    except BaseException:
        close(handle)
        raise
    return handle, found


def is_directory(attributes: int) -> bool:
    return bool(attributes & _FILE_ATTRIBUTE_DIRECTORY)


def to_descriptor(handle: int) -> int:
    """Give ``handle`` to a C runtime descriptor, which then owns closing it."""
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close(handle)
        raise


def handle_of(descriptor: int) -> int:
    return msvcrt.get_osfhandle(descriptor)


def final_path(descriptor: int) -> str:
    """The path of an open directory, as the file system now names it."""
    handle = msvcrt.get_osfhandle(descriptor)
    size = 1024
    while True:
        buffer = ctypes.create_unicode_buffer(size)
        written = _kernel32.GetFinalPathNameByHandleW(handle, buffer, size, _VOLUME_NAME_DOS)
        if written == 0:
            raise ctypes.WinError(ctypes.get_last_error())
        if written < size:
            return buffer.value
        size = written + 1


def local_app_data() -> str:
    """The account's LocalAppData from the shell, not from this process's settings."""
    # FOLDERID_LocalAppData
    folder = _Guid(
        0xF1B32785, 0x6FBA, 0x4FCF, (ctypes.c_ubyte * 8)(0x9D, 0x55, 0x7B, 0x8E, 0x7F, 0x15, 0x70, 0x91)
    )
    shell32 = ctypes.WinDLL("shell32")
    ole32 = ctypes.WinDLL("ole32")
    shell32.SHGetKnownFolderPath.restype = ctypes.c_long
    shell32.SHGetKnownFolderPath.argtypes = [
        ctypes.POINTER(_Guid),
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ole32.CoTaskMemFree.restype = None
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    pointer = ctypes.c_void_p()
    result = shell32.SHGetKnownFolderPath(ctypes.byref(folder), 0, None, ctypes.byref(pointer))
    if result != 0:
        raise ctypes.WinError(result & 0xFFFF)
    try:
        return ctypes.wstring_at(pointer.value)
    finally:
        ole32.CoTaskMemFree(pointer)


def retry_while_busy(operation, *, attempts: int = 100, delay: float = 0.02):
    """Run ``operation``, retrying while another handle briefly holds the file.

    Windows refuses to rename or replace an open file, and scanners and readers
    open new files for milliseconds. A file still held after the retries fails.
    """
    for remaining in range(attempts - 1, -1, -1):
        try:
            return operation()
        except PermissionError:
            if remaining == 0:
                raise
            time.sleep(delay)
    return None


def process_is_alive(pid: int) -> bool:
    """Whether ``pid`` runs. Anything uncertain, such as another user's process, is alive.

    ``os.kill(pid, 0)`` is not a probe here: 0 is ``CTRL_C_EVENT``, so it would send
    Ctrl-C to that process group.
    """
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ctypes.get_last_error() != _ERROR_INVALID_PARAMETER
    try:
        code = wintypes.DWORD()
        if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == _STILL_ACTIVE
    finally:
        close(handle)


def peak_working_set_bytes() -> int:
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not _kernel32.K32GetProcessMemoryInfo(
        _kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize


def lock_descriptor(descriptor: int, *, exclusive: bool, blocking: bool) -> None:
    """``flock`` for one guard file: released by closing it or by process exit.

    A blocking request polls so that Ctrl-C still reaches the waiting thread;
    ``LockFileEx`` cannot be interrupted once it is waiting.
    """
    handle = msvcrt.get_osfhandle(descriptor)
    flags = _LOCKFILE_FAIL_IMMEDIATELY | (_LOCKFILE_EXCLUSIVE_LOCK if exclusive else 0)
    while True:
        overlapped = _Overlapped()
        if _kernel32.LockFileEx(handle, flags, 0, 1, 0, ctypes.byref(overlapped)):
            return
        error = ctypes.get_last_error()
        if error not in {_ERROR_LOCK_VIOLATION, _ERROR_IO_PENDING}:
            raise ctypes.WinError(error)
        if not blocking:
            raise BlockingIOError(errno.EAGAIN, "package guard is held")
        time.sleep(_LOCK_POLL_SECONDS)
