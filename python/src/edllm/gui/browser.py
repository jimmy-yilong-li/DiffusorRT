"""Open the chat page as a standalone Edge window (no tabs, no address bar)."""
from __future__ import annotations

import logging
import os
import subprocess
import webbrowser
from pathlib import Path

_log = logging.getLogger("edllm.gui")
_APP_PATHS = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe"
WINDOW_SIZE = "1180,760"


def find_edge() -> Path | None:
    registered = _registered_edge()
    if registered is not None and registered.is_file():
        return registered
    for variable in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        root = os.environ.get(variable)
        if root:
            candidate = Path(root) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if candidate.is_file():
                return candidate
    return None


def open_window(url: str, *, profile_dir: Path) -> subprocess.Popen | None:
    """Edge in app mode with its own profile, so its process lives as long as the window.

    Without Edge the page opens in the default browser and None is returned; the
    page's polling then decides when the chat is closed.
    """
    edge = find_edge()
    if edge is None:
        _log.warning("Edge was not found; opening the chat in the default browser")
        webbrowser.open(url)
        return None
    command = [
        str(edge),
        f"--app={url}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        f"--window-size={WINDOW_SIZE}",
    ]
    return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def _registered_edge() -> Path | None:
    if os.name != "nt":
        return None
    import winreg

    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, _APP_PATHS) as key:
                value, _ = winreg.QueryValueEx(key, None)
        except OSError:
            continue
        if value:
            return Path(value)
    return None
