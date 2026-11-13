"""The WebKit window that shows the Mac page. Model and history ownership stays in the controller.

author:Opus -- taken over with the Mac windows; pywebview stays an optional dependency.
"""
from __future__ import annotations

import threading
from collections.abc import Callable


def _webview():
    try:
        import webview
    except ImportError as error:
        from edllm import _extra_install_command

        # The release's own install command, as every other missing-extra message
        # gives it: the wheel is not on a package index, so a bare
        # `pip install "diffusor-rt[gui]"` finds nothing.
        raise RuntimeError(
            "The Mac window needs the gui extra. Install it with "
            f"`{_extra_install_command('gui')}`, or use `diffusorrt gui --browser`."
        ) from error
    return webview


class MacWindow:
    def __init__(self, url: str, request_close: Callable[[], object], closed: threading.Event):
        self._webview = _webview()
        self._closed = closed
        self._request_close = request_close
        self._webview.settings["ALLOW_FILE_URLS"] = False
        self._webview.settings["ALLOW_DOWNLOADS"] = False
        self.window = self._webview.create_window(
            "DiffusorRT", url, width=1180, height=760, min_size=(900, 620),
            text_select=True, background_color="#f7f8fb",
        )
        self.window.events.closing += self._on_closing

    def _on_closing(self) -> bool:
        if self._closed.is_set():
            return True
        self._request_close()
        return False

    def run(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("The Mac window must run on the main thread.")
        # pywebview starts this callback off the GUI thread and marshals destroy
        # to Cocoa's main loop. Never block the native closing callback.
        self._webview.start(self._finish_when_released, private_mode=True)

    def _finish_when_released(self) -> None:
        self._closed.wait()
        self.window.destroy()

    def choose_folder(self) -> str | None:
        # Cocoa's pywebview adapter marshals the dialog to the main thread.
        # Called only by the authenticated transport, never a Python JS bridge.
        paths = self.window.create_file_dialog(self._webview.FileDialog.FOLDER)
        return str(paths[0]) if paths else None
