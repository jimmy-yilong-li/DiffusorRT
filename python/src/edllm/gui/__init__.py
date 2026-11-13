"""The DiffusorRT chat window.

A loopback page shown in a standalone Edge window. Importing this package imports
no GUI toolkit and opens nothing; `edllm.gui.app.run` starts the window.

On a Mac the window is another one, with its own code in `edllm.gui.mac`; the
two share nothing. `run` below starts the one for this platform.
"""
from __future__ import annotations

import sys


def run(command_spec, *, options, log_to_file):
    """Open this platform's chat window and return once it has closed.

    author:Opus -- the one place that knows there are two windows. Neither
    window's code is imported on the other's platform.
    """
    if sys.platform == "darwin":
        from .mac import page_app as window
    else:
        from . import app as window
    try:
        return window.run(command_spec, options=options, log_to_file=log_to_file)
    except Exception as error:  # noqa: BLE001 - a Windows launcher has no console
        if sys.platform == "darwin" or not log_to_file:
            raise
        window.report_launch_failure(error)
        return 1
