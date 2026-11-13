"""Double-click entry for the chat window (`diffusorrt-gui`, a gui-scripts launcher).

No arguments and no console: the window picks the model itself and logs to a file
under the application-data root.
"""
from __future__ import annotations


def main() -> int:
    from edllm import gui

    return gui.run(None, options=None, log_to_file=True)


__all__ = ["main"]
