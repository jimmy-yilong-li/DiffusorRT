"""The Mac windows.

Two windows live here: the page that `diffusorrt gui` shows on a Mac (in a
WebKit window or a browser tab), and the engine host behind the native app in
`apps/macos/`. Both show the decode as it happens, can stop a reply, and know
when the engine is busy.

Everything in this package runs only on a Mac, and it stands by itself: it
imports nothing from the parent package, which is the window for the other
platforms and another owner's code. Below it are only the SDK's engine,
conversations, model opening and model repository.

author:Opus
"""
