"""Compiler error taxonomy.

All subclass ``CompilerError``, itself a ``RuntimeError``, so a caller can catch
coarsely (any compile failure) or precisely (this one) without knowing the
hierarchy. Each message names the remedy, not just the condition.
"""
from __future__ import annotations


class CompilerError(RuntimeError):
    """Base for every compiler failure."""


class UnsupportedModelError(CompilerError):
    """The model id is not in the pinned registry. v0 supports exactly one."""


class RevisionMismatchError(CompilerError):
    """A caller-supplied revision disagrees with the registry pin.

    A pin a caller can silently override is not a pin.
    """


class InsufficientDiskError(CompilerError):
    """Free space is below the estimated requirement, checked BEFORE download."""


class ModelAccessError(CompilerError):
    """The repository is gated, private, or the token is missing/unauthorised."""


class SnapshotNotCachedError(CompilerError):
    """Offline compile: the pinned revision is not in the standard HF cache.

    Distinct from ModelAccessError on purpose. Nothing is wrong with the
    repository or the credentials -- the bytes are simply not on this machine
    yet, and the remedy is a download the user performs with their own Hugging
    Face tooling, not a retry.
    """


class RemoteCodeNotTrustedError(CompilerError):
    """Compiling requires executing repository-authored code and it was not
    explicitly permitted."""


class CompilerDependencyError(CompilerError):
    """An optional dependency of the compile path is absent.

    Names the extra to install rather than surfacing an ImportError on an
    internal module path.
    """
