"""The trust sentinel and its error, in a module with no heavy imports.

Lives apart from `tokenizer.py` on purpose. `deployment.py` and `cli.py` are
imported to answer `diffusorrt --help`, and both need the sentinel as a default
argument -- a module-level `from .tokenizer import ...` there would put the
whole tokenizer stack behind `--help`, which
test_importing_cli_does_not_import_runtime_dependency_graph forbids.

It has to be ONE object, not one per module: `expected_module is not UNSPECIFIED`
is the check, and per-module sentinels would make every cross-module comparison
silently false.
"""
from __future__ import annotations


class _Unspecified:
    """`expected_module=None` is a real value -- "this package must declare NO
    custom module" -- so it cannot double as "the caller did not say"."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unspecified>"


UNSPECIFIED = _Unspecified()


class RemoteCodeIdentityError(RuntimeError):
    """The module about to be imported is not the one consent was given for.

    WHAT THIS BINDS, precisely: the module NAME the package declares. It does
    not hash the ``.py`` file. A package that keeps the declared name and
    changes the file's contents passes this check.

    Stating the limit because the guard is easy to over-read. What it prevents
    is a package substituting a DIFFERENT module between the moment consent was
    given and the moment transformers imports one -- the swap the CLI-level
    re-check could not see. Content integrity is the compile receipt's job, and
    it covers the package as published, not a directory mutated afterwards.

    Raised at the LAST instant a check can still prevent the import: against
    the materialized tokenizer bundle, immediately before AutoTokenizer reads
    it. A check anywhere earlier inspects a different directory at a different
    moment, and two callers checked *after* the load entirely -- by which point
    the swapped module had already executed.
    """
