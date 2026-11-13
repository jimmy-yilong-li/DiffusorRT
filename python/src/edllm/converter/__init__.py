"""Converter entry points, exposed via LAZY attribute access.

``import edllm.converter`` must stay free of the heavy ML stack
(torch / peft / transformers). The ``merge`` submodule historically imported
those at module top, so eagerly importing it here pulled torch into every
``import edllm.converter`` (even to call a torch-free stub builder).

To keep the import torch-free, this package imports NO submodule at import time.
Each public name is resolved on first access via :func:`__getattr__`, which
imports the owning submodule only then. ``merge`` itself now also imports its
heavy deps lazily inside its functions, so resolving e.g. ``build_stub_package``
still does not require torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Map each public symbol to the submodule that defines it. Resolved lazily so no
# submodule (and none of its imports) loads until the symbol is first accessed.
_LAZY_EXPORTS: dict[str, str] = {
    "build_package_from_hf_masked_diffusion": ".convert_hf_masked_diffusion",
    "convert_hf_lora_fixture": ".hf_import",
    "MERGE_VERSION": ".merge",
    "build_merge_plan": ".merge",
    "build_stub_package": ".merge",
    "build_tiny_stub_package": ".merge",
    "convert_real_lora_checkpoint": ".merge",
    "make_old_fixture_package": ".merge",
    "is_compatible": ".package_writer",
    "read_manifest": ".package_writer",
    "validate_package_version": ".package_writer",
    "write_execution_package": ".package_writer",
}

# sorted() returns a list, which is exactly what __all__ must be. PLE0605
# cannot see through the call, and hand-duplicating the key list here would
# create a second copy to keep in sync with _LAZY_EXPORTS.
__all__ = sorted(_LAZY_EXPORTS)  # noqa: PLE0605


def __getattr__(name: str):
    """Lazily import and return a public converter symbol (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


if TYPE_CHECKING:  # pragma: no cover - import-time type visibility only
    # `X as X` is the re-export spelling. These names ARE public -- __all__ is
    # built from _LAZY_EXPORTS and __getattr__ resolves them -- but written
    # plainly they read as unused imports, which is what F401 reported.
    from .convert_hf_masked_diffusion import build_package_from_hf_masked_diffusion as build_package_from_hf_masked_diffusion
    from .hf_import import convert_hf_lora_fixture as convert_hf_lora_fixture
    from .merge import (
        MERGE_VERSION as MERGE_VERSION,
        build_merge_plan as build_merge_plan,
        build_stub_package as build_stub_package,
        build_tiny_stub_package as build_tiny_stub_package,
        convert_real_lora_checkpoint as convert_real_lora_checkpoint,
        make_old_fixture_package as make_old_fixture_package,
    )
    from .package_writer import (
        is_compatible as is_compatible,
        read_manifest as read_manifest,
        validate_package_version as validate_package_version,
        write_execution_package as write_execution_package,
    )
