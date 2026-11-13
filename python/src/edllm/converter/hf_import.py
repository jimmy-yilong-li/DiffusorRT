from __future__ import annotations

import warnings
from pathlib import Path

from .merge import build_tiny_stub_package


def convert_hf_lora_fixture(output_dir: str | Path) -> Path:
    warnings.warn(
        "convert_hf_lora_fixture is now a stub-package alias. Use build_tiny_stub_package for CI stubs "
        "and convert_real_lora_checkpoint for real merges.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_tiny_stub_package(output_dir)
