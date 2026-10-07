#!/usr/bin/env python3
"""Run one supported DiffusorRT model through the public Python API."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


DEFAULT_MODEL = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
DEFAULT_PROMPT = "Explain diffusion language models in one short paragraph."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download or reuse a supported model package and generate one reply."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Supported model ID or prepared package directory (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--revision", default=None, help="Optional exact revision pin.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--backend",
        default="auto",
        help="Runtime backend (default: auto). Invalid or unavailable values fail closed.",
    )
    parser.add_argument(
        "--precision",
        default=None,
        help="Required with cuda_native: bf16 or int8_weight_only. Omit for MLX.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached source or a prepared package without accessing the network.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow repository tokenizer code when the selected model requires it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.max_tokens < 0:
        parser.error("--max-tokens must be 0 or greater")
    try:
        from diffusor_rt import Engine
    except ModuleNotFoundError as error:
        if error.name not in {"diffusor_rt", "edllm"}:
            raise
        raise SystemExit(
            "DiffusorRT is not installed. Install the wheel first: "
            "python -m pip install "
            "'diffusor-rt[compiler,mlx-local] @ "
            "https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/"
            "v0.2.2/diffusor_rt-0.2.2-py3-none-any.whl'"
        ) from error

    with Engine.from_pretrained(
        args.model,
        revision=args.revision,
        offline=args.offline,
        trust_remote_code=args.trust_remote_code,
        backend=args.backend,
        precision=args.precision,
    ) as engine:
        result = engine.generate(args.prompt, max_tokens=args.max_tokens)
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
