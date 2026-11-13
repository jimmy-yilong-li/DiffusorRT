from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation-only: torch, MLX and transformers are heavy optional imports
    # that must not run at module import. The names still have to resolve for
    # a type checker.
    import mlx.core as mx
    import torch
    from transformers import AutoConfig

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Heavy ML deps (torch / peft / transformers / huggingface_hub) are imported
# LAZILY inside the functions that need them, so ``import edllm.converter`` (which
# re-exports this module's stub builders) stays torch-free. Only the real
# merge/export paths -- ``convert_real_lora_checkpoint`` and its helpers -- pull
# the heavy stack, and only when actually run.
_MX: object | None = None

from ..graph import build_graph_ir
from ..runtime_config import normalize_rope_scaling
from .package_writer import write_execution_package

MERGE_VERSION = "1.0.0"
FROZEN_BASE_REF = "Dream-org/Dream-v0-Instruct-7B"
FROZEN_ADAPTER_REF = "SJTU-DENG-Lab/D2F_Dream_Instruct_7B_Lora"
FROZEN_BASE_SHA = "05334cb9faaf763692dcf9d8737c642be2b2a6ae"
FROZEN_ADAPTER_SHA = "e7ef056117d81fa3c669a8945ed62fb5d2b47029"
STUB_MODEL_CONFIG = {
    "architecture": "dream_stub",
    "layer_count": 2,
    "hidden_size": 128,
    "attention_heads": 4,
    "vocab_size": 1000,
    "head_dim": 32,
    "num_kv_heads": 4,
    # Present here too, so a stub package and a real one describe the same six
    # geometry fields. A stub that is missing one would make the native
    # geometry path testable only against the 28 GiB package.
    "max_position_embeddings": 2048,
    "rope_theta": 1000000.0,
    "rope_scaling": {"rope_type": "linear", "factor": 1.0},
    "intermediate_size": 256,
    "rms_norm_eps": 1e-06,
    "block_size": 32,
    "step_budget": 4,
    "mask_token_id": 6,
    # attention_mask_kind is deliberately ABSENT. It used to say "d2f_hybrid",
    # which no adapter's default strategy (block_threshold) can satisfy: the
    # package loaded and then raised "d2f_hybrid requires mask geometry" at the
    # first forward. Setting "bidirectional" instead would be just as wrong --
    # apply_policy_profile() is {**profile, **config}, so an explicit key here
    # OVERRIDES the d2f presets and breaks the validated throughput profile.
    # Omitting it lets the runtime default apply AND lets a preset set it,
    # which is exactly what the working real package does.
    "block_schedule": {"kind": "fixed", "block_size": 32},
}


@dataclass(slots=True)
class MergePlan:
    base_model_ref: str
    adapter_ref: str
    output_dir: Path
    source_checkpoint_sha: str
    source_base_model_sha: str
    merge_script_version: str


def build_merge_plan(
    *,
    base_model_ref: str,
    adapter_ref: str,
    cache_root: str | Path | None = None,
    base_revision: str | None = None,
    adapter_revision: str | None = None,
    resolve_remote: bool = False,
) -> MergePlan:
    cache_root = Path(cache_root or Path.home() / ".cache" / "edllm" / "packages")
    source_checkpoint_sha = _resolve_model_sha(
        adapter_ref,
        revision=adapter_revision,
        resolve_remote=resolve_remote,
    )
    source_base_model_sha = _resolve_model_sha(
        base_model_ref,
        revision=base_revision,
        resolve_remote=resolve_remote,
    )
    merge_script_version = MERGE_VERSION
    package_key = hashlib.sha256(
        f"{base_model_ref}:{adapter_ref}:{source_checkpoint_sha}:{source_base_model_sha}:{merge_script_version}".encode()
    ).hexdigest()[:16]
    return MergePlan(
        base_model_ref=base_model_ref,
        adapter_ref=adapter_ref,
        output_dir=cache_root / package_key,
        source_checkpoint_sha=source_checkpoint_sha,
        source_base_model_sha=source_base_model_sha,
        merge_script_version=merge_script_version,
    )


def build_tiny_stub_package(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    graph = build_graph_ir(STUB_MODEL_CONFIG)
    manifest = {
        "version": {"major": 2, "minor": 0, "patch": 0},
        "fixture_mode": True,
        "source_checkpoint_reference": FROZEN_ADAPTER_REF,
        "source_base_model_reference": FROZEN_BASE_REF,
        "decode_family": "d2f_hybrid_block_parallel",
        "backend_memory_owner": "mlx",
        "forward_executor": "apple_mlx_backend",
        "runtime_impl": "python_scaffold",
        "merged_weights_ref": "weights.npz",
        "source_checkpoint_sha": FROZEN_ADAPTER_SHA,
        "source_base_model_sha": FROZEN_BASE_SHA,
        "merge_script_version": MERGE_VERSION,
        "model_config": STUB_MODEL_CONFIG,
        "graph_ref": "graph.json",
        "ir_version": dict(graph.ir_version),
    }
    tokenizer_manifest = {
        "vocab": {
            "<s>": 0,
            "</s>": 1,
            "<unk>": 2,
            "draft": 3,
            "response": 4,
            "for": 5,
            "<|mask|>": 6,
        },
    }
    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "mask_token": "<|mask|>",
    }
    package_dir = write_execution_package(
        output_dir,
        manifest,
        tokenizer_manifest=tokenizer_manifest,
        special_tokens_map=special_tokens_map,
        weights=None,
        graph=graph,
    )
    _write_stub_weights(package_dir / "weights.npz")
    manifest["output_hash"] = _sha256_file(package_dir / "weights.npz")
    (package_dir / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return package_dir


def convert_real_lora_checkpoint(
    output_dir: str | Path,
    *,
    base_model_ref: str,
    adapter_ref: str,
    base_revision: str | None = None,
    adapter_revision: str | None = None,
) -> Path:
    plan = build_merge_plan(
        base_model_ref=base_model_ref,
        adapter_ref=adapter_ref,
        base_revision=base_revision,
        adapter_revision=adapter_revision,
        resolve_remote=True,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = _load_model_config(base_model_ref, revision=base_revision)
    graph = build_graph_ir(model_config)
    manifest = {
        "version": {"major": 2, "minor": 0, "patch": 0},
        "fixture_mode": False,
        "source_checkpoint_reference": adapter_ref,
        "source_base_model_reference": base_model_ref,
        "decode_family": "d2f_hybrid_block_parallel",
        "backend_memory_owner": "mlx",
        "forward_executor": "apple_mlx_backend",
        "runtime_impl": "python_mlx_baseline",
        "merged_weights_ref": "weights.npz",
        "source_checkpoint_sha": plan.source_checkpoint_sha,
        "source_base_model_sha": plan.source_base_model_sha,
        "merge_script_version": plan.merge_script_version,
        "model_config": model_config,
        "graph_ref": "graph.json",
        "ir_version": dict(graph.ir_version),
    }
    package_dir = write_execution_package(
        output_dir,
        manifest,
        weights=None,
        graph=graph,
    )
    _save_real_tokenizer_assets(package_dir, base_model_ref=base_model_ref, revision=base_revision)
    _merge_real_weights_to_npz(
        package_dir / "weights.npz",
        base_model_ref=base_model_ref,
        adapter_ref=adapter_ref,
        base_revision=base_revision,
        adapter_revision=adapter_revision,
    )
    manifest["output_hash"] = _sha256_file(package_dir / "weights.npz")
    (package_dir / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return package_dir


def make_old_fixture_package(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    manifest = {
        "version": {"major": 1, "minor": 0, "patch": 0},
        "fixture_mode": True,
    }
    tokenizer_manifest = {
        "vocab": {"<s>": 0, "</s>": 1, "<unk>": 2},
    }
    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
    }
    return write_execution_package(
        output_dir,
        manifest,
        tokenizer_manifest=tokenizer_manifest,
        special_tokens_map=special_tokens_map,
    )


def read_package_meta(package_dir: str | Path) -> dict:
    package_dir = Path(package_dir)
    return json.loads((package_dir / "model.json").read_text(encoding="utf-8"))


def build_stub_package(output_dir: str | Path) -> Path:
    return build_tiny_stub_package(output_dir)


def _resolve_model_sha(repo_id: str, *, revision: str | None, resolve_remote: bool) -> str:
    if not resolve_remote:
        if repo_id == FROZEN_BASE_REF:
            return FROZEN_BASE_SHA
        if repo_id == FROZEN_ADAPTER_REF:
            return FROZEN_ADAPTER_SHA
        return hashlib.sha256(f"{repo_id}:{revision or 'main'}".encode()).hexdigest()[:40]
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision)
    return str(info.sha)


def _write_stub_weights(path: Path) -> None:
    vocab_size = int(STUB_MODEL_CONFIG["vocab_size"])
    hidden_size = int(STUB_MODEL_CONFIG["hidden_size"])
    arrays = {
        "model.embed_tokens.weight": np.zeros((vocab_size, hidden_size), dtype=np.float16),
        "model.norm.weight": np.ones((hidden_size,), dtype=np.float16),
        "lm_head.weight": np.zeros((vocab_size, hidden_size), dtype=np.float16),
    }
    for layer_idx in range(int(STUB_MODEL_CONFIG["layer_count"])):
        prefix = f"model.layers.{layer_idx}"
        arrays[f"{prefix}.input_layernorm.weight"] = np.ones((hidden_size,), dtype=np.float16)
        arrays[f"{prefix}.post_attention_layernorm.weight"] = np.ones((hidden_size,), dtype=np.float16)
        arrays[f"{prefix}.self_attn.q_proj.weight"] = np.eye(hidden_size, dtype=np.float16)
        arrays[f"{prefix}.self_attn.k_proj.weight"] = np.eye(hidden_size, dtype=np.float16)
        arrays[f"{prefix}.self_attn.v_proj.weight"] = np.eye(hidden_size, dtype=np.float16)
        arrays[f"{prefix}.self_attn.o_proj.weight"] = np.eye(hidden_size, dtype=np.float16)
        arrays[f"{prefix}.self_attn.q_proj.bias"] = np.zeros((hidden_size,), dtype=np.float16)
        arrays[f"{prefix}.self_attn.k_proj.bias"] = np.zeros((hidden_size,), dtype=np.float16)
        arrays[f"{prefix}.self_attn.v_proj.bias"] = np.zeros((hidden_size,), dtype=np.float16)
        arrays[f"{prefix}.mlp.gate_proj.weight"] = np.ones((hidden_size, hidden_size), dtype=np.float16)
        arrays[f"{prefix}.mlp.up_proj.weight"] = np.ones((hidden_size, hidden_size), dtype=np.float16)
        arrays[f"{prefix}.mlp.down_proj.weight"] = np.ones((hidden_size, hidden_size), dtype=np.float16)
    np.savez(str(path), **arrays)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_model_config(base_model_ref: str, *, revision: str | None) -> dict[str, int | str]:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(base_model_ref, revision=revision, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_ref, revision=revision, trust_remote_code=True)
    architecture = None
    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, list) and architectures:
        architecture = architectures[0]
    if not architecture:
        architecture = config.__class__.__name__
    return {
        "architecture": architecture,
        "layer_count": int(getattr(config, "num_hidden_layers", 0) or getattr(config, "n_layer", 0)),
        "hidden_size": int(getattr(config, "hidden_size", 0) or getattr(config, "n_embd", 0)),
        "attention_heads": int(getattr(config, "num_attention_heads", 0) or getattr(config, "n_head", 0)),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
        "head_dim": int(
            getattr(config, "head_dim", 0)
            or (
                (getattr(config, "hidden_size", 0) or getattr(config, "n_embd", 0))
                // (getattr(config, "num_attention_heads", 1) or getattr(config, "n_head", 1))
            )
        ),
        "num_kv_heads": int(getattr(config, "num_key_value_heads", 0) or getattr(config, "num_attention_heads", 0)),
        # The sixth DiffusorRTModelGeometry field, and the only one the package
        # did not carry. It exists in every source config this converter reads
        # (131072 for the pinned Dream) and was simply never propagated, so a
        # native caller asking for geometry could be given five of six values
        # or a guess. Packages built before this land without it and report
        # DIFFUSORRT_UNSET rather than a default -- a defaulted geometry field
        # is a silently wrong model (abi/lifetimes.py refuses to default one).
        "max_position_embeddings": int(
            getattr(config, "max_position_embeddings", 0)
            or getattr(config, "n_positions", 0)
        ),
        "rope_theta": float(getattr(config, "rope_theta", 10000.0)),
        "rope_scaling": _normalized_rope_scaling(config),
        "intermediate_size": int(getattr(config, "intermediate_size", 0)),
        "rms_norm_eps": float(getattr(config, "rms_norm_eps", 1e-6)),
        "block_size": 32,
        "step_budget": 4,
        "mask_token_id": _coerce_mask_token_id(tokenizer),
        # See STUB_MODEL_CONFIG: attention_mask_kind is deliberately ABSENT.
        # Writing "d2f_hybrid" here made every freshly merged package raise
        # "d2f_hybrid requires mask geometry" on a plain generate(); writing
        # "bidirectional" would override the d2f presets instead. The real
        # package on disk predates this line, carries no such key, and is
        # precisely why both plain generate() and the presets work on it.
        "block_schedule": {"kind": "fixed", "block_size": 32},
    }


def _save_real_tokenizer_assets(package_dir: Path, *, base_model_ref: str, revision: str | None) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_ref, revision=revision, trust_remote_code=True)
    for stale_name in ("tokenizer.json", "special_tokens_map.json"):
        stale_path = package_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    tokenizer.save_pretrained(str(package_dir))


def _merge_real_weights_to_npz(
    output_path: Path,
    *,
    base_model_ref: str,
    adapter_ref: str,
    base_revision: str | None,
    adapter_revision: str | None,
) -> None:
    from peft import PeftModel
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(base_model_ref, revision=base_revision, trust_remote_code=True)
    config.rope_scaling = _normalized_rope_scaling(config)

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_ref,
            revision=base_revision,
            config=config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except (ValueError, KeyError):
        # Only fall back when the architecture has no causal-LM head mapping
        # (AutoModelForCausalLM raises ValueError/KeyError there). OOM, network,
        # auth and other Runtime/OS errors must propagate — not be swallowed.
        base_model = AutoModel.from_pretrained(
            base_model_ref,
            revision=base_revision,
            config=config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    adapter_model = PeftModel.from_pretrained(
        base_model,
        adapter_ref,
        revision=adapter_revision,
    )
    merged_model = adapter_model.merge_and_unload()
    state_dict = merged_model.state_dict()
    arrays = {
        name: _tensor_to_mlx_array(tensor)
        for name, tensor in state_dict.items()
    }
    _get_mlx().savez(str(output_path), **arrays)


def _tensor_to_mlx_array(tensor: "torch.Tensor") -> "mx.array":
    import torch

    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype == torch.bfloat16:
        cpu_tensor = cpu_tensor.to(torch.float16)
    return _get_mlx().array(cpu_tensor.numpy())


def _get_mlx():
    global _MX
    if _MX is not None:
        return _MX
    try:
        import mlx.core as mx
    except Exception as exc:
        raise RuntimeError("MLX runtime unavailable; install/use MLX with Metal support before running merge/export paths.") from exc
    _MX = mx
    return mx


def _normalized_rope_scaling(config: AutoConfig) -> dict[str, float | str]:
    return normalize_rope_scaling(getattr(config, "rope_scaling", None))


def _coerce_mask_token_id(tokenizer) -> int:
    """One rule, shared with the Reference Compiler and the runtime.

    See edllm.control_tokens: an explicitly declared valid id wins, otherwise
    the exact `<|mask|>` must be in the vocabulary, otherwise refuse. bool is
    not an id -- isinstance(True, int) is True in Python, and this function
    used to return True, which became token id 1 in the manifest.
    """
    from ..control_tokens import resolve_from_vocab

    return resolve_from_vocab(
        getattr(tokenizer, "mask_token_id", None),
        tokenizer.get_vocab(),
        source="merge converter",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build DiffusorRT execution packages from Dream base + D2F LoRA."
    )
    parser.add_argument("--base", dest="base_model_ref", help="Base model repository id.")
    parser.add_argument("--adapter", dest="adapter_ref", help="Adapter repository id.")
    parser.add_argument("--output", required=True, help="Output package directory.")
    parser.add_argument("--base-revision", default=None)
    parser.add_argument("--adapter-revision", default=None)
    parser.add_argument("--stub", action="store_true", help="Build the tiny CI stub package instead of a real package.")
    args = parser.parse_args(argv)

    if args.stub:
        build_tiny_stub_package(args.output)
        return 0

    if not args.base_model_ref or not args.adapter_ref:
        parser.error("--base and --adapter are required unless --stub is used")
    convert_real_lora_checkpoint(
        args.output,
        base_model_ref=args.base_model_ref,
        adapter_ref=args.adapter_ref,
        base_revision=args.base_revision,
        adapter_revision=args.adapter_revision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
