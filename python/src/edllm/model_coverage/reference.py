"""Single parameterised external-oracle reference generator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np
from numpy.lib import format as npformat

from .producer_descriptors import (
    DecodeContract,
    ModelDescriptor,
    descriptor_for_key,
    load_case_corpus,
    load_oracle_lock,
    validate_execution_package,
    validate_snapshot,
)


@dataclass(frozen=True, slots=True)
class ReferenceAdapter:
    key: str
    loader: str


REFERENCE_ADAPTERS: dict[str, ReferenceAdapter] = {
    "dream_hf_446_v1": ReferenceAdapter(
        key="dream_hf_446_v1",
        loader="auto_model",
    ),
    "llada_hf_446_v1": ReferenceAdapter(
        key="llada_hf_446_v1",
        loader="auto_model",
    ),
    "qwen3_mdlm_hf_457_v1": ReferenceAdapter(
        key="qwen3_mdlm_hf_457_v1",
        loader="qwen3_snapshot_module",
    ),
    "qwen3_bd3lm_hf_457_v1": ReferenceAdapter(
        key="qwen3_bd3lm_hf_457_v1",
        loader="qwen3_snapshot_module",
    ),
}


def reference_adapter(key: str) -> ReferenceAdapter:
    try:
        return REFERENCE_ADAPTERS[key]
    except KeyError:
        raise ValueError(f"unknown_reference_adapter: {key}") from None


def validate_oracle_environment(descriptor: ModelDescriptor) -> dict[str, object]:
    """Require the locked Python/platform/dependency identity exactly."""
    lock = load_oracle_lock(descriptor)
    expected_python = lock["python"]
    actual_python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }
    if actual_python != expected_python:
        raise ValueError(
            f"oracle_python_mismatch: expected={expected_python!r} "
            f"actual={actual_python!r}"
        )
    expected_platform = lock["platform"]
    actual_platform = {"system": platform.system(), "machine": platform.machine()}
    if actual_platform != expected_platform:
        raise ValueError(
            f"oracle_platform_mismatch: expected={expected_platform!r} "
            f"actual={actual_platform!r}"
        )
    actual_packages: dict[str, str] = {}
    mismatches: list[str] = []
    for name, expected_version in sorted(lock["packages"].items()):
        try:
            actual_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{name}=missing(expected {expected_version})")
            continue
        actual_packages[name] = actual_version
        if actual_version != expected_version:
            mismatches.append(f"{name}={actual_version}(expected {expected_version})")
    if mismatches:
        raise ValueError("oracle_package_mismatch: " + "; ".join(mismatches))
    return {
        "profile": lock["profile"],
        "lock_digest": descriptor.oracle_lock_digest,
        "python": actual_python,
        "platform": actual_platform,
        "packages": actual_packages,
    }


def _load_model(
    adapter: ReferenceAdapter,
    snapshot: Path,
    *,
    trust_remote_code: bool,
):
    # Heavy imports occur only after every identity/digest/environment preflight.
    import torch

    if adapter.loader == "qwen3_snapshot_module":
        if not trust_remote_code:
            raise ValueError("reference_model_code_not_authorized")
        module_path = snapshot / "modeling_qwen3.py"
        if not module_path.is_file():
            raise ValueError("reference_modeling_module_missing: modeling_qwen3.py")
        module_name = f"diffusorrt_qwen3_frozen_oracle_{_sha256(module_path)}"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ValueError("reference_modeling_module_unloadable")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise
        model_class = getattr(module, "A2DQwen3LMHeadModel", None)
        if model_class is None:
            raise ValueError("reference_model_class_missing: A2DQwen3LMHeadModel")
        model = model_class.from_pretrained(
            str(snapshot),
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
    elif adapter.loader == "auto_model":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
    else:
        raise ValueError(f"unknown_reference_loader: {adapter.loader}")
    model.eval()
    return model


def _bd3_staircase_attention(
    tokens, *, block_size: int
) -> tuple[object, object]:
    """Build BD3's absolute-block mask in eager-attention additive form.

    Upstream constructs a boolean allow-mask for SDPA. The frozen parity oracle
    deliberately selects eager attention; eager Qwen3 adds a 4-D mask to its
    scores, so the same allow relation must be encoded as 0 / dtype-min rather
    than passed as booleans (where ``True`` would incorrectly add one).
    """
    import torch

    sequence_length = tokens.shape[1]
    position_ids = torch.arange(sequence_length, device=tokens.device).unsqueeze(0)
    block_ids = torch.arange(sequence_length, device=tokens.device) // block_size
    query_blocks = block_ids.view(1, 1, sequence_length, 1)
    key_blocks = block_ids.view(1, 1, 1, sequence_length)
    allowed = key_blocks <= query_blocks
    attention_mask = torch.zeros(
        allowed.shape, device=tokens.device, dtype=torch.float32
    )
    attention_mask.masked_fill_(~allowed, torch.finfo(torch.float32).min)
    return attention_mask, position_ids


def _forward_selected(
    model,
    input_ids: list[int],
    selected_rows: list[int],
    *,
    decode_contract: DecodeContract | None = None,
) -> np.ndarray:
    import torch

    tokens = torch.tensor([input_ids], dtype=torch.long)
    # Legacy frozen cases are single, unpadded sequences. Omitting the mask is
    # semantically identical to an all-valid mask and preserves their existing
    # bidirectional oracle path. Decode-contract families supply their exact
    # model-specific mask and position semantics below.
    forward_kwargs = {
        "input_ids": tokens,
        "attention_mask": None,
        "use_cache": False,
    }
    if decode_contract is not None:
        if (
            decode_contract.decode_strategy != "bd3_fixed_quota"
            or decode_contract.attention_mask_kind != "d2f_hybrid"
            or decode_contract.mask_geometry_prompt_len != 0
        ):
            raise ValueError("reference_decode_contract_unsupported")
        attention_mask, position_ids = _bd3_staircase_attention(
            tokens, block_size=decode_contract.block_size
        )
        forward_kwargs["attention_mask"] = attention_mask
        forward_kwargs["position_ids"] = position_ids
    with torch.inference_mode():
        output = model(**forward_kwargs)
    logits = output.logits[0, selected_rows].detach().to(torch.float32).cpu().numpy()
    logits = np.asarray(logits, dtype=np.float32)
    if not np.isfinite(logits).all():
        raise ValueError("reference_logits_nonfinite")
    return logits


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for name in sorted(arrays):
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                with archive.open(info, "w", force_zip64=True) as handle:
                    npformat.write_array(
                        handle,
                        np.asarray(arrays[name], dtype=np.float32),
                        allow_pickle=False,
                    )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_manifest_path(reference_path: str | Path) -> Path:
    path = Path(reference_path)
    return path.with_name(path.name + ".json")


def generate_reference(
    *,
    candidate_key: str,
    snapshot: str | Path,
    package: str | Path,
    output: str | Path,
) -> tuple[Path, Path]:
    """Generate every frozen case through one descriptor-selected oracle."""
    descriptor = descriptor_for_key(candidate_key)
    descriptor.validate()
    cases = load_case_corpus(descriptor)
    adapter = reference_adapter(descriptor.reference_adapter)
    environment = validate_oracle_environment(descriptor)
    snapshot_path = Path(snapshot).resolve()
    validate_snapshot(descriptor, snapshot_path)
    _, package_hash = validate_execution_package(
        descriptor, package, verify_weights_hash=True
    )

    # No model/weight import or open occurs before the complete preflight above.
    model = _load_model(
        adapter,
        snapshot_path,
        trust_remote_code=descriptor.trust_remote_code,
    )
    arrays: dict[str, np.ndarray] = {}
    case_records: list[dict[str, object]] = []
    try:
        for index, case in enumerate(cases["cases"]):
            array_key = f"case_{index:03d}"
            forward_args = (
                model,
                list(case["input_ids"]),
                list(case["selected_rows"]),
            )
            if descriptor.decode_contract is None:
                logits = _forward_selected(*forward_args)
            else:
                logits = _forward_selected(
                    *forward_args,
                    decode_contract=descriptor.decode_contract,
                )
            arrays[array_key] = logits
            case_records.append(
                {
                    "name": case["name"],
                    "array_key": array_key,
                    "input_ids": case["input_ids"],
                    "masked_positions": case["masked_positions"],
                    "selected_rows": case["selected_rows"],
                    "shape": list(logits.shape),
                    "dtype": "float32",
                }
            )
    finally:
        del model

    output_path = Path(output).resolve()
    _write_deterministic_npz(output_path, arrays)
    artifact_sha256 = _sha256(output_path)
    manifest = {
        "schema_version": 1,
        "candidate_key": descriptor.key,
        "model_id": descriptor.model_id,
        "resolved_revision": descriptor.revision,
        "architecture": list(descriptor.expected_architectures),
        "descriptor_digest": descriptor.descriptor_digest,
        "case_digest": descriptor.parity_case_digest,
        "environment_lock_digest": descriptor.oracle_lock_digest,
        "reference_adapter": descriptor.reference_adapter,
        "logits_convention": descriptor.logits_convention.value,
        "comparison_dtype": descriptor.comparison_dtype,
        "numeric_tier": {
            "key": descriptor.numeric_tier.key,
            "atol": descriptor.numeric_tier.atol,
            "rtol": descriptor.numeric_tier.rtol,
            "exact_argmax": descriptor.numeric_tier.exact_argmax,
            "exact_ordered_top_k": descriptor.numeric_tier.exact_ordered_top_k,
        },
        "environment": environment,
        "package_weights_sha256": package_hash,
        "cases": case_records,
        "artifact_sha256": artifact_sha256,
    }
    manifest_path = reference_manifest_path(output_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output_path, manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact, manifest = generate_reference(
        candidate_key=args.candidate,
        snapshot=args.snapshot,
        package=args.package,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "artifact": artifact.name,
                "manifest": manifest.name,
                "artifact_sha256": _sha256(artifact),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REFERENCE_ADAPTERS",
    "ReferenceAdapter",
    "generate_reference",
    "main",
    "reference_adapter",
    "reference_manifest_path",
    "validate_oracle_environment",
]
