"""Descriptor-driven real-model parity gate for frozen reference artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import traceback
import zipfile

import numpy as np
from numpy.lib import format as npformat

from edllm.graph import GraphIR, build_graph_ir
from edllm.package_artifacts import (
    artifact_source_stat,
    open_package_artifact,
    read_open_artifact,
    sha256_stream,
)

from .producer_descriptors import (
    descriptor_for_key,
    load_case_corpus,
    load_oracle_lock,
    validate_execution_package,
)
from .reference import reference_manifest_path


_MAX_PACKAGE_GRAPH_BYTES = 64 * 1024 * 1024


_REFERENCE_CASE_FIELDS = frozenset(
    {
        "name",
        "array_key",
        "input_ids",
        "masked_positions",
        "selected_rows",
        "shape",
        "dtype",
    }
)
_REFERENCE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_key",
        "model_id",
        "resolved_revision",
        "architecture",
        "descriptor_digest",
        "case_digest",
        "environment_lock_digest",
        "reference_adapter",
        "logits_convention",
        "comparison_dtype",
        "numeric_tier",
        "environment",
        "package_weights_sha256",
        "cases",
        "artifact_sha256",
    }
)


class _StrictManifestError(ValueError):
    pass


class _StrictGraphError(ValueError):
    pass


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _StrictManifestError("reference_manifest_duplicate_key")
        result[key] = value
    return result


def _reject_nonfinite_json(_constant: str):
    raise _StrictManifestError("reference_manifest_nonfinite_number")


def _strict_graph_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _StrictGraphError("package_graph_duplicate_key")
        result[key] = value
    return result


def _reject_nonfinite_graph_json(_constant: str):
    raise _StrictGraphError("package_graph_nonfinite_number")


def _exact_json_equal(actual: object, expected: object) -> bool:
    """JSON equality that never aliases bool/int/float or nested container types."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(expected, float):
        return math.isfinite(actual) and math.isfinite(expected) and actual == expected
    return actual == expected


def _manifest_value_matches(actual: object, expected: object) -> bool:
    try:
        return _exact_json_equal(actual, expected)
    except RecursionError as exc:
        raise ValueError("reference_manifest_invalid") from exc


def _ordered_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    token_ids = np.arange(logits.shape[-1], dtype=np.int64)
    ordered = [np.lexsort((token_ids, -row))[:k] for row in logits]
    return np.asarray(ordered, dtype=np.int64)


def _numeric_tier_manifest(descriptor) -> dict[str, object]:
    tier = descriptor.numeric_tier
    return {
        "key": tier.key,
        "atol": tier.atol,
        "rtol": tier.rtol,
        "exact_argmax": tier.exact_argmax,
        "exact_ordered_top_k": tier.exact_ordered_top_k,
    }


def _parity_attention_inputs(descriptor) -> dict[str, object]:
    """Resolve the descriptor-owned attention inputs for full-logit replay."""
    contract = descriptor.decode_contract
    if contract is None:
        return {"attention_mask_kind": "bidirectional"}
    if (
        contract.decode_strategy != "bd3_fixed_quota"
        or contract.attention_mask_kind != "d2f_hybrid"
        or contract.mask_geometry_prompt_len != 0
    ):
        raise ValueError("parity_decode_contract_unsupported")
    return {
        "attention_mask_kind": contract.attention_mask_kind,
        "mask_geometry": {
            "prompt_len": contract.mask_geometry_prompt_len,
            "block_size": contract.block_size,
        },
    }


def _oracle_environment_manifest(descriptor) -> dict[str, object]:
    lock = load_oracle_lock(descriptor)
    return {
        "profile": lock["profile"],
        "lock_digest": descriptor.oracle_lock_digest,
        "python": lock["python"],
        "platform": lock["platform"],
        "packages": lock["packages"],
    }


def _comparison_dtype(descriptor) -> np.dtype:
    try:
        dtype = np.dtype(descriptor.comparison_dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference_comparison_dtype_unsupported") from exc
    if dtype != np.dtype(np.float32):
        raise ValueError("reference_comparison_dtype_unsupported")
    return dtype


def _validate_reference_manifest(
    descriptor, reference_path: Path, corpus: dict
) -> dict:
    manifest_path = reference_manifest_path(reference_path)
    if not manifest_path.is_file():
        raise ValueError("reference_manifest_missing")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except _StrictManifestError as exc:
        raise ValueError(str(exc)) from exc
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("reference_manifest_invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("reference_manifest_invalid")
    if set(manifest) != _REFERENCE_MANIFEST_FIELDS:
        raise ValueError("reference_manifest_fields_mismatch")
    expected = {
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
        "numeric_tier": _numeric_tier_manifest(descriptor),
        "environment": _oracle_environment_manifest(descriptor),
    }
    for field, value in expected.items():
        if not _manifest_value_matches(manifest.get(field), value):
            raise ValueError(
                f"reference_{field}_mismatch: expected={value!r} "
                f"actual={manifest.get(field)!r}"
            )
    _comparison_dtype(descriptor)

    reference_cases = manifest.get("cases")
    corpus_cases = corpus.get("cases")
    if not isinstance(reference_cases, list):
        raise ValueError("reference_cases_missing")
    if not isinstance(corpus_cases, list):
        raise ValueError("reference_case_corpus_invalid")
    if len(reference_cases) != len(corpus_cases):
        raise ValueError("reference_case_count_mismatch")
    array_keys: set[str] = set()
    literal_fields = ("input_ids", "masked_positions", "selected_rows")
    for case, record in zip(corpus_cases, reference_cases, strict=True):
        if not isinstance(record, dict):
            raise ValueError("reference_case_record_invalid")
        if set(record) != _REFERENCE_CASE_FIELDS:
            raise ValueError("reference_case_fields_mismatch")
        if not _manifest_value_matches(record["name"], case["name"]):
            raise ValueError("reference_case_name_mismatch")
        for field in literal_fields:
            if not _manifest_value_matches(record[field], case[field]):
                raise ValueError(f"reference_case_{field}_mismatch")
        array_key = record["array_key"]
        if not isinstance(array_key, str) or not array_key:
            raise ValueError("reference_case_array_key_invalid")
        if array_key in array_keys:
            raise ValueError("reference_case_array_key_duplicate")
        array_keys.add(array_key)
        if record["dtype"] != descriptor.comparison_dtype:
            raise ValueError("reference_case_dtype_mismatch")
        shape = record["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or shape[0] != len(case["selected_rows"])
        ):
            raise ValueError("reference_case_shape_mismatch")
    return manifest


def _reference_npz_call(operation):
    """Normalize untrusted NumPy/ZIP parser failures without catching signals/OOM."""
    try:
        return operation()
    except MemoryError:
        raise
    except Exception as exc:
        raise ValueError("reference_artifact_invalid") from exc


def _close_reference_archive(archive, active_error: BaseException | None) -> None:
    try:
        _reference_npz_call(lambda: archive.close())
    except BaseException as cleanup_error:
        if active_error is None:
            raise
        add_note = getattr(active_error, "add_note", None)
        if callable(add_note):
            add_note(f"cleanup_failed: {type(cleanup_error).__name__}: {cleanup_error}")


def _validate_reference_arrays(
    descriptor,
    reference_path: Path,
    manifest: dict,
    corpus: dict,
    *,
    vocab_size: int,
) -> dict[str, np.ndarray]:
    """Load the small oracle payload only after its exact schema is proven."""
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("package_vocab_size_invalid")
    comparison_dtype = _comparison_dtype(descriptor)
    records = manifest["cases"]
    expected_keys = [record["array_key"] for record in records]
    loaded: dict[str, np.ndarray] = {}
    try:
        with open_package_artifact(
            package_path=reference_path.parent,
            artifact_ref=reference_path.name,
            error_prefix="reference",
            artifact_name="artifact",
        ) as (reference_handle, _display_path, opened_stat):
            opened_identity = artifact_source_stat(opened_stat)
            actual_hash = sha256_stream(reference_handle)
            if (
                artifact_source_stat(os.fstat(reference_handle.fileno()))
                != opened_identity
            ):
                raise ValueError("reference_artifact_changed")
            if manifest.get("artifact_sha256") != actual_hash:
                raise ValueError("reference_artifact_digest_mismatch")
            reference_handle.seek(0)
            archive = _reference_npz_call(
                lambda: np.load(reference_handle, allow_pickle=False)
            )
            archive_error = None
            try:
                actual_keys = _reference_npz_call(
                    lambda: getattr(archive, "files", None)
                )
                if not isinstance(actual_keys, list):
                    raise ValueError("reference_artifact_invalid")
                if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != set(
                    expected_keys
                ):
                    raise ValueError("reference_array_key_set_mismatch")
                for case, record in zip(corpus["cases"], records, strict=True):
                    array_key = record["array_key"]
                    array = _reference_npz_call(
                        lambda array_key=array_key: np.asarray(archive[array_key])
                    )
                    if array.dtype != comparison_dtype:
                        raise ValueError(
                            "reference_array_dtype_mismatch: "
                            f"key={array_key} expected={comparison_dtype.name} "
                            f"actual={array.dtype.name}"
                        )
                    expected_shape = (len(case["selected_rows"]), vocab_size)
                    if tuple(record["shape"]) != expected_shape:
                        raise ValueError(
                            "reference_case_shape_mismatch: "
                            f"key={array_key} expected={expected_shape!r} "
                            f"actual={tuple(record['shape'])!r}"
                        )
                    if array.shape != expected_shape:
                        raise ValueError(
                            "reference_array_shape_mismatch: "
                            f"key={array_key} expected={expected_shape!r} "
                            f"actual={array.shape!r}"
                        )
                    loaded[array_key] = array
                if (
                    artifact_source_stat(os.fstat(reference_handle.fileno()))
                    != opened_identity
                ):
                    raise ValueError("reference_artifact_changed")
            except BaseException as exc:
                archive_error = exc
                raise
            finally:
                _close_reference_archive(archive, archive_error)
    except MemoryError:
        raise
    return loaded


def _positive_config_int(config: dict, key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"package_model_config_geometry_invalid: {key}")
    return value


def _load_exact_package_graph(
    package: str | Path, manifest: dict, model_config: dict
) -> GraphIR:
    """Validate raw graph JSON exactly before any GraphIR coercion/defaulting."""
    graph_ref = manifest.get("graph_ref")
    if not isinstance(graph_ref, str) or not graph_ref:
        raise ValueError("package_graph_ref_missing")
    try:
        with open_package_artifact(
            package_path=package,
            artifact_ref=graph_ref,
            error_prefix="package",
            artifact_name="graph",
            max_bytes=_MAX_PACKAGE_GRAPH_BYTES,
        ) as (handle, _path, _stat):
            payload = read_open_artifact(
                handle,
                max_bytes=_MAX_PACKAGE_GRAPH_BYTES,
                error_prefix="package",
                artifact_name="graph",
            )
        raw_graph = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_graph_object,
            parse_constant=_reject_nonfinite_graph_json,
        )
    except _StrictGraphError as exc:
        raise ValueError(str(exc)) from exc
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        code = str(exc)
        if code.startswith("package_graph_"):
            raise
        raise ValueError("package_graph_invalid") from exc
    canonical = build_graph_ir(model_config)
    try:
        exact_contract = _exact_json_equal(raw_graph, canonical.to_dict())
    except RecursionError as exc:
        raise ValueError("package_graph_invalid") from exc
    if not exact_contract:
        raise ValueError("package_graph_contract_mismatch")
    if not _exact_json_equal(manifest.get("ir_version"), raw_graph["ir_version"]):
        raise ValueError("package_ir_version_mismatch")
    try:
        graph = GraphIR.from_dict(raw_graph)
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError("package_graph_invalid") from exc
    return graph


def _expected_package_weight_shapes(graph, config: dict) -> dict[str, tuple[int, ...]]:
    """Derive the complete dense-v0 NPZ contract from GraphIR plus geometry."""
    hidden = _positive_config_int(config, "hidden_size")
    vocab = _positive_config_int(config, "vocab_size")
    intermediate = _positive_config_int(config, "intermediate_size")
    n_heads = _positive_config_int(config, "attention_heads")
    num_kv_heads = _positive_config_int(config, "num_kv_heads")
    head_dim = _positive_config_int(config, "head_dim")
    layer_count = _positive_config_int(config, "layer_count")
    canonical_graph = build_graph_ir(config)
    if not _exact_json_equal(graph.to_dict(), canonical_graph.to_dict()):
        raise ValueError("package_graph_contract_mismatch")

    q_op = canonical_graph.layers[0].ops[1]
    qk_norm_op = canonical_graph.layers[0].ops[2]
    bias_present = q_op.params["bias_present"] is True
    apply_qk_norm = qk_norm_op.params["apply_qk_norm"] is True
    expected: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    projection_shapes = {
        "q": (n_heads * head_dim, hidden),
        "k": (num_kv_heads * head_dim, hidden),
        "v": (num_kv_heads * head_dim, hidden),
        "o": (hidden, n_heads * head_dim),
    }
    for layer_index in range(layer_count):
        prefix = f"model.layers.{layer_index}"
        expected[f"{prefix}.input_layernorm.weight"] = (hidden,)
        expected[f"{prefix}.post_attention_layernorm.weight"] = (hidden,)
        for role, shape in projection_shapes.items():
            projection = f"{prefix}.self_attn.{role}_proj"
            expected[f"{projection}.weight"] = shape
            if bias_present and role != "o":
                expected[f"{projection}.bias"] = (shape[0],)
        if apply_qk_norm:
            expected[f"{prefix}.self_attn.q_norm.weight"] = (head_dim,)
            expected[f"{prefix}.self_attn.k_norm.weight"] = (head_dim,)
        expected[f"{prefix}.mlp.gate_proj.weight"] = (intermediate, hidden)
        expected[f"{prefix}.mlp.up_proj.weight"] = (intermediate, hidden)
        expected[f"{prefix}.mlp.down_proj.weight"] = (hidden, intermediate)
    return expected


def _npy_header(handle) -> tuple[tuple[int, ...], np.dtype]:
    try:
        version = npformat.read_magic(handle)
        if version == (1, 0):
            shape, _fortran_order, dtype = npformat.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _fortran_order, dtype = npformat.read_array_header_2_0(handle)
        else:
            raise ValueError("package_weights_npy_version_unsupported")
    except ValueError as exc:
        if str(exc) == "package_weights_npy_version_unsupported":
            raise
        raise ValueError("package_weights_header_invalid") from exc
    except (OSError, EOFError) as exc:
        raise ValueError("package_weights_header_invalid") from exc
    return tuple(int(dimension) for dimension in shape), np.dtype(dtype)


def _validate_package_weights_npz(
    package: str | Path,
    weights_ref: str,
    graph,
    config: dict,
    *,
    precision: str,
) -> None:
    """Validate every NPZ key/dtype/shape by headers, without loading weights."""
    expected_dtype_by_precision = {"fp32": np.dtype(np.float32)}
    expected_dtype = expected_dtype_by_precision.get(precision)
    if expected_dtype is None:
        raise ValueError(f"package_weights_precision_unsupported: {precision}")
    expected_shapes = _expected_package_weight_shapes(graph, config)
    try:
        with (
            open_package_artifact(
                package_path=package,
                artifact_ref=weights_ref,
                error_prefix="package",
                artifact_name="weights",
            ) as (weights_handle, _weights_path, _weights_stat),
            zipfile.ZipFile(weights_handle) as archive,
        ):
            infos = [info for info in archive.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("package_weights_duplicate_key")
            if any(not name.endswith(".npy") for name in names):
                raise ValueError("package_weights_key_set_mismatch")
            actual_keys = {name[:-4] for name in names}
            if actual_keys != set(expected_shapes):
                raise ValueError("package_weights_key_set_mismatch")
            for info in infos:
                key = info.filename[:-4]
                with archive.open(info) as handle:
                    shape, dtype = _npy_header(handle)
                if dtype != expected_dtype:
                    raise ValueError(
                        "package_weights_dtype_mismatch: "
                        f"key={key} expected={expected_dtype.name} "
                        f"actual={dtype.name}"
                    )
                if shape != expected_shapes[key]:
                    raise ValueError(
                        "package_weights_shape_mismatch: "
                        f"key={key} expected={expected_shapes[key]!r} "
                        f"actual={shape!r}"
                    )
    except ValueError:
        raise
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError("package_weights_archive_invalid") from exc


def _close_resources(
    resources: tuple[object | None, ...], active_error: BaseException | None
) -> None:
    cleanup_errors: list[BaseException] = []
    for resource in resources:
        if resource is None:
            continue
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as exc:  # cleanup must continue for every owner
            cleanup_errors.append(exc)
    if not cleanup_errors:
        return
    notes = [f"cleanup_failed: {type(exc).__name__}: {exc}" for exc in cleanup_errors]
    if active_error is not None:
        add_note = getattr(active_error, "add_note", None)
        if callable(add_note):
            for note in notes:
                add_note(note)
        return
    primary = cleanup_errors[0]
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        for note in notes[1:]:
            add_note(note)
    raise primary


def run_parity(
    *,
    candidate_key: str,
    package: str | Path,
    reference: str | Path,
    output: str | Path | None = None,
) -> dict[str, object]:
    descriptor = descriptor_for_key(candidate_key)
    descriptor.validate()
    from edllm.model import (
        PackageConfigError,
        _canonicalize_runtime_manifest_snapshot,
        _preflight_runtime_descriptor_identity,
        _read_runtime_manifest_snapshot,
        _validate_runtime_manifest_snapshot,
    )

    manifest_snapshot = _read_runtime_manifest_snapshot(package)
    preflight = _validate_runtime_manifest_snapshot(manifest_snapshot)
    try:
        _preflight_runtime_descriptor_identity(preflight)
    except ValueError as exc:
        raise PackageConfigError(str(exc)) from exc
    _validated_manifest, package_hash = validate_execution_package(
        descriptor,
        package,
        verify_weights_hash=True,
        manifest_snapshot=preflight.manifest,
    )

    canonical = _canonicalize_runtime_manifest_snapshot(
        package,
        validated_snapshot=preflight,
    )
    package_manifest = canonical.manifest
    package_model_config = canonical.model_config
    exact_graph = _load_exact_package_graph(
        package, package_manifest, package_model_config
    )
    corpus = load_case_corpus(descriptor)
    reference_path = Path(reference).absolute()
    reference_manifest = _validate_reference_manifest(
        descriptor, reference_path, corpus
    )
    if reference_manifest.get("package_weights_sha256") != package_hash:
        raise ValueError("reference_package_weights_digest_mismatch")
    reference_arrays = _validate_reference_arrays(
        descriptor,
        reference_path,
        reference_manifest,
        corpus,
        vocab_size=package_model_config.get("vocab_size"),
    )
    _validate_package_weights_npz(
        package,
        package_manifest["merged_weights_ref"],
        exact_graph,
        package_model_config,
        precision=package_manifest["model_precision"],
    )

    from edllm.backends.mlx_dream import MLXDreamEngine

    engine = None
    resident_state = None
    expected = None
    actual = None
    full_logits = None
    forwarded = None
    results: list[dict[str, object]] = []
    verdict = "pass"
    body_error = None
    try:
        engine = MLXDreamEngine(
            package_dir=Path(package),
            merged_weights_ref=package_manifest["merged_weights_ref"],
            model_config=copy.deepcopy(package_model_config),
            graph=exact_graph,
        )
        engine._resident_weights(require_source_identity=True)
        resident_state = engine._weight_state
        resident_digest = getattr(resident_state, "source_digest", None)
        if type(resident_digest) is not str or resident_digest != package_hash:
            raise ValueError("package_resident_weights_digest_mismatch")
        comparison_dtype = _comparison_dtype(descriptor)
        for case, record in zip(
            corpus["cases"], reference_manifest["cases"], strict=True
        ):
            array_key = record["array_key"]
            expected = reference_arrays[array_key]
            forwarded = engine.forward_full_logits(
                token_ids=list(case["input_ids"]),
                **_parity_attention_inputs(descriptor),
            )
            if engine._weight_state is not resident_state:
                raise ValueError("package_resident_weights_reloaded")
            full_logits = np.asarray(forwarded)
            if full_logits.dtype != comparison_dtype:
                raise ValueError(
                    "parity_actual_logits_dtype_mismatch: "
                    f"expected={comparison_dtype.name} actual={full_logits.dtype.name}"
                )
            expected_full_shape = (
                len(case["input_ids"]),
                package_model_config["vocab_size"],
            )
            if full_logits.shape != expected_full_shape:
                raise ValueError(
                    "parity_actual_logits_shape_mismatch: "
                    f"expected={expected_full_shape!r} actual={full_logits.shape!r}"
                )
            actual = full_logits[case["selected_rows"]]
            problems: list[str] = []
            if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                problems.append("nonfinite_logits")
            max_abs_diff = None
            if not problems:
                max_abs_diff = float(np.max(np.abs(actual - expected)))
                try:
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        atol=descriptor.numeric_tier.atol,
                        rtol=descriptor.numeric_tier.rtol,
                    )
                except AssertionError:
                    problems.append("allclose_failed")
                if descriptor.numeric_tier.exact_argmax and not np.array_equal(
                    actual.argmax(axis=-1), expected.argmax(axis=-1)
                ):
                    problems.append("argmax_mismatch")
                top_k = descriptor.numeric_tier.exact_ordered_top_k
                if not np.array_equal(
                    _ordered_top_k(actual, top_k),
                    _ordered_top_k(expected, top_k),
                ):
                    problems.append(f"ordered_top_{top_k}_mismatch")
            case_verdict = "pass" if not problems else "fail"
            if problems:
                verdict = "fail"
            results.append(
                {
                    "name": case["name"],
                    "shape": list(actual.shape),
                    "max_abs_diff": max_abs_diff,
                    "verdict": case_verdict,
                    "reasons": problems,
                }
            )
    except BaseException as exc:
        body_error = exc
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        try:
            _close_resources((engine,), body_error)
        finally:
            engine = None
            resident_state = None
            reference_arrays = None
            expected = None
            actual = None
            full_logits = None
            forwarded = None
            exact_graph = None
            body_error = None

    receipt: dict[str, object] = {
        "schema_version": 1,
        "candidate_key": descriptor.key,
        "model_id": descriptor.model_id,
        "revision": descriptor.revision,
        "descriptor_digest": descriptor.descriptor_digest,
        "case_digest": descriptor.parity_case_digest,
        "lock_digest": descriptor.oracle_lock_digest,
        "package_weights_sha256": package_hash,
        "reference_artifact_sha256": reference_manifest["artifact_sha256"],
        "backend": package_manifest["admitted_backend"],
        "precision": package_manifest["model_precision"],
        "numeric_tier": descriptor.numeric_tier.key,
        "verdict": verdict,
        "cases": results,
    }
    if output is not None:
        Path(output).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_parity(
        candidate_key=args.candidate,
        package=args.package,
        reference=args.reference,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_parity"]
