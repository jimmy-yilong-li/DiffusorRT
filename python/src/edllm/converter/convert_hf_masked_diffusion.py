"""Build canonical DiffusorRT packages from registered HF dLLM snapshots.

One reader streams single-file or sharded safetensors through the selected
declarative tensor mapping. The same build writes canonical GraphIR, model
geometry, tokenizer assets, and provenance. Model families supply definitions;
they do not supply loaders or package pipelines.

The module imports Torch only while converting weights and never imports MLX.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from ..graph import build_graph_ir
from ..model_definitions import (
    ModelDefinition,
    TokenizerLayout,
    converter_adapter_for_key,
    model_build_digest,
    model_definition_digest as digest_model_definition,
    model_definition_for_descriptor,
    model_definition_payload,
    runtime_architecture_for_key,
    validate_model_snapshot,
)
from ..weight_store import (
    WeightArtifactSpec,
    WrittenWeightArtifact,
    write_safetensors_weight_store,
)

# Mirror the Dream real-package manifest shape (see converter/merge.py).
MASKED_DIFFUSION_CONVERTER_VERSION = "1.0.1"

# Bundled canonical Qwen3 chat template (verbatim from ``Qwen/Qwen3-0.6B``), used
# as a fallback when the masked-diffusion snapshot ships none. Lives next to this
# module so it travels with the package source.
_BUNDLED_QWEN3_CHAT_TEMPLATE_PATH = Path(__file__).with_name(
    "qwen3_chat_template.jinja"
)

# Tokenizer / aux files copied verbatim from the HF snapshot into the package.
# tokenizer.json + tokenizer_config.json + vocab.json + merges.txt are what
# AutoTokenizer.from_pretrained needs; special_tokens_map.json + added_tokens.json
# carry the <|mask|> special token the engine resolves. generation_config.json and
# the model_type's remote modeling file are copied for provenance/completeness.
_TOKENIZER_ASSET_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    # Custom slow-tokenizer source resolved via auto_map. Dream / Dream-Coder ship
    # NO tokenizer.json: AutoTokenizer(trust_remote_code=True) imports these to
    # build DreamTokenizer. Missing for Qwen3/LLaDA, so the copy loop skips them.
    "tokenization_dream.py",
    "configuration_dream.py",
)

def _resolve_provenance(*, source_ref, source_sha, source_license):
    """Require literal provenance; architecture-derived identities are forbidden."""
    if not source_ref:
        raise ValueError("source_ref_required")
    if not isinstance(source_sha, str) or len(source_sha) != 40:
        raise ValueError("source_sha_must_be_40_char_revision")
    if not source_license:
        raise ValueError("source_license_required")
    return {
        "source_ref": source_ref,
        "source_sha": source_sha,
        "source_license": source_license,
    }


def build_package_from_hf_masked_diffusion(
    hf_snapshot_dir: str | Path,
    out_dir: str | Path,
    *,
    candidate_key: str | None = None,
    model_definition: ModelDefinition | None = None,
    model_definition_digest: str | None = None,
    converter_adapter: str | None = None,
    source_ref: str | None = None,
    source_sha: str | None = None,
    source_license: str | None = None,
    backend: str = "mlx_local",
    precision: str = "fp32",
) -> Path:
    """Build a DiffusorRT execution package from an HF masked-diffusion snapshot.

    Reads ``config.json`` + the snapshot's safetensors + tokenizer files from
    ``hf_snapshot_dir`` and writes a loadable package to ``out_dir``:

    - a canonical weight artifact keyed with the engine's llama-style tensor
      names (``model.embed_tokens.weight``, ``model.layers.N....weight``,
      ``model.norm.weight``, ``lm_head.weight``). Product packages retain their
      admitted NPZ bytes; Local packages use dtype-preserving safetensors shards.
      Each definition supplies only a tensor-name mapper. Dream/Qwen3 use the
      identity mapper; LLaDA uses the declarative OLMo-to-GraphIR mapping.
    - ``model.json`` mirroring the Dream real-package manifest shape, with the
      model_config geometry (explicit head_dim, num_kv_heads, rope_theta, ...).
    - the transformers tokenizer assets, plus a ``chat_template.jinja`` when the
      snapshot ships one (LLaDA's inline LLaMA-3-style template is carried).

    A Product descriptor or explicit Local ``ModelDefinition`` selects one
    conversion contract. Every layout uses the same single-file-or-sharded
    safetensors reader and streaming writer; only its tensor/config/tokenizer
    mapping data differs. Returns the package directory path.

    ``source_ref`` / ``source_sha`` / ``source_license`` record where the package
    actually came from. A catalog candidate supplies them through its descriptor;
    historical direct callers must supply all three literally with an explicit
    converter adapter. Architecture-derived provenance is rejected because a
    Dream-family checkpoint that is not Dream-Coder must never be labelled as
    Dream-Coder. An artifact that cannot say which repository and commit it was
    built from, and under what licence, makes the distribution boundary in
    ``COPYRIGHT_AND_LICENSING.md`` unenforceable.
    """
    hf_snapshot_dir = Path(hf_snapshot_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _read_json(hf_snapshot_dir / "config.json")
    if candidate_key is not None and model_definition is not None:
        raise ValueError("candidate_key_and_model_definition_are_exclusive")
    descriptor = None
    definition = model_definition
    if candidate_key is not None:
        from ..model_coverage.producer_descriptors import runtime_descriptor_for_key

        descriptor = runtime_descriptor_for_key(candidate_key)
        definition = model_definition_for_descriptor(descriptor)
        descriptor.resolve_backend(backend, precision)
        validate_model_snapshot(definition, hf_snapshot_dir, config=config)
        if (
            converter_adapter is not None
            and converter_adapter != descriptor.converter_adapter
        ):
            raise ValueError(
                "converter_adapter_mismatch: "
                f"expected={descriptor.converter_adapter} actual={converter_adapter}"
            )
        adapter = definition.converter
        source_ref = definition.model_id
        source_sha = definition.revision
        source_license = definition.source_license
    elif definition is not None:
        expected_digest = digest_model_definition(definition)
        if model_definition_digest != expected_digest:
            raise ValueError("model_definition_digest_mismatch")
        validate_model_snapshot(definition, hf_snapshot_dir, config=config)
        if (
            converter_adapter is not None
            and converter_adapter != definition.converter.key
        ):
            raise ValueError(
                "converter_adapter_mismatch: "
                f"expected={definition.converter.key} actual={converter_adapter}"
            )
        for name, supplied, expected in (
            ("source_ref", source_ref, definition.model_id),
            ("source_sha", source_sha, definition.revision),
            ("source_license", source_license, definition.source_license),
        ):
            if supplied is not None and supplied != expected:
                raise ValueError(f"model_definition_{name}_mismatch")
        adapter = definition.converter
        source_ref = definition.model_id
        source_sha = definition.revision
        source_license = definition.source_license
    else:
        if model_definition_digest is not None:
            raise ValueError("model_definition_digest_without_definition")
        if converter_adapter is None:
            raise ValueError("candidate_key_or_converter_adapter_required")
        adapter = converter_adapter_for_key(converter_adapter)
        _resolve_provenance(
            source_ref=source_ref,
            source_sha=source_sha,
            source_license=source_license,
        )
    adapter.validate_config(config)
    provenance = _resolve_provenance(
        source_ref=source_ref,
        source_sha=source_sha,
        source_license=source_license,
    )
    tie_word_embeddings = _tie_word_embeddings(
        config, upstream_field=adapter.upstream_tying_field
    )
    if (
        adapter.required_tie_word_embeddings is not None
        and tie_word_embeddings is not adapter.required_tie_word_embeddings
    ):
        raise ValueError(
            f"converter_{adapter.upstream_tying_field}_mismatch: "
            f"expected={adapter.required_tie_word_embeddings!r} "
            f"actual={tie_word_embeddings!r}"
        )
    if (
        descriptor is not None
        and tie_word_embeddings is not definition.tie_word_embeddings
    ):
        raise ValueError(
            "converter_tie_word_embeddings_mismatch: "
            f"expected={definition.tie_word_embeddings!r} "
            f"actual={tie_word_embeddings!r}"
        )

    # 1) Resolve config and GraphIR before reading or writing model weights.
    #    Unsupported source semantics must fail before expensive conversion.
    mask_token_id = _resolve_mask_token_id(hf_snapshot_dir, config)
    if definition is not None and mask_token_id != definition.mask_token_id:
        raise ValueError(
            "converter_mask_token_id_mismatch: "
            f"expected={definition.mask_token_id!r} actual={mask_token_id!r}"
        )

    # The selected definition owns config translation. The shared loader does
    # not branch on a model/config-layout tag.
    model_config = adapter.build_model_config(
        config,
        mask_token_id=mask_token_id,
        tie_word_embeddings=tie_word_embeddings,
    )
    model_config["architecture_family"] = adapter.runtime_adapter
    if definition is not None and definition.decode_contract is not None:
        model_config.update(definition.decode_contract.package_model_config())
    resolved_source_ref = provenance["source_ref"]
    resolved_source_sha = provenance["source_sha"]
    runtime = (
        definition.runtime
        if definition is not None
        else runtime_architecture_for_key(adapter.runtime_adapter)
    )
    graph = build_graph_ir(model_config, runtime=runtime)

    # 2) Every family uses one mapped-tensor stream. Storage is an orthogonal
    #    sink decision: preserve already-admitted Product NPZ bytes, while Local
    #    packages use the scalable dtype-preserving store.
    tied_weight_keys = adapter.tied_weight_keys if tie_word_embeddings else ()
    local_package = definition is not None and descriptor is None
    if local_package:
        weight_artifact = _write_mapped_weights_store(
            hf_snapshot_dir,
            out_dir,
            tensor_name_mapper=adapter.tensor_name_mapper,
            tied_weight_keys=tied_weight_keys,
        )
        weight_keys = weight_artifact.tensor_names
    else:
        weights_path = out_dir / "weights.npz"
        _write_mapped_weights_npz(
            hf_snapshot_dir,
            weights_path,
            tensor_name_mapper=adapter.tensor_name_mapper,
            tied_weight_keys=tied_weight_keys,
        )
        with np.load(str(weights_path), allow_pickle=False) as archive:
            weight_keys = frozenset(archive.keys())
        weight_artifact = WrittenWeightArtifact(
            spec=WeightArtifactSpec.legacy_npz(
                "weights.npz",
                digest=_sha256_file(weights_path),
            ),
            tensor_names=weight_keys,
            tensor_count=len(weight_keys),
            total_tensor_bytes=weights_path.stat().st_size,
        )
    _copy_tokenizer_assets(
        hf_snapshot_dir,
        out_dir,
        qwen_chat_template_fallback=adapter.qwen_chat_template_fallback,
        tokenizer_layout=adapter.tokenizer_layout,
    )

    # I4 converter half: fail closed BEFORE shipping. The GraphIR declares the
    # exact canonical keys each op reads (including q/k/v bias when bias_present);
    # a declared key absent from the written artifact would let the mlx path
    # silently run bias-free while numpyref raises. Assert the package is
    # self-consistent here, at build time, so no runtime path ever sees it.
    _assert_declared_weights_present(graph, weight_keys)
    manifest = {
        "version": {"major": 2, "minor": 0, "patch": 0},
        "fixture_mode": False,
        "source_checkpoint_reference": resolved_source_ref,
        "source_base_model_reference": resolved_source_ref,
        "decode_family": "masked_diffusion",
        "backend_memory_owner": "mlx",
        "forward_executor": "apple_mlx_backend",
        "runtime_impl": "python_mlx_baseline",
        "merged_weights_ref": weight_artifact.spec.primary_ref,
        "source_checkpoint_sha": resolved_source_sha,
        "source_base_model_sha": resolved_source_sha,
        "source_license": provenance["source_license"],
        "merge_script_version": MASKED_DIFFUSION_CONVERTER_VERSION,
        "model_config": model_config,
        "graph_ref": "graph.json",
        "ir_version": dict(graph.ir_version),
        "output_hash": weight_artifact.spec.digest,
    }
    if descriptor is not None or definition is None:
        manifest.update(
            {
                "model_precision": precision,
                "admitted_backend": backend,
            }
        )
    if descriptor is not None:
        # Keep Product package bytes compatible with the established converter.
        # A descriptor already names the exact model definition; duplicating a
        # second identity field changed model.json and invalidated provider
        # package hashes without adding information.
        manifest.update(
            {
                "model_descriptor_key": descriptor.key,
                "model_descriptor_digest": descriptor.descriptor_digest,
                "graph_ir_family": definition.graph_ir_family,
            }
        )
    elif definition is not None:
        # Local packages carry their complete callable-free plan. They load the
        # same way with or without the original application catalog; the catalog
        # is an optional pin, never a source of missing runtime behavior.
        manifest.update(
            {
                "model_definition_key": definition.key,
                "model_definition_digest": model_definition_digest,
                "model_build_digest": model_build_digest(definition),
                "model_definition": model_definition_payload(definition),
                "graph_ir_family": definition.graph_ir_family,
                "package_format": weight_artifact.spec.format,
                "weight_storage_dtype": weight_artifact.spec.storage_dtype,
            }
        )
    (out_dir / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    (out_dir / "graph.json").write_text(graph.to_json(indent=2), encoding="utf-8", newline="\n")
    return out_dir


def _graph_declared_weight_keys(graph) -> set[str]:
    """All weight keys the GraphIR declares, from embedding, every layer op,
    final_norm, and lm_head (including q/k/v bias when bias_present)."""
    keys: set[str] = set()
    keys.update(graph.embedding.weight_keys.values())
    keys.update(graph.final_norm.weight_keys.values())
    keys.update(graph.lm_head.weight_keys.values())
    for layer in graph.layers:
        for op in layer.ops:
            keys.update(op.weight_keys.values())
    return keys


def _assert_declared_weights_present(graph, artifact_keys) -> None:
    """Fail closed if a GraphIR-declared key is absent from the weight artifact.

    A declared-but-missing key (e.g. a dropped q/k/v bias) would let the mlx
    forward run bias-free while the numpyref oracle raises — a package the two
    sanctioned paths disagree on. Reject it at build time.
    """
    missing = sorted(_graph_declared_weight_keys(graph) - set(artifact_keys))
    if missing:
        raise ValueError(
            "GraphIR declares weight keys absent from the weight artifact: "
            + ", ".join(missing[:8])
            + (f" ... (+{len(missing) - 8} more)" if len(missing) > 8 else "")
        )


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #


def _write_arrays_npz_streaming(
    weights_path,
    items,
    *,
    tied_weight_keys: tuple[str, str] | None = None,
) -> None:
    """Write an npz one array at a time, never holding them all.

    ``items`` is an iterable of ``(name, array)`` consumed lazily, so peak memory
    is one array rather than the whole model. Dream-7B's fp32 payload is
    30,462,466,048 B; the largest single tensor is 2.03 GiB and its conversion
    peaks at a MEASURED 3.218 GiB -- the bf16 source and the fp32 result are alive
    together, so it is 1.5x the tensor, not 1.0x. That is roughly an 8.8x
    reduction, and it is a property of the conversion path rather than of this
    writer.

    ``force_zip64`` is required: entries above 4 GiB would otherwise raise, and a
    writer that fails only on large models is worse than one that fails on all.

    Output is byte-equal to ``np.savez`` -- asserted by test, because a rewrite
    that changed the bytes would silently invalidate every existing package.

    When tied weights are declared, bounded-chunk content signatures compare the
    two tensors without retaining either tensor or materializing the model.
    """
    import stat
    import tempfile
    import zipfile

    from numpy.lib import format as npformat

    tied_signatures: dict[str, tuple[tuple[int, ...], str, str]] = {}
    iterator = iter(items)
    partial = None
    try:
        try:
            required_tied = set(tied_weight_keys or ())
            if tied_weight_keys is not None and len(required_tied) != 2:
                raise ValueError("declared_tied_embeddings_keys_invalid")

            target = Path(weights_path)
            target_mode = (
                stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
            )
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".partial",
                delete=False,
            ) as handle:
                partial = Path(handle.name)
            seen: set[str] = set()
            with zipfile.ZipFile(
                partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as archive:
                while True:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        break
                    name = array = None
                    try:
                        name, array = item
                        if name in seen:
                            raise ValueError(
                                f"duplicate tensor name across shards: {name!r}"
                            )
                        seen.add(name)
                        if name in required_tied:
                            signature = _array_content_signature(array)
                            if tied_signatures and signature != next(
                                iter(tied_signatures.values())
                            ):
                                raise ValueError("declared_tied_embeddings_mismatch")
                            tied_signatures[name] = signature
                        member = zipfile.ZipInfo(f"{name}.npy")
                        member.compress_type = zipfile.ZIP_STORED
                        # The same bytes on every platform: the archive's digest is
                        # part of the package identity. 3 is what Unix records.
                        member.create_system = 3
                        with archive.open(member, "w", force_zip64=True) as entry:
                            npformat.write_array(entry, array, allow_pickle=False)
                    finally:
                        del name, array, item
            missing_tied = sorted(required_tied - tied_signatures.keys())
            if missing_tied:
                raise ValueError(
                    "declared_tied_embeddings_missing: " + ", ".join(missing_tied)
                )
        except BaseException:
            with contextlib.suppress(Exception):
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            raise
        else:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        if partial is None:
            raise RuntimeError("streaming archive did not create a temporary file")
        partial.chmod(target_mode)
        partial.replace(target)
    except BaseException:
        if partial is not None:
            partial.unlink(missing_ok=True)
        raise


def _array_content_signature(array) -> tuple[tuple[int, ...], str, str]:
    """Hash an array in bounded chunks without retaining another tensor copy."""
    value = np.asarray(array)
    digest = hashlib.sha256()
    chunks = np.nditer(
        value,
        flags=("external_loop", "buffered", "zerosize_ok"),
        op_flags=("readonly",),
        order="C",
        buffersize=1 << 18,
    )
    for chunk in chunks:
        digest.update(memoryview(np.asarray(chunk)).cast("B"))
    return tuple(value.shape), value.dtype.str, digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _SafetensorsShardPlan:
    path: Path
    expected_tensor_names: frozenset[str] | None


def _validated_safetensors_shard_keys(
    handle, shard: _SafetensorsShardPlan
) -> list[str]:
    """Return deterministic keys after enforcing an indexed shard's exact set."""
    actual = set(handle.keys())
    expected = shard.expected_tensor_names
    if expected is not None and actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "safetensors_index_tensor_set_mismatch: "
            f"shard={shard.path.name!r} missing={missing!r} extra={extra!r}"
        )
    return sorted(actual)


def _mapped_weight_tensors(
    hf_snapshot_dir: Path,
    *,
    tensor_name_mapper,
):
    """Yield canonical names and source-dtype CPU tensors exactly once."""
    from safetensors import safe_open  # lazy: keep module import torch/mlx-free

    seen_engine_keys: set[str] = set()
    for shard in _safetensors_shard_plan(hf_snapshot_dir):
        with safe_open(str(shard.path), framework="pt") as handle:
            for source_key in _validated_safetensors_shard_keys(handle, shard):
                engine_key = tensor_name_mapper(source_key)
                if engine_key in seen_engine_keys:
                    raise ValueError(
                        f"duplicate engine key {engine_key!r} from {source_key!r}"
                    )
                seen_engine_keys.add(engine_key)
                tensor = handle.get_tensor(source_key)
                try:
                    yield engine_key, tensor
                finally:
                    del tensor


def _write_mapped_weights_npz(
    hf_snapshot_dir: Path,
    weights_path: Path,
    *,
    tensor_name_mapper,
    tied_weight_keys: tuple[str, ...] = (),
) -> None:
    """Write the established Product NPZ encoding from the shared tensor stream."""
    import torch  # lazy: only needed when actually building a package

    def _stream():
        for engine_key, tensor in _mapped_weight_tensors(
            hf_snapshot_dir,
            tensor_name_mapper=tensor_name_mapper,
        ):
            array = _tensor_to_numpy_float32(tensor, torch_module=torch)
            del tensor
            yield engine_key, array
            del array

    # Streamed, not accumulated: holding every fp32 tensor first meant a
    # 30,462,466,048 B payload for Dream-7B before one byte reached disk. The
    # duplicate-name guard moved into the writer, so the fail-closed behaviour
    # against a shard/index miss is preserved.
    _write_arrays_npz_streaming(
        weights_path,
        _stream(),
        tied_weight_keys=(tied_weight_keys or None),
    )


def _write_mapped_weights_store(
    hf_snapshot_dir: Path,
    package_dir: Path,
    *,
    tensor_name_mapper,
    tied_weight_keys: tuple[str, ...] = (),
) -> WrittenWeightArtifact:
    """Write the Local dtype-preserving store from the shared tensor stream."""

    return write_safetensors_weight_store(
        package_dir,
        _mapped_weight_tensors(
            hf_snapshot_dir,
            tensor_name_mapper=tensor_name_mapper,
        ),
        tied_weight_keys=tied_weight_keys,
    )


def _tensor_to_numpy_float32(tensor, *, torch_module):
    """Cast a torch tensor to a numpy array, promoting any float dtype to float32.

    bf16 (and fp16) cannot be represented by numpy here, and the engine upcasts
    all float weights to float32 on load anyway (``mlx_dream._load_mlx_weight_state``),
    so storing float32 is both necessary and lossless relative to what runs.
    Integer/bool tensors (if any) are passed through unchanged.
    """
    cpu_tensor = tensor.detach().to("cpu")
    if cpu_tensor.dtype.is_floating_point:
        cpu_tensor = cpu_tensor.to(dtype=torch_module.float32)
    return cpu_tensor.numpy()


def _safetensors_shard_paths(hf_snapshot_dir: Path) -> list[Path]:
    """Return lexical shard paths while retaining the internal indexed plan."""
    return [shard.path for shard in _safetensors_shard_plan(hf_snapshot_dir)]


def _safetensors_shard_plan(hf_snapshot_dir: Path) -> list[_SafetensorsShardPlan]:
    """Resolve safetensors shards and their authoritative indexed tensor sets.

    An index is authoritative when present and is parsed strictly, including
    duplicate-key preservation for ``weight_map``. Without an index, only the
    canonical single-file ``model.safetensors`` layout is accepted. Shard refs
    are checked lexically before joining, while ``is_file()`` intentionally keeps
    Hugging Face cache symlinks valid.
    """
    index_path = hf_snapshot_dir / "model.safetensors.index.json"

    class _JSONObjectPairs(list):
        pass

    if index_path.exists() or index_path.is_symlink():

        def _reject_non_rfc_constant(value):
            raise ValueError(f"non-RFC JSON constant: {value}")

        try:
            index = json.loads(
                index_path.read_text(encoding="utf-8"),
                object_pairs_hook=_JSONObjectPairs,
                parse_constant=_reject_non_rfc_constant,
            )
        except (OSError, UnicodeError, ValueError, RecursionError):
            raise ValueError(f"safetensors_index_malformed: {index_path}") from None

        if not isinstance(index, _JSONObjectPairs):
            raise ValueError(f"safetensors_index_malformed: {index_path}")

        root_fields = {}
        for field_name, value in index:
            if field_name in root_fields:
                raise ValueError(
                    f"safetensors_index_malformed: duplicate root field {field_name!r}"
                )
            root_fields[field_name] = value

        if "weight_map" not in root_fields:
            raise ValueError(f"safetensors_index_empty: {index_path}")
        weight_map = root_fields["weight_map"]
        if not isinstance(weight_map, _JSONObjectPairs):
            raise ValueError(f"safetensors_index_malformed: {index_path}")
        if not weight_map:
            raise ValueError(f"safetensors_index_empty: {index_path}")

        tensor_shards: dict[str, str] = {}
        shard_tensor_names: dict[str, set[str]] = {}
        for tensor_name, shard_name in weight_map:
            if (
                not isinstance(tensor_name, str)
                or not tensor_name
                or not isinstance(shard_name, str)
                or not shard_name
            ):
                raise ValueError(
                    "safetensors_index_invalid_entry: "
                    f"tensor={tensor_name!r} shard={shard_name!r}"
                )

            if tensor_name in tensor_shards:
                previous_shard = tensor_shards[tensor_name]
                error = (
                    "safetensors_index_duplicate_tensor"
                    if shard_name == previous_shard
                    else "safetensors_index_inconsistent_tensor"
                )
                raise ValueError(
                    f"{error}: tensor={tensor_name!r} "
                    f"shards=({previous_shard!r}, {shard_name!r})"
                )

            posix_ref = PurePosixPath(shard_name)
            windows_ref = PureWindowsPath(shard_name)
            if (
                "\0" in shard_name
                or "/" in shard_name
                or "\\" in shard_name
                or ":" in shard_name
                or shard_name in {".", ".."}
                or posix_ref.is_absolute()
                or windows_ref.is_absolute()
                or bool(windows_ref.drive)
                or posix_ref.name != shard_name
                or windows_ref.name != shard_name
                or not shard_name.endswith(".safetensors")
            ):
                raise ValueError(f"safetensors_index_shard_ref_unsafe: {shard_name!r}")
            tensor_shards[tensor_name] = shard_name
            shard_tensor_names.setdefault(shard_name, set()).add(tensor_name)

        shard_plan = []
        for shard_name in sorted(shard_tensor_names):
            shard_path = hf_snapshot_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"safetensors_index_shard_missing: {shard_name!r}"
                )
            shard_plan.append(
                _SafetensorsShardPlan(
                    path=shard_path,
                    expected_tensor_names=frozenset(shard_tensor_names[shard_name]),
                )
            )
        return shard_plan

    shard_paths = sorted(
        path for path in hf_snapshot_dir.glob("*.safetensors") if path.is_file()
    )
    if not shard_paths:
        raise FileNotFoundError(f"no safetensors shards found in {hf_snapshot_dir}")
    if len(shard_paths) > 1:
        names = ", ".join(path.name for path in shard_paths)
        raise ValueError(f"safetensors_unindexed_ambiguous: {names}")
    if shard_paths[0].name != "model.safetensors":
        raise ValueError(f"safetensors_unindexed_noncanonical: {shard_paths[0].name!r}")
    return [
        _SafetensorsShardPlan(
            path=shard_paths[0],
            expected_tensor_names=None,
        )
    ]


def _tie_word_embeddings(
    config: dict, *, upstream_field: str = "tie_word_embeddings"
) -> bool:
    if upstream_field not in config:
        raise ValueError(f"converter_{upstream_field}_missing")
    value = config[upstream_field]
    if type(value) is not bool:
        raise ValueError(
            f"converter_{upstream_field}_invalid: expected=bool actual={value!r}"
        )
    return value


# --------------------------------------------------------------------------- #
# Tokenizer assets
# --------------------------------------------------------------------------- #


def _copy_tokenizer_assets(
    hf_snapshot_dir: Path,
    out_dir: Path,
    *,
    qwen_chat_template_fallback: bool = False,
    tokenizer_layout: TokenizerLayout = TokenizerLayout.VERBATIM,
) -> None:
    """Copy the tokenizer files into the package and bridge a chat template.

    Copies the standard transformers tokenizer assets verbatim. The chat template
    is resolved in priority order:

    1. A standalone ``chat_template.jinja`` in the snapshot is copied as-is.
    2. An INLINE chat template in ``tokenizer_config.json`` is written out as
       ``chat_template.jinja`` so the loader's ``.jinja`` bridge (tokenizer.py
       ``_load_package_chat_template``) can pick it up.
    3. Otherwise, fall back to the bundled canonical Qwen3 chat template. The
       Qwen3-MDLM snapshot ships NO chat template, yet it IS a Qwen3 chat model
       (``<|im_start|>``/``<|im_end|>`` turns, ``enable_thinking`` gate); the
       engine's chat path needs a template, so this fallback supplies the
       upstream Qwen3 one (verbatim from ``Qwen/Qwen3-0.6B``).

    ``dream_qwen2_builtin`` is a pinned equivalence normalization: the upstream
    Dream tokenizer is the Transformers Qwen2 slow tokenizer under a custom
    class name. Its declaration is rewritten to ``Qwen2Tokenizer`` and the
    remote Python modules are excluded, so product loading remains
    ``trust_remote_code=False``.

    Missing optional tokenizer assets are skipped silently.
    """
    tokenizer_config_path = hf_snapshot_dir / "tokenizer_config.json"
    tokenizer_config = None
    if tokenizer_layout == "dream_qwen2_builtin":
        if not tokenizer_config_path.is_file():
            raise FileNotFoundError(
                f"Dream tokenizer config missing: {tokenizer_config_path}"
            )
        tokenizer_config = _read_json(tokenizer_config_path)
        expected_auto_map = {
            "AutoTokenizer": ["tokenization_dream.DreamTokenizer", None]
        }
        if (
            tokenizer_config.get("tokenizer_class") != "DreamTokenizer"
            or tokenizer_config.get("auto_map") != expected_auto_map
        ):
            raise ValueError("dream_tokenizer_builtin_equivalence_contract_mismatch")
        tokenizer_config = dict(tokenizer_config)
        tokenizer_config["tokenizer_class"] = "Qwen2Tokenizer"
        tokenizer_config.pop("auto_map")
    elif tokenizer_layout != "verbatim":
        raise ValueError(f"unknown_tokenizer_layout: {tokenizer_layout}")

    for name in _TOKENIZER_ASSET_NAMES:
        if tokenizer_layout == "dream_qwen2_builtin" and name in {
            "tokenizer_config.json",
            "tokenization_dream.py",
            "configuration_dream.py",
        }:
            continue
        source = hf_snapshot_dir / name
        if source.exists():
            shutil.copyfile(source, out_dir / name)
    if tokenizer_config is not None:
        (out_dir / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config, indent=2, sort_keys=True),
            encoding="utf-8", newline="\n",
        )

    standalone_template = hf_snapshot_dir / "chat_template.jinja"
    if standalone_template.exists():
        shutil.copyfile(standalone_template, out_dir / "chat_template.jinja")
        return

    if tokenizer_config_path.exists():
        source_tokenizer_config = _read_json(tokenizer_config_path)
        chat_template = source_tokenizer_config.get("chat_template")
        if isinstance(chat_template, str) and chat_template.strip():
            (out_dir / "chat_template.jinja").write_text(chat_template, encoding="utf-8", newline="\n")
            return

    if qwen_chat_template_fallback:
        fallback = _BUNDLED_QWEN3_CHAT_TEMPLATE_PATH
        if not fallback.exists():
            raise FileNotFoundError(f"Qwen chat template missing: {fallback}")
        (out_dir / "chat_template.jinja").write_text(
            fallback.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )


def _resolve_mask_token_id(hf_snapshot_dir: Path, config: dict) -> int:
    """One rule, shared with the merge converter and the runtime.

    Delegates to edllm.control_tokens, which first accepts a valid explicit
    model-config id, then reads added_tokens.json or tokenizer_config.json's
    added_tokens_decoder -- and refuses rather than returning a sentinel. It
    returned -1 once, and every caller that treated a negative as "absent"
    then invented something.
    """
    from ..control_tokens import resolve_from_snapshot

    return resolve_from_snapshot(hf_snapshot_dir, config)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_revision(hf_snapshot_dir: Path) -> str:
    """A stable >=7-char provenance string for the snapshot.

    The HF cache lays snapshots out under ``snapshots/<commit_sha>/``; use that
    directory name when it looks like a hex sha, else hash the absolute path. The
    manifest schema requires source_*_sha minLength 7.
    """
    name = Path(hf_snapshot_dir).name
    if len(name) >= 7 and all(ch in "0123456789abcdef" for ch in name.lower()):
        return name
    return hashlib.sha256(str(Path(hf_snapshot_dir).resolve()).encode()).hexdigest()
