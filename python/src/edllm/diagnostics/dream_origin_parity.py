from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Sequence


Layer0Trace = dict[str, dict[str, Any]]

_LAYER0_TRACE_STAGE_ORDER = (
    "embed",
    "input_layernorm",
    "q_linear",
    "k_linear",
    "v_linear",
    "q_after_rope",
    "k_after_rope",
    "attention_output",
    "attention_projected",
    "layer0_output",
)


@dataclass(frozen=True, slots=True)
class FirstStepInputs:
    prompt: str
    prompt_token_ids: list[int]
    mask_token_id: int
    max_tokens: int
    diffusion_steps: int
    local_rows: list[list[float]]
    upstream_rows: list[list[float]]
    random_draws: list[float]
    local_raw_rows: list[list[float]] | None = None
    upstream_raw_rows: list[list[float]] | None = None
    local_layer0_trace: Layer0Trace | None = None
    upstream_layer0_trace: Layer0Trace | None = None
    eps: float = 1e-3
    top_k: int = 5


def build_self_test_report() -> dict:
    rows = [
        _peaked_row(256, 16, secondary_ids=(15, 11, 220, 17)),
        _peaked_row(256, 15, secondary_ids=(16, 11, 220, 17)),
        _peaked_row(256, 11, secondary_ids=(16, 15, 220, 17)),
        _peaked_row(256, 220, secondary_ids=(16, 15, 11, 17)),
    ]
    layer0_trace = {
        "embed": {
            "shape": [7, 2],
            "rows": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]],
        }
    }
    for stage in _LAYER0_TRACE_STAGE_ORDER:
        layer0_trace.setdefault(
            stage,
            {
                "shape": [7, 1, 2],
                "rows": [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 0.0]]],
            },
        )
    return build_first_step_report(
        FirstStepInputs(
            prompt="Hello, ",
            prompt_token_ids=[9707, 11, 220],
            mask_token_id=151666,
            max_tokens=4,
            diffusion_steps=4,
            local_rows=rows,
            upstream_rows=[list(row) for row in rows],
            local_raw_rows=[list(row) for row in rows],
            upstream_raw_rows=[list(row) for row in rows],
            local_layer0_trace=layer0_trace,
            upstream_layer0_trace=layer0_trace,
            random_draws=[0.0, 1.0, 1.0, 1.0],
        ),
        mode="self_test",
    )


def build_first_step_report(inputs: FirstStepInputs, *, mode: str) -> dict:
    if inputs.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if inputs.diffusion_steps <= 1:
        raise ValueError("diffusion_steps must be greater than 1")
    if len(inputs.local_rows) != inputs.max_tokens:
        raise ValueError("local_rows must have max_tokens rows")
    if len(inputs.upstream_rows) != inputs.max_tokens:
        raise ValueError("upstream_rows must have max_tokens rows")
    if inputs.local_raw_rows is not None and len(inputs.local_raw_rows) != inputs.max_tokens:
        raise ValueError("local_raw_rows must have max_tokens rows")
    if inputs.upstream_raw_rows is not None and len(inputs.upstream_raw_rows) != inputs.max_tokens:
        raise ValueError("upstream_raw_rows must have max_tokens rows")

    p_transfer = _first_step_origin_transfer_probability(inputs.diffusion_steps, inputs.eps)
    mask_positions = list(range(inputs.max_tokens))
    selected_positions = _origin_selected_positions(
        random_draws=inputs.random_draws,
        p_transfer=p_transfer,
        positions=mask_positions,
    )
    local_topk = [_topk(row, inputs.top_k) for row in inputs.local_rows]
    upstream_topk = [_topk(row, inputs.top_k) for row in inputs.upstream_rows]
    local_raw_topk = (
        [_topk(row, inputs.top_k) for row in inputs.local_raw_rows]
        if inputs.local_raw_rows is not None
        else None
    )
    upstream_raw_topk = (
        [_topk(row, inputs.top_k) for row in inputs.upstream_raw_rows]
        if inputs.upstream_raw_rows is not None
        else None
    )
    comparisons = _compare_topk(local_topk, upstream_topk)
    max_abs_diff = _max_abs_diff(inputs.local_rows, inputs.upstream_rows)
    comparisons["max_abs_diff"] = max_abs_diff
    comparisons["logits_match"] = max_abs_diff == 0.0
    # The origin-transfer selection is derived from the SHARED random_draws (one
    # draw set for both backends), so it is identical by construction and is NOT
    # an independent local-vs-upstream parity signal. It is surfaced under
    # first_step.selected_positions for inspection but intentionally excluded from
    # the parity verdict below — a hardcoded match here would be fake evidence.
    raw_rows_match = True
    if local_raw_topk is not None and upstream_raw_topk is not None:
        raw_comparisons = _compare_topk(local_raw_topk, upstream_raw_topk)
        comparisons["raw_top1_token_ids_match"] = raw_comparisons["top1_token_ids_match"]
        comparisons["raw_topk_token_ids_match"] = raw_comparisons["topk_token_ids_match"]
        raw_max_abs_diff = _max_abs_diff(inputs.local_raw_rows or [], inputs.upstream_raw_rows or [])
        comparisons["raw_max_abs_diff"] = raw_max_abs_diff
        comparisons["raw_logits_match"] = raw_max_abs_diff == 0.0
        raw_rows_match = (
            comparisons["raw_top1_token_ids_match"]
            and comparisons["raw_topk_token_ids_match"]
            and comparisons["raw_logits_match"]
        )
    verdict = (
        "match"
        if comparisons["top1_token_ids_match"]
        and comparisons["topk_token_ids_match"]
        and comparisons["logits_match"]
        and raw_rows_match
        else "diverge"
    )

    local = {"topk": local_topk, "shifted_mask_topk": local_topk}
    upstream = {"topk": upstream_topk, "shifted_mask_topk": upstream_topk}
    if local_raw_topk is not None:
        local["raw_topk"] = local_raw_topk
    if upstream_raw_topk is not None:
        upstream["raw_topk"] = upstream_raw_topk

    report = {
        "mode": mode,
        "prompt": inputs.prompt,
        "prompt_token_ids": inputs.prompt_token_ids,
        "mask_token_id": inputs.mask_token_id,
        "max_tokens": inputs.max_tokens,
        "diffusion_steps": inputs.diffusion_steps,
        "alg": "origin",
        "temperature": 0.0,
        "eps": inputs.eps,
        "first_step": {
            "p_transfer": p_transfer,
            "mask_positions": mask_positions,
            "random_draws": inputs.random_draws,
            "selected_positions": selected_positions,
        },
        "raw": {
            "absolute_positions": _shift_source_positions(
                len(inputs.prompt_token_ids),
                inputs.max_tokens,
            )
        },
        "local": local,
        "upstream": upstream,
        "comparisons": comparisons,
        "verdict": verdict,
    }
    if inputs.local_layer0_trace is not None and inputs.upstream_layer0_trace is not None:
        report["layer0_trace"] = _summarize_layer0_trace(
            local_trace=inputs.local_layer0_trace,
            upstream_trace=inputs.upstream_layer0_trace,
            positions=_shift_source_positions(len(inputs.prompt_token_ids), inputs.max_tokens),
        )
        # A clean top-k/logits match at the OUTPUT is not enough if an intermediate
        # layer-0 stage already diverged: the verdict must reflect that, else the
        # report claims "match" while the model internals disagree (label != evidence).
        if report["layer0_trace"].get("first_divergent_stage") is not None:
            report["verdict"] = "diverge"
            comparisons["first_divergent_stage"] = report["layer0_trace"]["first_divergent_stage"]
    return report


def build_real_first_step_report(
    *,
    local_package_dir: str | Path,
    hf_model: str,
    hf_adapter: str | None = None,
    prompt: str,
    max_tokens: int,
    diffusion_steps: int,
    device: str = "cpu",
    top_k: int = 5,
) -> dict:
    from edllm.model import load_runtime

    runtime = load_runtime(local_package_dir)
    try:
        if runtime.tokenizer is None:
            raise ValueError("runtime has no tokenizer")
        model_config = runtime.model_config or {}
        prompt_token_ids = runtime.tokenizer.encode(prompt)
        mask_token_id = int(
            model_config.get(
                "mask_token_id", runtime.tokenizer.token_to_id("<|mask|>")
            )
        )
        adapter_ref = hf_adapter or _infer_adapter_ref(
            runtime=runtime, hf_model=hf_model
        )
        local_raw_rows, local_rows = _local_first_step_raw_and_shifted_rows(
            runtime=runtime,
            prompt_token_ids=prompt_token_ids,
            mask_token_id=mask_token_id,
            max_tokens=max_tokens,
        )
        upstream_raw_rows, upstream_rows = _upstream_first_step_raw_and_shifted_rows(
            hf_model=hf_model,
            hf_adapter=adapter_ref,
            prompt_token_ids=prompt_token_ids,
            mask_token_id=mask_token_id,
            max_tokens=max_tokens,
            device=device,
        )
        attention_mask_kind = str(
            (runtime.model_config or {}).get(
                "attention_mask_kind", "bidirectional"
            )
        )
        local_layer0_trace = _local_layer0_trace(
            runtime=runtime,
            prompt_token_ids=prompt_token_ids,
            mask_token_id=mask_token_id,
            max_tokens=max_tokens,
            attention_mask_kind=attention_mask_kind,
        )
        upstream_layer0_trace = _upstream_layer0_trace(
            hf_model=hf_model,
            hf_adapter=adapter_ref,
            prompt_token_ids=prompt_token_ids,
            mask_token_id=mask_token_id,
            max_tokens=max_tokens,
            device=device,
        )
        report = build_first_step_report(
            FirstStepInputs(
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                mask_token_id=mask_token_id,
                max_tokens=max_tokens,
                diffusion_steps=diffusion_steps,
                local_rows=local_rows,
                upstream_rows=upstream_rows,
                local_raw_rows=local_raw_rows,
                upstream_raw_rows=upstream_raw_rows,
                local_layer0_trace=local_layer0_trace,
                upstream_layer0_trace=upstream_layer0_trace,
                # Deterministic diagnostic draws: first position transfers,
                # others hold. This avoids claiming parity from unseeded RNG.
                random_draws=[0.0, *([1.0] * max(0, max_tokens - 1))],
                top_k=top_k,
            ),
            mode="real_first_step",
        )
        report["local"]["package_dir"] = str(local_package_dir)
        report["local"]["attention_mask_kind"] = attention_mask_kind
        report["upstream"]["hf_model"] = hf_model
        report["upstream"]["hf_adapter"] = adapter_ref
        report["upstream"]["device"] = device
        return report
    finally:
        runtime.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dream origin first-step local/upstream parity diagnostic."
    )
    parser.add_argument("--self-test", action="store_true", help="emit a synthetic report")
    parser.add_argument("--output-json", action="store_true", help="write the report as JSON")
    parser.add_argument("--local-package-dir", help="local merged Dream execution package")
    parser.add_argument("--hf-model", help="HF Dream model id or local path")
    parser.add_argument("--hf-adapter", help="optional HF LoRA adapter id or local path")
    parser.add_argument("--prompt", default="Hello, ")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            report = build_self_test_report()
        else:
            if not args.local_package_dir or not args.hf_model:
                parser.error("real mode requires --local-package-dir and --hf-model")
            report = build_real_first_step_report(
                local_package_dir=args.local_package_dir,
                hf_model=args.hf_model,
                hf_adapter=args.hf_adapter,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                diffusion_steps=args.diffusion_steps,
                device=args.device,
                top_k=args.top_k,
            )
    except Exception as exc:
        report = {
            "mode": "real_first_step",
            "verdict": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        if args.output_json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.output_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"verdict={report['verdict']}")
    return 0


def _first_step_origin_transfer_probability(diffusion_steps: int, eps: float) -> float:
    # Matches torch.linspace(1, eps, steps + 1) for the first origin step.
    s = 1.0 + (eps - 1.0) / float(diffusion_steps)
    return 1.0 - s


def _origin_selected_positions(
    *,
    random_draws: list[float],
    p_transfer: float,
    positions: list[int],
) -> list[int]:
    if len(random_draws) < len(positions):
        raise ValueError("random_draws must cover every mask position")
    return [position for position, draw in zip(positions, random_draws, strict=True) if draw < p_transfer]


def _local_first_step_raw_and_shifted_rows(
    *,
    runtime,
    prompt_token_ids: list[int],
    mask_token_id: int,
    max_tokens: int,
) -> tuple[list[list[float]], list[list[float]]]:
    from edllm.executor import ExecutorInput

    executor = runtime.make_executor()
    block_token_ids = [mask_token_id] * max_tokens
    mask_pattern = [False] * len(prompt_token_ids) + [True] * max_tokens
    attention_mask_kind = str((runtime.model_config or {}).get("attention_mask_kind", "bidirectional"))
    executor_input = ExecutorInput(
        prompt_token_ids=prompt_token_ids,
        block_token_ids=block_token_ids,
        block_position=0,
        mask_pattern=mask_pattern,
        attention_mask_kind=attention_mask_kind,
    )
    full_logits = executor._forward(
        [*prompt_token_ids, *block_token_ids],
        attention_mask_kind=attention_mask_kind,
        mask_pattern=mask_pattern,
    )
    raw_rows = full_logits[_shift_source_positions(len(prompt_token_ids), max_tokens)]
    block_logits = executor.step(executor_input)
    return raw_rows.astype("float16").tolist(), block_logits.values


def _upstream_first_step_raw_and_shifted_rows(
    *,
    hf_model: str,
    hf_adapter: str | None,
    prompt_token_ids: list[int],
    mask_token_id: int,
    max_tokens: int,
    device: str,
) -> tuple[list[list[float]], list[list[float]]]:
    import torch

    model = _load_upstream_dream_model(hf_model=hf_model, hf_adapter=hf_adapter, device=device)
    input_ids = torch.tensor(
        [prompt_token_ids + [mask_token_id] * max_tokens],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask="full").logits
        shifted_logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        block = shifted_logits[:, len(prompt_token_ids): len(prompt_token_ids) + max_tokens, :]
        raw = logits[:, _shift_source_positions(len(prompt_token_ids), max_tokens), :]
    return (
        raw[0].detach().cpu().float().numpy().tolist(),
        block[0].detach().cpu().float().numpy().tolist(),
    )


def _local_layer0_trace(
    *,
    runtime,
    prompt_token_ids: list[int],
    mask_token_id: int,
    max_tokens: int,
    attention_mask_kind: str,
) -> Layer0Trace:
    import numpy as np

    executor = runtime.make_executor()
    token_ids = [*prompt_token_ids, *([mask_token_id] * max_tokens)]
    positions = _shift_source_positions(len(prompt_token_ids), max_tokens)
    hidden_states = executor._weight("model.embed_tokens.weight")[token_ids].astype(np.float32)

    trace: Layer0Trace = {"embed": _numpy_trace_stage(hidden_states, positions)}

    residual = hidden_states
    normalized = executor._rms_norm(
        hidden_states,
        executor._weight("model.layers.0.input_layernorm.weight"),
    )
    trace["input_layernorm"] = _numpy_trace_stage(normalized, positions)

    q = executor._linear(
        normalized,
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
    )
    k = executor._linear(
        normalized,
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
    )
    v = executor._linear(
        normalized,
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.bias",
    )

    seq_len = q.shape[0]
    q = q.reshape(seq_len, executor._n_attention_heads, executor._head_dim)
    k = k.reshape(seq_len, executor._num_kv_heads, executor._head_dim)
    v = v.reshape(seq_len, executor._num_kv_heads, executor._head_dim)
    trace["q_linear"] = _numpy_trace_stage(q, positions)
    trace["k_linear"] = _numpy_trace_stage(k, positions)
    trace["v_linear"] = _numpy_trace_stage(v, positions)
    q, k, v = executor._apply_rope(q, k, v)
    trace["q_after_rope"] = _numpy_trace_stage(q, positions)
    trace["k_after_rope"] = _numpy_trace_stage(k, positions)

    if executor._num_kv_heads != executor._n_attention_heads:
        k = np.repeat(k, executor._kv_repeat, axis=1)
        v = np.repeat(v, executor._kv_repeat, axis=1)

    attention_output = executor._attention(q, k, v, attention_mask_kind=attention_mask_kind)
    trace["attention_output"] = _numpy_trace_stage(attention_output, positions)

    attention_projected = executor._linear(
        attention_output,
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.o_proj.bias",
    )
    trace["attention_projected"] = _numpy_trace_stage(attention_projected, positions)

    hidden_states = residual + attention_projected
    residual = hidden_states
    hidden_states = executor._rms_norm(
        hidden_states,
        executor._weight("model.layers.0.post_attention_layernorm.weight"),
    )
    gate = executor._silu(
        executor._linear(
            hidden_states,
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_proj.bias",
        )
    )
    up = executor._linear(
        hidden_states,
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.bias",
    )
    hidden_states = executor._linear(gate * up, "model.layers.0.mlp.down_proj.weight")
    trace["layer0_output"] = _numpy_trace_stage(residual + hidden_states, positions)
    return trace


def _upstream_layer0_trace(
    *,
    hf_model: str,
    hf_adapter: str | None,
    prompt_token_ids: list[int],
    mask_token_id: int,
    max_tokens: int,
    device: str,
) -> Layer0Trace:
    import importlib

    import torch
    import torch.nn.functional as F

    model = _load_upstream_dream_model(hf_model=hf_model, hf_adapter=hf_adapter, device=device)
    input_ids = torch.tensor(
        [prompt_token_ids + [mask_token_id] * max_tokens],
        dtype=torch.long,
        device=device,
    )
    positions = _shift_source_positions(len(prompt_token_ids), max_tokens)

    with torch.no_grad():
        model_inner = model.model
        layer0 = model_inner.layers[0]
        self_attn = layer0.self_attn
        hidden_states = model_inner.embed_tokens(input_ids)

        trace: Layer0Trace = {"embed": _torch_trace_stage(hidden_states[0], positions)}

        residual = hidden_states
        normalized = layer0.input_layernorm(hidden_states)
        trace["input_layernorm"] = _torch_trace_stage(normalized[0], positions)

        bsz, q_len, _hidden = normalized.size()
        num_heads = int(getattr(self_attn, "num_heads", model.config.num_attention_heads))
        num_kv_heads = int(
            getattr(
                self_attn,
                "num_key_value_heads",
                getattr(model.config, "num_key_value_heads", num_heads),
            )
        )
        head_dim = int(
            getattr(
                self_attn,
                "head_dim",
                self_attn.q_proj.out_features // max(1, num_heads),
            )
        )

        query_states = self_attn.q_proj(normalized)
        key_states = self_attn.k_proj(normalized)
        value_states = self_attn.v_proj(normalized)
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        trace["q_linear"] = _torch_trace_stage(query_states[0].transpose(0, 1), positions)
        trace["k_linear"] = _torch_trace_stage(key_states[0].transpose(0, 1), positions)
        trace["v_linear"] = _torch_trace_stage(value_states[0].transpose(0, 1), positions)

        position_ids = torch.arange(q_len, device=input_ids.device).unsqueeze(0)
        cos, sin = _call_upstream_rotary_emb(model_inner.rotary_emb, value_states, position_ids)
        module = importlib.import_module(self_attn.__class__.__module__)
        apply_rotary_pos_emb = module.apply_rotary_pos_emb
        query_states, key_states = _call_upstream_apply_rotary_pos_emb(
            apply_rotary_pos_emb,
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )
        trace["q_after_rope"] = _torch_trace_stage(query_states[0].transpose(0, 1), positions)
        trace["k_after_rope"] = _torch_trace_stage(key_states[0].transpose(0, 1), positions)

        repeat_kv = getattr(module, "repeat_kv", None)
        n_rep = max(1, num_heads // max(1, num_kv_heads))
        if n_rep != 1:
            if repeat_kv is not None:
                key_states = repeat_kv(key_states, n_rep)
                value_states = repeat_kv(value_states, n_rep)
            else:
                key_states = key_states.repeat_interleave(n_rep, dim=1)
                value_states = value_states.repeat_interleave(n_rep, dim=1)
        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(bsz, q_len, num_heads * head_dim)
        trace["attention_output"] = _torch_trace_stage(attention_output[0], positions)

        attention_projected = self_attn.o_proj(attention_output)
        trace["attention_projected"] = _torch_trace_stage(attention_projected[0], positions)

        hidden_states = residual + attention_projected
        residual = hidden_states
        hidden_states = layer0.post_attention_layernorm(hidden_states)
        hidden_states = layer0.mlp(hidden_states)
        trace["layer0_output"] = _torch_trace_stage((residual + hidden_states)[0], positions)
        return trace


def _load_upstream_dream_model(*, hf_model: str, hf_adapter: str | None, device: str):
    from transformers import AutoModel
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    ensure_default_rope_registry(ROPE_INIT_FUNCTIONS)
    model = AutoModel.from_pretrained(hf_model, trust_remote_code=True)
    if hf_adapter:
        from peft import PeftModel

        _ensure_torch_dtensor_namespace_for_peft()
        model = PeftModel.from_pretrained(model, hf_adapter).merge_and_unload()
    model.eval()
    model.to(device)
    return model


def _ensure_torch_dtensor_namespace_for_peft() -> None:
    try:
        import torch.distributed.tensor  # noqa: F401
    except Exception:
        return


def _call_upstream_rotary_emb(rotary_emb, value_states, position_ids):
    try:
        return rotary_emb(value_states, position_ids)
    except TypeError:
        return rotary_emb(value_states, seq_len=position_ids.shape[-1])


def _call_upstream_apply_rotary_pos_emb(
    apply_rotary_pos_emb,
    query_states,
    key_states,
    cos,
    sin,
    position_ids,
):
    try:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    except TypeError:
        try:
            return apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                position_ids=position_ids,
            )
        except TypeError:
            return apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                unsqueeze_dim=1,
            )


def _numpy_trace_stage(array, positions: list[int]) -> dict[str, Any]:
    import numpy as np

    selected = np.asarray(array[positions], dtype=np.float32)
    return {
        "shape": [int(dim) for dim in array.shape],
        "rows": selected.tolist(),
    }


def _torch_trace_stage(tensor, positions: list[int]) -> dict[str, Any]:
    import torch

    position_tensor = torch.tensor(positions, dtype=torch.long, device=tensor.device)
    selected = tensor.index_select(0, position_tensor)
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "rows": selected.detach().cpu().float().numpy().tolist(),
    }


def _summarize_layer0_trace(
    *,
    local_trace: Layer0Trace,
    upstream_trace: Layer0Trace,
    positions: list[int],
) -> dict[str, Any]:
    local_stages = set(local_trace)
    upstream_stages = set(upstream_trace)
    common_stages = local_stages.intersection(upstream_stages)
    ordered_stages = [
        stage for stage in _LAYER0_TRACE_STAGE_ORDER if stage in common_stages
    ]
    ordered_stages.extend(sorted(common_stages.difference(ordered_stages)))
    stages = {
        stage: _summarize_trace_stage(local_trace[stage], upstream_trace[stage])
        for stage in ordered_stages
    }
    divergence_threshold = 1.0
    return {
        "positions": [int(position) for position in positions],
        "divergence_threshold": divergence_threshold,
        "missing_stages": {
            "local": _ordered_missing_stages(upstream_stages.difference(local_stages)),
            "upstream": _ordered_missing_stages(local_stages.difference(upstream_stages)),
        },
        "first_divergent_stage": _first_divergent_stage(
            stages=stages,
            ordered_stages=ordered_stages,
            threshold=divergence_threshold,
        ),
        "stages": stages,
    }


def _ordered_missing_stages(stages: set[str]) -> list[str]:
    ordered = [stage for stage in _LAYER0_TRACE_STAGE_ORDER if stage in stages]
    ordered.extend(sorted(stages.difference(ordered)))
    return ordered


def _first_divergent_stage(
    *,
    stages: dict[str, dict[str, Any]],
    ordered_stages: list[str],
    threshold: float,
) -> str | None:
    for stage in ordered_stages:
        if float(stages[stage]["comparison"]["max_abs_diff"]) > threshold:
            return stage
    return None


def _summarize_trace_stage(local_stage: dict[str, Any], upstream_stage: dict[str, Any]) -> dict:
    local_rows = _trace_stage_rows(local_stage)
    upstream_rows = _trace_stage_rows(upstream_stage)
    return {
        "local": _trace_side_summary(local_stage, local_rows),
        "upstream": _trace_side_summary(upstream_stage, upstream_rows),
        "comparison": _trace_stage_comparison(local_rows, upstream_rows),
    }


def _trace_stage_rows(stage: dict[str, Any]) -> list[Any]:
    rows = stage.get("rows")
    if rows is None:
        raise ValueError("layer0 trace stage is missing rows")
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    return list(rows)


def _trace_side_summary(stage: dict[str, Any], rows: list[Any]) -> dict[str, Any]:
    shape = stage.get("shape")
    return {
        "shape": _json_shape(shape) if shape is not None else _nested_shape(rows),
        "selected_row_count": len(rows),
        "selected_row_l2_norms": [_l2_norm(_flatten_numbers(row)) for row in rows],
    }


def _trace_stage_comparison(local_rows: list[Any], upstream_rows: list[Any]) -> dict[str, float]:
    if len(local_rows) != len(upstream_rows):
        raise ValueError("local and upstream trace stages must have matching row counts")
    max_abs_diff = 0.0
    squared_diff_sum = 0.0
    for local_row, upstream_row in zip(local_rows, upstream_rows, strict=True):
        local_values = _flatten_numbers(local_row)
        upstream_values = _flatten_numbers(upstream_row)
        if len(local_values) != len(upstream_values):
            raise ValueError("local and upstream trace rows must have matching flattened size")
        for local_value, upstream_value in zip(local_values, upstream_values, strict=True):
            diff = float(local_value) - float(upstream_value)
            max_abs_diff = max(max_abs_diff, abs(diff))
            squared_diff_sum += diff * diff
    return {
        "max_abs_diff": max_abs_diff,
        "l2_diff": math.sqrt(squared_diff_sum),
    }


def _json_shape(shape: Any) -> list[int]:
    if hasattr(shape, "tolist"):
        shape = shape.tolist()
    return [int(dim) for dim in shape]


def _nested_shape(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        return [len(value), *_nested_shape(value[0])]
    return []


def _flatten_numbers(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Real):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numbers(item))
        return flattened
    raise TypeError(f"unsupported trace row value: {type(value).__name__}")


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _shift_source_positions(prompt_token_count: int, max_tokens: int) -> list[int]:
    block_positions = range(prompt_token_count, prompt_token_count + max_tokens)
    return [0 if position == 0 else position - 1 for position in block_positions]


def _infer_adapter_ref(*, runtime, hf_model: str) -> str | None:
    adapter_ref = str(getattr(runtime, "source_checkpoint_reference", "") or "")
    base_ref = str(getattr(runtime, "source_base_model_reference", "") or "")
    if adapter_ref and adapter_ref not in {hf_model, base_ref}:
        return adapter_ref
    return None


def ensure_default_rope_registry(registry: dict | None = None) -> bool:
    if registry is None:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        registry = ROPE_INIT_FUNCTIONS
    if "default" in registry:
        return False
    registry["default"] = _compute_default_rope_parameters
    return True


def _compute_default_rope_parameters(
    config=None,
    device=None,
    seq_len: int | None = None,
    layer_type: str | None = None,
):
    del seq_len
    import torch

    if hasattr(config, "standardize_rope_params"):
        config.standardize_rope_params()
        rope_parameters = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
        base = rope_parameters["rope_theta"]
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
    else:
        base = getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


def _topk(row: Sequence[float], k: int) -> dict:
    ranked = sorted(enumerate(row), key=lambda item: item[1], reverse=True)[:k]
    top1_value = float(ranked[0][1]) if ranked else None
    top1_margin = (
        top1_value - float(ranked[1][1])
        if top1_value is not None and len(ranked) > 1
        else None
    )
    topk_margin = (
        top1_value - float(ranked[-1][1])
        if top1_value is not None and ranked
        else None
    )
    return {
        "token_ids": [int(token_id) for token_id, _value in ranked],
        "logits": [float(value) for _token_id, value in ranked],
        "top1_margin": top1_margin,
        "topk_margin": topk_margin,
    }


def _compare_topk(local_topk: list[dict], upstream_topk: list[dict]) -> dict[str, bool]:
    return {
        "top1_token_ids_match": [
            row["token_ids"][0] for row in local_topk
        ]
        == [row["token_ids"][0] for row in upstream_topk],
        "topk_token_ids_match": [
            row["token_ids"] for row in local_topk
        ]
        == [row["token_ids"] for row in upstream_topk],
    }


def _max_abs_diff(local_rows: list[list[float]], upstream_rows: list[list[float]]) -> float:
    max_diff = 0.0
    for local_row, upstream_row in zip(local_rows, upstream_rows, strict=True):
        if len(local_row) != len(upstream_row):
            raise ValueError("local and upstream rows must have matching vocab size")
        for local_value, upstream_value in zip(local_row, upstream_row, strict=True):
            max_diff = max(max_diff, abs(float(local_value) - float(upstream_value)))
    return max_diff


def _peaked_row(
    vocab_size: int,
    peak_id: int,
    *,
    secondary_ids: tuple[int, ...],
) -> list[float]:
    row = [-10.0] * vocab_size
    row[peak_id] = 10.0
    for offset, token_id in enumerate(secondary_ids, start=1):
        row[token_id] = 10.0 - offset
    return row


if __name__ == "__main__":
    raise SystemExit(main())
