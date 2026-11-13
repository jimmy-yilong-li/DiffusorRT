from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import ExitStack
import importlib
from pathlib import Path
import math
import re
from typing import Any
import weakref

import numpy as np

from .cache import DelayedKVCache
from .graph.numpyref import _validated_linear_rope_factor
from .package_artifacts import open_package_artifact

_MX = None

PROBED_MLX_EXECUTOR_SOURCE = "pooled_embedding_probe"
TRANSFORMER_NUMPY_EXECUTOR_SOURCE = "transformer_numpy_forward"
_BIDIRECTIONAL_LAYER0_KV_CACHE = "bidirectional_layer0_kv"


@dataclass(slots=True)
class BlockLogits:
    values: list[list[float]]
    shape: tuple[int, int]
    dtype: str
    block_span: tuple[int, int]
    vocab_size: int
    source: str = "python_scaffold_logits"
    cache_state_ref: str | None = None
    attention_metadata: dict[str, str | int] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutorInput:
    prompt_token_ids: list[int]
    block_token_ids: list[int]
    block_position: int
    generation_cursor: int | None = None
    mask_pattern: list[bool] | None = None
    cache_state_ref: str | None = None
    attention_mask_kind: str = "bidirectional"
    attention_metadata: dict[str, str | int] = field(default_factory=dict)


class FakeExecutor:
    def __init__(self, *, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    def step(self, executor_input: ExecutorInput) -> BlockLogits:
        values: list[list[float]] = []
        for idx, token_id in enumerate(executor_input.block_token_ids):
            row = [-8.0] * self._vocab_size
            peak_token_id = min(max(0, token_id), self._vocab_size - 1)
            if peak_token_id == self._vocab_size - 1 and self._vocab_size > 3:
                peak_token_id = 3 + ((executor_input.block_position + idx) % (self._vocab_size - 3))
            row[peak_token_id] = 1.0
            if self._vocab_size > 1:
                secondary = max(0, peak_token_id - 1)
                row[secondary] = 0.1
            values.append(row)
        return BlockLogits(
            values=values,
            shape=(len(executor_input.block_token_ids), self._vocab_size),
            dtype="float32",
            block_span=(
                executor_input.block_position,
                executor_input.block_position + len(executor_input.block_token_ids),
            ),
            vocab_size=self._vocab_size,
            cache_state_ref=executor_input.cache_state_ref,
            attention_metadata={
                "mask_kind": executor_input.attention_mask_kind,
                **executor_input.attention_metadata,
            },
        )


class PooledEmbeddingProbeExecutor:
    """Minimal NumPy-backed probe executor.

    This does not implement the real D2F transformer forward path.
    It projects embedded token context with a pooled mean as a stop-gap
    forward signal so the rest of the runtime can be exercised.
    """

    def __init__(
        self,
        *,
        package_dir: str | Path,
        merged_weights_ref: str,
        vocab_size: int,
        mask_token_id: int,
    ) -> None:
        mx = _get_mlx()
        package_dir = Path(package_dir)
        with open_package_artifact(
            package_path=package_dir,
            artifact_ref=merged_weights_ref,
            error_prefix="numpy",
            artifact_name="weights",
        ) as (handle, _path, _stat):
            with np.load(handle, allow_pickle=False) as archive:
                self._embed_tokens = mx.array(archive["model.embed_tokens.weight"])
                self._lm_head = mx.array(archive["lm_head.weight"])
                norm = (
                    archive["model.norm.weight"]
                    if "model.norm.weight" in archive
                    else None
                )
        self._norm = mx.array(norm) if norm is not None else None
        self._vocab_size = vocab_size
        self._mask_token_id = mask_token_id

    def step(self, executor_input: ExecutorInput) -> BlockLogits:
        rows: list[list[float]] = []
        prompt_context = [
            int(token_id)
            for token_id in executor_input.prompt_token_ids
            if 0 <= int(token_id) < self._vocab_size
        ]
        if not prompt_context:
            prompt_context = [0]

        for position, _ in enumerate(executor_input.block_token_ids):
            committed_prefix = [
                int(token_id)
                for token_id in executor_input.block_token_ids[:position]
                if 0 <= int(token_id) < self._vocab_size and int(token_id) != self._mask_token_id
            ]
            logits = self._next_token_logits(prompt_context + committed_prefix)
            row = np.asarray(logits, dtype=np.float16)
            rows.append(row.tolist())

        return BlockLogits(
            values=rows,
            shape=(len(rows), self._vocab_size),
            dtype="float16",
            block_span=(
                executor_input.block_position,
                executor_input.block_position + len(executor_input.block_token_ids),
            ),
            vocab_size=self._vocab_size,
            source=PROBED_MLX_EXECUTOR_SOURCE,
            cache_state_ref=executor_input.cache_state_ref,
            attention_metadata={
                "mask_kind": executor_input.attention_mask_kind,
                **executor_input.attention_metadata,
            },
        )

    def _next_token_logits(self, context_token_ids: list[int]) -> np.ndarray:
        mx = _get_mlx()
        context_window = context_token_ids[-16:] if len(context_token_ids) > 16 else context_token_ids
        token_matrix = self._embed_tokens[mx.array(context_window, dtype=mx.int32)]
        hidden = mx.mean(token_matrix, axis=0)
        if self._norm is not None:
            hidden = hidden * self._norm
        logits = mx.matmul(hidden.astype(mx.float16), self._lm_head.T)
        return np.asarray(logits, dtype=np.float16)


class DreamTransformerExecutor:
    """Reference forward path for Dream/Qwen-style Transformer blocks."""

    def __init__(
        self,
        *,
        package_dir: str | Path,
        merged_weights_ref: str,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        package_dir = Path(package_dir)
        config = dict(model_config or {})
        weights_stack = ExitStack()
        try:
            handle, _path, _stat = weights_stack.enter_context(
                open_package_artifact(
                    package_path=package_dir,
                    artifact_ref=merged_weights_ref,
                    error_prefix="numpy",
                    artifact_name="weights",
                )
            )
            self._weights = np.load(handle, mmap_mode="r", allow_pickle=False)
        except BaseException:
            weights_stack.close()
            raise
        self._weights_cleanup = weakref.finalize(
            self,
            _close_numpy_archive,
            self._weights,
            weights_stack,
        )

        self._vocab_size = int(config.get("vocab_size", 0) or self._infer_vocab_size())
        self._hidden_size = int(config.get("hidden_size", 0) or self._infer_hidden_size())
        configured_layer_count = config.get("layer_count")
        self._n_layers = int(
            configured_layer_count
            if configured_layer_count is not None
            else self._infer_layer_count()
        )
        self._n_attention_heads = int(config.get("attention_heads", 0))
        self._head_dim = int(config.get("head_dim", 0))
        self._num_kv_heads = int(config.get("num_kv_heads", 0))
        self._rms_norm_eps = float(config.get("rms_norm_eps", 1e-6))

        if self._n_attention_heads <= 0:
            self._n_attention_heads = self._infer_attention_heads(self._hidden_size, self._head_dim)

        if self._head_dim <= 0:
            self._head_dim = max(1, self._hidden_size // max(1, self._n_attention_heads))

        if self._num_kv_heads <= 0:
            self._num_kv_heads = self._infer_num_kv_heads(self._head_dim)

        if self._n_attention_heads % self._num_kv_heads != 0:
            raise ValueError("attention heads must be divisible by key/value heads")
        self._kv_repeat = max(1, self._n_attention_heads // self._num_kv_heads)

        configured_intermediate_size = config.get("intermediate_size")
        self._intermediate_size = int(
            configured_intermediate_size
            if configured_intermediate_size is not None
            else self._infer_intermediate_size()
        )
        self._rope_theta = float(config.get("rope_theta", 10000.0))
        # I1 third path: fail closed on unsupported rope_scaling types (yarn/
        # dynamic/su) at construction, mirroring the numpyref + mlx paths. Only
        # linear/default (which read the `factor` key) are implemented here.
        self._rope_factor = _validated_linear_rope_factor(config.get("rope_scaling"))
        self._delayed_kv_cache = DelayedKVCache()
        self._last_cache_metadata: dict[str, str | int] = {}

    def step(self, executor_input: ExecutorInput) -> BlockLogits:
        # Clamp out-of-vocab ids to 0 rather than DROPPING them: block_start /
        # block_end below index by ORIGINAL position, so filtering an OOV token
        # would shift the slice and silently return the wrong block rows.
        sequence_ids = [
            int(token_id) if 0 <= int(token_id) < self._vocab_size else 0
            for token_id in (*executor_input.prompt_token_ids, *executor_input.block_token_ids)
        ]
        if not sequence_ids:
            sequence_ids = [0]

        block_start = len(executor_input.prompt_token_ids)
        block_end = block_start + len(executor_input.block_token_ids)

        self._last_cache_metadata = {}
        logits = self._forward(
            sequence_ids,
            attention_mask_kind=executor_input.attention_mask_kind,
            mask_pattern=executor_input.mask_pattern,
            cache_action=str(executor_input.attention_metadata.get("kv_cache_action", "")),
            cache_block_start=int(executor_input.attention_metadata.get("kv_cache_block_start", block_start)),
        )
        shifted_logits = np.concatenate([logits[:1], logits[:-1]], axis=0)
        block_logits = shifted_logits[block_start:block_end]

        return BlockLogits(
            values=[row.tolist() for row in block_logits.astype(np.float16)],
            shape=(len(block_logits), self._vocab_size),
            dtype="float16",
            block_span=(block_start, block_end),
            vocab_size=self._vocab_size,
            source=TRANSFORMER_NUMPY_EXECUTOR_SOURCE,
            cache_state_ref=executor_input.cache_state_ref,
            attention_metadata={
                "mask_kind": executor_input.attention_mask_kind,
                **executor_input.attention_metadata,
                **self._last_cache_metadata,
            },
        )

    def _forward(
        self,
        token_ids: list[int],
        *,
        attention_mask_kind: str,
        mask_pattern: list[bool] | None = None,
        cache_action: str = "",
        cache_block_start: int = 0,
    ) -> np.ndarray:
        if not token_ids:
            token_ids = [0]

        cache_metadata: dict[str, str | int] = {}
        if cache_action == "refresh":
            had_entries = self._delayed_kv_cache.has_entries()
            self._delayed_kv_cache.clear()
            if had_entries:
                cache_metadata["kv_cache_kind"] = _BIDIRECTIONAL_LAYER0_KV_CACHE
                cache_metadata["kv_cache_refreshed"] = 1

        reused_positions = 0
        hidden_states = self._weight("model.embed_tokens.weight")[token_ids].astype(np.float32)
        for layer_index in range(self._n_layers):
            hidden_states, layer_reused_positions = self._transformer_layer_forward(
                hidden_states,
                layer_index,
                token_ids=token_ids,
                attention_mask_kind=attention_mask_kind,
                mask_pattern=mask_pattern,
                cache_action=cache_action,
                cache_block_start=cache_block_start,
            )
            reused_positions += layer_reused_positions

        hidden_states = self._rms_norm(hidden_states, self._weight("model.norm.weight"))
        lm_head = self._weight("lm_head.weight").astype(np.float32)
        if reused_positions > 0:
            cache_metadata["kv_cache_kind"] = _BIDIRECTIONAL_LAYER0_KV_CACHE
            cache_metadata["kv_cache_reused"] = reused_positions
        self._last_cache_metadata = cache_metadata
        return hidden_states.astype(np.float32) @ lm_head.T

    def _transformer_layer_forward(
        self,
        hidden_states: np.ndarray,
        layer_index: int,
        *,
        token_ids: list[int],
        attention_mask_kind: str,
        mask_pattern: list[bool] | None,
        cache_action: str,
        cache_block_start: int,
    ) -> tuple[np.ndarray, int]:
        residual = hidden_states
        normalized = self._rms_norm(
            hidden_states,
            self._weight(f"model.layers.{layer_index}.input_layernorm.weight"),
        )

        q = self._linear(
            normalized,
            f"model.layers.{layer_index}.self_attn.q_proj.weight",
            f"model.layers.{layer_index}.self_attn.q_proj.bias",
        )
        k = self._linear(
            normalized,
            f"model.layers.{layer_index}.self_attn.k_proj.weight",
            f"model.layers.{layer_index}.self_attn.k_proj.bias",
        )
        v = self._linear(
            normalized,
            f"model.layers.{layer_index}.self_attn.v_proj.weight",
            f"model.layers.{layer_index}.self_attn.v_proj.bias",
        )

        seq_len = q.shape[0]
        q = q.reshape(seq_len, self._n_attention_heads, self._head_dim)
        k = k.reshape(seq_len, self._num_kv_heads, self._head_dim)
        v = v.reshape(seq_len, self._num_kv_heads, self._head_dim)
        q, k, v = self._apply_rope(q, k, v)
        if self._num_kv_heads != self._n_attention_heads:
            k = np.repeat(k, self._kv_repeat, axis=1)
            v = np.repeat(v, self._kv_repeat, axis=1)

        reused_positions = 0
        if layer_index == 0 and cache_action in {"reuse", "refresh"}:
            reusable_positions = self._layer0_kv_reusable_positions(
                mask_pattern=mask_pattern,
                seq_len=seq_len,
                cache_block_start=cache_block_start,
            )
            if cache_action == "reuse":
                k, v, cache_event = self._delayed_kv_cache.apply_layer(
                    layer_index=layer_index,
                    token_ids=token_ids,
                    key=k,
                    value=v,
                    reusable_positions=reusable_positions,
                )
                reused_positions += cache_event.reused_positions
            self._delayed_kv_cache.update_layer(
                layer_index=layer_index,
                token_ids=token_ids,
                key=k,
                value=v,
                reusable_positions=reusable_positions,
            )

        attn = self._attention(q, k, v, attention_mask_kind=attention_mask_kind)
        attn = self._linear(
            attn,
            f"model.layers.{layer_index}.self_attn.o_proj.weight",
            f"model.layers.{layer_index}.self_attn.o_proj.bias",
        )
        hidden_states = residual + attn

        residual = hidden_states
        hidden_states = self._rms_norm(
            hidden_states,
            self._weight(f"model.layers.{layer_index}.post_attention_layernorm.weight"),
        )
        gate = self._silu(
            self._linear(
                hidden_states,
                f"model.layers.{layer_index}.mlp.gate_proj.weight",
                f"model.layers.{layer_index}.mlp.gate_proj.bias",
            )
        )
        up = self._linear(
            hidden_states,
            f"model.layers.{layer_index}.mlp.up_proj.weight",
            f"model.layers.{layer_index}.mlp.up_proj.bias",
        )
        hidden_states = self._linear(gate * up, f"model.layers.{layer_index}.mlp.down_proj.weight")
        return residual + hidden_states, reused_positions

    def _layer0_kv_reusable_positions(
        self,
        *,
        mask_pattern: list[bool] | None,
        seq_len: int,
        cache_block_start: int,
    ) -> list[bool]:
        if mask_pattern is None or len(mask_pattern) != seq_len:
            return [False] * seq_len
        # v1 intentionally caches only committed decoded-output positions. Prompt
        # K/V reuse is a higher-leverage future optimization, but it is outside
        # this narrow refresh_real evidence path.
        return [
            position >= cache_block_start and not bool(is_masked)
            for position, is_masked in enumerate(mask_pattern)
        ]

    def _apply_rope(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if q.shape[-1] != self._head_dim:
            return q, k, v

        seq_len = q.shape[0]
        positions = np.arange(seq_len, dtype=np.float32)
        half_dim = self._head_dim // 2
        if half_dim <= 0:
            return q, k, v

        factor = self._rope_factor
        inv_freq = 1.0 / (self._rope_theta ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
        inv_freq = inv_freq / max(1e-6, factor)
        freqs = np.einsum("i,j->ij", positions, inv_freq)
        cos = np.cos(freqs)
        sin = np.sin(freqs)

        cos = cos[:, None, :]
        sin = sin[:, None, :]
        q_rot = np.empty_like(q)
        k_rot = np.empty_like(k)
        v_rot = np.asarray(v, copy=True)

        q_rot[..., :half_dim] = q[..., :half_dim] * cos - q[..., half_dim:] * sin
        q_rot[..., half_dim:] = q[..., :half_dim] * sin + q[..., half_dim:] * cos
        k_rot[..., :half_dim] = k[..., :half_dim] * cos - k[..., half_dim:] * sin
        k_rot[..., half_dim:] = k[..., :half_dim] * sin + k[..., half_dim:] * cos
        return q_rot, k_rot, v_rot

    def _attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        *,
        attention_mask_kind: str,
    ) -> np.ndarray:
        scores = np.einsum("t h d, s h d -> h t s", q, k) / math.sqrt(float(self._head_dim))

        if attention_mask_kind == "causal":
            mask = np.triu(np.full((q.shape[0], k.shape[0]), -1e9, dtype=np.float32), k=1)
            scores = scores + mask[None, :, :]

        max_scores = np.max(scores, axis=-1, keepdims=True)
        scores = np.exp(scores - max_scores)
        probs = scores / np.sum(scores, axis=-1, keepdims=True)
        output = np.einsum("h t s, s h d -> t h d", probs, v)
        return output.reshape(output.shape[0], -1)

    def _rms_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        if weight is None:
            return x
        mean_square = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
        x_norm = x.astype(np.float32) / np.sqrt(mean_square + self._rms_norm_eps)
        return x_norm * weight.astype(np.float32)

    def _silu(self, x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    def _linear(self, x: np.ndarray, weight_key: str, bias_key: str | None = None) -> np.ndarray:
        weight = self._weight(weight_key).astype(np.float32)
        if bias_key is not None and bias_key in self._weights:
            bias = self._weight(bias_key).astype(np.float32)
        else:
            bias = None
        y = x.astype(np.float32) @ weight.T
        if bias is not None:
            y = y + bias
        return y

    def _weight(self, key: str) -> np.ndarray:
        if key not in self._weights:
            raise KeyError(f"missing weight key: {key}")
        return self._weights[key]

    def _infer_vocab_size(self) -> int:
        return int(self._weight("model.embed_tokens.weight").shape[0])

    def _infer_hidden_size(self) -> int:
        return int(self._weight("model.embed_tokens.weight").shape[1])

    def _infer_layer_count(self) -> int:
        pattern = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
        max_layer = -1
        for key in self._weights.files:
            match = pattern.match(key)
            if match is not None:
                max_layer = max(max_layer, int(match.group(1)))
        if max_layer < 0:
            raise ValueError("unable to infer model layer count from qkv weights")
        return max_layer + 1

    def _infer_attention_heads(self, hidden_size: int, head_dim: int) -> int:
        q = self._weight("model.layers.0.self_attn.q_proj.weight")
        if head_dim > 0 and q.shape[0] % head_dim == 0:
            return max(1, q.shape[0] // head_dim)
        if q.ndim == 2 and hidden_size > 0:
            return max(1, q.shape[0] // hidden_size)
        return max(1, hidden_size // max(1, head_dim))

    def _infer_num_kv_heads(self, head_dim: int) -> int:
        k = self._weight("model.layers.0.self_attn.k_proj.weight")
        if head_dim > 0:
            return max(1, k.shape[0] // head_dim)
        if k.ndim == 2 and k.shape[1] > 0:
            return max(1, min(self._n_attention_heads, k.shape[0] // max(1, (self._hidden_size // max(1, self._n_attention_heads)))))
        return 1

    def _infer_intermediate_size(self) -> int:
        w = self._weight("model.layers.0.mlp.gate_proj.weight")
        return int(w.shape[0])


def _get_mlx():
    global _MX
    if _MX is not None:
        return _MX
    try:
        mx = importlib.import_module("mlx.core")
    except Exception as exc:  # pragma: no cover - exercised when MLX unavailable
        raise RuntimeError(
            "MLX runtime unavailable; install MLX with Metal support before using transformer executor"
        ) from exc
    _MX = mx
    return mx


def _close_numpy_archive(weights: Any, stack: ExitStack) -> None:
    try:
        close = getattr(weights, "close", None)
        if callable(close):
            close()
    finally:
        stack.close()
