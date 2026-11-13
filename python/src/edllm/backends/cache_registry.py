"""State-cache selection registry.

Focused home for the engine's optional KV-cache *selection*: the single factory
``select_state_cache_manager`` and the ``new_*`` strategy constructors it goes
through. This logic used to live inline in ``mlx_dream.py`` (the monolith that
owns the forward); it is relocated here so future KV-cache work stops piling into
``mlx_dream``. K2-3 narrows this boundary to the surviving approximate
prefix/block caches and an injected native exact-D2F strategy. Removed owner
controls remain exact neutral tombstones and otherwise fail closed.

Import discipline: this module MUST NOT import ``mlx_dream`` (it is imported BY
it). It imports only the layer-0 cache policy and keeps the MLX-free scheduling
manager import local to its constructors.
"""

from __future__ import annotations

from typing import Any

from . import mlx_cache


# Geometry required by the legacy layer-0 cache constructor. The production
# layer trace consumes the compiled backend plan instead of duplicating this
# model-config contract.
_REQUIRED_LAYER0_CONFIG_KEYS = (
    "hidden_size",
    "attention_heads",
    "head_dim",
    "num_kv_heads",
    "rms_norm_eps",
    "rope_theta",
)


def _is_exact_neutral(actual: Any, expected: Any) -> bool:
    """Compare a retired setting without bool/int numeric aliases."""
    return type(actual) is type(expected) and actual == expected


def _require_layer0_config(config: dict[str, Any]) -> None:
    for key in _REQUIRED_LAYER0_CONFIG_KEYS:
        if key not in config or config[key] is None or config[key] == "":
            raise ValueError(f"MLX layer-0 cache requires model_config[{key!r}]")


def _rope_signature(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "theta": float(model_config["rope_theta"]),
        "scaling": model_config.get("rope_scaling"),
    }


def new_layer0_kv_cache_state(model_config: dict[str, Any], *, attention_mask_kind: str):
    _require_layer0_config(model_config)
    policy = mlx_cache.MLXKVCachePolicy(
        strategy="bidirectional_layer0_kv",
        attention_mask_kind=attention_mask_kind,
        rope_signature=_rope_signature(model_config),
        head_dim=int(model_config["head_dim"]),
        num_kv_heads=int(model_config["num_kv_heads"]),
        dtype="float32",
    )
    return policy.new_state()


def new_prefix_kv_state_cache_manager(*, prompt_len: int, refresh_interval: int = 0):
    """Construct the StateCacheManager carrying the approximate prefix KV cache.

    Local import keeps the (mlx-free) manager module off the import-light path of
    this engine; the manager itself imports no mlx.
    """
    from . import mlx_state_cache

    return mlx_state_cache.StateCacheManager.for_prefix_kv(
        prompt_len=int(prompt_len),
        refresh_interval=int(refresh_interval),
    )


def new_block_kv_state_cache_manager(*, prompt_len: int, refresh_interval: int = 0):
    """Construct the StateCacheManager carrying the approximate block KV cache.

    Local import keeps the (mlx-free) manager module off the import-light path; the
    manager itself imports no mlx.
    """
    from . import mlx_state_cache

    return mlx_state_cache.StateCacheManager.for_block_kv(
        prompt_len=int(prompt_len),
        refresh_interval=int(refresh_interval),
    )


def new_d2f_exact_kv_state_cache_manager(
    *,
    prompt_len: int,
    block_size: int,
    window_mode: str,
    enable_block_kv_cache: bool = False,
    enable_prefix_kv_cache: bool = False,
    skip_committed_rows: bool = False,
    skip_refresh_interval: int = 4,
    exact_strategy: Any = None,
):
    """Wrap the required native exact-D2F strategy in the scheduling manager."""
    if enable_block_kv_cache or enable_prefix_kv_cache:
        raise ValueError(
            "the exact d2f KV cache (mlx_diffusion_d2f_exact_kv_cache) cannot "
            "be combined with the approximate prefix/block KV caches; enable "
            "exactly one"
        )
    if exact_strategy is None:
        raise ValueError(
            "exact-D2F requires the native strategy; the legacy owner is no "
            "longer supported in DiffusorRT 0.2"
        )
    if skip_committed_rows:
        raise ValueError(
            "d2f_skip_committed_rows is no longer supported in DiffusorRT 0.2"
        )
    if skip_refresh_interval != 4:
        raise ValueError(
            "d2f_skip_refresh_interval is no longer supported in DiffusorRT 0.2"
        )
    expected_geometry = {
        "prompt_len": prompt_len,
        "block_size": block_size,
        "window_mode": window_mode,
    }
    for field, expected in expected_geometry.items():
        actual = getattr(exact_strategy, field, None)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                "the injected native exact-D2F strategy disagrees with "
                f"{field}: {actual!r} != {expected!r}"
            )
    from . import mlx_state_cache

    return mlx_state_cache.StateCacheManager(d2f_exact_kv=exact_strategy)


def select_state_cache_manager(
    *,
    enable_d2f_exact_kv_cache: bool,
    enable_block_kv_cache: bool,
    enable_prefix_kv_cache: bool,
    prompt_len: int,
    block_size: int,
    decode_strategy: str,
    d2f_skip_committed_rows: bool,
    d2f_skip_refresh_interval: int,
    block_kv_refresh_interval: int,
    prefix_kv_refresh_interval: int,
    enable_delayed_kv: bool,
    delayed_kv_max_stale_steps,
    delayed_kv_refresh_interval,
    delayed_kv_scope,
    delayed_kv_stale_epoch_budget,
    delayed_kv_upper_layer_range,
    exact_strategy: Any = None,
):
    """Pick one surviving K/V scheduling strategy.

    Single selection point for the engine's load-bearing throughput strategies.
    All are OFF by default -> returns ``None`` -> the forward is byte-identical.
    At most one is active per turn:

    * prefix KV cache: cache the fixed conversation-prefix K/V once and forward
      the whole block block-only (per-step cost stops scaling with prefix
      length).
    * block KV cache (Phase D, the bigger lever): ALSO cache committed-block
      K/V, so each step forwards ONLY the still-masked frontier — erasing the
      51–55% re-forward of already-committed positions the profile found. It is
      a strict superset of the prefix cache, so when BOTH are enabled the block
      cache subsumes the prefix one (it caches the prefix too).
    * d2f exact KV cache (D2F-2, valid only under d2f_hybrid): cache the prompt
      + frozen-block K/V EXACTLY (block causality, no drift) and forward only the
      active block. Exclusive with the approximate caches (fail-closed); the
      D2F-3 block scheduler is what drives freezing during a real decode.

    The ``new_*_state_cache_manager`` constructors are the construction seams
    (tests monkeypatch them), so this factory must go through them rather than
    constructing the strategies inline. Returns the ``StateCacheManager`` (which
    may be ``None``).
    """
    retired = {
        "d2f_skip_committed_rows": (d2f_skip_committed_rows, False),
        "d2f_skip_refresh_interval": (d2f_skip_refresh_interval, 4),
        "enable_delayed_kv": (enable_delayed_kv, False),
        "delayed_kv_max_stale_steps": (delayed_kv_max_stale_steps, None),
        "delayed_kv_refresh_interval": (delayed_kv_refresh_interval, None),
        "delayed_kv_scope": (delayed_kv_scope, "layer0"),
        "delayed_kv_stale_epoch_budget": (delayed_kv_stale_epoch_budget, None),
        "delayed_kv_upper_layer_range": (delayed_kv_upper_layer_range, None),
    }
    for name, (actual, expected) in retired.items():
        if not _is_exact_neutral(actual, expected):
            raise ValueError(
                f"{name} is no longer supported in DiffusorRT 0.2"
            )

    if exact_strategy is not None and not enable_d2f_exact_kv_cache:
        raise ValueError(
            "exact_strategy requires enable_d2f_exact_kv_cache=True"
        )

    if enable_d2f_exact_kv_cache:
        state_cache_manager = new_d2f_exact_kv_state_cache_manager(
            prompt_len=prompt_len,
            block_size=block_size,
            window_mode=(
                "tail"
                if decode_strategy == "d2f_block_pipeline"
                else "single_block"
            ),
            enable_block_kv_cache=enable_block_kv_cache,
            enable_prefix_kv_cache=enable_prefix_kv_cache,
            skip_committed_rows=d2f_skip_committed_rows,
            skip_refresh_interval=d2f_skip_refresh_interval,
            exact_strategy=exact_strategy,
        )
    elif enable_block_kv_cache:
        state_cache_manager = new_block_kv_state_cache_manager(
            prompt_len=prompt_len,
            refresh_interval=block_kv_refresh_interval,
        )
    elif enable_prefix_kv_cache:
        state_cache_manager = new_prefix_kv_state_cache_manager(
            prompt_len=prompt_len,
            refresh_interval=prefix_kv_refresh_interval,
        )
    else:
        state_cache_manager = None

    return state_cache_manager
