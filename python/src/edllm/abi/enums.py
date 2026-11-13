"""Integer encodings for string-valued config fields.

Member VALUES are part of the ABI: adding a member is backward-compatible,
renumbering is not. Values follow the sorted order of the domains declared in
runtime_config.py so the mapping is reproducible rather than incidental; a
member added later is appended after them instead, which keeps every earlier
value.
"""
from __future__ import annotations

from enum import IntEnum, IntFlag


class DecodeStrategy(IntEnum):
    BLOCK_THRESHOLD = 0
    CONFIDENCE_THRESHOLD = 1
    D2F_BLOCK_PIPELINE = 2
    DILATED_UNMASK = 3
    ENTROPY_BUDGET = 4
    ENTROPY_COUNT = 5
    # Appended 2026-11-12 for cuda_native's fixed-quota device sessions (review R2).
    FIXED_QUOTA = 6
    BD3_FIXED_QUOTA = 7


class CommitPolicy(IntEnum):
    DILATED = 0
    ENTROPY_BUDGET = 1
    THRESHOLD = 2
    # Appended 2026-11-12: commit a fixed number of rows per step (review R2).
    FIXED_QUOTA = 3


class ComputeDtype(IntEnum):
    BF16 = 0
    FP32 = 1


class AttentionMaskKind(IntEnum):
    BIDIRECTIONAL = 0
    D2F_HYBRID = 1
    FULL = 2


class WindowExec(IntEnum):
    AUTO = 0
    OFF = 1
    ON = 2


class TokenAction(IntEnum):
    """Mirrors backends.window_plan.TokenAction (a str Enum there)."""
    ACTIVE = 0
    DIRTY = 1
    REMASK = 2
    FROZEN = 3
    DEPENDENCY_RISK = 4


class CacheAction(IntEnum):
    """Mirrors backends.window_plan.CacheAction."""
    RECOMPUTE = 0
    REUSE_EXACT = 1
    # Reserved wire value. ABI v1 validators reject this experimental,
    # Python-only action; keeping 2 prevents accidental enum renumbering.
    REUSE_APPROX = 2


class RecoveryDetectors(IntFlag):
    LOW_CONFIDENCE = 1
    REPETITION = 2
    KL_DRIFT = 4


class ReuseSource(IntEnum):
    """Mirrors the provenance tags of backends.window_plan reuse_source."""
    FROZEN_BLOCK = 0
    DELAYED_KV_LAYER0_EXACT = 1


DECODE_STRATEGY_BY_NAME = {
    "block_threshold": DecodeStrategy.BLOCK_THRESHOLD,
    "confidence_threshold": DecodeStrategy.CONFIDENCE_THRESHOLD,
    "d2f_block_pipeline": DecodeStrategy.D2F_BLOCK_PIPELINE,
    "dilated_unmask": DecodeStrategy.DILATED_UNMASK,
    "entropy_budget": DecodeStrategy.ENTROPY_BUDGET,
    "entropy_count": DecodeStrategy.ENTROPY_COUNT,
    "fixed_quota": DecodeStrategy.FIXED_QUOTA,
    "bd3_fixed_quota": DecodeStrategy.BD3_FIXED_QUOTA,
}
COMMIT_POLICY_BY_NAME = {
    "dilated": CommitPolicy.DILATED,
    "entropy_budget": CommitPolicy.ENTROPY_BUDGET,
    "threshold": CommitPolicy.THRESHOLD,
    "fixed_quota": CommitPolicy.FIXED_QUOTA,
}
COMPUTE_DTYPE_BY_NAME = {
    "bf16": ComputeDtype.BF16, "bfloat16": ComputeDtype.BF16,
    "fp32": ComputeDtype.FP32, "float32": ComputeDtype.FP32,
}
ATTENTION_MASK_KIND_BY_NAME = {
    "bidirectional": AttentionMaskKind.BIDIRECTIONAL,
    "d2f_hybrid": AttentionMaskKind.D2F_HYBRID,
    "full": AttentionMaskKind.FULL,
}
CACHE_ACTION_BY_NAME = {
    "recompute": CacheAction.RECOMPUTE,
    "reuse_exact": CacheAction.REUSE_EXACT,
    "reuse_approx": CacheAction.REUSE_APPROX,
}
REUSE_SOURCE_BY_NAME = {
    "frozen_block": ReuseSource.FROZEN_BLOCK,
    "delayed_kv_layer0_exact": ReuseSource.DELAYED_KV_LAYER0_EXACT,
}


def encode(mapping: dict, name: str, field: str) -> int:
    try:
        return int(mapping[name])
    except KeyError:
        raise ValueError(
            f"{field}={name!r} has no ABI encoding; add it to edllm/abi/enums.py "
            "before it can cross the boundary"
        ) from None
