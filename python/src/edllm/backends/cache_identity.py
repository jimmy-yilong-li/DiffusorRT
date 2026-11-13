"""Stable policy identity retained by snapshots and native exact-D2F."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyVersion:
    """A deterministic digest of the decode policy."""

    digest: str

    @classmethod
    def of(
        cls,
        *,
        decode_strategy: str,
        commit_policy: str,
        remask_policy: str,
        thresholds: tuple,
        block_size: int,
    ) -> "PolicyVersion":
        threshold_text = ",".join(repr(value) for value in tuple(thresholds))
        canonical = "|".join(
            (
                f"decode_strategy={decode_strategy}",
                f"commit_policy={commit_policy}",
                f"remask_policy={remask_policy}",
                f"thresholds=[{threshold_text}]",
                f"block_size={int(block_size)}",
            )
        )
        digest = hashlib.sha256(
            b"policy_version:" + canonical.encode("utf-8")
        ).hexdigest()
        return cls(digest=digest)


__all__ = ["PolicyVersion"]
