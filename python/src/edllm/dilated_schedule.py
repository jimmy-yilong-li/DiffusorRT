"""Dilated Unmasking Scheduler (DUS) — schedule generator.

Implements the official dilated unmasking schedule from
Luxembourg et al., "Plan for Speed" (arXiv:2506.19037, ICML 2026).
Source: https://github.com/omerlux/DUS/blob/fe0e8e4b283dc86d559f57648981fa1f0566f170/generate.py

Algorithm (``dilated_unmask_levels`` + ``merge_last_level``):
  stride = length // base  (skip_exp=1 by default)
  while stride >= 1:
      pick positions at this stride (not already revealed)
      stride //= base
  pick any leftover
  merge last level into second-to-last if smaller

This naturally handles non-power-of-base B without empty rounds (the while
loop stops when stride < 1). Zero dependencies (stdlib only); import-pure.
"""
from __future__ import annotations


def dilated_schedule(
    block_size: int,
    *,
    base: int = 2,
) -> list[list[int]]:
    """Return the DUS unmasking schedule for a block of *block_size* positions.

    Positions are 1-indexed. Each group is a sorted list of positions to unmask
    in one denoiser forward. Groups are pairwise disjoint and their union is
    {1, ..., block_size}. The ``merge_last_level`` rule (official) may collapse
    the final group into the previous one when it is smaller — so for non-
    power-of-base B (e.g. B=5) the result may have fewer groups than
    ``ceil(log_base(B))``.

    Raises ValueError on block_size < 1 or base < 2.
    """
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if not isinstance(base, int) or isinstance(base, bool) or base < 2:
        raise ValueError("base must be an integer ≥ 2")

    start = 1
    end = block_size
    stride = block_size // base
    levels: list[list[int]] = []
    revealed: set[int] = set()

    while stride >= 1:
        this_round = sorted(
            i
            for i in range(start, end + 1)
            if (i - start) % stride == 0 and i not in revealed
        )
        if this_round:
            levels.append(this_round)
            revealed.update(this_round)
        stride //= base

    # Final round — any leftovers not revealed by the dilated rounds
    remainder = sorted(i for i in range(start, end + 1) if i not in revealed)
    if remainder:
        levels.append(remainder)

    # merge_last_level (official): if the last group is smaller than the
    # second-to-last, merge it in. This prevents tiny final rounds.
    if len(levels) >= 2 and len(levels[-1]) < len(levels[-2]):
        levels[-2].extend(levels[-1])
        levels[-2].sort()
        levels.pop()

    return levels
