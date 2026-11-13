"""The single control-token contract: one validator, two input adapters.

There were four resolvers with four policies, and each round of "fix every
call site" closed some and missed others. This module is the policy; the two
adapters differ only in where they read from.

THE POLICY, in priority order:

1. An explicitly declared VALID id wins. It is the model's own answer, and a
   vocabulary entry is not permitted to override it.
2. Otherwise the EXACT token ``<|mask|>`` must be present, and its id must
   itself be valid.
3. Otherwise refuse.

No substitution, no near-miss token, no sentinel. Every value that crosses the
boundary goes through :func:`valid_token_id` -- an earlier version validated
the DECLARATION and then trusted whatever the vocabulary held, so `True`, `-1`
and `"6"` all came back out.
"""
from __future__ import annotations

import json
from pathlib import Path

MASK_TOKEN = "<|mask|>"


class MaskTokenError(ValueError):
    """No mask token could be resolved, and none may be invented.

    A substitute -- UNK, EOS, ``<|endoftext|>``, or 0 -- lets the diffusion
    loop run to completion and return plausible nonsense, which is worse than
    a refusal because nothing downstream can tell.
    """


def valid_token_id(value: object) -> int | None:
    """``value`` as a token id, or None. The ONLY way a value becomes an id.

    ``bool`` is rejected explicitly. ``isinstance(True, int)`` is True in
    Python, so every `isinstance(x, int)` guard in this codebase accepted
    ``mask_token_id: true`` and wrote token id 1 into the manifest.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _from_vocabulary(vocab: dict, *, source: str) -> int:
    """The exact ``<|mask|>``, validated. Never a near-miss token."""
    if MASK_TOKEN not in vocab:
        raise MaskTokenError(
            f"{source}: no valid mask_token_id was declared and {MASK_TOKEN!r} "
            "is not in the tokenizer vocabulary."
        )
    resolved = valid_token_id(vocab[MASK_TOKEN])
    if resolved is None:
        raise MaskTokenError(
            f"{source}: the vocabulary maps {MASK_TOKEN!r} to "
            f"{vocab[MASK_TOKEN]!r}, which is not a token id."
        )
    return resolved


def resolve_from_vocab(declared: object, vocab: dict, *, source: str) -> int:
    """Adapter: a live tokenizer. ``vocab`` is its exact token -> id map.

    A vocabulary LOOKUP, never ``convert_tokens_to_ids``: that returns the UNK
    id for a token the vocabulary lacks, which is indistinguishable from a real
    match and produced manifests whose mask id was UNK.
    """
    explicit = valid_token_id(declared)
    if explicit is not None:
        return explicit
    return _from_vocabulary(vocab, source=source)


def resolve_from_snapshot(snapshot_dir: Path, config: dict) -> int:
    """Adapter: a Hugging Face snapshot on disk, before a tokenizer exists.

    Same priority as the live adapter. An earlier version read added_tokens and
    the decoder BEFORE the config, so an explicit ``mask_token_id: 42`` lost to
    a vocabulary entry -- and it followed ``tokenizer_config["mask_token"]``
    wherever it pointed, which let a snapshot naming ``<|endoftext|>`` as its
    mask resolve to EOT.
    """
    explicit = valid_token_id(config.get("mask_token_id"))
    if explicit is not None:
        return explicit

    snapshot_dir = Path(snapshot_dir)
    vocab: dict = {}

    added = _read_json(snapshot_dir / "added_tokens.json")
    if MASK_TOKEN in added:
        vocab[MASK_TOKEN] = added[MASK_TOKEN]
    else:
        # added_tokens_decoder is id -> {"content": token}. Only an entry whose
        # content IS "<|mask|>" counts; tokenizer_config["mask_token"] is NOT
        # consulted, because following it is how <|endoftext|> became a mask.
        decoder = _read_json(snapshot_dir / "tokenizer_config.json").get(
            "added_tokens_decoder"
        )
        if isinstance(decoder, dict):
            for token_id, spec in decoder.items():
                if isinstance(spec, dict) and spec.get("content") == MASK_TOKEN:
                    try:
                        vocab[MASK_TOKEN] = int(token_id)
                    except (TypeError, ValueError):
                        pass
                    break

    return _from_vocabulary(vocab, source="snapshot")


def _read_json(path: Path) -> dict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
