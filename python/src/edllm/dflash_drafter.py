"""DFlash drafter package-identity contract.

A DFlash drafter is a diffusion draft model co-trained against a specific AR
target (selected target hidden layers, fixed ``target_layer_ids``), sharing the
target's tokenizer. It is a recognized package identity that requires a
speculative coordinator to run; it is NOT a standalone generative model.

This module is dependency-free and import-pure. It only recognizes and validates
the identity; it does not load architectures, serve, or train.

The recognized shape is anchored on the project's pinned ``z-lab/Qwen3.5-4B-DFlash``
revision ``96899cc`` (``DRAFT_REVISION`` in ``dflash_mlx_reference.py``). At that
revision the HF layout is ``architectures=["DFlashDraftModel"]`` with a TOP-LEVEL
``block_size`` and ``num_target_layers``, and a ``dflash_config`` block carrying
only ``mask_token_id`` and ``target_layer_ids``. In this repo's package model the
drafter identity is gathered under ``model_config["dflash_drafter"]`` (one block,
all conditioning fields + pairing target metadata, which the upstream HF config
does not carry).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class DFlashDrafterConfigError(ValueError):
    """A model_config claimed to be a DFlash drafter but was malformed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DFlashDrafterStandaloneRefused(ValueError):
    """A DFlash drafter package cannot run standalone — it needs a coordinator.

    A drafter is a recognized package identity (its ``dflash_drafter`` block
    validates), but it has no standalone decode path: it consumes AR-target
    features produced by a speculative coordinator. ``load_runtime`` raises this
    so the refusal is precise and distinguishable from a generic
    ``unsupported_architecture_family`` load failure.
    """

    code = "dflash_drafter_standalone_refused"

    def __init__(self, identity: DFlashDrafterIdentity) -> None:
        super().__init__(
            "dflash_drafter_standalone_refused: this package is a DFlash drafter "
            f"(paired target {identity.paired_target_id!r}) and cannot run "
            "standalone; it requires a speculative coordinator"
        )
        self.identity = identity


@dataclass(frozen=True, slots=True)
class DFlashDrafterIdentity:
    """The validated identity of a DFlash drafter package."""

    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    paired_target_id: str
    paired_target_revision: str
    paired_target_tokenizer_identity: str


def recognize_drafter(
    model_config: Mapping[str, object],
) -> DFlashDrafterIdentity | None:
    """Recognize a DFlash drafter identity in a ``model_config``.

    Returns the validated :class:`DFlashDrafterIdentity` when ``model_config``
    declares a ``dflash_drafter`` block; returns ``None`` only when the key is
    ABSENT (a plain standalone model). Raises :class:`DFlashDrafterConfigError`
    if the key is present but the block is null/non-object/malformed, or if the
    declared architecture conflicts with the drafter declaration — a claim to be
    a drafter must fail closed rather than be silently treated as standalone.
    """
    if "dflash_drafter" not in model_config:
        # Finding 1: a declared DFlashDraftModel architecture without an
        # identity block is an incomplete drafter declaration — fail closed
        # rather than silently load as a standalone model.
        if (
            isinstance(model_config.get("architecture"), str)
            and model_config["architecture"] == _DFLASH_DRAFTER_ARCHITECTURE
        ):
            raise DFlashDrafterConfigError(
                "dflash_drafter_block_missing",
                "architecture is DFlashDraftModel but no dflash_drafter block "
                "is present",
            )
        return None
    block = model_config["dflash_drafter"]
    if not isinstance(block, Mapping):
        raise DFlashDrafterConfigError(
            "dflash_drafter_block_not_object",
            "dflash_drafter must be an object",
        )
    architecture = model_config.get("architecture")
    if (
        isinstance(architecture, str)
        and architecture
        and architecture != _DFLASH_DRAFTER_ARCHITECTURE
    ):
        raise DFlashDrafterConfigError(
            "dflash_architecture_conflict",
            "a dflash_drafter block is present but architecture is "
            f"{architecture!r}, not {_DFLASH_DRAFTER_ARCHITECTURE!r}",
        )
    return _parse_identity(block)


def _parse_identity(block: Mapping[str, object]) -> DFlashDrafterIdentity:
    block_size = _require_int(block.get("block_size"), "dflash_block_size_invalid")
    if block_size < 2:
        raise DFlashDrafterConfigError(
            "dflash_block_size_invalid",
            "dflash_drafter.block_size must be an integer greater than 1 "
            "(consistent with the DFlash measurement harness)",
        )
    mask_token_id = _require_int(
        block.get("mask_token_id"), "dflash_mask_token_id_invalid"
    )
    if mask_token_id < 0:
        raise DFlashDrafterConfigError(
            "dflash_mask_token_id_invalid",
            "dflash_drafter.mask_token_id must be a non-negative integer",
        )
    # Parse num_target_layers first so layer ids can be bounds-checked against it
    # (enforces the plan's num_target_layers >= max(target_layer_ids)+1 invariant).
    num_target_layers = _require_int(
        block.get("num_target_layers"), "dflash_num_target_layers_invalid"
    )
    if num_target_layers <= 0:
        raise DFlashDrafterConfigError(
            "dflash_num_target_layers_invalid",
            "dflash_drafter.num_target_layers must be a positive integer",
        )
    raw_ids = block.get("target_layer_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise DFlashDrafterConfigError(
            "dflash_target_layer_ids_empty",
            "dflash_drafter.target_layer_ids must be a non-empty list",
        )
    target_layer_ids: list[int] = []
    seen: set[int] = set()
    for item in raw_ids:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise DFlashDrafterConfigError(
                "dflash_target_layer_ids_invalid",
                "dflash_drafter.target_layer_ids must be non-negative integers",
            )
        if item >= num_target_layers:
            raise DFlashDrafterConfigError(
                "dflash_target_layer_id_out_of_range",
                "dflash_drafter.target_layer_ids must all be < num_target_layers",
            )
        if item in seen:
            raise DFlashDrafterConfigError(
                "dflash_target_layer_ids_duplicate",
                "dflash_drafter.target_layer_ids must be unique",
            )
        seen.add(item)
        target_layer_ids.append(item)
    pairing = block.get("paired_target")
    if not isinstance(pairing, Mapping):
        raise DFlashDrafterConfigError(
            "dflash_paired_target_missing",
            "dflash_drafter.paired_target must name the co-trained target",
        )
    paired_id = pairing.get("id")
    paired_rev = pairing.get("revision")
    paired_tok = pairing.get("tokenizer_identity")
    if not isinstance(paired_id, str) or not paired_id:
        raise DFlashDrafterConfigError(
            "dflash_paired_target_id_missing",
            "dflash_drafter.paired_target.id must be a non-empty string",
        )
    if not isinstance(paired_rev, str) or not paired_rev:
        raise DFlashDrafterConfigError(
            "dflash_paired_target_revision_missing",
            "dflash_drafter.paired_target.revision must be a non-empty string",
        )
    if not isinstance(paired_tok, str) or not paired_tok:
        raise DFlashDrafterConfigError(
            "dflash_paired_target_tokenizer_identity_missing",
            "dflash_drafter.paired_target.tokenizer_identity must be a non-empty string",
        )
    return DFlashDrafterIdentity(
        block_size=block_size,
        mask_token_id=mask_token_id,
        target_layer_ids=tuple(target_layer_ids),
        num_target_layers=num_target_layers,
        paired_target_id=paired_id,
        paired_target_revision=paired_rev,
        paired_target_tokenizer_identity=paired_tok,
    )


def _require_int(value: object, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DFlashDrafterConfigError(
            code, "expected an integer"
        )
    return value


_DFLASH_DRAFTER_ARCHITECTURE = "DFlashDraftModel"


def _is_dflash_drafter_hf_config(hf_config: Mapping[str, object]) -> bool:
    """A DFlash drafter HF checkpoint declares BOTH the DFlashDraftModel
    architecture AND a dflash_config block (Finding 2: requiring both refuses a
    conflicting architecture that merely happens to carry a dflash_config field)."""
    architectures = hf_config.get("architectures")
    has_dflash_architecture = (
        isinstance(architectures, list)
        and architectures
        and str(architectures[0]) == _DFLASH_DRAFTER_ARCHITECTURE
    )
    return has_dflash_architecture and isinstance(
        hf_config.get("dflash_config"), Mapping
    )


def build_drafter_block(
    hf_config: Mapping[str, object],
    *,
    paired_target_id: str,
    paired_target_revision: str,
    paired_target_tokenizer_identity: str,
) -> dict[str, object]:
    """Build a ``dflash_drafter`` model_config block from a DFlash drafter HF
    ``config.json`` plus caller-supplied pairing target identity.

    The HF checkpoint carries the conditioning fields (``dflash_config``:
    ``block_size``/``mask_token_id``/``target_layer_ids`` and top-level
    ``num_target_layers``) but NOT the identity of the AR target it was
    co-trained against — that pairing is supplied by the caller converting the
    checkpoint, since it is what makes the drafter meaningful in this framework.

    Field parsing only: this does NOT load the custom ``DFlashDraftModel``
    architecture (no ``trust_remote_code`` model load). The returned block
    round-trips through :func:`recognize_drafter`. Raises
    :class:`DFlashDrafterConfigError` for a non-drafter HF config or any
    malformed/missing field.
    """
    if not _is_dflash_drafter_hf_config(hf_config):
        raise DFlashDrafterConfigError(
            "not_a_dflash_drafter",
            "HF config is not a DFlash drafter (no DFlashDraftModel architecture "
            "and no dflash_config block)",
        )
    dflash_config = hf_config.get("dflash_config")
    if not isinstance(dflash_config, Mapping):
        raise DFlashDrafterConfigError(
            "dflash_config_missing",
            "DFlash drafter HF config must carry a dflash_config block",
        )
    # HF layout (pinned 96899cc): block_size and num_target_layers are TOP-LEVEL;
    # dflash_config carries only mask_token_id + target_layer_ids. Our package
    # gathers all drafter-identity fields into one dflash_drafter block.
    raw_layer_ids = dflash_config.get("target_layer_ids")
    block: dict[str, object] = {
        "block_size": hf_config.get("block_size"),
        "mask_token_id": dflash_config.get("mask_token_id"),
        "target_layer_ids": (
            list(raw_layer_ids) if isinstance(raw_layer_ids, list) else raw_layer_ids
        ),
        "num_target_layers": hf_config.get("num_target_layers"),
        "paired_target": {
            "id": paired_target_id,
            "revision": paired_target_revision,
            "tokenizer_identity": paired_target_tokenizer_identity,
        },
    }
    # Validate via the canonical recognizer so the produced block is guaranteed
    # to round-trip (raises DFlashDrafterConfigError on any malformed field).
    recognize_drafter({"dflash_drafter": block})
    return block
