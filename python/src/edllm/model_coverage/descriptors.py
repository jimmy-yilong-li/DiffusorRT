"""Admission control plane over producer-only model descriptors.

Producer facts live in :mod:`producer_descriptors`.  This module owns mutable
admission state and receipt pointers, and joins them only when constructing the
release-visible product registry.
"""

from __future__ import annotations

from typing import Mapping

from ..model_definitions import PRODUCT_MODEL_DEFINITION_KEYS
from .admission import load_admission_evidence
from .admission_catalog import ADMISSION_CATALOG, admission_metadata_for_key
from .admission_types import AdmissionMetadata, AdmissionState
from .producer_descriptors import (
    DREAM_CODER_MODEL_ID,
    DREAM_MODEL_ID,
    LLADA_MODEL_ID,
    MODEL_CATALOG,
    PRODUCER_CATALOG,
    QWEN3_BD3LM_MODEL_ID,
    QWEN3_MDLM_MODEL_ID,
    BackendAdmission,
    CapabilityCase,
    DecodeContract,
    LogitsConvention,
    ModelDescriptor,
    NumericTier,
    ProducerDescriptor,
    _canonical_required_signature_function as _canonical_required_signature_function,
    _validate_case as _validate_case,
    descriptor_for_identity,
    descriptor_for_key,
    load_case_corpus,
    load_oracle_lock,
    validate_execution_package,
    validate_snapshot,
)

def _validate_admission_catalog(
    catalog: Mapping[str, AdmissionMetadata],
) -> None:
    if set(catalog) != set(PRODUCER_CATALOG):
        raise ValueError("admission_catalog_descriptor_set_mismatch")
    for key, metadata in catalog.items():
        if key != metadata.key:
            raise ValueError(f"admission_metadata_key_mismatch: {key}")
        metadata.validate()
    admitted = {
        key
        for key, metadata in catalog.items()
        if metadata.state == AdmissionState.PRODUCT_SUPPORTED
    }
    if admitted != set(PRODUCT_MODEL_DEFINITION_KEYS):
        raise ValueError("runtime_product_catalog_admission_state_mismatch")
_validate_admission_catalog(ADMISSION_CATALOG)


def product_registry(
    *, provisional_bootstrap_candidate: str | None = None
) -> dict[str, ModelDescriptor]:
    """Join producer descriptors with Product admission metadata.

    A provisional v2 receipt is allowed only for the explicit bootstrap W6.
    A transition receipt exposes the ordinary registry long enough to run the
    default-path W6. Repository and release validation reject both intermediate
    stages; only a final v2 receipt can ship.
    """
    if provisional_bootstrap_candidate is not None and (
        type(provisional_bootstrap_candidate) is not str
        or provisional_bootstrap_candidate not in ADMISSION_CATALOG
    ):
        raise ValueError("admission_provisional_bootstrap_invalid")

    registry: dict[str, ModelDescriptor] = {}
    bootstrap_consumed = False
    for key, metadata in ADMISSION_CATALOG.items():
        if metadata.state != AdmissionState.PRODUCT_SUPPORTED:
            continue
        descriptor = descriptor_for_key(key)
        receipt = load_admission_evidence(descriptor, metadata=metadata)
        kind = receipt["evidence_kind"]
        if kind == "legacy-product-migration-v1":
            registry[descriptor.model_id] = descriptor
            continue
        if kind != "model-coverage-v2":
            raise ValueError(f"admission_evidence_not_release_final: {key}")
        stage = receipt["stage"]
        if stage == "provisional":
            if provisional_bootstrap_candidate != key:
                continue
            bootstrap_consumed = True
        elif stage not in {"transition", "final"}:
            raise ValueError(f"admission_evidence_not_release_final: {key}")
        registry[descriptor.model_id] = descriptor

    if provisional_bootstrap_candidate is not None and not bootstrap_consumed:
        raise ValueError("admission_provisional_bootstrap_not_available")
    return registry


__all__ = [
    "ADMISSION_CATALOG",
    "DREAM_CODER_MODEL_ID",
    "DREAM_MODEL_ID",
    "LLADA_MODEL_ID",
    "MODEL_CATALOG",
    "PRODUCER_CATALOG",
    "QWEN3_BD3LM_MODEL_ID",
    "QWEN3_MDLM_MODEL_ID",
    "AdmissionMetadata",
    "AdmissionState",
    "BackendAdmission",
    "CapabilityCase",
    "DecodeContract",
    "LogitsConvention",
    "ModelDescriptor",
    "NumericTier",
    "ProducerDescriptor",
    "admission_metadata_for_key",
    "descriptor_for_identity",
    "descriptor_for_key",
    "load_case_corpus",
    "load_oracle_lock",
    "product_registry",
    "validate_execution_package",
    "validate_snapshot",
]
