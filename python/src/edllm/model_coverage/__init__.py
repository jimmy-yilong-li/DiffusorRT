"""Lazy public API for producer descriptors and admission control."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "ADMISSION_CATALOG": ".descriptors",
    "AdmissionMetadata": ".descriptors",
    "AdmissionState": ".descriptors",
    "admission_metadata_for_key": ".descriptors",
    "product_registry": ".descriptors",
    "load_admission_evidence": ".admission",
    "validate_repository_admission_evidence": ".admission",
    "DREAM_MODEL_ID": ".producer_descriptors",
    "MODEL_CATALOG": ".producer_descriptors",
    "PRODUCER_CATALOG": ".producer_descriptors",
    "QWEN3_BD3LM_MODEL_ID": ".producer_descriptors",
    "QWEN3_MDLM_MODEL_ID": ".producer_descriptors",
    "BackendAdmission": ".producer_descriptors",
    "CapabilityCase": ".producer_descriptors",
    "COVERAGE_RECORDS": ".producer_descriptors",
    "CoverageRecord": ".producer_descriptors",
    "DecodeContract": ".producer_descriptors",
    "DeploymentContract": ".producer_descriptors",
    "LogitsConvention": ".producer_descriptors",
    "ModelDescriptor": ".producer_descriptors",
    "NumericTier": ".producer_descriptors",
    "ProducerDescriptor": ".producer_descriptors",
    "descriptor_for_identity": ".producer_descriptors",
    "descriptor_for_key": ".producer_descriptors",
    "coverage_record_for_key": ".producer_descriptors",
    "load_case_corpus": ".producer_descriptors",
    "load_oracle_lock": ".producer_descriptors",
    "validate_execution_package": ".producer_descriptors",
    "validate_snapshot": ".producer_descriptors",
}


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "ADMISSION_CATALOG",
    "COVERAGE_RECORDS",
    "DREAM_MODEL_ID",
    "MODEL_CATALOG",
    "PRODUCER_CATALOG",
    "QWEN3_BD3LM_MODEL_ID",
    "QWEN3_MDLM_MODEL_ID",
    "AdmissionMetadata",
    "AdmissionState",
    "BackendAdmission",
    "CapabilityCase",
    "CoverageRecord",
    "DecodeContract",
    "DeploymentContract",
    "LogitsConvention",
    "ModelDescriptor",
    "NumericTier",
    "ProducerDescriptor",
    "admission_metadata_for_key",
    "coverage_record_for_key",
    "descriptor_for_identity",
    "descriptor_for_key",
    "load_admission_evidence",
    "load_case_corpus",
    "load_oracle_lock",
    "product_registry",
    "validate_execution_package",
    "validate_repository_admission_evidence",
    "validate_snapshot",
]
