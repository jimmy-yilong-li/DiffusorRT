"""Strict loader for declarative admission state and receipt pointers."""

from __future__ import annotations

from importlib import resources

from .admission import _parse_json, _require_exact_keys
from .admission_types import AdmissionMetadata, AdmissionState
from .producer_descriptors import PRODUCER_CATALOG


ADMISSION_CATALOG_RESOURCE = "data/admission_catalog.json"


def _load_admission_catalog() -> dict[str, AdmissionMetadata]:
    try:
        raw = resources.files(__package__).joinpath(
            ADMISSION_CATALOG_RESOURCE
        ).read_bytes()
    except (OSError, TypeError, ValueError, AttributeError):
        raise ValueError("admission_catalog_resource_unreadable") from None
    document = _require_exact_keys(
        _parse_json(raw),
        {"schema_version", "entries"},
        error="admission_catalog_schema_invalid",
    )
    if type(document["schema_version"]) is not int:
        raise ValueError("admission_catalog_schema_invalid")
    if document["schema_version"] != 1:
        raise ValueError("admission_catalog_schema_unsupported")
    entries = document["entries"]
    if type(entries) is not list:
        raise ValueError("admission_catalog_entries_invalid")

    catalog: dict[str, AdmissionMetadata] = {}
    ordered_keys: list[str] = []
    for raw_entry in entries:
        entry = _require_exact_keys(
            raw_entry,
            {"key", "state", "evidence_resource", "evidence_digest"},
            error="admission_catalog_entry_invalid",
        )
        key = entry["key"]
        state = entry["state"]
        if type(key) is not str or not key or type(state) is not str:
            raise ValueError("admission_catalog_entry_invalid")
        if key in catalog:
            raise ValueError("admission_catalog_key_duplicate")
        try:
            metadata = AdmissionMetadata(
                key=key,
                state=AdmissionState(state),
                evidence_resource=entry["evidence_resource"],
                evidence_digest=entry["evidence_digest"],
            )
        except (TypeError, ValueError):
            raise ValueError("admission_catalog_entry_invalid") from None
        metadata.validate()
        catalog[key] = metadata
        ordered_keys.append(key)
    if ordered_keys != sorted(ordered_keys):
        raise ValueError("admission_catalog_order_invalid")
    if set(catalog) != set(PRODUCER_CATALOG):
        raise ValueError("admission_catalog_descriptor_set_mismatch")
    return catalog


ADMISSION_CATALOG = _load_admission_catalog()


def admission_metadata_for_key(key: str) -> AdmissionMetadata:
    try:
        return ADMISSION_CATALOG[key]
    except KeyError:
        raise ValueError(f"admission_metadata_missing: {key}") from None


__all__ = [
    "ADMISSION_CATALOG",
    "ADMISSION_CATALOG_RESOURCE",
    "admission_metadata_for_key",
]
