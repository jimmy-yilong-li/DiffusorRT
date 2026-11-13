"""Stable admission types shared by control state and registry policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .admission import validate_admission_pointer
from .producer_descriptors import PRODUCER_CATALOG


class AdmissionState(StrEnum):
    HISTORICAL = "historical"
    CANDIDATE = "candidate"
    PARITY_RESTORED = "parity-restored"
    EXPERIMENTALLY_GATED = "experimentally-gated"
    PRODUCT_SUPPORTED = "product-supported"


@dataclass(frozen=True, slots=True)
class AdmissionMetadata:
    key: str
    state: AdmissionState
    evidence_resource: str | None = None
    evidence_digest: str | None = None

    def validate(self) -> None:
        if self.key not in set(PRODUCER_CATALOG):
            raise ValueError(f"admission_metadata_unknown_descriptor: {self.key}")
        validate_admission_pointer(self)
        if (
            self.state
            in {
                AdmissionState.PARITY_RESTORED,
                AdmissionState.EXPERIMENTALLY_GATED,
                AdmissionState.PRODUCT_SUPPORTED,
            }
            and self.evidence_resource is None
        ):
            raise ValueError(f"admission_evidence_missing: {self.key}")


__all__ = ["AdmissionMetadata", "AdmissionState"]
