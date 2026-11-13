"""Product registry generated from the canonical model-descriptor catalog."""

from __future__ import annotations

import os

from ..model_definitions import (
    PRODUCT_MODEL_DEFINITION_KEYS,
    model_definition_for_descriptor,
)
from ..model_coverage.producer_descriptors import (
    DREAM_MODEL_ID,
    MODEL_CATALOG,
    ModelDescriptor,
    QWEN3_MDLM_MODEL_ID,
    runtime_descriptor_for_key,
)


# Kept as the stable Dream compatibility name used by existing callers. It is
# not a claim that the catalog contains only one model.
PINNED_MODEL_ID = DREAM_MODEL_ID

PROVISIONAL_BOOTSTRAP_ENV = (
    "DIFFUSORRT_INTERNAL_PROVISIONAL_BOOTSTRAP_CANDIDATE"
)

def _runtime_registry() -> dict[str, ModelDescriptor]:
    """Resolve release configuration without loading certification receipts.

    Admission evidence verifies a release; it is not runtime configuration.
    Importing ``Engine`` or opening a package must therefore remain possible
    while source edits have invalidated receipts that the integrator will later
    refresh.  Release gates still validate every Product receipt independently.
    """

    provisional = os.environ.get(PROVISIONAL_BOOTSTRAP_ENV)
    if provisional is not None:
        try:
            runtime_descriptor_for_key(provisional)
        except ValueError:
            raise ValueError("admission_provisional_bootstrap_invalid") from None
    keys = set(PRODUCT_MODEL_DEFINITION_KEYS)
    if provisional is not None:
        keys.add(provisional)
    descriptors = tuple(runtime_descriptor_for_key(key) for key in sorted(keys))
    for descriptor in descriptors:
        model_definition_for_descriptor(descriptor)
    return {descriptor.model_id: descriptor for descriptor in descriptors}


# The runtime registry contains Product-configured descriptors only. Candidate
# descriptors remain available to the certification subsystem via MODEL_CATALOG.
REGISTRY: dict[str, ModelDescriptor] = _runtime_registry()

# Compatibility alias for code that imported the old type name. There is no
# second structure: RegistryEntry and ModelDescriptor are the same type.
RegistryEntry = ModelDescriptor


__all__ = [
    "MODEL_CATALOG",
    "PINNED_MODEL_ID",
    "PROVISIONAL_BOOTSTRAP_ENV",
    "QWEN3_MDLM_MODEL_ID",
    "REGISTRY",
    "RegistryEntry",
]
