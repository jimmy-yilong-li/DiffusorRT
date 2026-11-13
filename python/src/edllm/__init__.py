from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.2.6"


def _extra_install_command(extra: str) -> str:
    """Render a release installation command without assuming a PyPI upload."""
    wheel = f"diffusor_rt-{__version__}-py3-none-any.whl"
    url = f"https://github.com/jimmy-yilong-li/DiffusorRT/releases/download/v{__version__}/{wheel}"
    return f'python -m pip install "diffusor-rt[{extra}] @ {url}"'


_EXPORTS = {
    "CompilerError": (".compiler.errors", "CompilerError"),
    "CompilerDependencyError": (".compiler.errors", "CompilerDependencyError"),
    "UnsupportedModelError": (".compiler.errors", "UnsupportedModelError"),
    "RevisionMismatchError": (".compiler.errors", "RevisionMismatchError"),
    "InsufficientDiskError": (".compiler.errors", "InsufficientDiskError"),
    "ModelAccessError": (".compiler.errors", "ModelAccessError"),
    "SnapshotNotCachedError": (".compiler.errors", "SnapshotNotCachedError"),
    "RemoteCodeNotTrustedError": (".compiler.errors", "RemoteCodeNotTrustedError"),
    "ConfigurationError": (".application_state", "ConfigurationError"),
    "ConversationError": (".conversation", "ConversationError"),
    "ConversationConflictError": (".conversation", "ConversationConflictError"),
    "ConversationIdentityError": (".conversation", "ConversationIdentityError"),
    "ConversationContextError": (".conversation", "ConversationContextError"),
    "RunProfile": (".application_state", "RunProfile"),
    "Conversation": (".conversation", "Conversation"),
    "ConversationInfo": (".conversation", "ConversationInfo"),
    "ConversationReply": (".conversation", "ConversationReply"),
    "ConversationSummary": (".conversation", "ConversationSummary"),
    "MemoryItem": (".conversation", "MemoryItem"),
    "MemoryProvider": (".conversation", "MemoryProvider"),
    "Engine": (".engine", "Engine"),
    "DebugBundle": (".result", "DebugBundle"),
    "GenerateResult": (".result", "GenerateResult"),
    "EmptyGenerationError": (".result", "EmptyGenerationError"),
    "CancellationToken": (".cancellation", "CancellationToken"),
    "GenerationCancelled": (".cancellation", "GenerationCancelled"),
    "InfillResult": (".result", "InfillResult"),
    "BlockLogits": (".executor", "BlockLogits"),
    "ExecutorInput": (".executor", "ExecutorInput"),
    "RuntimeHandle": (".model", "RuntimeHandle"),
    "TokenizerSpec": (".tokenizer", "TokenizerSpec"),
    "LocalModelEntry": (".model_catalog", "LocalModelEntry"),
    "ModelCatalog": (".model_catalog", "ModelCatalog"),
    "ModelRepository": (".local_models", "ModelRepository"),
    "ModelResources": (".model_catalog", "ModelResources"),
    "model_definition_digest": (".model_catalog", "model_definition_digest"),
    "DecodeContract": (".model_definitions", "DecodeContract"),
    "DecodePolicy": (".model_definitions", "DecodePolicy"),
    "ArchitectureDefinition": (".model_definitions", "ArchitectureDefinition"),
    "CheckpointMetadata": (".model_definitions", "CheckpointMetadata"),
    "DeploymentContract": (".model_definitions", "DeploymentContract"),
    "DeploymentPolicy": (".model_definitions", "DeploymentPolicy"),
    "DeploymentTarget": (".model_definitions", "DeploymentTarget"),
    "ModelDefinition": (".model_definitions", "ModelDefinition"),
    "converter_adapter_for_key": (
        ".model_definitions",
        "converter_adapter_for_key",
    ),
    "runtime_architecture_for_key": (
        ".model_definitions",
        "runtime_architecture_for_key",
    ),
    "validate_model_snapshot": (
        ".model_definitions",
        "validate_model_snapshot",
    ),
    "load_runtime": (".model", "load_runtime"),
    "try_load_package": (".model", "try_load_package"),
}

__all__ = [
    "ArchitectureDefinition",
    "BlockLogits",
    "CheckpointMetadata",
    "CompilerDependencyError",
    "CompilerError",
    "ConfigurationError",
    "Conversation",
    "ConversationConflictError",
    "ConversationContextError",
    "ConversationError",
    "ConversationIdentityError",
    "ConversationInfo",
    "ConversationReply",
    "ConversationSummary",
    "DebugBundle",
    "DecodeContract",
    "DecodePolicy",
    "DeploymentContract",
    "DeploymentPolicy",
    "DeploymentTarget",
    "EmptyGenerationError",
    "Engine",
    "ExecutorInput",
    "GenerateResult",
    "InfillResult",
    "InsufficientDiskError",
    "LocalModelEntry",
    "MemoryItem",
    "MemoryProvider",
    "ModelAccessError",
    "ModelCatalog",
    "ModelDefinition",
    "ModelRepository",
    "ModelResources",
    "RemoteCodeNotTrustedError",
    "RevisionMismatchError",
    "RunProfile",
    "RuntimeHandle",
    "SnapshotNotCachedError",
    "TokenizerSpec",
    "UnsupportedModelError",
    "__version__",
    "converter_adapter_for_key",
    "load_runtime",
    "model_definition_digest",
    "runtime_architecture_for_key",
    "try_load_package",
    "validate_model_snapshot",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
