from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from edllm.validation import _passes_text_quality_gate

try:
    from transformers import AutoModel, AutoTokenizer
    from transformers import __version__ as TRANSFORMERS_VERSION
except Exception:  # pragma: no cover - exercised when optional deps are absent.
    AutoModel = None
    AutoTokenizer = None
    TRANSFORMERS_VERSION = ""

try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
except Exception:  # pragma: no cover - exercised when optional deps are absent.
    ROPE_INIT_FUNCTIONS = None

try:
    from peft import PeftModel
    from peft import __version__ as PEFT_VERSION
except Exception:  # pragma: no cover - exercised when optional deps are absent.
    PeftModel = None
    PEFT_VERSION = ""


@dataclass(slots=True)
class HFReferenceResult:
    text: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HFReferenceState:
    tokenizer: Any
    model: Any
    base_model_ref: str
    adapter_ref: str
    device: str


def load_reference_state(
    *,
    base_model_ref: str,
    adapter_ref: str | None = None,
) -> HFReferenceState:
    if not base_model_ref:
        raise ValueError("base_model_ref is required")
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError("transformers is required for hf_dream_reference")
    _require_supported_transformers()
    tokenizer = AutoTokenizer.from_pretrained(base_model_ref, trust_remote_code=True)
    _ensure_default_rope_init_function()
    model = AutoModel.from_pretrained(base_model_ref, trust_remote_code=True)
    if adapter_ref is not None:
        if PeftModel is None:
            raise RuntimeError("peft is required for hf_dream_reference adapter loading")
        _require_supported_peft()
        _ensure_torch_dtensor_namespace_for_peft()
        model = PeftModel.from_pretrained(model, adapter_ref).merge_and_unload()
    model.eval()
    device = _select_device()
    model.to(device)
    if not hasattr(model, "diffusion_generate"):
        raise RuntimeError("hf_dream_reference model does not expose diffusion_generate")
    return HFReferenceState(
        tokenizer=tokenizer,
        model=model,
        base_model_ref=base_model_ref,
        adapter_ref=adapter_ref or "",
        device=str(device),
    )


def generate_reference(
    *,
    state: HFReferenceState,
    prompt: str,
    chat_messages: list[dict] | None = None,
    max_tokens: int,
) -> HFReferenceResult:
    input_ids, attention_mask = _reference_inputs(
        tokenizer=state.tokenizer,
        prompt=prompt,
        chat_messages=chat_messages,
        device=state.device,
    )
    generation_config = _dream_generation_config(
        state.model,
        tokenizer=state.tokenizer,
        max_tokens=max_tokens,
    )
    generate_kwargs = {"inputs": input_ids}
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask
    if generation_config is not None:
        output_ids = state.model.diffusion_generate(
            generation_config=generation_config,
            **generate_kwargs,
        )
    else:
        output_ids = state.model.diffusion_generate(max_new_tokens=max_tokens, **generate_kwargs)
    generated_ids = _generated_tail(output_ids, prompt_token_count=input_ids.shape[-1])
    text = state.tokenizer.decode(generated_ids, skip_special_tokens=True)
    quality_passed = _passes_text_quality_gate(text)
    return HFReferenceResult(
        text=text,
        metrics={
            "reference_tokenizer_real": True,
            "reference_text_quality_passed": quality_passed,
            "reference_backend_usable": quality_passed,
            "reference_model": state.base_model_ref,
            "reference_adapter": state.adapter_ref,
            "reference_device": state.device,
        },
    )


def _select_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _require_supported_transformers() -> None:
    if not str(TRANSFORMERS_VERSION).startswith("4.46."):
        raise RuntimeError(
            "hf_dream_reference requires transformers 4.46.x; "
            f"found {TRANSFORMERS_VERSION or 'unknown'}"
        )


def _require_supported_peft() -> None:
    if _version_tuple(PEFT_VERSION) < (0, 19, 0):
        raise RuntimeError(
            "hf_dream_reference adapter loading requires peft >=0.19; "
            f"found {PEFT_VERSION or 'unknown'}"
        )


def _ensure_torch_dtensor_namespace_for_peft() -> None:
    try:
        import torch.distributed.tensor  # noqa: F401
    except Exception:
        return


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in str(version).split("."):
        digits = "".join(char for char in piece if char.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _reference_inputs(
    *,
    tokenizer,
    prompt: str,
    chat_messages: list[dict] | None = None,
    device: str,
):
    messages = (
        [dict(message) for message in chat_messages]
        if chat_messages is not None
        else [{"role": "user", "content": prompt}]
    )
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        inputs = apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
    else:
        rendered_prompt = (
            _plain_chat_transcript(messages)
            if chat_messages is not None
            else prompt
        )
        inputs = tokenizer(rendered_prompt, return_tensors="pt")
    input_ids = _input_value(inputs, "input_ids").to(device)
    attention_mask = _input_value(inputs, "attention_mask", default=None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _plain_chat_transcript(messages: list[dict]) -> str:
    lines = [
        f"{str(message['role']).upper()}: {message['content']}"
        for message in messages
    ]
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _input_value(inputs, key: str, default=None):
    value = getattr(inputs, key, None)
    if value is not None:
        return value
    if isinstance(inputs, dict):
        return inputs.get(key, default)
    return default


def _ensure_default_rope_init_function() -> None:
    if ROPE_INIT_FUNCTIONS is None or "default" in ROPE_INIT_FUNCTIONS:
        return
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _compute_default_rope_parameters(
    config,
    device=None,
    seq_len: int | None = None,
    layer_type: str | None = None,
):
    del seq_len
    import torch

    standardize = getattr(config, "standardize_rope_params", None)
    if callable(standardize):
        standardize()
    rope_parameters = getattr(config, "rope_parameters", None)
    params = {}
    if isinstance(rope_parameters, dict):
        layer_params = rope_parameters.get(layer_type) if layer_type is not None else None
        params = layer_params if isinstance(layer_params, dict) else rope_parameters
    base = float(params.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
    partial_rotary_factor = float(params.get("partial_rotary_factor", 1.0))
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    positions = torch.arange(0, dim, 2, dtype=torch.int64)
    if device is not None:
        positions = positions.to(device=device, dtype=torch.float)
    else:
        positions = positions.to(dtype=torch.float)
    inv_freq = 1.0 / (base ** (positions / dim))
    return inv_freq, 1.0


def _dream_generation_config(model, *, tokenizer, max_tokens: int):
    config_cls = _dream_generation_config_cls(model)
    if config_cls is None:
        return None
    return config_cls(
        max_new_tokens=max_tokens,
        steps=512,
        temperature=0.2,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
        **_generation_special_token_ids(model, tokenizer),
    )


def _dream_generation_config_cls(model):
    method = getattr(model, "diffusion_generate", None)
    func = getattr(method, "__func__", method)
    seen = set()
    while func is not None and id(func) not in seen:
        seen.add(id(func))
        globals_dict = getattr(func, "__globals__", {})
        config_cls = globals_dict.get("DreamGenerationConfig")
        if config_cls is not None:
            return config_cls
        func = getattr(func, "__wrapped__", None)
    return None


def _generation_special_token_ids(model, tokenizer) -> dict[str, int]:
    model_config = getattr(model, "config", None)
    token_ids: dict[str, int] = {}
    for key in ("bos_token_id", "eos_token_id", "pad_token_id", "mask_token_id"):
        value = getattr(model_config, key, None) if model_config is not None else None
        if value is None:
            value = getattr(tokenizer, key, None)
        if value is not None:
            token_ids[key] = int(value)
    if "pad_token_id" not in token_ids and "eos_token_id" in token_ids:
        token_ids["pad_token_id"] = token_ids["eos_token_id"]
    return token_ids


def _generated_tail(output_ids, *, prompt_token_count: int):
    sequences = getattr(output_ids, "sequences", output_ids)
    first_sequence = sequences[0]
    return first_sequence[prompt_token_count:]
