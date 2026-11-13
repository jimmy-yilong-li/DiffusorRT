from __future__ import annotations

import argparse
import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_MERGES_BYTES = 64 * 1024 * 1024
_SPECIAL_ROLES = ("bos", "eos", "pad", "unk", "mask")
_MASK_LIKE_TOKENS = frozenset(("<|mask|>", "<|mask_token|>", "[MASK]"))


class TokenContractError(ValueError):
    def __init__(self, code: str, message: str | None = None) -> None:
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True, slots=True)
class SpecialToken:
    content: str | None
    ids: tuple[int, ...]
    model_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "content": self.content,
            "ids": list(self.ids),
            "model_ids": list(self.model_ids),
            "model_ids_match": not self.model_ids or self.model_ids == self.ids,
        }


@dataclass(frozen=True, slots=True)
class TokenizerContract:
    source_path: str
    resolved_path: str
    revision: str | None
    declared_revisions: dict[str, str]
    tokenizer_class: str | None
    base_token_to_id: dict[str, int]
    added_token_to_id: dict[str, int]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    merge_rules: tuple[str, ...]
    model_vocab_size: int
    specials: dict[str, SpecialToken]
    chat_template: str | None
    warnings: tuple[str, ...]

    @property
    def defined_ids(self) -> set[int]:
        return set(self.id_to_token)

    @property
    def undefined_head_ids(self) -> set[int]:
        return set(range(self.model_vocab_size)).difference(self.defined_ids)

    def summary(self) -> dict[str, object]:
        semantic_payload = {
            "base_token_to_id": self.base_token_to_id,
            "added_token_to_id": self.added_token_to_id,
            "merge_rules": self.merge_rules,
            "specials": {
                role: {
                    "content": value.content,
                    "ids": value.ids,
                }
                for role, value in self.specials.items()
            },
        }
        return {
            "source_path": self.source_path,
            "resolved_path": self.resolved_path,
            "revision": self.revision,
            "declared_revisions": dict(sorted(self.declared_revisions.items())),
            "tokenizer_class": self.tokenizer_class,
            "base_vocab_size": len(self.base_token_to_id),
            "defined_token_count": len(self.token_to_id),
            "model_vocab_size": self.model_vocab_size,
            "base_vocab_digest": _digest(self.base_token_to_id),
            "merge_rules_digest": _digest(self.merge_rules),
            "semantic_digest": _digest(semantic_payload),
            "chat_template_digest": (
                _digest(self.chat_template) if self.chat_template is not None else None
            ),
            "special_tokens": {
                role: value.to_dict() for role, value in self.specials.items()
            },
            "undefined_head_id_ranges": _compact_ranges(self.undefined_head_ids),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class TokenContractReport:
    verdict: str
    prompt_owner: str
    reasons: tuple[str, ...]
    draft: dict[str, object] | None
    target: dict[str, object] | None
    comparison: dict[str, object]
    prompt_contract: dict[str, object]
    proposal_filter: dict[str, object]
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "verdict": self.verdict,
            "prompt_owner": self.prompt_owner,
            "reasons": list(self.reasons),
            "draft": self.draft,
            "target": self.target,
            "comparison": self.comparison,
            "prompt_contract": self.prompt_contract,
            "proposal_filter": self.proposal_filter,
        }
        if self.error is not None:
            payload["error"] = dict(self.error)
        return payload


def diagnose_token_contract(
    draft_path: str | Path,
    target_path: str | Path,
    *,
    prompt_owner: str = "target",
) -> TokenContractReport:
    if prompt_owner not in {"target", "draft", "pretokenized"}:
        return _invalid_report(
            prompt_owner=prompt_owner,
            code="prompt_owner_invalid",
            message=f"unsupported prompt owner: {prompt_owner}",
        )

    try:
        draft = _load_contract(draft_path, role="draft")
        target = _load_contract(target_path, role="target")
    except TokenContractError as exc:
        return _invalid_report(
            prompt_owner=prompt_owner,
            code=exc.code,
            message=str(exc),
        )
    return _compare_contracts(draft, target, prompt_owner=prompt_owner)


def _load_contract(path: str | Path, *, role: str) -> TokenizerContract:
    source_path = str(Path(path).expanduser())
    try:
        root = Path(path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise TokenContractError(f"{role}_root_missing") from exc
    if not root.is_dir():
        raise TokenContractError(f"{role}_root_not_directory")

    vocab_payload = _read_json(root, "vocab.json", role=role, name="vocab")
    base_token_to_id = _validate_token_map(
        vocab_payload,
        code_prefix=f"{role}_vocab",
    )
    if not base_token_to_id:
        raise TokenContractError(f"{role}_vocab_empty")
    merge_rules = _read_merge_rules(root, role=role)
    tokenizer_config = _read_json(
        root,
        "tokenizer_config.json",
        role=role,
        name="tokenizer_config",
    )
    if not isinstance(tokenizer_config, dict):
        raise TokenContractError(f"{role}_tokenizer_config_not_object")

    added_token_to_id = _load_added_tokens(root, tokenizer_config, role=role)
    token_to_id, id_to_token = _merge_token_maps(
        base_token_to_id,
        added_token_to_id,
        role=role,
    )
    model_payload, model_name = _load_model_payload(root, role=role)
    model_config = model_payload.get("model_config", model_payload)
    if not isinstance(model_config, dict):
        raise TokenContractError(f"{role}_{model_name}_model_config_not_object")
    model_vocab_size = _resolve_model_vocab_size(model_config, role=role)
    if id_to_token and max(id_to_token) >= model_vocab_size:
        raise TokenContractError(f"{role}_defined_token_outside_model_head")

    special_map = _read_json(
        root,
        "special_tokens_map.json",
        role=role,
        name="special_tokens",
        required=False,
    )
    if special_map is not None and not isinstance(special_map, dict):
        raise TokenContractError(f"{role}_special_tokens_not_object")
    specials, special_warnings = _load_specials(
        tokenizer_config=tokenizer_config,
        special_map=special_map or {},
        model_config=model_config,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        model_vocab_size=model_vocab_size,
        role=role,
    )
    chat_template = _load_chat_template(root, tokenizer_config, role=role)
    declared_revisions = _declared_revisions(model_payload)

    return TokenizerContract(
        source_path=source_path,
        resolved_path=str(root),
        revision=_infer_revision(root),
        declared_revisions=declared_revisions,
        tokenizer_class=_optional_string(tokenizer_config.get("tokenizer_class")),
        base_token_to_id=base_token_to_id,
        added_token_to_id=added_token_to_id,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        merge_rules=merge_rules,
        model_vocab_size=model_vocab_size,
        specials=specials,
        chat_template=chat_template,
        warnings=tuple(special_warnings),
    )


def _compare_contracts(
    draft: TokenizerContract,
    target: TokenizerContract,
    *,
    prompt_owner: str,
) -> TokenContractReport:
    draft_summary = draft.summary()
    target_summary = target.summary()
    base_match = draft.base_token_to_id == target.base_token_to_id
    merges_match = draft.merge_rules == target.merge_rules
    chat_template_match = draft.chat_template == target.chat_template
    special_match = all(
        (draft.specials[role].content, draft.specials[role].ids)
        == (target.specials[role].content, target.specials[role].ids)
        for role in _SPECIAL_ROLES
    )

    shared_tokens = set(draft.token_to_id).intersection(target.token_to_id)
    shared_token_id_mismatches = {
        token
        for token in shared_tokens
        if draft.token_to_id[token] != target.token_to_id[token]
    }
    shared_ids = set(draft.id_to_token).intersection(target.id_to_token)
    shared_id_content_mismatches = {
        token_id
        for token_id in shared_ids
        if draft.id_to_token[token_id] != target.id_to_token[token_id]
    }

    identity_ids = {
        token_id
        for token_id, token in draft.id_to_token.items()
        if target.id_to_token.get(token_id) == token
    }
    mask_like_ids = {
        token_id
        for token_id, token in draft.id_to_token.items()
        if token in _MASK_LIKE_TOKENS
    }
    special_semantics_ids: set[int] = set()
    for role in _SPECIAL_ROLES:
        draft_special = draft.specials[role]
        target_special = target.specials[role]
        if (draft_special.content, draft_special.ids) != (
            target_special.content,
            target_special.ids,
        ):
            special_semantics_ids.update(draft_special.ids)

    undefined_head_ids = draft.undefined_head_ids
    not_identity_mapped = set(range(draft.model_vocab_size)).difference(identity_ids)
    forbidden_ids = (
        not_identity_mapped
        | mask_like_ids
        | special_semantics_ids
        | undefined_head_ids
    )
    identity_ids.difference_update(forbidden_ids)
    all_draft_head_ids = set(range(draft.model_vocab_size))
    classified_ids = identity_ids | forbidden_ids
    unresolved_ids = all_draft_head_ids.difference(classified_ids)

    forbidden_by_reason: dict[str, list[list[int]]] = {}
    reason_sets = {
        "undefined_head": undefined_head_ids,
        "mask_like": mask_like_ids,
        "special_semantics": special_semantics_ids,
        "not_identity_mapped": not_identity_mapped,
    }
    for reason, token_ids in reason_sets.items():
        if token_ids:
            forbidden_by_reason[reason] = _compact_ranges(token_ids)

    target_termination_ids = sorted(
        set(target.specials["eos"].ids) | set(target.specials["eos"].model_ids)
    )
    target_prompt_unmapped = _unmapped_ids(source=target, destination=draft)
    draft_prompt_unmapped = _unmapped_ids(source=draft, destination=target)

    incompatible_reasons: list[str] = []
    if not base_match:
        incompatible_reasons.append("base_token_map_mismatch")
    if not merges_match:
        incompatible_reasons.append("merge_rules_mismatch")
    if shared_token_id_mismatches:
        incompatible_reasons.append("shared_token_id_mismatch")
    if shared_id_content_mismatches:
        incompatible_reasons.append("shared_id_content_mismatch")
    if prompt_owner == "target" and target_prompt_unmapped:
        incompatible_reasons.append("target_prompt_space_not_representable_by_draft")
    if prompt_owner == "draft" and draft_prompt_unmapped:
        incompatible_reasons.append("draft_prompt_space_not_representable_by_target")
    if not target_termination_ids:
        incompatible_reasons.append("target_termination_unresolved")
    if unresolved_ids:
        incompatible_reasons.append("draft_head_ids_unresolved")

    reasons = list(incompatible_reasons)
    if incompatible_reasons:
        verdict = "incompatible"
    elif forbidden_ids:
        verdict = "compatible_with_filter"
        reasons.append("draft_proposal_filter_required")
    else:
        verdict = "compatible"
    if not special_match:
        reasons.append("special_token_contract_differs")
    if not chat_template_match:
        reasons.append(f"chat_template_differs_{prompt_owner}_owned")

    prompt_contract = {
        "owner": prompt_owner,
        "chat_template_match": chat_template_match,
        "owner_chat_template_digest": (
            draft_summary["chat_template_digest"]
            if prompt_owner == "draft"
            else target_summary["chat_template_digest"]
            if prompt_owner == "target"
            else None
        ),
    }
    comparison = {
        "base_token_map_match": base_match,
        "merge_rules_match": merges_match,
        "chat_template_match": chat_template_match,
        "special_token_contract_match": special_match,
        "shared_token_count": len(shared_tokens),
        "shared_token_id_mismatch_count": len(shared_token_id_mismatches),
        "shared_id_content_mismatch_count": len(shared_id_content_mismatches),
        "target_prompt_unmapped_id_count": len(target_prompt_unmapped),
        "draft_prompt_unmapped_id_count": len(draft_prompt_unmapped),
    }
    proposal_filter = {
        "complete": not unresolved_ids
        and identity_ids.isdisjoint(forbidden_ids)
        and classified_ids == all_draft_head_ids,
        "identity_mapped_id_count": len(identity_ids),
        "identity_mapped_id_ranges": _compact_ranges(identity_ids),
        "forbidden_draft_id_count": len(forbidden_ids),
        "forbidden_draft_id_ranges": _compact_ranges(forbidden_ids),
        "forbidden_by_reason": forbidden_by_reason,
        "target_termination_ids": target_termination_ids,
        "unresolved_draft_id_count": len(unresolved_ids),
    }
    return TokenContractReport(
        verdict=verdict,
        prompt_owner=prompt_owner,
        reasons=tuple(dict.fromkeys(reasons)),
        draft=draft_summary,
        target=target_summary,
        comparison=comparison,
        prompt_contract=prompt_contract,
        proposal_filter=proposal_filter,
    )


def _load_model_payload(root: Path, *, role: str) -> tuple[dict[str, object], str]:
    for filename, name in (("model.json", "model"), ("config.json", "config")):
        payload = _read_json(
            root,
            filename,
            role=role,
            name=name,
            required=False,
        )
        if payload is None:
            continue
        if not isinstance(payload, dict):
            raise TokenContractError(f"{role}_{name}_not_object")
        return payload, name
    raise TokenContractError(f"{role}_model_config_missing")


def _resolve_model_vocab_size(model_config: dict[str, object], *, role: str) -> int:
    """Resolve the text vocabulary size, failing closed if genuinely absent.

    Multimodal configs (e.g. Qwen3.5 ``Qwen3_5ForConditionalGeneration``) nest
    the text vocab under ``text_config.vocab_size`` instead of a top-level
    ``vocab_size``. Read the nested location before declaring the size missing,
    so the preflight fails closed on genuinely-absent sizes instead of
    misreading the vocabulary boundary of multimodal targets.
    """
    candidates = [model_config.get("vocab_size")]
    text_config = model_config.get("text_config")
    if isinstance(text_config, dict):
        candidates.append(text_config.get("vocab_size"))
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise TokenContractError(f"{role}_model_vocab_size_missing")


def _resolve_model_special_token_id(
    model_config: dict[str, object], key: str
) -> object:
    """Resolve a special-token id (``eos_token_id`` etc.) for the cross-check.

    Multimodal configs (e.g. Qwen3.5) nest these under ``text_config`` with no
    top-level value, same shape as ``vocab_size``. Return the top-level value
    when present, otherwise fall back to ``text_config.<key>`` (None if neither
    declares it — a genuinely-absent token, which the caller treats as no-op).
    """
    top = model_config.get(key)
    if top is not None:
        return top
    text_config = model_config.get("text_config")
    if isinstance(text_config, dict):
        return text_config.get(key)
    return None


def _load_added_tokens(
    root: Path,
    tokenizer_config: dict[str, object],
    *,
    role: str,
) -> dict[str, int]:
    combined: dict[str, int] = {}
    decoder = tokenizer_config.get("added_tokens_decoder", {})
    if decoder is None:
        decoder = {}
    if not isinstance(decoder, dict):
        raise TokenContractError(f"{role}_added_tokens_decoder_not_object")
    for raw_id, raw_token in decoder.items():
        try:
            token_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise TokenContractError(f"{role}_added_token_id_invalid") from exc
        if isinstance(raw_token, dict):
            token = raw_token.get("content")
        else:
            token = raw_token
        if not isinstance(token, str):
            raise TokenContractError(f"{role}_added_token_content_invalid")
        _insert_token(combined, token, token_id, role=role)

    payload = _read_json(
        root,
        "added_tokens.json",
        role=role,
        name="added_tokens",
        required=False,
    )
    if payload is None:
        return combined
    parsed = _validate_token_map(payload, code_prefix=f"{role}_added_tokens")
    for token, token_id in parsed.items():
        _insert_token(combined, token, token_id, role=role)
    return combined


def _merge_token_maps(
    base: dict[str, int],
    added: dict[str, int],
    *,
    role: str,
) -> tuple[dict[str, int], dict[int, str]]:
    combined = dict(base)
    id_to_token = {token_id: token for token, token_id in base.items()}
    for token, token_id in added.items():
        existing_id = combined.get(token)
        if existing_id is not None and existing_id != token_id:
            raise TokenContractError(f"{role}_token_has_multiple_ids")
        existing_token = id_to_token.get(token_id)
        if existing_token is not None and existing_token != token:
            raise TokenContractError(f"{role}_token_id_collision")
        combined[token] = token_id
        id_to_token[token_id] = token
    return combined, id_to_token


def _load_specials(
    *,
    tokenizer_config: dict[str, object],
    special_map: dict[str, object],
    model_config: dict[str, object],
    token_to_id: dict[str, int],
    id_to_token: dict[int, str],
    model_vocab_size: int,
    role: str,
) -> tuple[dict[str, SpecialToken], list[str]]:
    specials: dict[str, SpecialToken] = {}
    warnings: list[str] = []
    for special_role in _SPECIAL_ROLES:
        key = f"{special_role}_token"
        tokenizer_declares_special = key in tokenizer_config
        fallback_declares_special = key in special_map
        if key in tokenizer_config:
            content = _token_content(tokenizer_config[key], role=role, name=key)
        else:
            content = _token_content(special_map.get(key), role=role, name=key)
        model_ids = _token_ids(
            _resolve_model_special_token_id(
                model_config, f"{special_role}_token_id"
            ),
            role=role,
            name=f"{special_role}_token_id",
        )
        if any(token_id >= model_vocab_size for token_id in model_ids):
            raise TokenContractError(
                f"{role}_{special_role}_token_id_outside_model_head"
            )
        if any(token_id not in id_to_token for token_id in model_ids):
            raise TokenContractError(f"{role}_{special_role}_token_id_not_defined")
        tokenizer_ids: tuple[int, ...] = ()
        if content is not None:
            token_id = token_to_id.get(content)
            if token_id is None:
                raise TokenContractError(
                    f"{role}_{special_role}_token_not_in_tokenizer"
                )
            tokenizer_ids = (token_id,)
        elif model_ids and not (
            tokenizer_declares_special or fallback_declares_special
        ):
            known = [token_id for token_id in model_ids if token_id in id_to_token]
            tokenizer_ids = tuple(known)
            if len(known) == 1:
                content = id_to_token[known[0]]

        if model_ids and model_ids != tokenizer_ids:
            warnings.append(f"model_{special_role}_ids_differ_from_tokenizer")
        specials[special_role] = SpecialToken(
            content=content,
            ids=tokenizer_ids,
            model_ids=model_ids,
        )
    return specials, warnings


def _load_chat_template(
    root: Path,
    tokenizer_config: dict[str, object],
    *,
    role: str,
) -> str | None:
    if "chat_template" in tokenizer_config:
        value = tokenizer_config["chat_template"]
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        raise TokenContractError(f"{role}_chat_template_invalid")

    payload = _read_bytes(
        root,
        "chat_template.jinja",
        role=role,
        name="chat_template",
        max_bytes=_MAX_JSON_BYTES,
        required=False,
    )
    if payload is None:
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TokenContractError(f"{role}_chat_template_invalid_utf8") from exc


def _read_merge_rules(root: Path, *, role: str) -> tuple[str, ...]:
    payload = _read_bytes(
        root,
        "merges.txt",
        role=role,
        name="merges",
        max_bytes=_MAX_MERGES_BYTES,
    )
    assert payload is not None
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise TokenContractError(f"{role}_merges_invalid_utf8") from exc
    if lines and lines[0].startswith("#version:"):
        lines = lines[1:]
    return tuple(line for line in lines if line)


def _read_json(
    root: Path,
    filename: str,
    *,
    role: str,
    name: str,
    required: bool = True,
) -> object | None:
    payload = _read_bytes(
        root,
        filename,
        role=role,
        name=name,
        max_bytes=_MAX_JSON_BYTES,
        required=required,
    )
    if payload is None:
        return None
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TokenContractError(f"{role}_{name}_invalid_json") from exc


def _read_bytes(
    root: Path,
    filename: str,
    *,
    role: str,
    name: str,
    max_bytes: int,
    required: bool = True,
) -> bytes | None:
    path = root / filename
    try:
        status = path.stat()
    except FileNotFoundError as exc:
        if not required:
            return None
        raise TokenContractError(f"{role}_{name}_missing") from exc
    except OSError as exc:
        raise TokenContractError(f"{role}_{name}_unreadable") from exc
    if not stat.S_ISREG(status.st_mode):
        raise TokenContractError(f"{role}_{name}_not_regular_file")
    if status.st_size > max_bytes:
        raise TokenContractError(f"{role}_{name}_too_large")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TokenContractError(f"{role}_{name}_unreadable") from exc
    if len(payload) > max_bytes:
        raise TokenContractError(f"{role}_{name}_too_large")
    return payload


def _validate_token_map(payload: object, *, code_prefix: str) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise TokenContractError(f"{code_prefix}_not_object")
    result: dict[str, int] = {}
    seen_ids: dict[int, str] = {}
    for token, token_id in payload.items():
        if not isinstance(token, str):
            raise TokenContractError(f"{code_prefix}_token_invalid")
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
        ):
            raise TokenContractError(f"{code_prefix}_id_invalid")
        previous = seen_ids.get(token_id)
        if previous is not None and previous != token:
            raise TokenContractError(f"{code_prefix}_id_collision")
        result[token] = token_id
        seen_ids[token_id] = token
    return result


def _insert_token(
    mapping: dict[str, int],
    token: str,
    token_id: int,
    *,
    role: str,
) -> None:
    if token_id < 0:
        raise TokenContractError(f"{role}_added_token_id_invalid")
    existing = mapping.get(token)
    if existing is not None and existing != token_id:
        raise TokenContractError(f"{role}_added_token_conflict")
    mapping[token] = token_id


def _token_content(value: object, *, role: str, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if content is None:
            return None
        if isinstance(content, str):
            return content
    raise TokenContractError(f"{role}_{name}_invalid")


def _token_ids(
    value: object,
    *,
    role: str,
    name: str,
) -> tuple[int, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, list) else [value]
    result: list[int] = []
    for token_id in values:
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
        ):
            raise TokenContractError(f"{role}_{name}_invalid")
        result.append(token_id)
    return tuple(result)


def _unmapped_ids(
    *,
    source: TokenizerContract,
    destination: TokenizerContract,
) -> set[int]:
    return {
        token_id
        for token_id, token in source.id_to_token.items()
        if destination.id_to_token.get(token_id) != token
    }


def _declared_revisions(model_payload: dict[str, object]) -> dict[str, str]:
    keys = (
        "source_checkpoint_reference",
        "source_checkpoint_sha",
        "source_base_model_reference",
        "source_base_model_sha",
        "output_hash",
        "_commit_hash",
    )
    return {
        key: value
        for key in keys
        if isinstance((value := model_payload.get(key)), str) and value
    }


def _infer_revision(root: Path) -> str | None:
    name = root.name
    if len(name) == 40 and all(character in "0123456789abcdef" for character in name):
        return name
    return None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compact_ranges(values: set[int]) -> list[list[int]]:
    if not values:
        return []
    ordered = sorted(values)
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append([start, previous])
        start = previous = value
    ranges.append([start, previous])
    return ranges


def _invalid_report(
    *,
    prompt_owner: str,
    code: str,
    message: str,
) -> TokenContractReport:
    return TokenContractReport(
        verdict="invalid_input",
        prompt_owner=prompt_owner,
        reasons=(code,),
        draft=None,
        target=None,
        comparison={},
        prompt_contract={"owner": prompt_owner},
        proposal_filter={},
        error={"code": code, "message": message},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a dLLM drafter and AR target token contract."
    )
    parser.add_argument("--draft", required=True, help="draft tokenizer/package root")
    parser.add_argument("--target", required=True, help="target tokenizer/package root")
    parser.add_argument(
        "--prompt-owner",
        choices=("target", "draft", "pretokenized"),
        default="target",
    )
    parser.add_argument("--output-json", action="store_true")
    args = parser.parse_args(argv)

    report = diagnose_token_contract(
        args.draft,
        args.target,
        prompt_owner=args.prompt_owner,
    )
    payload = report.to_dict()
    if args.output_json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(f"verdict={report.verdict}")
        for reason in report.reasons:
            print(f"reason={reason}")

    if report.verdict == "invalid_input":
        return 2
    if report.verdict == "incompatible":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
