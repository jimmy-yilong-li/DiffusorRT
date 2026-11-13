from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from tokenizers import Tokenizer

from .package_artifacts import open_package_artifact, read_open_artifact
from .trust_identity import UNSPECIFIED, RemoteCodeIdentityError

# NOTE: ``transformers`` is imported LAZILY inside ``from_package`` (it pulls in
# torch via ``transformers.utils.chat_template_utils``). Keeping it off the
# module top means ``import edllm`` / ``import edllm.converter`` stay torch-free
# on the default path; the real HF tokenizer load only happens when a non-fixture
# package is opened.

_MAX_TOKENIZER_ARTIFACT_BYTES = 128 * 1024 * 1024
_MAX_TOKENIZER_BUNDLE_BYTES = 256 * 1024 * 1024
_BASE_TOKENIZER_ARTIFACTS = (
    ("tokenizer_config.json", "tokenizer_config"),
    ("tokenizer.json", "tokenizer_json"),
    ("special_tokens_map.json", "special_tokens"),
    ("added_tokens.json", "added_tokens"),
    ("vocab.json", "vocab"),
    ("merges.txt", "merges"),
    ("tokenizer.model", "tokenizer_model"),
    ("sentencepiece.bpe.model", "sentencepiece_model"),
    ("chat_template.jinja", "chat_template"),
    ("config.json", "config"),
)


def _load_package_chat_template(backend: object, package_dir: Path) -> None:
    """Bridge transformers < 4.49, which does not auto-load a standalone
    ``chat_template.jinja``. When the loaded tokenizer has no chat template but the
    package ships one as a .jinja file, load it so ``apply_chat_template`` works.
    An existing (e.g. tokenizer_config.json inline) template is left untouched.
    """
    if getattr(backend, "chat_template", None):
        return
    jinja_path = package_dir / "chat_template.jinja"
    if jinja_path.exists():
        backend.chat_template = jinja_path.read_text(encoding="utf-8")


class MissingControlTokenError(RuntimeError):
    """A control token the runtime needs is absent from the tokenizer.

    Distinct from an unknown WORD: substituting UNK for `<|mask|>` gives the
    diffusion loop a token meaning "unknown word" to mask with, and it then
    runs to completion and returns plausible nonsense.
    """


@dataclass(slots=True)
class TokenizerSpec:
    kind: str
    bos_token_id: int
    eos_token_id: int
    unk_token_id: int
    vocab_size: int
    backend: object

    @classmethod
    def from_package(
        cls,
        package_dir: str | Path,
        *,
        fixture_mode: bool,
        trust_remote_code: bool = False,
        expected_module: object = UNSPECIFIED,
    ) -> "TokenizerSpec":
        """Load a tokenizer from a package.

        ``trust_remote_code`` defaults to **False**: a package is untrusted data,
        and custom tokenizer code (``auto_map`` ``.py`` files) must not execute
        without an explicit caller opt-in (review Critical C2). Real packages
        that require a custom tokenizer must be loaded through a caller that
        surfaces the trust decision (e.g. ``deploy``/``doctor``) and passes
        ``trust_remote_code=True``.

        ``expected_module`` is the module the caller obtained consent FOR. When
        supplied it is verified against the MATERIALIZED bundle, immediately
        before ``AutoTokenizer`` is invoked -- the last instant at which a
        check can still prevent an import. A CLI-level re-check cannot do this:
        it inspects the package directory at an earlier moment, and two callers
        checked *after* the load entirely, by which point the swapped module
        had already run.

        ``fixture_mode`` is package-manifest routing truth supplied by the
        caller's already-validated snapshot.  This loader deliberately never
        reopens ``model.json``: doing so could choose a tokenizer from different
        bytes than the runtime and descriptor validation used.
        """
        if type(fixture_mode) is not bool:
            raise ValueError("tokenizer_fixture_mode_invalid")
        package_dir = Path(package_dir)
        if fixture_mode:
            special_tokens = json.loads(
                _read_tokenizer_artifact(
                    package_dir,
                    "special_tokens_map.json",
                    "special_tokens",
                    required=True,
                ).decode("utf-8")
            )
            tokenizer_payload = _read_tokenizer_artifact(
                package_dir,
                "tokenizer.json",
                "tokenizer_json",
                required=True,
            )
            backend = Tokenizer.from_str(tokenizer_payload.decode("utf-8"))
            def _special(kind: str, default: str) -> int:
                """Refuse rather than int(None).

                ``Tokenizer.token_to_id`` returns None for a token the vocab
                does not contain, and ``int(None)`` raised TypeError from deep
                inside the loader -- a package whose special_tokens_map names a
                token its tokenizer.json lacks failed with a traceback about
                NoneType instead of the actual problem.
                """
                name = str(special_tokens.get(kind, default))
                resolved = backend.token_to_id(name)
                if resolved is None:
                    raise MissingControlTokenError(
                        f"special_tokens_map.json names {kind}={name!r}, but "
                        "the package's tokenizer.json has no id for it."
                    )
                return int(resolved)

            return cls(
                kind="tokenizers_backend",
                bos_token_id=_special("bos_token", "<s>"),
                eos_token_id=_special("eos_token", "</s>"),
                unk_token_id=_special("unk_token", "<unk>"),
                vocab_size=backend.get_vocab_size(),
                backend=backend,
            )
        # Materialize before entering the optional-backend error boundary so
        # package safety violations retain their precise fail-closed error.
        with _materialized_tokenizer_package(package_dir) as tokenizer_dir:
            if expected_module is not UNSPECIFIED:
                declared = package_declares_custom_tokenizer_code(tokenizer_dir)
                if declared != expected_module:
                    raise RemoteCodeIdentityError(
                        f"consent was given for {expected_module!r} but the "
                        f"tokenizer bundle about to be imported declares "
                        f"{declared!r}; refusing to load."
                    )
            try:
                # Lazy import: pulls torch via transformers; keep it off the
                # default import path. Transformers sees only files copied
                # through the package no-follow opener.
                from transformers import AutoTokenizer

                backend = AutoTokenizer.from_pretrained(
                    str(tokenizer_dir),
                    trust_remote_code=trust_remote_code,
                    local_files_only=True,
                )
                _load_package_chat_template(backend, tokenizer_dir)
                bos_token_id = _coerce_special_id(
                    backend.bos_token_id,
                    fallback_id=backend.eos_token_id,
                )
                eos_token_id = _coerce_special_id(
                    backend.eos_token_id,
                    fallback_id=backend.bos_token_id,
                )
                unk_token_id = _coerce_special_id(
                    backend.unk_token_id,
                    fallback_id=eos_token_id,
                )
                return cls(
                    kind="transformers_auto_tokenizer",
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    unk_token_id=unk_token_id,
                    vocab_size=int(len(backend)),
                    backend=backend,
                )
            except (
                AttributeError,
                OSError,
                ValueError,
                RuntimeError,
                ImportError,
                TypeError,
            ) as exc:
                raise RuntimeError(
                    "real_tokenizer_load_error: AutoTokenizer failed for non-fixture package"
                ) from exc

    def encode(self, text: str) -> list[int]:
        if not text.strip():
            return [self.bos_token_id]
        if self.kind == "transformers_auto_tokenizer":
            return list(self.backend.encode(text, add_special_tokens=True))
        # add_special_tokens=False, then ONE explicit BOS. The default applies
        # the tokenizer's post-processor, which for many vocabularies already
        # inserts BOS -- and the prepend below then added a second one. Making
        # the boundary explicit here is the only way both backends agree on
        # what the model sees.
        encoded = self.backend.encode(text, add_special_tokens=False)
        return [self.bos_token_id, *encoded.ids]

    def encode_raw(self, text: str) -> list[int]:
        """Encode document text without inserting sequence boundary tokens.

        Infill prefix/suffix spans are fixed text inside one model sequence.  An
        empty span must therefore stay empty, and whitespace must reach the
        tokenizer unchanged; replacing either with BOS silently changes the
        context the model sees.
        """
        if self.kind == "transformers_auto_tokenizer":
            return list(self.backend.encode(text, add_special_tokens=False))
        encoded = self.backend.encode(text, add_special_tokens=False)
        return list(encoded.ids)

    def decode(self, tokens: list[int], *, strip: bool = True) -> str:
        """Decode output tokens; infill keeps whitespace at fragment boundaries."""
        visible = [
            token_id
            for token_id in tokens
            if token_id not in {self.bos_token_id, self.eos_token_id}
        ]
        if not visible:
            return ""
        decoded = self.backend.decode(visible)
        return decoded.strip() if strip else decoded

    def control_token_vocab(self, token: str) -> dict:
        """A one-entry exact vocabulary view, for edllm.control_tokens.

        The shared contract wants a token -> id MAP so it can tell "absent"
        from "present but not an id". A tokenizer answers one token at a time,
        so this adapts the shape without adding a second policy.
        """
        if self.kind == "transformers_auto_tokenizer":
            # `convert_tokens_to_ids` substitutes UNK for an absent token. The
            # shared contract requires membership, so inspect the exact map.
            vocab = self.backend.get_vocab()
            if token not in vocab:
                return {}
            resolved = vocab[token]
        else:
            resolved = self.backend.token_to_id(token)
            if resolved is None:
                return {}
        return {token: resolved}

    def require_control_token(self, token: str) -> int:
        """The id of a control token, or refuse.

        Delegates to edllm.control_tokens so the runtime, both converters and
        the conformance check share ONE policy rather than four.
        """
        from .control_tokens import MaskTokenError, resolve_from_vocab

        try:
            return resolve_from_vocab(
                None, self.control_token_vocab(token), source="tokenizer"
            )
        except MaskTokenError as error:
            raise MissingControlTokenError(str(error)) from error

    def token_to_id(self, token: str, *, required: bool = False) -> int:
        """Look a token up. ``required=True`` refuses to substitute UNK.

        The UNK fallback is right for ordinary text, and wrong for a control
        token: a missing ``<|mask|>`` resolved to the UNK id and the diffusion
        loop then masked with a token that means "unknown word", producing
        plausible nonsense instead of a refusal. Prefer
        :meth:`require_control_token` at call sites.
        """
        if self.kind == "transformers_auto_tokenizer":
            token_id = self.backend.convert_tokens_to_ids(token)
        else:
            token_id = self.backend.token_to_id(token)
        if token_id is None or (required and token_id == self.unk_token_id):
            if required:
                raise MissingControlTokenError(
                    f"the tokenizer has no id for the control token {token!r}. "
                    "Substituting UNK would mask with a token that means "
                    "'unknown word' and produce plausible nonsense; refusing."
                )
            return self.unk_token_id
        return int(token_id)


def _coerce_special_id(token_id: int | None, *, fallback_id: int | None) -> int:
    if token_id is not None:
        return int(token_id)
    if fallback_id is not None:
        return int(fallback_id)
    return 0


def package_declares_custom_tokenizer_code(package_dir: str | Path) -> str | None:
    """The package-local ``.py`` module ``AutoTokenizer`` would import, or None.

    Read-only and cheap: parses the package's ``tokenizer_config.json`` through
    the package opener and returns the ``auto_map.AutoTokenizer`` module name.
    Nothing is imported or executed, so a caller can obtain consent first.

    This asks the RIGHT question. "Does the package use the Transformers
    tokenizer?" is a different one, and answering that instead refuses every
    real package: on this machine, Dream declares
    ``auto_map = {"AutoTokenizer": ["tokenization_dream.DreamTokenizer", None]}``
    and needs consent, while Qwen3-MDLM declares ``auto_map = null`` and runs a
    stock PreTrainedTokenizerFast that executes nothing. Both are
    ``fixture_mode: false``, so a fixture_mode gate would have blocked Qwen3 and
    LLaDA for code they do not ship.

    Uses the loader's own parser, so a caller gating on trust cannot disagree
    with the loader about which module would be imported.

    Checks BOTH sources transformers checks -- tokenizer_config.json first, then
    config.json's auto_map, which it falls back to when the tokenizer config
    resolves no class. Reading only the first left a hole the loader does not
    have: a package declaring its module solely in config.json would load and
    execute it while this reported nothing to consent to.

    Raises ValueError for a config that is unreadable, not an object, or names
    an unsupported/unsafe module reference. That is deliberate: the caller must
    fail closed rather than treat "cannot tell" as "nothing to consent to".
    """
    package_dir = Path(package_dir)
    module = _custom_tokenizer_module(
        _read_tokenizer_artifact(
            package_dir, "tokenizer_config.json", "tokenizer_config", required=False
        )
    )
    if module is not None:
        return module

    config = _config_mapping(
        _read_tokenizer_artifact(
            package_dir, "config.json", "config", required=False
        ),
        error="tokenizer_config_json_invalid",
    )
    if config is None:
        return None
    return _auto_map_module(config.get("auto_map"))


def _read_tokenizer_artifact(
    package_dir: Path,
    artifact_ref: str,
    artifact_name: str,
    *,
    required: bool,
) -> bytes | None:
    try:
        with open_package_artifact(
            package_path=package_dir,
            artifact_ref=artifact_ref,
            error_prefix="tokenizer",
            artifact_name=artifact_name,
            max_bytes=_MAX_TOKENIZER_ARTIFACT_BYTES,
        ) as (handle, _path, _stat):
            return read_open_artifact(
                handle,
                max_bytes=_MAX_TOKENIZER_ARTIFACT_BYTES,
                error_prefix="tokenizer",
                artifact_name=artifact_name,
            )
    except ValueError as exc:
        missing = f"tokenizer_{artifact_name}_artifact_missing"
        if not required and str(exc) == missing:
            return None
        raise


def _auto_map_module(auto_map: object) -> str | None:
    """Resolve an ``auto_map`` entry to the ``.py`` AutoTokenizer would import.

    Mirrors transformers 4.46.3 ``tokenization_auto.py`` lines 861-900, because a
    trust gate that disagrees with the loader about which module runs is not a
    gate. Two accepted shapes:

        {"AutoTokenizer": [slow, fast]}   the current contract
        [slow, fast]                      the legacy dynamic-tokenizer form

    The slot choice is the LOADER'S: ``from_pretrained`` defaults to
    ``use_fast=True`` and takes slot 1 when it is not None, else slot 0. Taking
    slot 0 unconditionally reported "no custom code" for ``[None, fast]`` and
    named the wrong module for ``[slow, fast]``.

    Every other shape raises rather than returning None. A config we cannot
    interpret is not a config that declares nothing.
    """
    if auto_map is None:
        return None
    if isinstance(auto_map, (list, tuple)):
        entry: object = auto_map
    elif isinstance(auto_map, dict):
        entry = auto_map.get("AutoTokenizer")
    else:
        raise ValueError("tokenizer_auto_map_invalid")

    if entry is None:
        return None
    if isinstance(entry, str):
        target: object = entry
    elif isinstance(entry, (list, tuple)):
        if len(entry) != 2:
            raise ValueError("tokenizer_auto_map_invalid")
        slow, fast = entry
        target = fast if fast is not None else slow
    else:
        raise ValueError("tokenizer_auto_map_invalid")

    if not isinstance(target, str) or not target:
        raise ValueError("tokenizer_auto_map_invalid")
    module_name = target.split(".", 1)[0]
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module_name) is None:
        raise ValueError("tokenizer_custom_module_ref_unsafe")
    return f"{module_name}.py"


def _config_mapping(payload: bytes | None, *, error: str) -> dict | None:
    if payload is None:
        return None
    try:
        config = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if not isinstance(config, dict):
        # `null`, a bare list, a number. This used to reach .get() and raise
        # AttributeError straight out of the caller's ValueError boundary.
        raise ValueError(error)
    return config


def _custom_tokenizer_module(config_payload: bytes | None) -> str | None:
    config = _config_mapping(
        config_payload, error="tokenizer_tokenizer_config_invalid"
    )
    if config is None:
        return None
    return _auto_map_module(config.get("auto_map"))


@contextmanager
def _materialized_tokenizer_package(package_dir: Path) -> Iterator[Path]:
    config_payload = _read_tokenizer_artifact(
        package_dir,
        "tokenizer_config.json",
        "tokenizer_config",
        required=False,
    )
    artifacts = list(_BASE_TOKENIZER_ARTIFACTS)
    custom_module = _custom_tokenizer_module(config_payload)
    if custom_module is not None:
        artifacts.append((custom_module, "custom_module"))

    with TemporaryDirectory(prefix="edllm-tokenizer-") as temp_dir:
        target_dir = Path(temp_dir)
        total_bytes = 0
        for artifact_ref, artifact_name in artifacts:
            payload = (
                config_payload
                if artifact_ref == "tokenizer_config.json"
                else _read_tokenizer_artifact(
                    package_dir,
                    artifact_ref,
                    artifact_name,
                    required=False,
                )
            )
            if payload is None:
                continue
            total_bytes += len(payload)
            if total_bytes > _MAX_TOKENIZER_BUNDLE_BYTES:
                raise ValueError("tokenizer_bundle_too_large")
            (target_dir / artifact_ref).write_bytes(payload)
        yield target_dir
