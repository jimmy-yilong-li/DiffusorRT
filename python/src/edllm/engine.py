"""Public Engine API — the product entry point.

The Engine facade hides execution packages, GraphIR, presets, cache config,
and kernel selection behind a simple surface:

    engine = Engine.from_package("/path/to/package")
    result = engine.generate("Explain diffusion models.", max_tokens=128)
    engine.close()

This is a thin wrapper over the existing ``RuntimeHandle`` and MLX backend.
No new backend logic. The underlying runtime handles are reusable — Engine
owns the lifecycle (close releases weights).

For chat:

    engine = Engine.from_package(pkg)
    reply = engine.chat("Hello!", max_tokens=64)

For infill / edit:

    filled = engine.infill("prefix ", " suffix", max_fill_tokens=32)
    edited = engine.edit("Fix the bug in this code", "def foo():\n    pass")
"""

from __future__ import annotations

# These are real imports because they appear in the public runtime signatures
# and must resolve for introspection and generated documentation.
from .chat import ChatSession
from .conversation import Conversation, MemoryProvider
from .trust_identity import UNSPECIFIED

import contextlib
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from .backend_ids import (
    AUTO_BACKEND,
    CUDA_NATIVE_BACKEND,
    validate_backend as _validate_backend,
    validate_deployment_profile as _validate_deployment_profile,
    validate_device_options as _validate_device_options,
)
from .model import (
    ChatRequestCapacityError,
    RuntimeHandle,
    _close_preserving_primary,
    load_runtime_for_deployment,
)
from .model_catalog import ModelCatalog
from .result import GenerateResult, InfillResult


class Engine:
    """The public product entry point for DiffusorRT.

    An Engine wraps a loaded execution package and exposes a clean API for
    generation, chat, infill, and edit. It owns the runtime lifecycle:
    ``close()`` releases resident weights and invalidates the handle.

    Create from a prepared package::

        engine = Engine.from_package("/path/to/package")

    Create from a model ID (compiles it if the cache is cold)::

        engine = Engine.from_pretrained("Dream-org/Dream-v0-Instruct-7B")

    The default backend is ``"auto"`` (MLX on Apple Silicon, NumPy elsewhere).
    """

    __slots__ = (
        "_backend",
        "_closed",
        "_data_root",
        "_deployment_profile",
        "_deployment_target",
        "_device_memory_budget_bytes",
        "_device_ordinal",
        "_handle",
        "_lock",
        "_precision",
    )

    def __init__(
        self,
        handle: RuntimeHandle,
        *,
        backend: str = "auto",
        precision: str | None = None,
        device_ordinal: int = 0,
        device_memory_budget_bytes: int = 0,
        deployment_profile: str | None = None,
        data_root: str | Path | None = None,
    ) -> None:
        self._handle = handle
        self._backend = _validate_backend(backend)
        _validate_device_options(
            self._backend,
            precision,
            device_ordinal,
            device_memory_budget_bytes,
        )
        _validate_deployment_profile(deployment_profile)
        self._precision = precision
        self._device_ordinal = device_ordinal
        self._device_memory_budget_bytes = device_memory_budget_bytes
        self._deployment_profile = deployment_profile
        from .application_state import resolve_data_root

        self._data_root = resolve_data_root(data_root)
        self._deployment_target = None
        if self._backend == CUDA_NATIVE_BACKEND:
            plan = handle._resolved_model_plan
            if (
                plan is None
                or plan.load.backend != CUDA_NATIVE_BACKEND
                or plan.deployment is None
            ):
                raise ValueError("cuda_native requires an admitted executable plan")
            if plan.load.precision != precision:
                raise ValueError("cuda_native precision differs from executable plan")
            self._deployment_target = plan.deployment
            if self._deployment_profile is None:
                contract = self._deployment_target.deployment
                if contract is None:
                    raise ValueError("cuda_native deployment profile is missing")
                self._deployment_profile = contract.profile
        self._closed = False
        # Serialises every public call, and makes the open-check and the handle
        # dereference atomic. Without it a close landing between them left
        # ``self._handle is None`` and leaked
        # ``AttributeError: 'NoneType' object has no attribute 'generate'``.
        #
        # Re-entrant because public methods compose (``edit`` delegates to the
        # same guarded path); a plain Lock would self-deadlock there.
        #
        # Blocking rather than raising "busy" is the right semantic here: the
        # engine owns ONE resident model and the device runs one forward at a
        # time, so callers are serialised by the hardware regardless. Making that
        # explicit is friendlier than an error the caller must retry around.
        self._lock = threading.RLock()

    @property
    def backend(self) -> str:
        """The backend this engine was created with.

        Every generation call routes through it. ``"auto"`` defers the choice to
        the runtime's own resolution; an explicit value is passed through
        verbatim so a caller asking for the portable reference path gets it.
        """
        return self._backend

    @property
    def deployment_profile(self) -> str | None:
        """The exact optimization/deployment contract used by this Engine."""

        return self._deployment_profile

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_package(
        cls,
        package_dir: str | Path,
        *,
        backend: str = "auto",
        deployment_profile: str | None = None,
        precision: str | None = None,
        device_ordinal: int = 0,
        device_memory_budget_bytes: int = 0,
        trust_remote_code: bool = False,
        expected_module: object = UNSPECIFIED,
        catalog: ModelCatalog | None = None,
        data_root: str | Path | None = None,
    ) -> "Engine":
        """Load a prepared package through the application ModelRepository."""

        from .local_models import ModelRepository

        return ModelRepository(catalog=catalog, data_root=data_root).open_package(
            package_dir,
            engine_cls=cls,
            backend=backend,
            deployment_profile=deployment_profile,
            precision=precision,
            device_ordinal=device_ordinal,
            device_memory_budget_bytes=device_memory_budget_bytes,
            trust_remote_code=trust_remote_code,
            expected_module=expected_module,
        )

    @classmethod
    def _from_package_impl(
        cls,
        package_dir: str | Path,
        *,
        backend: str = "auto",
        deployment_profile: str | None = None,
        precision: str | None = None,
        device_ordinal: int = 0,
        device_memory_budget_bytes: int = 0,
        trust_remote_code: bool = False,
        expected_module: object = UNSPECIFIED,
        catalog: ModelCatalog | None = None,
        data_root: str | Path | None = None,
    ) -> "Engine":
        """Load a DiffusorRT execution package.

        Parameters
        ----------
        package_dir:
            Path to a prepared execution package (``model.json`` +
            ``graph.json`` + weights + tokenizer).
        backend:
            ``"auto"`` (default — selects the best available backend for the
            platform), ``"mlx_local"`` (Apple Silicon MLX), or
            ``"numpy_local"`` (portable NumPy reference).
        deployment_profile:
            Exact admitted deployment profile to apply before graph/runtime
            construction. The resolved package profile must match; it is never
            treated as a best-effort preference. When omitted, an exact
            definition-owned Product profile is selected automatically; an
            unregistered package keeps its own configuration.
        trust_remote_code:
            Whether to allow the package's custom tokenizer code to execute.
            Default ``False`` — a package is untrusted data.
        expected_module:
            The module the caller obtained consent FOR. Forwarded to
            the unified package loader, which verifies it against the
            materialized tokenizer bundle immediately before AutoTokenizer
            runs.
        catalog:
            Explicit Local definitions used to verify a locally identified
            package. Omit it for ordinary Product or unregistered packages.
        """
        resolved_backend = _validate_backend(backend)
        _validate_device_options(
            resolved_backend,
            precision,
            device_ordinal,
            device_memory_budget_bytes,
        )
        _validate_deployment_profile(deployment_profile)

        loaded = load_runtime_for_deployment(
            package_dir,
            requested_backend=resolved_backend,
            mlx_profile_name=deployment_profile,
            precision=precision,
            device_ordinal=device_ordinal,
            device_memory_budget_bytes=device_memory_budget_bytes,
            trust_remote_code=trust_remote_code,
            expected_module=expected_module,
            model_catalog=catalog,
        )
        handle = loaded.runtime
        resolved_deployment_profile = loaded.profile
        try:
            if (
                deployment_profile is not None
                and loaded.profile != deployment_profile
            ):
                raise ValueError(
                    "deployment profile mismatch: "
                    f"expected {deployment_profile!r}, got {loaded.profile!r}"
                )
            if (
                resolved_backend != AUTO_BACKEND
                and loaded.backend != resolved_backend
            ):
                raise ValueError(
                    "deployment backend mismatch: "
                    f"expected {resolved_backend!r}, got {loaded.backend!r}"
                )
        except BaseException as primary:
            _close_preserving_primary(
                handle.close,
                primary,
                label="runtime handle cleanup",
            )
            raise
        effective_backend = (
            loaded.backend if deployment_profile is not None else resolved_backend
        )
        # Keep an implicit ``auto`` request visible to the caller even though
        # profile resolution already determined which backend the package will
        # use. An explicitly named profile is bound to its resolved backend.
        try:
            return cls(
                handle,
                backend=effective_backend,
                precision=precision,
                device_ordinal=device_ordinal,
                device_memory_budget_bytes=device_memory_budget_bytes,
                deployment_profile=resolved_deployment_profile,
                data_root=data_root,
            )
        except BaseException as primary:
            _close_preserving_primary(
                handle.close,
                primary,
                label="runtime handle cleanup",
            )
            raise

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        cache_root: str | Path | None = None,
        offline: bool = False,
        trust_remote_code: bool = False,
        backend: str = "auto",
        deployment_profile: str | None = None,
        precision: str | None = None,
        device_ordinal: int = 0,
        device_memory_budget_bytes: int = 0,
        catalog: ModelCatalog | None = None,
        data_root: str | Path | None = None,
    ) -> "Engine":
        """Compile a supported model from its Hugging Face id, then load it.

        Without ``catalog``, the compiler accepts exact Product-registry
        identities. With an explicit ``ModelCatalog``, it accepts exactly that
        Local scope instead. A ``revision`` that disagrees with the selected
        entry's pin is refused.

        Download and package size depend on the selected model. The compiler
        checks the model's recorded disk budget before acquisition; the cache
        root is relocatable via ``cache_root`` or ``DIFFUSORRT_CACHE_ROOT``. A
        second call with the same (id, revision) reuses the cached package.

        ``offline=True`` requires the fixed revision in local cache and never
        reaches the network. ``trust_remote_code`` is required only when the
        admitted model names a repository-provided tokenizer module. When
        ``deployment_profile`` is omitted, its default admitted profile is
        applied. Local catalogs build one canonical storage package and
        currently execute it through MLX/fp32; CUDA needs a registered provider
        package and is rejected before acquisition.
        """
        if type(offline) is not bool:
            raise ValueError("offline must be a boolean")
        from .local_models import ModelRepository

        return ModelRepository(
            catalog=catalog,
            cache_root=cache_root,
            data_root=data_root,
        ).open(
            model_id,
            engine_cls=cls,
            revision=revision,
            offline=offline,
            trust_remote_code=trust_remote_code,
            backend=backend,
            deployment_profile=deployment_profile,
            precision=precision,
            device_ordinal=device_ordinal,
            device_memory_budget_bytes=device_memory_budget_bytes,
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str = "",
        *,
        max_tokens: int = 128,
        seed: int | None = None,
        return_debug: bool = False,
    ) -> GenerateResult:
        """Generate text from a prompt.

        Parameters
        ----------
        prompt:
            The input prompt string.
        max_tokens:
            Maximum number of tokens to generate.
        seed:
            Reserved. Seeded sampling is **not implemented** — the underlying
            runtime exposes no seed parameter, and decoding is greedy. Passing a
            non-``None`` value raises rather than being ignored: a caller who
            believes they suppressed non-determinism and did not is worse off
            than one who gets an error.

        Returns
        -------
        GenerateResult
            With ``.text``, ``.finish_reason``, and optional ``.debug``.
        """
        with self._guarded():
            self._reject_seed(seed)
            result = self._handle.generate(
                prompt,
                max_tokens=max_tokens,
                backend=self._backend,
                return_debug=return_debug,
                **self._runtime_device_options(),
            )
            if self._deployment_target is not None and result.debug is not None:
                result.debug.metrics.update(
                    {
                        "cuda_execution_profile": self._deployment_profile,
                        "cuda_minimum_device": self._deployment_target.minimum_device,
                    }
                )
            return result

    def chat(
        self,
        message: str,
        *,
        max_tokens: int = 128,
        system_prompt: str | None = None,
        session: "ChatSession | None" = None,
    ) -> str:
        """Chat with the model, single-turn or multi-turn.

        Without ``session`` this is a single turn: the message is rendered
        through the chat template on its own and no history is kept.

        With a ``session`` from :meth:`chat_session`, the message is appended to
        that session's history, the model conditions on the whole conversation,
        and the reply is appended back — so consecutive calls are a real
        multi-turn exchange. v0 documented this path but did not provide it: the
        session it handed out had no way to reach the engine.

        ``system_prompt`` must be omitted, or match the value already bound to
        a supplied session.

        Returns the assistant's reply text.
        """
        if session is not None:
            if system_prompt is not None and system_prompt != session.system_prompt:
                raise ValueError(
                    "system_prompt differs from the ChatSession-bound value"
                )
            return session._send(self, message, max_tokens=max_tokens).text
        conversation = self._new_conversation(
            system_prompt=system_prompt,
            persistent=False,
        )
        try:
            return conversation.send(message, max_tokens=max_tokens).text
        finally:
            conversation.close()

    def conversation(
        self,
        *,
        session_id: str | None = None,
        system_prompt: str | None = None,
        context_policy: str = "recent_only",
        memory_provider: MemoryProvider | None = None,
        memory_token_budget: int = 512,
    ) -> Conversation:
        """Create or resume one durable local conversation.

        History is stored below this Engine's application-data root. No
        framework persona or long-term memory provider is installed by default.
        """

        return self._new_conversation(
            session_id=session_id,
            system_prompt=system_prompt,
            context_policy=context_policy,
            memory_provider=memory_provider,
            memory_token_budget=memory_token_budget,
            persistent=True,
        )

    def _new_conversation(
        self,
        *,
        session_id: str | None = None,
        system_prompt: str | None = None,
        context_policy: str = "recent_only",
        memory_provider: MemoryProvider | None = None,
        memory_token_budget: int = 512,
        persistent: bool,
    ) -> Conversation:
        from .conversation import _ConversationStore, new_session_id

        with self._guarded():
            identity = self._conversation_identity_locked(
                require_verified_package=persistent
            )
            store = (
                _ConversationStore.persistent(self._data_root)
                if persistent
                else _ConversationStore.ephemeral()
            )
            try:
                return Conversation(
                    _engine=self,
                    _store=store,
                    _identity=identity,
                    session_id=session_id or new_session_id(),
                    system_prompt=system_prompt,
                    context_policy=context_policy,
                    memory_provider=memory_provider,
                    memory_token_budget=memory_token_budget,
                )
            except BaseException:
                store.close()
                raise

    def chat_session(
        self,
        *,
        system_prompt: str | None = None,
    ) -> "ChatSession":
        """Start a multi-turn chat session.

        Returns a :class:`~edllm.chat.ChatSession` that accumulates history.
        Pass it back to :meth:`chat` as ``session=`` to take turns; the session
        itself is a compatibility view over the same conversation transaction.
        """
        conversation = self._new_conversation(
            system_prompt=system_prompt,
            persistent=False,
        )
        return ChatSession(_conversation=conversation)

    def infill(
        self,
        prefix: str,
        suffix: str,
        *,
        max_fill_tokens: int = 64,
        return_debug: bool = False,
    ) -> InfillResult:
        """Fill text between a prefix and a suffix (dLLM-native bidirectional).

        The forward attends to BOTH sides — the differentiation from
        autoregressive models, which could only continue the prefix.

        Parameters
        ----------
        prefix:
            Fixed text before the fill region.
        suffix:
            Fixed text after the fill region.
        max_fill_tokens:
            Maximum tokens to generate in the fill region.

        Returns
        -------
        InfillResult
            With ``.filled_text``, ``.full_text``, and ``.finish_reason``.
        """
        with self._guarded():
            # ``RuntimeHandle.infill`` is fail-closed to ``mlx_local``: it is the only
            # backend that implements bidirectional infill today. So this method's
            # backend domain is NARROWER than ``generate``'s, and an engine created
            # with an explicit non-MLX backend must be told so rather than being
            # quietly routed onto MLX anyway.
            if self._backend not in ("auto", "mlx_local", CUDA_NATIVE_BACKEND):
                raise NotImplementedError(
                    "infill requires mlx_local or cuda_native; this "
                    f"engine was created with backend={self._backend!r}"
                )
            backend = (
                CUDA_NATIVE_BACKEND
                if self._backend == CUDA_NATIVE_BACKEND
                else "mlx_local"
            )
            return self._handle.infill(
                prefix,
                suffix,
                max_fill_tokens=max_fill_tokens,
                backend=backend,
                return_debug=return_debug,
                **self._runtime_device_options(),
            )

    def edit(
        self,
        text: str,
        start: int,
        end: int,
        *,
        max_fill_tokens: int = 128,
        return_debug: bool = False,
    ) -> InfillResult:
        """Replace ``text[start:end]`` with newly generated content, respecting
        BOTH the surrounding prefix AND suffix.

        This is the dLLM-native span edit — the bidirectional rewrite an
        autoregressive model cannot do (AR conditions only on the left context).
        Built on :meth:`RuntimeHandle.edit`, which wraps ``infill(prefix, suffix)``.

        ``start``/``end`` are character offsets into ``text``;
        ``0 <= start <= end <= len(text)``; ``start == end`` is a pure insertion.

        Parameters
        ----------
        text:
            The full document to edit.
        start:
            Character offset where the rewrite begins (inclusive).
        end:
            Character offset where the rewrite ends (exclusive).
        max_fill_tokens:
            Maximum tokens to generate in the rewritten span.

        Returns
        -------
        InfillResult
            With ``.filled_text`` (the rewritten span), ``.full_text``
            (prefix + filled + suffix), and ``.finish_reason``.
        """
        with self._guarded():
            # ``RuntimeHandle.edit`` is fail-closed to mlx_local (the only backend
            # that implements bidirectional infill today). Same domain as infill().
            if self._backend not in ("auto", "mlx_local", CUDA_NATIVE_BACKEND):
                raise NotImplementedError(
                    "edit requires mlx_local or cuda_native; this "
                    f"engine was created with backend={self._backend!r}"
                )
            backend = (
                CUDA_NATIVE_BACKEND
                if self._backend == CUDA_NATIVE_BACKEND
                else "mlx_local"
            )
            return self._handle.edit(
                text,
                start,
                end,
                max_fill_tokens=max_fill_tokens,
                backend=backend,
                return_debug=return_debug,
                **self._runtime_device_options(),
            )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def capabilities(self) -> dict[str, Any]:
        """Return the engine's backend capabilities and model metadata."""
        with self._guarded():
            cfg = self._handle.model_config or {}
            pkg_dir = str(self._handle.package_dir) if self._handle.package_dir else ""
            backend_caps: dict[str, Any] = {}
            if pkg_dir:
                from .capabilities import diagnose_loaded_runtime

                # v0 read ``report.recommended_backend``, which RuntimeCapabilityReport
                # does not define, so every call raised AttributeError into a bare
                # ``except Exception`` and returned a dict with no backend keys at all.
                # The real fields are auto_backend / selected_backend / backends, and
                # only genuine probe failures are tolerated below — a typo must not be
                # indistinguishable from an unprobeable environment.
                try:
                    report = diagnose_loaded_runtime(
                        self._handle,
                        package_path=pkg_dir,
                        requested_backend=self._backend,
                        cuda_precision=self._precision,
                    )
                except (OSError, ImportError, ValueError) as exc:
                    backend_caps = {"capability_probe_error": f"{type(exc).__name__}: {exc}"}
                else:
                    # "available" must mean usable. The unfiltered picture stays
                    # reachable under known_backends, so a planned backend remains
                    # diagnostic-visible without being reported as runnable.
                    backend_caps = {
                        "available_backends": [
                            capability.backend_id
                            for capability in report.backends
                            if capability.usable
                        ],
                        "known_backends": [
                            capability.backend_id for capability in report.backends
                        ],
                        "auto_backend": report.auto_backend,
                        "selected_backend": report.selected_backend,
                        "healthy": report.healthy,
                    }
            return {
                "backend": self._handle.runtime_impl,
                "requested_backend": self._backend,
                "deployment_profile": self._deployment_profile,
                # The artifact's own redistribution terms and origin. Empty for
                # packages built before the compiler recorded them;
                # COPYRIGHT_AND_LICENSING.md's boundary is unenforceable if a
                # loaded package cannot state what it was built from.
                "source_license": self._handle.source_license,
                "source_reference": self._handle.source_checkpoint_reference,
                "model_architecture": cfg.get("architecture"),
                "layer_count": cfg.get("layer_count"),
                "hidden_size": cfg.get("hidden_size"),
                "vocab_size": cfg.get("vocab_size"),
                "mask_token_id": cfg.get("mask_token_id"),
                **backend_caps,
            }

    @property
    def is_closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Invalidate the engine and drop its reference to the runtime.

        Releases the handle's resident backend state, then drops the reference.
        Both halves matter: ``Engine.from_package`` hands the caller no handle,
        but ``Engine(handle)`` does, and a caller holding one kept every MLX
        buffer alive after close() returned. Dropping a reference is not a
        release when someone else holds another.

        This USED to be reference-dropping alone, correctly documented as
        unable to promise deallocation. RuntimeHandle.close() exists now, so the
        promise can be kept instead of disclaimed.

        The handle stays usable as read-only metadata afterwards -- package
        version and source reference -- but its first close attempt also ends
        direct handle execution permanently.

        A cleanup failure is propagated and leaves the handle attached only so
        another ``close()`` can retry it. The engine itself remains closed from
        the first attempt and refuses new work; a native owner may already be
        shut down even when device cleanup still reports debt.

        This engine IS thread-safe, by serialisation: every public call holds an
        engine-level re-entrant lock, so concurrent callers queue rather than
        interleave. That matches the hardware — one resident model, one forward at
        a time — and it is why close() takes the lock too: it WAITS for an
        in-flight generation instead of yanking the handle out from under it.

        Serialised is not parallel. Two callers do not get two forwards at once;
        they get them one after the other. For real concurrency, load one Engine
        per worker (and budget the weights per Engine).
        """
        # Not ``_guarded()``: that requires the engine to be open, and close()
        # must stay idempotent. Take the lock directly so an in-flight call
        # completes first.
        with self._lock:
            if self._closed and self._handle is None:
                return
            # Stop new public work from the first close attempt. A backend can
            # fail AFTER its owner is already shut down (cleanup debt), so
            # leaving the Engine open would admit generation against a manager
            # that only supports cleanup retries. The handle remains until its
            # checked close succeeds, and another close() retries it.
            self._closed = True
            handle = self._handle
            if handle is not None:
                # Keep the handle reachable until checked cleanup succeeds. A
                # close that first nulled it and swallowed the exception
                # destroyed the only retry path.
                handle.close()
            self._handle = None  # type: ignore[assignment]

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # The ONE place a bare `pass` is right. __del__ can run during
        # interpreter shutdown, when the logging module is already torn down,
        # so logging here can raise inside a destructor -- which Python prints
        # and ignores anyway. Explicit close() propagates release failures;
        # this only guards the non-deterministic finalizer.
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _conversation_identity_locked(self, *, require_verified_package: bool):
        """Bind durable history to the loaded package, tokenizer, and graph."""

        from .compiler.build import compile_receipt_digest
        from .conversation import _ConversationIdentity

        handle = self._handle
        if handle.package_dir is not None:
            package_digest = compile_receipt_digest(
                handle.package_dir,
                _package_guard=handle._package_lease,
            )
            if (
                package_digest is None
                and require_verified_package
                and not handle.fixture_mode
            ):
                raise ValueError(
                    "conversation requires a readable verified package receipt"
                )
        else:
            package_digest = None
        graph_json = handle.graph.to_json() if handle.graph is not None else ""
        graph_digest = hashlib.sha256(graph_json.encode("utf-8")).hexdigest()
        if package_digest is None:
            fallback = {
                "package": (
                    str(handle.package_dir.expanduser().resolve(strict=False))
                    if handle.package_dir is not None
                    else ""
                ),
                "model_config": handle.model_config or {},
                "graph_digest": graph_digest,
                "source_revision": handle.source_checkpoint_sha,
                "model_definition_digest": (
                    handle.model_definition_digest or handle.model_descriptor_digest
                ),
            }
            package_digest = hashlib.sha256(
                json.dumps(
                    fallback,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
        tokenizer = handle.tokenizer
        if tokenizer is None:
            raise ValueError("conversation requires a package tokenizer")
        tokenizer_identity = hashlib.sha256(
            (
                f"{package_digest}:{tokenizer.kind}:{tokenizer.bos_token_id}:"
                f"{tokenizer.eos_token_id}:{tokenizer.unk_token_id}:"
                f"{tokenizer.vocab_size}"
            ).encode("utf-8")
        ).hexdigest()
        return _ConversationIdentity(
            package_digest=package_digest,
            tokenizer_identity=tokenizer_identity,
            graph_digest=graph_digest,
            model_definition_digest=(
                handle.model_definition_digest or handle.model_descriptor_digest
            ),
            source_revision=handle.source_checkpoint_sha,
        )

    def _admit_conversation_locked(self, messages, *, max_tokens: int):
        return self._handle.admit_chat_messages(messages, max_tokens=max_tokens)

    def _admit_conversation_if_fits_locked(self, messages, *, max_tokens: int):
        """Return exact admission, or ``None`` only for sequence capacity."""

        try:
            return self._admit_conversation_locked(messages, max_tokens=max_tokens)
        except ChatRequestCapacityError:
            return None

    def _generate_conversation_locked(self, messages, *, max_tokens: int):
        return self._handle.generate(
            chat_messages=messages,
            max_tokens=max_tokens,
            backend=self._backend,
            **self._runtime_device_options(),
        )

    def _count_conversation_text_locked(self, text: str) -> int:
        tokenizer = self._handle.tokenizer
        if tokenizer is None:
            raise ValueError("conversation requires a package tokenizer")
        return len(tokenizer.encode_raw(text))

    def _conversation_max_sequence_locked(self) -> int:
        plan = self._handle._resolved_model_plan
        if plan is None:
            raise ValueError("conversation executable plan is not resolved")
        maximum = plan.model_config.get("max_position_embeddings")
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("conversation requires a positive max_position_embeddings")
        return maximum

    @contextlib.contextmanager
    def _guarded(self):
        """Hold the engine lock for a whole call, with the open-check inside it.

        The check and the ``self._handle`` dereference must be atomic: a close
        landing between them leaves the handle ``None`` and leaks an
        ``AttributeError`` where the contract promises ``RuntimeError``.
        """
        with self._lock:
            self._require_open()
            yield

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Engine is closed; create a new one with from_package()")

    def _runtime_device_options(self) -> dict[str, object]:
        if self._backend != CUDA_NATIVE_BACKEND:
            return {}
        return {
            "precision": self._precision,
            "device_ordinal": self._device_ordinal,
            "device_memory_budget_bytes": self._device_memory_budget_bytes,
        }

    @staticmethod
    def _reject_seed(seed: int | None) -> None:
        if seed is not None:
            raise NotImplementedError(
                "seed is not implemented: the runtime exposes no seed parameter and "
                "decoding is greedy. Passing a seed silently would hand back "
                "non-determinism you believe you had suppressed."
            )
