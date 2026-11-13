"""§10 device-backend contract for the exact K/V cache manager.

§10 gives a backend FIVE responsibilities and no others:

1. allocation and copy primitives for its device;
2. consumption of a ``KVReadPlan``;
3. production of captures named by a ``KVCapturePlan``;
4. the synchronisation required before pages become readable or releasable;
5. honest physical-byte and movement accounting.

It implements no identity, no eviction, and no publication policy. Those live in
the manager, and a backend that decided any of them would be the second owner
§17.8 forbids.

WHAT THIS MODULE DELIBERATELY DOES NOT OWN. The native manager owns world
identity, publication, leases, and eviction; the device backend owns only its
opaque buffer registry and physical release primitives. The vocabulary here is
allocate, copy, synchronise, and free because a device backend that also chose
which world was reusable would be the second policy owner §14 refuses.

WHY THE MOVEMENT COUNTERS LIVE HERE. §12 requires physical K/V bytes read,
written, concatenated and copied "when known". The manager cannot know them: it
holds opaque payload handles by design. The backend is the only layer that can
count them honestly.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import threading
from typing import Any, Mapping, Protocol, runtime_checkable

from .kv_manager import (
    CapturePermit,
    KVFragment, KVReadPlan, ReleaseOutcome, UnknownTensorSize,
)

# Re-exported: a backend author reads this module, and the spec is half the
# contract even though the manager owns and validates it.
from .kv_manager import KVTensorSpec as KVTensorSpec  # noqa: F401


# §8's refusal is one class, defined with the contract that needs it. The alias
# keeps the device-facing name usable without a second exception meaning the
# same thing.
UnknownPhysicalBytes = UnknownTensorSize


class StaleDeviceBuffer(KeyError):
    """§17.6 for device memory: a stale handle refuses, never resolves.

    Raised for a handle issued by ANOTHER backend, one this backend never
    issued, one already released, and one whose fields disagree with the record
    it names -- the last because a caller-authored `DeviceBuffer` claiming 1
    byte for a 4096-byte allocation removed the real allocation while charging
    one byte for it. Every one of those was measured.
    """


@dataclass(frozen=True, slots=True)
class DeviceBuffer:
    """A payload handle: opaque to the manager, canonical to the backend.

    THIS IS THE ONE PAYLOAD TYPE. It used to sit beside the raw device arrays
    that `capture` actually put into fragments, so the lifetime the backend
    could account for and the lifetime the manager held were two disconnected
    paths -- `free` worked only on buffers nobody had captured, and a captured
    payload raised `AttributeError` when handed to it. Fragments now carry these,
    so capture, read, evict, abort and close all name the same object.

    It exposes `shape` and `dtype` because `KVSlice` requires a payload to
    describe itself, and because the tensor spec is checked against what the
    backend recorded rather than against what a caller asserted. Every field is
    filled in by the backend from the real allocation; none is caller-supplied.
    """

    # WHICH BACKEND ISSUED IT. Handles are per-instance counters, so two
    # backends both mint handle 1 and one of them resolved the other's --
    # measured, backend B returned its own tensor for backend A's handle, which
    # is §17.6's "a stale handle resolves to a refusal, never to another object"
    # broken in the most direct way.
    owner_id: int
    handle: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    device: str = ""


@dataclass(frozen=True, slots=True)
class KVCaptureMutation:
    """One layer's captured K/V for one span, provider-neutral.

    The typed form of `WindowExecutionResult.cache_mutations`, which is `()`
    today and documented as "reserved for a future consumer". This is that type,
    so the write path can stop being a side effect on a cache object without
    every backend inventing its own tuple shape.
    """

    layer: int
    start: int
    end: int
    key: Any
    value: Any

    @property
    def rows(self) -> int:
        return self.end - self.start


@dataclass
class MovementCounters:
    """Honest physical movement, in bytes, for one backend instance.

    These are DISTINCT quantities from forward-work metrics (§12): rows
    forwarded, K/V projection skips and tokens/second are not bytes.

    `bytes_free_requested` and `bytes_freed` are SEPARATE on purpose. The first
    is what a caller asked to let go; the second is what the device actually
    gave back, measured. Reporting the first as the second is the accounting lie
    an audit caught here, and one number cannot say both.
    """

    bytes_allocated: int = 0
    bytes_read: int = 0
    bytes_concatenated: int = 0
    bytes_written: int = 0
    bytes_copied: int = 0
    bytes_free_requested: int = 0
    # Asked for, and refused because somebody else still holds the buffer. The
    # difference between this and `bytes_freed` is the whole reason both exist:
    # a free that could not free must say so in a number, not only by returning
    # a smaller one that noise could have produced.
    bytes_free_held: int = 0
    bytes_freed: int = 0
    synchronizations: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "kv_backend_bytes_allocated": self.bytes_allocated,
            "kv_backend_bytes_read": self.bytes_read,
            "kv_backend_bytes_concatenated": self.bytes_concatenated,
            "kv_backend_bytes_written": self.bytes_written,
            "kv_backend_bytes_copied": self.bytes_copied,
            "kv_backend_bytes_free_requested": self.bytes_free_requested,
            "kv_backend_bytes_free_held": self.bytes_free_held,
            "kv_backend_bytes_freed": self.bytes_freed,
            "kv_backend_synchronizations": self.synchronizations,
        }


@dataclass(frozen=True)
class ContiguousKV:
    """A contiguous K/V span within one layer.

    ``materialize`` returns one pair for the whole read window. The optional
    ``read_parts`` backend method returns an ordered tuple of these spans per
    layer without assembling them. Both require the caller to hold the read
    lease until device work completes. `rows` is explicit rather than inferred
    from a backend-specific tensor layout.
    """

    layer: int
    rows: int
    key: Any
    value: Any


@runtime_checkable
class KVDeviceBackend(Protocol):
    """§10's five responsibilities, and nothing that decides policy."""

    @property
    def device(self) -> str:
        """A stable name for the device this backend owns."""

    # 1. allocation and copy primitives
    def allocate(self, shape: tuple[int, ...], dtype: Any) -> DeviceBuffer:
        """Zeroed device memory this backend owns until `free`."""

    def view(self, buffer: DeviceBuffer) -> Any:
        """The device array behind a handle, for the caller to read or write."""

    def copy(self, value: Any) -> DeviceBuffer:
        """A new buffer this backend owns, with the same contents. NEVER an
        alias."""

    def free(self, *buffers: DeviceBuffer) -> ReleaseOutcome:
        """Release ownership; report what the device actually gave back.

        MEASURED, not computed. A buffer somebody else still holds is not freed:
        it becomes cleanup DEBT, keeps its handle live, and is retried by
        `sweep`. A handle this backend did not issue is a refusal.
        """

    def sweep(self) -> int:
        """Retry deferred releases; return the bytes that finally came back.

        Without it a buffer the caller was still using at `free` time was lost
        to accounting forever: the handle was deleted, the memory stayed
        resident, and nothing could record that it eventually went.
        """

    # 2. consumption of a read plan
    def materialize(self, plan: KVReadPlan) -> dict[int, ContiguousKV]:
        """One contiguous key/value pair per declared layer, in row order.

        Providers may additionally implement
        ``read_parts(plan) -> dict[int, tuple[ContiguousKV, ...]]`` to expose
        ordered borrowed views without concatenation. Empty layers return ().
        This optional representation keeps older installed providers usable;
        it changes neither the owner nor the lease/capture contract.
        """

    # 3. production of captures
    def capture(self, permit: CapturePermit,
                produced: Mapping[int, tuple[Any, Any]]) -> tuple[KVFragment, ...]:
        """Fragments for exactly the permit's requests, as owned payload handles.

        A PERMIT, NOT A PLAN. §7 step 2 wants the failure before the unsafe
        allocation, and while this took a plan that was a calling convention:
        capture-then-admit was spellable, and the integration tests spelled it.
        Measured on the real backend -- capture first, 2 handles and 128 bytes
        live, admission then refused, and afterwards still 2 handles and 128
        bytes. A permit can only come from a manager that already admitted the
        transaction, so the wrong order has no spelling.

        The spec comes from the permit's plan, which the manager has already
        checked against the transaction's identity. It is NOT a parameter here:
        a spec the caller chooses proves only that the tensors match what that
        caller said, which an audit demonstrated by committing float32 tensors
        under an identity keyed bfloat16 with a spec naming a quantization that
        does not exist.
        """

    # 4. synchronisation
    def synchronize(self, *values: Any) -> None:
        """Make `values` readable, and their sources releasable."""

    # 5. honest accounting
    def physical_bytes(self, value: Any) -> int:
        """Real bytes, or `UnknownPhysicalBytes`. Never a guess, never zero."""

    def movement(self) -> Mapping[str, int]:
        """The §12 physical-movement counters for this backend."""


class BackendConformanceError(TypeError):
    """A provider whose surface does not match `KVDeviceBackend`."""


def check_backend_conformance(candidate: Any) -> None:
    """Refuse a provider whose SIGNATURES do not match, not just its names.

    `runtime_checkable` compares attribute names and nothing else: an object
    whose every method takes no arguments at all satisfies
    `isinstance(x, KVDeviceBackend)`. Measured -- so the test that claimed the
    Protocol "refuses an incomplete provider" only ever proved it refuses a
    MISSING method.

    A CUDA provider is the reason this matters. It will be written against this
    contract by a different author in a different lane, and "it type-checks"
    has to mean the calls the manager makes will actually bind.
    """
    for name, expected in inspect.getmembers(KVDeviceBackend, inspect.isfunction):
        if name.startswith("_"):
            continue
        found = getattr(type(candidate), name, None)
        if found is None:
            raise BackendConformanceError(
                f"{type(candidate).__name__} implements no {name!r}"
            )
        want = inspect.signature(expected)
        got = inspect.signature(found)
        want_params = list(want.parameters.values())
        got_params = list(got.parameters.values())
        if len(want_params) != len(got_params):
            raise BackendConformanceError(
                f"{type(candidate).__name__}.{name}{got} does not match "
                f"{name}{want}: {len(got_params)} parameter(s) against "
                f"{len(want_params)}"
            )
        for wanted, actual in zip(want_params, got_params, strict=True):
            if wanted.kind != actual.kind:
                raise BackendConformanceError(
                    f"{type(candidate).__name__}.{name}: parameter "
                    f"{actual.name!r} is {actual.kind.description}, and the "
                    f"contract has {wanted.name!r} as {wanted.kind.description}"
                )
        if inspect.iscoroutinefunction(found):
            # A provider whose methods return coroutines type-checks by name and
            # by arity while every call the manager makes returns an unawaited
            # object. §11 keeps this contract serial per session; an async
            # backend is a different contract, not a compatible one.
            raise BackendConformanceError(
                f"{type(candidate).__name__}.{name} is async, and this contract "
                "is synchronous"
            )
        # AND THE RETURN TYPE. Parameters alone let a backend whose `free`
        # returns a bare `int` through -- the contract SAID `-> int` while every
        # implementation returned a `ReleaseOutcome`, which is exactly the drift
        # this catches. Compared only where the contract annotates one, so an
        # unannotated method is not silently held to `None`.
        # `Any` in the contract means UNCONSTRAINED, so an implementation that
        # narrows it is conforming rather than diverging. Only a stated type has
        # to match.
        if want.return_annotation not in (inspect.Signature.empty, Any, "Any"):
            if got.return_annotation is inspect.Signature.empty:
                # A MISSING annotation is refused, not waved through. This check
                # can only catch a WRONG claim, so a provider that makes no
                # claim would slip past exactly the drift it exists to stop --
                # and this contract is meant to hold a provider written in
                # another lane by another author.
                raise BackendConformanceError(
                    f"{type(candidate).__name__}.{name} declares no return "
                    f"type; the contract returns {want.return_annotation!r} "
                    "and conformance is checked against what is declared"
                )
            if got.return_annotation != want.return_annotation:
                raise BackendConformanceError(
                    f"{type(candidate).__name__}.{name} returns "
                    f"{got.return_annotation!r} where the contract returns "
                    f"{want.return_annotation!r}"
                )
    if not isinstance(getattr(candidate, "device", None), str):
        raise BackendConformanceError(
            f"{type(candidate).__name__}.device is not a string"
        )


class BackendUnavailable(RuntimeError):
    """A known provider that THIS BUILD cannot run.

    Distinct from "no such provider" on purpose. `open_backend("mlx")` used to
    hand back a backend on a machine with no MLX, and the failure arrived at the
    first allocation as a bare `ModuleNotFoundError` -- fail-open, from the one
    function whose job is to admit a provider.
    """


@dataclass(frozen=True, slots=True)
class _Registration:
    factory: Any
    available: Any            # () -> bool, probed at open, never at import


# GUARDED. A bare module-level dict mutated from any importing thread is the
# same unguarded-shared-state class this package fixed inside the backend.
_BACKEND_LOCK = threading.Lock()
_BACKENDS: dict[str, _Registration] = {}


def register_backend(name: str, factory: Any, *, available: Any = None) -> None:
    """Register a device backend factory under `name`.

    `available` is probed at `open_backend`, not here: importing a provider
    module must not require its device, and a build that cannot run a provider
    should still be able to name it.
    """
    with _BACKEND_LOCK:
        if name in _BACKENDS:
            raise ValueError(f"a backend is already registered as {name!r}")
        _BACKENDS[name] = _Registration(factory=factory,
                                        available=available or (lambda: True))


def open_backend(name: str) -> KVDeviceBackend:
    """Construct a registered backend, refusing one this build cannot run.

    Two refusals, deliberately different: an unknown NAME is a `KeyError`, and a
    known provider whose device is absent is `BackendUnavailable`. Collapsing
    them would tell a caller with no MLX that they had misspelled "mlx".

    Conformance is checked HERE rather than at registration because a factory is
    not a backend: the surface being validated is the object it produces.
    """
    with _BACKEND_LOCK:
        registration = _BACKENDS.get(name)
    if registration is None:
        raise KeyError(
            f"no device backend registered as {name!r}; known: "
            f"{registered_backends()}"
        )
    if not registration.available():
        raise BackendUnavailable(
            f"the {name!r} device backend is registered but this build cannot "
            "run it"
        )
    backend = registration.factory()
    check_backend_conformance(backend)
    return backend


def registered_backends() -> tuple[str, ...]:
    """Every provider this build KNOWS, runnable or not."""
    with _BACKEND_LOCK:
        return tuple(sorted(_BACKENDS))


def available_backends() -> tuple[str, ...]:
    """Every provider this build can actually run.

    Separate from `registered_backends` because "known" and "runnable" are
    different facts, and a caller choosing a device needs the second.
    """
    with _BACKEND_LOCK:
        items = list(_BACKENDS.items())
    return tuple(sorted(name for name, reg in items if reg.available()))


class StoreCollector:
    """Collects what the forward stored, so a side effect becomes a value.

    STRICTLY TEMPORARY AND OWNS NOTHING. It allocates nothing, publishes
    nothing, caches nothing, and holds no reference after `drain()`. Every call
    is delegated to the underlying cache unchanged, so behaviour is identical
    with it in place; the only thing it adds is a record of what crossed.

    It exists to migrate the write path. Today the forward mutates a cache
    object by side effect and `WindowExecutionResult.cache_mutations` is always
    `()`. This turns those calls into `KVCaptureMutation` values a manager can
    stage, WITHOUT a second store: the moment the runner returns typed
    mutations itself, this class is deleted rather than kept as a cache.
    """

    def __init__(self, underlying: Any, *, prompt_len: int = 0,
                 block_size: int = 0) -> None:
        self._underlying = underlying
        self._prompt_len = prompt_len
        self._block_size = block_size
        self._mutations: list[KVCaptureMutation] = []

    def __getattr__(self, name: str) -> Any:
        # Everything not intercepted passes through untouched: a collector that
        # hid part of the cache's surface would change behaviour it exists to
        # preserve.
        return getattr(self._underlying, name)

    # DELEGATE FIRST, RECORD SECOND. The reverse order recorded a capture the
    # product cache had refused: with the underlying store raising, `drain()`
    # still returned a mutation, and a manager staging it would commit a payload
    # nothing ever accepted. Measured.
    def store_prompt_layer(self, *, layer_index: int, key: Any, value: Any) -> None:
        self._underlying.store_prompt_layer(layer_index=layer_index, key=key,
                                            value=value)
        self._mutations.append(
            KVCaptureMutation(layer_index, 0, self._prompt_len, key, value))

    def store_block_layer(self, *, block_index: int, layer_index: int,
                          key: Any, value: Any) -> None:
        self._underlying.store_block_layer(block_index=block_index,
                                           layer_index=layer_index, key=key,
                                           value=value)
        start = self._prompt_len + block_index * self._block_size
        self._mutations.append(
            KVCaptureMutation(layer_index, start, start + self._block_size,
                              key, value))

    def drain(self) -> tuple[KVCaptureMutation, ...]:
        """Hand over what was recorded and forget it.

        Draining CLEARS, so a second drain returns nothing rather than the same
        mutations again: a collector that kept them would be a cache, and a
        caller staging them twice would commit the same rows under one identity.
        """
        drained = tuple(self._mutations)
        self._mutations.clear()
        return drained
