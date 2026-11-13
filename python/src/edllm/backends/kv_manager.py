"""The K/V ownership contract, at the semantics §17 freezes.

`docs/product/kv-cache-manager-v1.md` §17 is normative and was written before
either implementation, so neither becomes the other's definition of correct.
This module is the vocabulary half: it stores no tensor and owns no memory.

WHAT CHANGED FROM THE REJECTED FIRST ATTEMPT, because the shapes below only make
sense against what they replace:

  * a fragment carries its LAYER (§17.1). Without it, two layers captured over
    the same rows collapsed into one -- measured -- and 28 layers could not be
    expressed at all.
  * a read plan is per-layer and must tile EVERY declared layer. A plan covering
    27 of 28 layers is refused, not accepted as a smaller plan.
  * `CommittedWorld` exists and nothing but a successful commit constructs one
    (§17.4). The interface that took transaction, identity and fragment as
    independent arguments is gone; it allowed publishing against a transaction
    that never existed.
  * `Lease` exists (§17.5), because a reuse that pins a page with no release
    path turns ordinary memory pressure into permanent admission failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Protocol, runtime_checkable

from .kv_slices import KVSlice

__all__ = [
    "TRACE_SCHEMA_VERSION", "KVPageState", "KVIdentity", "KVFragment",
    "KVReadPlan", "KVCaptureRequest", "KVCapturePlan", "CommittedWorld",
    "Lease", "KVOperation", "KVCacheManager", "IncompleteIdentityError",
    "IDENTITY_FIELDS",
    "SESSION_FIXED_FIELDS",
    "ReleaseOutcome",
    "PayloadReleaser",
]

# §17.7. Bumped whenever a trace's meaning changes; an unknown version is
# refused rather than best-effort parsed, because a differential oracle that
# guesses at an older shape compares two things it cannot name.
TRACE_SCHEMA_VERSION = 3


class IncompleteIdentityError(ValueError):
    """A bound identity field was absent, unknown, or not comparable.

    §6: "An incomplete identity is not a weak hit. It is a miss or a refused
    store." Raising rather than returning a partial identity is what stops a
    caller assembling one and then using it as a key.
    """


class KVPageState(Enum):
    """The five states of §5. Every page is in exactly one."""

    REQUEST_PROMPT = "request_prompt"
    ACTIVE_WINDOW = "active_window"
    TRANSACTION_STAGING = "transaction_staging"
    COMMITTED_PROMPT = "committed_prompt"
    COMMITTED_FROZEN_BLOCK = "committed_frozen_block"

    @property
    def mutable(self) -> bool:
        return self in {KVPageState.REQUEST_PROMPT, KVPageState.ACTIVE_WINDOW,
                        KVPageState.TRANSACTION_STAGING}

    @property
    def shareable(self) -> bool:
        return self in {KVPageState.COMMITTED_PROMPT,
                        KVPageState.COMMITTED_FROZEN_BLOCK}

    @property
    def evictable_while_live(self) -> bool:
        """§4.3: mutable K/V never enters a shared index, so it is never a
        candidate for eviction."""
        return self.shareable


# §6's bound fields, in the specification's order. A tuple, so the completeness
# check is driven by the specification rather than by whatever the dataclass
# happens to declare.
# §4.2: a session is one decode-run lifetime against ONE model for ONE tenant.
# These are the fields a transaction inside it may not replace. The rest --
# token ids, span, freeze state, epoch, cache salt, policy version -- vary per
# request by design, and keying different worlds is exactly what they are for.
#
# The session identity used to be stored by both implementations and read by
# neither: accepted-and-ignored, the same class as `Fragment.state` was.
# Measured: a session opened `privacy_scope="private-A"` accepted and committed
# a transaction whose identity said `private-B`, which is the one field that
# exists to keep one tenant's cache away from another's.
SESSION_FIXED_FIELDS: tuple[str, ...] = (
    "package_digest", "weights_digest", "tokenizer_identity", "graph_digest",
    "attention_mask_kind", "model_geometry", "dtype", "quantization_identity",
    "privacy_scope",
)
# NOT `layer_set`, deliberately. It names the layers THIS world covers, which is
# what makes a layer-0 world different from a layer-1 world -- the whole reason
# it is bound into the identity. One session legitimately captures different
# layer subsets in different transactions, and fixing it at the session would
# refuse the ordinary case. The MODEL's layer count belongs to
# `model_geometry`, which is fixed.

IDENTITY_FIELDS: tuple[str, ...] = (
    "package_digest", "weights_digest", "tokenizer_identity", "graph_digest",
    "attention_mask_kind", "model_geometry", "layer_set", "dtype",
    "quantization_identity",
    "token_ids", "position_start", "position_end", "freeze_state",
    "state_epoch", "policy_version", "cache_salt", "privacy_scope",
)


def _is_unset(value: Any) -> bool:
    """Absent, empty, or explicitly unknown.

    An empty string counts as absent deliberately: a digest nobody filled in and
    a digest that is "" carry the same evidence, and only one looks like a value.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() == "unknown"
    if isinstance(value, (tuple, list, dict)):
        return len(value) == 0
    return False


@dataclass(frozen=True, slots=True)
class KVIdentity:
    """Everything that can change K/V bytes or how they are read (§6)."""

    package_digest: str
    weights_digest: str
    tokenizer_identity: str
    graph_digest: str
    attention_mask_kind: str
    model_geometry: tuple[tuple[str, int], ...]
    # §6 lists the LAYER SET among the fields an identity binds, and neither
    # implementation carried one. Measured: committing layer 0 and then layer 1
    # under one identity left `shared_worlds == 1` -- by the key they WERE the
    # same world -- and reading layer 0 back failed with "no owner". The layer
    # check kept wrong K/V from being served, so it was a reachability bug
    # rather than a correctness one, but a legitimately cached layer became
    # permanently unreachable.
    layer_set: tuple[int, ...]
    dtype: str
    quantization_identity: str
    token_ids: tuple[int, ...]
    position_start: int
    position_end: int
    freeze_state: str
    state_epoch: int
    policy_version: str
    cache_salt: str
    privacy_scope: str

    def __post_init__(self) -> None:
        missing = [name for name in IDENTITY_FIELDS
                   if _is_unset(getattr(self, name, None))]
        if missing:
            raise IncompleteIdentityError(
                "identity is missing " + ", ".join(sorted(missing))
                + "; §6 makes an incomplete identity a miss or a refused store, "
                "never a weak hit"
            )
        # THE GEOMETRY NAMES ITS K/V SHAPE. `model_geometry` is a free-form
        # pair list, so `agrees_with` could only check the fields it happened to
        # contain -- "presence, not validity" -- and a session whose geometry
        # said `layers=2` accepted a request-local spec of 1x8 against a world
        # of 2x4 with the same dtype and quantization. Two different tensors,
        # one key, and no check anywhere that could have said so. Requiring the
        # two fields makes the comparison unconditional.
        geometry_names = tuple(name for name, _value in self.model_geometry)
        if len(set(geometry_names)) != len(geometry_names):
            raise IncompleteIdentityError(
                "model_geometry names one field more than once; one identity "
                "field must have one value"
            )
        geometry = dict(self.model_geometry)
        for name in ("kv_heads", "head_dim"):
            value = geometry.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise IncompleteIdentityError(
                    f"model_geometry must name a positive integer {name}; §6 "
                    "keys K/V bytes, and their shape is part of what they are"
                )
        if self.position_start < 0 or self.position_end <= self.position_start:
            raise IncompleteIdentityError(
                f"position span [{self.position_start}, {self.position_end}) is "
                "not a forward span"
            )
        if self.state_epoch < 0:
            raise IncompleteIdentityError(
                f"state_epoch is {self.state_epoch}; an epoch counts forward"
            )
        if tuple(sorted(set(self.layer_set))) != tuple(self.layer_set):
            # CANONICAL, like the token ids: `(1, 0)` and `(0, 1)` name the same
            # layers and would key two different worlds, and `(0, 0)` names one
            # layer twice.
            raise IncompleteIdentityError(
                f"layer_set {self.layer_set} is not strictly ascending; one "
                "layer set has one spelling or it keys two worlds"
            )
        if any(layer < 0 for layer in self.layer_set):
            raise IncompleteIdentityError(
                f"layer_set {self.layer_set} has a negative layer"
            )
        if len(self.token_ids) != self.position_end - self.position_start:
            raise IncompleteIdentityError(
                f"{len(self.token_ids)} token ids for a span of "
                f"{self.position_end - self.position_start}; the tokens and the "
                "span must describe the same rows"
            )

    @property
    def span(self) -> tuple[int, int]:
        return (self.position_start, self.position_end)

    @property
    def short(self) -> str:
        """A stable, comparable label for a trace entry."""
        return f"{self.graph_digest[:8]}:{self.cache_salt}:{self.position_start}-{self.position_end}"


@dataclass(frozen=True, slots=True)
class KVFragment:
    """One contiguous run of rows OF ONE LAYER, and who owns it (§17.1).

    `layer` is the field the rejected model lacked. `slice_` carries the payload
    references through `kv_slices.KVSlice`, which §14 names as the immutable
    fragment contract.
    """

    layer: int
    slice_: KVSlice
    state: KVPageState
    identity: KVIdentity | None = None

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.state.shareable and self.identity is None:
            raise ValueError(
                f"a {self.state.value} fragment is shareable and must carry the "
                "exact identity it is shared by"
            )
        if not self.state.shareable and self.identity is not None:
            raise ValueError(
                f"a {self.state.value} fragment is request-local; giving it a "
                "reuse identity would offer a mutable page for sharing"
            )

    @property
    def rows(self) -> int:
        return self.slice_.end - self.slice_.start


@dataclass(frozen=True, slots=True)
class KVReadPlan:
    """The fragments one forward consumes, PER LAYER (§17.1, §4.8).

    Well formed only when EVERY declared layer tiles `[0, total_rows)` with no
    gap or overlap. A plan covering 27 of 28 layers is refused rather than
    accepted as a smaller plan: the missing layer's rows would have no owner and
    the forward would read whatever happened to be there.

    The tiling starts at row 0 because attention reads the whole consumed
    history; a window-relative plan would satisfy the same check while leaving
    the prefix unaccounted for.
    """

    layers: tuple[int, ...]
    by_layer: tuple[tuple[int, tuple[KVFragment, ...]], ...]
    total_rows: int

    def __post_init__(self) -> None:
        if self.total_rows < 0:
            raise ValueError(f"total_rows is {self.total_rows}; rows are counted")
        if not self.layers and self.total_rows > 0:
            # An empty layer set trivially satisfies a per-layer tiling check
            # while leaving every row unowned -- exactly what §17.1 refuses. The
            # C++ manager rejected this and the oracle accepted it, so the two
            # disagreed about which plans exist at all.
            raise ValueError(
                f"a read plan over {self.total_rows} rows declares no layers; "
                "every row would be unowned"
            )
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("the declared layer set contains a duplicate")
        present = {layer for layer, _ in self.by_layer}
        declared = set(self.layers)
        if present != declared:
            raise ValueError(
                "read plan does not cover its declared layers; missing "
                f"{sorted(declared - present)}, unexpected {sorted(present - declared)}"
            )
        for layer, fragments in self.by_layer:
            cursor = 0
            for fragment in fragments:
                if fragment.layer != layer:
                    raise ValueError(
                        f"fragment for layer {fragment.layer} filed under layer {layer}"
                    )
                if fragment.slice_.start != cursor:
                    raise ValueError(
                        f"layer {layer} must tile with no gap or overlap; "
                        f"expected row {cursor}, got {fragment.slice_.start}"
                    )
                cursor = fragment.slice_.end
            if cursor != self.total_rows:
                raise ValueError(
                    f"layer {layer} covers {cursor} rows, not {self.total_rows}"
                )

    def fragments(self, layer: int) -> tuple[KVFragment, ...]:
        for candidate, fragments in self.by_layer:
            if candidate == layer:
                return fragments
        raise KeyError(layer)

    @property
    def reused_rows(self) -> int:
        """Summed ACROSS LAYERS: 28 layers of 4 reused rows is 112 rows of work
        avoided, not 4. The rejected model could not express the difference."""
        return sum(f.rows for _, frags in self.by_layer for f in frags
                   if f.state.shareable)

    @property
    def request_local_rows(self) -> int:
        return sum(f.rows for _, frags in self.by_layer for f in frags
                   if not f.state.shareable)


@dataclass(frozen=True, slots=True)
class KVCaptureRequest:
    layer: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"[{self.start}, {self.end}) is not a forward span")


# The production manager counts bytes in int64, so this contract does too --
# even where the language would not force it.
_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class ReleaseOutcome:
    """What a releaser actually managed to do, as a value rather than a promise.

    §9: "A post-commit cleanup failure records cleanup debt; it does not roll
    back an already durable logical commit or expose a partial one." A bare
    callback cannot express that -- it either returns nothing or throws, and
    both were measured going wrong: a releaser that freed nothing still counted
    a release, and one that raised escaped out of `abort` leaving the
    transaction gone, the payloads unreleased and the failure unrecorded.

    `failed` is separate from `deferred_bytes` deliberately. Deferred means the
    backend understood and could not free yet; failed means the call did not
    complete. Both become debt, and only one of them is a fault.
    """

    freed_bytes: int = 0
    deferred_bytes: int = 0
    # BYTES THAT LEFT THE BACKEND WITHOUT THE DEVICE CONFIRMING THEIR RETURN.
    # A third bucket because two could not express what a real backend measures:
    # an MLX graph node holding a buffer is not a Python reference, so the
    # backend gives up its handle -- the caller asked it to -- while the
    # allocator still reports the memory resident. Calling that "freed" claims a
    # return nobody observed; calling it "deferred" tells the manager to retry a
    # handle that no longer exists, and the retry is refused forever. It is
    # neither: ownership is gone, confirmation is pending, and the BACKEND
    # reconciles it in `sweep` against the allocator.
    retired_bytes: int = 0
    failed: bool = False
    detail: str = ""

    def __post_init__(self) -> None:
        # A NUMBER THAT CANNOT BE TRUE IS NOT ACCOUNTING. Measured before this:
        # a releaser reporting `freed_bytes=-7, deferred_bytes=-11` put both
        # straight into the manager's metrics.
        if (self.freed_bytes < 0 or self.deferred_bytes < 0
                or self.retired_bytes < 0):
            raise ValueError(
                f"a release cannot free {self.freed_bytes}, defer "
                f"{self.deferred_bytes} and retire {self.retired_bytes} bytes; "
                "all three are counts"
            )

    @property
    def accounted_bytes(self) -> int:
        return self.freed_bytes + self.deferred_bytes + self.retired_bytes

    def conserves(self, requested: int) -> bool:
        """Whether this outcome accounts for EVERY byte it was handed, once.

        A ceiling was not enough. `accounted_bytes <= requested` let a backend
        under-report and the remainder left the ledger AND the retry queue at
        the same time: measured at `requested=256`, a releaser answering
        `freed=1, deferred=0` produced `fragments_released=1, owed=0,
        deferred_bytes=0, cleanup_failures=0`, and 255 bytes simply stopped
        existing. Nothing downstream could notice, because nothing downstream
        was told a number that was wrong -- it was told a number that was small.

        SO THE OUTCOME IS A PARTITION. Every byte handed over is either freed
        or deferred, never both and never neither:

            freed + deferred == requested,  0 <= freed <= requested

        `failed=True` means the call did not complete, so nothing was freed and
        the whole request is owed. A releaser that freed PART of what it was
        handed and could not finish reports the partition -- `failed=False,
        freed=k, deferred=requested-k` -- because that is a description of what
        happened, not a fault. Reporting it as a failure instead would make a
        retry free the same bytes twice.
        """
        if self.failed:
            return (self.freed_bytes == 0 and self.retired_bytes == 0
                    and self.deferred_bytes == requested)
        return self.accounted_bytes == requested

    def is_atomic(self, requested: int) -> bool:
        """Whether what will be RETRIED is all of this request or none of it.

        Only `deferred_bytes` is retried, and the manager's debt unit is the
        fragment: it keeps whole fragments and cannot subdivide one. So a call
        that defers PART of what it was handed can only be retried by offering
        back the payloads it already released, and a backend that refuses a
        handle it no longer holds turns that retry into a debt nothing can pay.

        `freed` and `retired` may be mixed freely -- neither is retried.
        """
        return self.deferred_bytes in (0, requested)


class AdmissionRequired(RuntimeError):
    """Device memory was about to be allocated for a transaction that has not
    been admitted, or is no longer open (§7 step 2, §17.2)."""


class CapturePermit:
    """Proof that a transaction was ADMITTED before any device memory was
    allocated for it.

    "Admission before allocation" was a CALLING CONVENTION and nothing more.
    The public interface let a caller run `backend.capture(plan, produced)` and
    then `begin_transaction`, and the integration tests did exactly that --
    sizing the reservation from the tensors they had already allocated.
    Measured on the real MLX backend: capture first, 2 handles and 128 bytes
    live; `begin_transaction` then refused admission; and afterwards, still 2
    handles and 128 bytes. The spec requires the failure to occur BEFORE the
    unsafe allocation, and the API allowed the opposite order.

    A permit closes that by construction: it can only be minted by a manager
    that has already admitted the transaction, and `capture` takes a permit
    rather than a plan, so there is no way to spell the wrong order. It is
    single-use, because one admitted transaction is one capture: claiming it
    twice would allocate a second set of tensors against one reservation.
    """

    __slots__ = ("_transaction", "_plan", "_issuer", "_claimed")

    def __init__(self, transaction: str, plan: KVCapturePlan,
                 issuer: "KVCacheManager") -> None:
        self._transaction = transaction
        self._plan = plan
        self._issuer = issuer
        self._claimed = False

    @property
    def transaction(self) -> str:
        return self._transaction

    @property
    def claimed(self) -> bool:
        return self._claimed

    def claim(self) -> KVCapturePlan:
        """Consume the permit and hand over the plan it admitted.

        Checked against the manager's LIVE state, not just against having been
        issued: a permit for a transaction that has since been aborted or
        committed must not authorise an allocation either.
        """
        # The MANAGER owns the once-only fact. Keeping it only on this token
        # allowed two permits for one transaction to claim independently, and
        # let callers bypass permits entirely by calling stage/commit directly.
        self._issuer._claim_capture(self._transaction)
        self._claimed = True
        return self._plan


@runtime_checkable
class PayloadReleaser(Protocol):
    """What the manager needs of whatever owns the payloads' device memory.

    TWO METHODS, because releasing and RETRYING a release are different events
    with different owners. `free` is driven by the manager when a page stops
    being reachable; `sweep` retries what `free` could not complete, and without
    an owner it was called from nowhere but two backend unit tests -- the first
    release was wired and the debt behind it was left to whoever remembered.

    `MLXKVDeviceBackend` satisfies this by shape; the manager imports no
    backend and never inspects a payload.
    """

    def free(self, *payloads: Any) -> ReleaseOutcome:
        """Release these payloads. Reports what it managed; never raises."""

    def sweep(self) -> int:
        """Retry deferred releases; return the bytes that finally went back."""


class TensorSpecMismatch(ValueError):
    """A tensor, or a spec, does not match the identity the world is keyed by."""


@dataclass(frozen=True, slots=True)
class KVTensorSpec:
    """The tensor geometry an identity is keyed by (§6, §7 step 5).

    WHY THIS LIVES IN THE CONTRACT AND NOT IN A BACKEND. When the spec was a
    parameter the backend's caller supplied, it proved only that the tensors
    matched what THAT CALLER SAID. Measured: an identity naming bfloat16 and
    `int8-weight-only` accepted float32 tensors under a spec claiming
    quantization `totally-wrong`, and the manager committed it. A spec the
    caller chooses cannot validate the identity the manager keys by; only a spec
    the manager holds can.

    `row_stride_bytes` is derived rather than passed: two callers computing it
    differently is how one spec becomes two.

    NO `compute_dtype` FIELD. An earlier version carried one and nothing could
    check it -- a tensor does not know what precision produced it, and
    `KVIdentity` binds a single `dtype`. Carrying an unvalidatable field is the
    decoration this package keeps finding; a storage/compute split needs an
    identity field first.
    """

    dtype: str
    quantization_identity: str
    kv_heads: int
    head_dim: int
    layout: str = "rows_heads_dim"

    def __post_init__(self) -> None:
        for name in ("dtype", "quantization_identity", "layout"):
            if not getattr(self, name):
                raise ValueError(f"a tensor spec must name its {name}")
        if self.kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError(
                f"kv_heads={self.kv_heads} head_dim={self.head_dim}; both are counts"
            )
        if self.layout != "rows_heads_dim":
            # Refused rather than assumed: a layout this contract cannot check
            # would make every shape assertion below meaningless.
            raise ValueError(f"unsupported layout {self.layout!r}")
        # ONE ROW MUST BE COUNTABLE IN INT64. Python integers do not overflow,
        # so this side would happily carry a spec whose byte count no
        # production manager can represent -- and the C++ manager, which counts
        # in int64, refuses it. Measured by the differential: `kv_heads = 2^60`
        # wrapped to zero bytes there and was accepted here, so the two
        # disagreed about which specs exist at all. The limit is shared rather
        # than each side having its own.
        if self.row_stride_bytes > _INT64_MAX:
            raise ValueError(
                f"one row of {self.kv_heads}x{self.head_dim} in {self.dtype} is "
                f"{self.row_stride_bytes} bytes, which no int64 byte count can "
                "represent"
            )

    def expected_shape(self, rows: int) -> tuple[int, int, int]:
        return (rows, self.kv_heads, self.head_dim)

    @property
    def itemsize(self) -> int:
        sizes = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "uint8": 1}
        if self.dtype not in sizes:
            raise UnknownTensorSize(
                f"no byte size is known for dtype {self.dtype!r}; §8 forbids "
                "charging it as zero"
            )
        return sizes[self.dtype]

    @property
    def row_stride_bytes(self) -> int:
        return self.kv_heads * self.head_dim * self.itemsize

    def nbytes(self, rows: int) -> int:
        return rows * self.row_stride_bytes

    def agrees_with(self, identity: KVIdentity) -> None:
        """Refuse a spec that describes tensors the identity does not name.

        §6 binds dtype and quantization identity into the key, so a world keyed
        `bfloat16`/`int8-weight-only` holding float32 unquantized bytes is
        serving the wrong K/V under a name that says otherwise.
        """
        if self.dtype != identity.dtype:
            raise TensorSpecMismatch(
                f"the capture spec stores {self.dtype!r} while the identity is "
                f"keyed by dtype {identity.dtype!r}"
            )
        if self.quantization_identity != identity.quantization_identity:
            raise TensorSpecMismatch(
                f"the capture spec says quantization "
                f"{self.quantization_identity!r} while the identity is keyed by "
                f"{identity.quantization_identity!r}"
            )
        geometry = dict(identity.model_geometry)
        for name, value in (("kv_heads", self.kv_heads),
                            ("head_dim", self.head_dim)):
            # UNCONDITIONAL. `KVIdentity` now requires both fields, so there is
            # no longer an identity this comparison cannot make -- it used to
            # check only what the geometry happened to spell, which is presence
            # standing in for validity.
            if int(geometry[name]) != value:
                raise TensorSpecMismatch(
                    f"the capture spec has {name}={value} while the identity's "
                    f"geometry says {geometry[name]}"
                )

    def validate(self, tensor: Any, rows: int, *, what: str) -> None:
        """Refuse anything that is not exactly this spec at this row count.

        Checks dtype AND the whole shape, not the leading axis: a tensor with
        the right row count and the wrong head count is the same wrong bytes
        with a more convincing disguise.
        """
        shape = getattr(tensor, "shape", None)
        if shape is None:
            raise TensorSpecMismatch(f"{what} has no shape")
        actual = tuple(int(dimension) for dimension in shape)
        if actual != self.expected_shape(rows):
            raise TensorSpecMismatch(
                f"{what} has shape {actual}, not {self.expected_shape(rows)} "
                f"(rows={rows}, kv_heads={self.kv_heads}, head_dim={self.head_dim})"
            )
        name = tensor_dtype_name(tensor)
        if name != self.dtype:
            raise TensorSpecMismatch(
                f"{what} has dtype {name!r}, not the spec's {self.dtype!r}; the "
                "identity is keyed by dtype and would name bytes it does not "
                "describe"
            )


class UnknownTensorSize(RuntimeError):
    """§8: a payload whose size cannot be determined is refused, never zero."""


def tensor_dtype_name(tensor: Any) -> str:
    """A backend-neutral dtype name.

    MLX renders `mlx.core.bfloat16` and numpy renders `dtype('float32')`; both
    end in the name a spec uses, and taking the last dotted component is the one
    rule that holds for both without importing either.
    """
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        raise TensorSpecMismatch("tensor carries no dtype")
    return str(dtype).rsplit(".", 1)[-1]


@dataclass(frozen=True, slots=True)
class KVCapturePlan:
    """Which layer/span outputs the manager will stage (§4.8).

    The layer set is declared here so completeness at commit is checked against
    a declaration rather than against whatever arrived (§4.7).
    """

    requests: tuple[KVCaptureRequest, ...]
    expected_layers: tuple[int, ...]
    # The tensor geometry this transaction will produce. REQUIRED of every
    # transaction: an earlier version made it optional for request-local ones,
    # and an audit committed a (2,1) int8 payload through one by omitting it.
    # A request-local world is read by the same forward and charged against the
    # same budget; nothing about it makes its shape unimportant.
    #
    # REQUIRED AT CONSTRUCTION, not merely refused later. It was keyword-
    # defaulted to None so `begin_transaction` could name the refusal, which
    # meant a plan with no spec was a thing that existed -- and `total_bytes`
    # had to carry a branch for it.
    spec: KVTensorSpec
    # Which committed state a shareable world lands in (§5). Without it
    # COMMITTED_FROZEN_BLOCK was unreachable in Python while the C++ manager
    # could produce it -- a state the two implementations disagreed about.
    committed_state: KVPageState = KVPageState.COMMITTED_PROMPT

    def __post_init__(self) -> None:
        if not self.committed_state.shareable:
            raise ValueError(
                "a transaction must commit into a shareable state (§5)"
            )
        # The field is required, so OMITTING it is a TypeError from the
        # dataclass itself. Passing it explicitly as None is not, and that is
        # the shape callers actually wrote while it had a default.
        if not isinstance(self.spec, KVTensorSpec):
            raise ValueError(
                "a capture plan must declare its tensor spec (§7 step 5); got "
                f"{type(self.spec).__name__}"
            )
        if not self.expected_layers:
            raise ValueError("a capture plan must name the layers it expects")
        if len(set(self.expected_layers)) != len(self.expected_layers):
            raise ValueError("expected_layers contains a duplicate")
        named = {request.layer for request in self.requests}
        expected = set(self.expected_layers)
        if named != expected:
            raise ValueError(
                "capture requests do not match the expected layer set; missing "
                f"{sorted(expected - named)}, unexpected {sorted(named - expected)}"
            )
        # ONE REQUEST PER LAYER, checked by COUNT and not only by set equality.
        # Two requests for layer 0 satisfy `named == expected` -- the sets are
        # both {0} -- while `total_bytes` sums over the requests. Measured: a
        # duplicated single-layer request reserved 512 bytes for a world that
        # commits 128, so the arena charged a transaction for memory no
        # fragment would ever occupy and §17.2's "the whole reservation" named
        # a number belonging to no plan.
        if len(self.requests) != len(self.expected_layers):
            counted = sorted(r.layer for r in self.requests)
            raise ValueError(
                f"a capture plan holds one request per layer; got {counted} "
                f"for layers {sorted(self.expected_layers)}"
            )
        # ONE SPAN. `span` refuses a disagreeing plan when asked, but nothing
        # asked before `commit`, so a plan whose layers covered different rows
        # was admitted and reserved against.
        spans = {(request.start, request.end) for request in self.requests}
        if len(spans) != 1:
            raise ValueError(
                f"every layer of one transaction captures the same span; got "
                f"{sorted(spans)}"
            )
        # AND THE WHOLE PLAN FITS AN INT64, not just one row of it. The spec
        # bounds a row; a plan multiplies that by rows and by layers, and this
        # side counts in unbounded integers. Measured: a 4-layer plan totalling
        # 147573952589676412928 bytes was accepted here and cannot be
        # represented at all by the C++ manager that has to honour it.
        if self.total_bytes > _INT64_MAX:
            raise ValueError(
                f"this plan totals {self.total_bytes} bytes, which no int64 "
                "byte count can represent"
            )

    @property
    def total_bytes(self) -> int:
        """What this plan will produce, derived rather than asserted.

        `begin_transaction` used to take the caller's word for it: a two-layer
        plan needing 512 bytes opened with `total_bytes=1`, and the refusal
        arrived at `stage` -- after the caller had already allocated the real
        device tensors. "Admission before allocation" was a recommended calling
        order rather than an invariant of the API.
        """
        return sum(2 * self.spec.nbytes(request.end - request.start)
                   for request in self.requests)

    @property
    def span(self) -> tuple[int, int]:
        """Every layer of one transaction captures the SAME span.

        A transaction whose layers disagreed about their rows would commit a
        world with no single span to index, and §17.4's token could not name it.
        """
        starts = {request.start for request in self.requests}
        ends = {request.end for request in self.requests}
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError(
                "every layer of one transaction must capture the same span"
            )
        return (next(iter(starts)), next(iter(ends)))


@dataclass(frozen=True, slots=True)
class CommittedWorld:
    """What a successful commit returns, and the only thing a reader may name.

    §17.4. Nothing else constructs one: a caller cannot assemble a world out of
    a transaction it did not commit, an identity it did not declare and a
    fragment it did not stage -- which is exactly what the deleted publish
    interface allowed, measured as publishing against a transaction that never
    existed.
    """

    world_id: str
    transaction_id: str
    session: str
    identity: KVIdentity | None      # None: request-local, never indexed
    layers: tuple[int, ...]
    span: tuple[int, int]
    nbytes: int


@dataclass(frozen=True, slots=True)
class Lease:
    """A held claim on pinned pages (§17.5).

    Every reuse is taken under one and every path releases it: success, failure
    and session close. A lease never released is a page never evictable, which
    turns ordinary memory pressure into permanent admission failure.
    """

    lease_id: str
    session: str
    worlds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KVOperation:
    """One entry in the maintained operation-trace corpus.

    `detail` is canonical -- sorted keys, primitives all the way down. A trace
    that embeds an object compares memory addresses and fails between two runs
    of one implementation, let alone two implementations.
    """

    op: str
    session: str
    detail: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.op:
            raise ValueError("op must be named")
        for key, value in self.detail:
            if not isinstance(key, str):
                raise ValueError("detail keys must be strings")
            _require_comparable(key, value)

    @classmethod
    def of(cls, op: str, session: str, **detail: Any) -> KVOperation:
        return cls(op=op, session=session, detail=tuple(sorted(detail.items())))


def _require_comparable(key: str, value: Any) -> None:
    """Reject anything a second implementation could not reproduce by value.

    Recursive: a tuple of arrays satisfies `isinstance(value, tuple)` and would
    compare memory addresses. A container is only as comparable as what it holds.
    """
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _require_comparable(f"{key}[{index}]", item)
        return
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError(
            f"detail[{key!r}] is {type(value).__name__}; a trace entry holds "
            "primitives so two implementations can be compared by value"
        )


@runtime_checkable
class KVCacheManager(Protocol):
    """The sole production authority for K/V ownership (§4.1).

    §7's lifecycle at §17's semantics. Note what is ABSENT: there is no publish
    operation. §17.3 deletes it and §17.4 makes commit the only thing that
    produces a visible world.
    """

    def open_session(self, session: str, identity: KVIdentity,
                     reserve_bytes: int = 0) -> None:
        """Admit a session, proving its non-evictable reservation first (§17.2)."""

    def resolve_read_plan(
        self,
        session: str,
        layers: Iterable[int],
        total_rows: int,
        candidates: Iterable[tuple[KVIdentity, int, int]] = (),
    ) -> tuple[KVReadPlan, Lease]:
        """Resolve reusable pages and pin them under a lease (§7 step 3, §17.5).

        A miss is a request-local fragment, never a partial hit. If anything
        fails after a page is pinned, every pin this call took is released
        before the failure returns -- a partially pinned plan is never
        observable.
        """

    def release(self, lease: Lease) -> None:
        """Release a lease. Idempotent; a stale lease is refused (§17.5, §17.6)."""

    def begin_transaction(self, session: str, identity: KVIdentity | None,
                          plan: KVCapturePlan, total_bytes: int) -> str:
        """Prove the WHOLE staging reservation, then open (§17.2, §17.3).

        `identity=None` is a REQUEST-LOCAL transaction: the world lands in the
        session's arena, is never indexed and is never evictable while the
        session lives (§5, §4.3).
        """

    def transaction_is_open(self, transaction: str) -> bool:
        """Whether `transaction` is still open. False for unknown, committed and
        aborted ones alike -- a permit must not outlive its reservation.

        A native implementation may need an authoritative boundary query. A
        boundary failure propagates, and reconciling stale local bookkeeping is
        an allowed side effect; callers must not treat this as a no-fail,
        in-memory predicate.
        """

    def capture_permit(self, transaction: str) -> CapturePermit:
        """Mint the permit `KVDeviceBackend.capture` requires (§7 step 2).

        Only a manager can issue one, and only for a transaction it admitted,
        which is what makes "admission before allocation" a property of the API
        instead of a note about the order to call things in.
        """

    def _claim_capture(self, transaction: str) -> None:
        """Atomically consume this transaction's one capture admission.

        Private to the manager/permit contract: callers receive a
        :class:`CapturePermit`, not an alternate way around it.
        """

    def stage(self, transaction: str, fragment: KVFragment) -> None:
        """Place one captured layer into staging, consuming reservation already
        proven -- never asking for more (§17.2)."""

    def commit(self, transaction: str) -> CommittedWorld:
        """Validate, reserve, make visible and index, atomically (§17.4)."""

    def abort(self, transaction: str) -> None:
        """Release staging and publish nothing (§7 step 7)."""

    def close_session(self, session: str) -> None:
        """Release every request-local reference, staging and lease. Idempotent."""

    def metrics(self) -> dict[str, int]:
        """One coherent snapshot (§12)."""

    def trace(self) -> tuple[KVOperation, ...]:
        """The operations performed, in order."""
