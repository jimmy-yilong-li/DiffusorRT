"""Command-line product surface over the existing DiffusorRT runtime APIs.

The CLI owns preflight and interaction only. It does not duplicate generation,
decode, cache, or backend execution logic.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import contextlib
import json
import logging
import math
import sys
import time
from typing import NamedTuple
from pathlib import Path

from . import __version__, _extra_install_command
from ._log import configure_logging
from .backend_ids import (
    AUTO_BACKEND,
    CUDA_NATIVE_BACKEND,
    CUDA_PRECISION_IDS,
    MLX_LOCAL_BACKEND,
    NUMPY_LOCAL_BACKEND,
    SELECTABLE_BACKEND_IDS,
    PLANNED_BACKEND_IDS,
)
from .deployment import DeploymentError

_log = logging.getLogger("edllm.cli")

# Module-level placeholder retained for the existing low-level test injection.
# Production run and chat commands go through ``ModelRepository``; direct
# package commands retain the public ``Engine.from_package`` compatibility
# path. Neither loader is imported at module load time:
# ``edllm.model`` pulls in heavy backend deps, and we want both
# ``python -m diffusor_rt generate --help`` and the compatibility
# ``python -m edllm generate --help`` to stay free of model-loading work. The
# real loader is late-imported only when a model is actually opened.
load_runtime = None  # type: ignore[assignment]
diagnose_runtime = None  # type: ignore[assignment]
prepare_local_deployment = None  # type: ignore[assignment]

BACKEND_CHOICES = list(SELECTABLE_BACKEND_IDS)
PLANNED_BACKEND_CHOICES = list(PLANNED_BACKEND_IDS)
GENERATION_BACKEND_CHOICES = [AUTO_BACKEND, *BACKEND_CHOICES]
DIAGNOSTIC_BACKEND_CHOICES = [
    *GENERATION_BACKEND_CHOICES,
    *PLANNED_BACKEND_CHOICES,
]
DEPLOYMENT_TARGET_CHOICES = [
    AUTO_BACKEND,
    CUDA_NATIVE_BACKEND,
    MLX_LOCAL_BACKEND,
    NUMPY_LOCAL_BACKEND,
]


def _record_profile_override(namespace: argparse.Namespace, field: str) -> None:
    fields = set(getattr(namespace, "_run_profile_overrides", ()))
    fields.add(field)
    namespace._run_profile_overrides = fields


class _StoreProfileOverride(argparse.Action):
    """Store a value while retaining whether the user actually supplied it."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        _record_profile_override(namespace, self.dest)


class _StoreConstProfileOverride(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs) -> None:
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        del parser, values, option_string
        setattr(namespace, self.dest, self.const)
        _record_profile_override(namespace, self.dest)


def _add_sampling_options(parser: argparse.ArgumentParser, *, window: bool = False) -> None:
    parser.add_argument(
        "--temperature",
        type=float,
        # The window starts at the temperature the user last set unless told one.
        default=None if window else 0.0,
        help=(
            "Sample each token at this temperature, as the model cards do "
            "(cuda_native MDLM and BD3 models). 0 is greedy"
            + ("; omitted, the window keeps the one last set." if window
               else ", the default.")
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Repeat a sampled reply. Omitted, a seed is drawn for each reply.",
    )


def _sampling_problem(args, *, maximum: float | None = None) -> str | None:
    temperature = getattr(args, "temperature", None)
    seed = getattr(args, "seed", None)
    if temperature is not None and (not math.isfinite(temperature) or temperature < 0):
        return f"--temperature must be a finite number 0 or greater; got {temperature}"
    if temperature is not None and maximum is not None and temperature > maximum:
        return f"--temperature must be at most {maximum} in the window; got {temperature}"
    if seed is not None and not 0 <= seed < 2**64:
        return f"--seed must be between 0 and 2**64 - 1; got {seed}"
    return None


def _sampling(args) -> dict:
    from .runtime_config import sampling_arguments

    return sampling_arguments(getattr(args, "temperature", None) or 0.0, getattr(args, "seed", None))


def _add_cuda_device_options(
    parser: argparse.ArgumentParser,
    *,
    track_run_profile: bool = False,
) -> None:
    store_action = _StoreProfileOverride if track_run_profile else "store"
    precision_choices = (
        ("fp32", *CUDA_PRECISION_IDS)
        if track_run_profile
        else CUDA_PRECISION_IDS
    )
    parser.add_argument(
        "--precision",
        choices=precision_choices,
        default=None,
        action=store_action,
        help=(
            "With --run-profile, override deployment precision (including fp32). "
            "Otherwise omit for MLX; cuda_native requires bf16 or int8_weight_only."
            if track_run_profile
            else "Required with cuda_native: bf16 or int8_weight_only."
        ),
    )
    parser.add_argument(
        "--device-ordinal",
        type=int,
        default=0,
        action=store_action,
        help="CUDA device ordinal (default: 0).",
    )
    parser.add_argument(
        "--device-memory-budget-bytes",
        type=int,
        default=0,
        action=store_action,
        help=(
            "Whole-session CUDA allocation limit; 0 uses device capacity. "
            "Fixed-quota packages refuse a nonzero limit and run unbounded at 0."
        ),
    )


def _add_run_profile_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-profile",
        default=None,
        metavar="NAME",
        help=(
            "Use a saved run profile. Explicit backend, device, generation, "
            "and context options override its saved values."
        ),
    )


def _add_model_catalog_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-catalog",
        default=None,
        metavar="PATH",
        help=(
            "Use an application-scoped Local model catalog TOML. The catalog "
            "is loaded once and shared by planning, compilation, diagnostics, "
            "and runtime loading."
        ),
    )


def _add_application_state_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="PATH",
        help=(
            "Override local DiffusorRT configuration and conversation storage. "
            "The default is the platform application-data directory."
        ),
    )


def _add_cache_root_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-root", default=None, metavar="PATH",
        help="Override the DiffusorRT compiled-package cache root.",
    )


def _repository_for_args(args: argparse.Namespace):
    """Return one repository instance for the whole command transaction."""

    cached = getattr(args, "_model_repository", None)
    if cached is not None:
        return cached
    from edllm.local_models import ModelRepository
    from edllm.model_catalog import ModelCatalog

    catalog_path = getattr(args, "model_catalog", None)
    catalog = None if catalog_path is None else ModelCatalog.from_toml(catalog_path)
    repository = ModelRepository(
        catalog=catalog,
        cache_root=getattr(args, "cache_root", None),
        data_root=getattr(args, "data_root", None),
        catalog_path=catalog_path,
    )
    args._model_repository = repository
    return repository


def _cuda_device_kwargs(args: object, backend: str) -> dict[str, object]:
    if backend != CUDA_NATIVE_BACKEND:
        return {}
    return {
        "precision": getattr(args, "precision", None),
        "device_ordinal": getattr(args, "device_ordinal", 0),
        "device_memory_budget_bytes": getattr(
            args, "device_memory_budget_bytes", 0
        ),
    }


def _open_package_engine(
    package_path: Path,
    *,
    backend: str,
    trust_remote_code: bool,
    expected_module: object,
    device_options: dict[str, object],
    repository=None,
):
    """Open the Engine-owned package path used by package-backed commands."""
    from edllm import Engine

    if load_runtime is None:
        if repository is not None:
            return repository.open_package(
                package_path,
                backend=backend,
                trust_remote_code=trust_remote_code,
                expected_module=expected_module,
                **device_options,
            )
        return Engine.from_package(
            package_path,
            backend=backend,
            trust_remote_code=trust_remote_code,
            expected_module=expected_module,
            **device_options,
        )

    # The injected lower-level loader keeps focused tests able to provide a
    # tiny RuntimeHandle substitute. Shipping code never enters this branch.
    runtime = load_runtime(
        package_path,
        trust_remote_code=trust_remote_code,
        expected_module=expected_module,
    )
    try:
        return Engine(
            runtime,
            backend=backend,
            data_root=(None if repository is None else repository.data_root),
            **device_options,
        )
    except BaseException:
        runtime.close()
        raise


def _run_with_checked_cleanup(
    action: Callable[[], int],
    *cleanups: tuple[str, Callable[[], object]],
) -> int:
    """Run all cleanup, preserving a primary failure and rejecting false success."""

    def finish(primary: BaseException | None) -> bool:
        failed = False
        for label, cleanup in cleanups:
            try:
                cleanup()
            except Exception as error:  # cleanup must not replace the primary
                failed = True
                _log.error("%s failed: %s", label, error, exc_info=True)
                if primary is not None and hasattr(primary, "add_note"):
                    primary.add_note(
                        f"{label} also failed: {type(error).__name__}: {error}"
                    )
        return failed

    try:
        result = action()
    except ValueError as primary:
        # Request/policy validation is a normal CLI refusal, not a traceback.
        # Keep the Python API exception and still release every acquired owner.
        _log.error("request rejected: %s", primary, exc_info=_log.isEnabledFor(logging.DEBUG))
        finish(primary)
        return 2
    except BaseException as primary:
        finish(primary)
        raise
    cleanup_failed = finish(None)
    return 1 if result == 0 and cleanup_failed else result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diffusorrt",
        description="DiffusorRT command-line interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging to stderr.",
    )
    # Shared -v so "diffusorrt run MODEL -v" works as well as "diffusorrt -v
    # run MODEL". SUPPRESS is load-bearing: with a normal store_true default,
    # parsing the subcommand writes False over the True the top-level -v just
    # set, and the documented "diffusorrt -v run MODEL" silently does nothing.
    _verbose_parent = argparse.ArgumentParser(add_help=False)
    _verbose_parent.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
        help="Enable DEBUG-level logging to stderr.",
    )
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser(
        "generate", parents=[_verbose_parent],
        help="Run a single one-shot generation against a runtime package.",
    )
    generate.add_argument(
        "--package",
        required=True,
        help="Path to the execution package directory to load.",
    )
    _add_cuda_device_options(generate)
    _add_sampling_options(generate)
    generate.add_argument(
        "--prompt",
        required=True,
        help="Prompt text to generate from.",
    )
    generate.add_argument(
        "--backend",
        default=AUTO_BACKEND,
        choices=GENERATION_BACKEND_CHOICES,
        help="Generation backend to use (default: auto).",
    )
    generate.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate (default: 32).",
    )
    generate.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON object with the text and debug metrics.",
    )
    generate.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Permit the package's own tokenizer module to be imported. "
            "Required non-interactively; on a terminal you are asked instead."
        ),
    )
    generate.set_defaults(func=_cmd_generate)

    chat = subparsers.add_parser(
        "chat", parents=[_verbose_parent],
        help="Create or resume a durable local conversation.",
        description=(
            "Create or resume a durable local conversation. History is stored "
            "as plain local application data; it is not encrypted or synchronized."
        ),
    )
    chat.add_argument(
        "model",
        nargs="?",
        metavar="MODEL",
        help="A supported model ID, Local catalog entry, or prepared package.",
    )
    chat.add_argument(
        "--package",
        default=None,
        help="Compatibility spelling for a prepared package path.",
    )
    _add_model_catalog_option(chat)
    _add_application_state_option(chat)
    _add_run_profile_option(chat)
    _add_cuda_device_options(chat, track_run_profile=True)
    _add_sampling_options(chat)
    chat.add_argument(
        "--backend",
        default=AUTO_BACKEND,
        choices=GENERATION_BACKEND_CHOICES,
        action=_StoreProfileOverride,
        help="Generation backend to use (default: auto).",
    )
    chat.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        action=_StoreProfileOverride,
        help="A limit for every reply. By default a reply runs until the model ends it "
        "(on MLX, up to 1,024 tokens).",
    )
    chat.add_argument(
        "--system",
        default=None,
        action=_StoreProfileOverride,
        help="Explicit system prompt. The framework supplies none by default.",
    )
    chat.add_argument(
        "--session",
        default=None,
        metavar="NAME",
        help="Create or resume this durable conversation.",
    )
    chat.add_argument(
        "--context-policy",
        choices=("recent_only", "bounded_summary"),
        default="recent_only",
        action=_StoreProfileOverride,
        help="Context compaction policy for a durable session.",
    )
    chat.add_argument(
        "--once",
        metavar="TEXT",
        default=None,
        help="Run a single user message, print the reply, and exit.",
    )
    chat.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Permit the package's own tokenizer module to be imported. "
            "Required non-interactively; on a terminal you are asked instead."
        ),
    )
    chat_network = chat.add_mutually_exclusive_group()
    chat_network.add_argument(
        "--offline",
        action=_StoreConstProfileOverride,
        const=True,
        default=False,
        help="Never reach the network; require the fixed revision locally.",
    )
    chat_network.add_argument(
        "--online",
        dest="offline",
        action=_StoreConstProfileOverride,
        const=False,
        default=False,
        help="Permit exact-revision acquisition, overriding an offline profile.",
    )
    _add_cache_root_option(chat)
    management = chat.add_mutually_exclusive_group()
    management.add_argument(
        "--list-sessions",
        action="store_true",
        help="List durable sessions without loading a model.",
    )
    management.add_argument(
        "--delete-session",
        default=None,
        metavar="NAME",
        help="Delete one exact durable session without loading a model.",
    )
    chat.add_argument(
        "--expected-revision",
        type=int,
        default=None,
        metavar="N",
        help="Required with --delete-session to prevent lost updates.",
    )
    chat.add_argument(
        "--json",
        action="store_true",
        help="Emit session-management output as JSON.",
    )
    chat.set_defaults(func=_cmd_chat)

    gui = subparsers.add_parser(
        "gui", parents=[_verbose_parent],
        help="Open the chat window.",
        description=(
            "Open the chat window. With MODEL, open exactly that model; without it, "
            "open the model the window used last, or the only prepared one, or ask. "
            "The window never downloads or compiles a model."
        ),
    )
    gui.add_argument(
        "model",
        nargs="?",
        metavar="MODEL",
        help="A supported model ID, Local catalog entry, or prepared package.",
    )
    _add_model_catalog_option(gui)
    _add_application_state_option(gui)
    _add_cuda_device_options(gui)
    _add_sampling_options(gui, window=True)
    gui.add_argument(
        "--backend",
        default=AUTO_BACKEND,
        choices=GENERATION_BACKEND_CHOICES,
        help="Generation backend to use (default: auto).",
    )
    gui.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        # author:Opus -- the Mac windows have no output limit and do not read this flag.
        help="Maximum number of tokens to generate per reply (default: 128). "
        "Not used on a Mac, where a reply runs until the model ends it.",
    )
    gui.add_argument(
        "--system",
        default=None,
        help="Explicit system prompt. The framework supplies none by default.",
    )
    gui.add_argument(
        "--context-policy",
        choices=("recent_only", "bounded_summary"),
        default="recent_only",
        help="Context compaction policy for durable conversations.",
    )
    gui.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Permit the package's own tokenizer module to be imported without asking.",
    )
    gui.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the chat page's local address instead of opening a window.",
    )
    gui.add_argument("--browser", action="store_true",
                     help="On a Mac, show the page in the default browser instead of its own window.")
    _add_cache_root_option(gui)
    gui.set_defaults(func=_cmd_gui)

    pull = subparsers.add_parser(
        "pull", parents=[_verbose_parent],
        help="Prepare a supported model into a runnable execution package.",
    )
    pull.add_argument("model", help="A supported model ID.")
    _add_model_catalog_option(pull)
    _add_cache_root_option(pull)
    _add_application_state_option(pull)
    pull.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Use only the pinned revision already in the standard Hugging "
            "Face cache; never reach the network. Without it, a missing "
            "revision is downloaded."
        ),
    )
    pull.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Permit the model's own tokenizer module to be imported. Required "
            "non-interactively; on a terminal you are asked instead."
        ),
    )
    pull.add_argument("--json", action="store_true", help="Emit the result as JSON.")
    pull.set_defaults(func=_cmd_pull)

    run = subparsers.add_parser(
        "run", parents=[_verbose_parent],
        help="Run a prepared model: one-shot with --prompt, otherwise a chat REPL.",
    )
    run.add_argument(
        "model",
        nargs="?",
        help="A supported model ID or package directory; omit with --run-profile.",
    )
    _add_model_catalog_option(run)
    _add_run_profile_option(run)
    prompt_input = run.add_mutually_exclusive_group()
    prompt_input.add_argument(
        "--prompt",
        default=None,
        help="Generate once for this prompt and exit. Omit for an interactive chat.",
    )
    prompt_input.add_argument(
        "--prompt-file", metavar="PATH",
        help="Read one UTF-8 prompt from a file, or - for standard input.",
    )
    _add_cuda_device_options(run, track_run_profile=True)
    _add_sampling_options(run)
    run.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        action=_StoreProfileOverride,
        help="A limit for the reply. By default it runs until the model ends it "
        "(on MLX, up to 1,024 tokens).",
    )
    run.add_argument(
        "--system",
        default=None,
        action=_StoreProfileOverride,
        help="System prompt for the chat session.",
    )
    _add_cache_root_option(run)
    _add_application_state_option(run)
    run.add_argument(
        "--json",
        action="store_true",
        help="Emit the one-shot result as JSON instead of bare text.",
    )
    run.add_argument(
        "--backend",
        default=AUTO_BACKEND,
        choices=GENERATION_BACKEND_CHOICES,
        action=_StoreProfileOverride,
    )
    run_network = run.add_mutually_exclusive_group()
    run_network.add_argument(
        "--offline",
        action=_StoreConstProfileOverride,
        const=True,
        default=False,
        help="Never reach the network; require the revision to be cached already.",
    )
    run_network.add_argument(
        "--online",
        dest="offline",
        action=_StoreConstProfileOverride,
        const=False,
        default=False,
        help="Permit exact-revision acquisition, overriding an offline profile.",
    )
    run.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Permit the package's own tokenizer module to be imported.",
    )
    run.set_defaults(func=_cmd_run)

    serve = subparsers.add_parser(
        "serve", parents=[_verbose_parent],
        help="Serve a prepared model over HTTP in the foreground.",
    )
    serve.add_argument("model", help="A supported model ID, or a package directory.")
    _add_model_catalog_option(serve)
    _add_cache_root_option(serve)
    _add_application_state_option(serve)
    serve.add_argument("--host", default="127.0.0.1", help="Default: 127.0.0.1.")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--request-timeout", type=float, default=300.0,
        help="HTTP response wait limit in seconds (default: 300); running GPU work is not cancelled.",
    )
    serve.add_argument(
        "--backend",
        default=AUTO_BACKEND,
        choices=GENERATION_BACKEND_CHOICES,
    )
    _add_cuda_device_options(serve)
    serve.add_argument(
        "--offline",
        action="store_true",
        help="Never reach the network; require the revision to be cached already.",
    )
    serve.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Permit the package's own tokenizer module to be imported.",
    )
    serve.set_defaults(func=_cmd_serve)

    listing = subparsers.add_parser(
        "list",
        help="Show which supported models are present in the compiler cache.",
    )
    _add_model_catalog_option(listing)
    _add_cache_root_option(listing)
    _add_application_state_option(listing)
    listing.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Re-hash each cached package against its compile receipt. Without "
            "this, a package is reported as 'cached' (present), never 'valid'."
        ),
    )
    listing.add_argument(
        "--json",
        action="store_true",
        help="Emit the listing as machine-readable JSON.",
    )
    listing.set_defaults(func=_cmd_list)

    remove = subparsers.add_parser(
        "remove",
        parents=[_verbose_parent],
        help="Safely remove one repository-owned compiled model package.",
    )
    remove.add_argument("model", help="A supported model ID or its cache path.")
    _add_model_catalog_option(remove)
    _add_application_state_option(remove)
    _add_cache_root_option(remove)
    remove.add_argument(
        "--yes",
        action="store_true",
        help="Confirm package removal; required for non-interactive use.",
    )
    remove.add_argument("--json", action="store_true", help="Emit JSON output.")
    remove.set_defaults(func=_cmd_remove)

    models = subparsers.add_parser(
        "models",
        parents=[_verbose_parent],
        help="Browse models and features; use --list without the optional TUI.",
    )
    models.add_argument("model", nargs="?", help="Describe one exact model ID without loading it.")
    models.add_argument("--list", action="store_true", help="Print models instead of opening the TUI.")
    models.add_argument("--json", action="store_true", help="Describe models and supported operations as JSON.")
    _add_model_catalog_option(models)
    _add_application_state_option(models)
    _add_cache_root_option(models)
    models.set_defaults(func=_cmd_models)

    doctor = subparsers.add_parser(
        "doctor", parents=[_verbose_parent],
        help="Report local backend and execution-package capabilities.",
    )
    doctor.add_argument(
        "package",
        nargs="?",
        default=None,
        metavar="MODEL",
        help=(
            "A supported model ID or an execution package directory. Omit it "
            "to report what this machine can run. --offline is a mode, not a "
            "precondition: it stops the package being opened."
        ),
    )
    _add_model_catalog_option(doctor)
    _add_cache_root_option(doctor)
    _add_application_state_option(doctor)
    _add_cuda_device_options(doctor)
    doctor.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Pre-flight a supported model ID without downloading, converting, "
            "loading weights, or running tokenizer code."
        ),
    )
    doctor.add_argument(
        "--backend",
        default=AUTO_BACKEND,
        choices=DIAGNOSTIC_BACKEND_CHOICES,
        help="Backend to preflight (default: auto).",
    )
    doctor.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Permit opening a package that imports its own tokenizer module, "
            "which lets doctor check backend compatibility against the actual "
            "package instead of the machine alone."
        ),
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="Emit a stable machine-readable capability report.",
    )
    doctor.set_defaults(func=_cmd_doctor)
    doctor.add_argument(
        "--prompt-tokens", type=int,
        help="Tokenized prompt length, including chat-template and special tokens; pair with --max-tokens for an exact-D2F K/V budget.",
    )
    doctor.add_argument(
        "--max-tokens", type=int,
        help="Generation token limit for the K/V budget; requires --prompt-tokens. This is not a physical memory-fit check.",
    )

    deploy = subparsers.add_parser(
        "deploy", parents=[_verbose_parent],
        help="Preflight and run a local execution package.",
    )
    deploy.add_argument(
        "package",
        help="Path to the execution package directory to deploy.",
    )
    _add_cuda_device_options(deploy)
    deploy.add_argument(
        "--target",
        default=AUTO_BACKEND,
        choices=DEPLOYMENT_TARGET_CHOICES,
        help="Deployment backend target (default: auto).",
    )
    deploy.add_argument(
        "--once",
        metavar="TEXT",
        default=None,
        help="Run one prompt and exit; otherwise enter the chat loop.",
    )
    deploy.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens per user request (default: 128).",
    )
    deploy.add_argument(
        "--system",
        default=None,
        help="Explicit system prompt. The framework supplies none by default.",
    )
    deploy.add_argument(
        "--json",
        action="store_true",
        help="Emit one machine-readable result; requires --once.",
    )
    deploy.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Permit the package's own tokenizer module to be imported. "
            "Required non-interactively; on a terminal you are asked instead."
        ),
    )
    deploy.set_defaults(func=_cmd_deploy)
    return parser


def _consent_to_remote_code(model_id: str, module: str, *, granted: bool) -> bool:
    """Obtain consent BEFORE anything is fetched, converted, or imported.

    Non-interactive callers must pass the flag: prompting a pipe would either
    hang or read EOF and be taken as a "yes", and this is a decision to execute
    someone else's code.
    """
    if granted:
        return True
    if not sys.stdin.isatty():
        _log.error(
            f"{model_id} imports {module} from its own repository when it "
            "loads. Re-run with --trust-remote-code to permit that."
        )
        return False
    # The one diagnostic that stays a print: it is half of an interactive
    # question, not a record of something that happened. A log level must
    # never be able to suppress the text a `input()` prompt refers to.
    print(
        f"{model_id} imports {module} from its own repository when it loads.\n"
        "Only continue if you trust that repository.",
        file=sys.stderr,
    )
    try:
        answer = input("Permit it? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def _plan_or_report(
    args,
    *,
    allow_local_dir: bool = True,
    spec: str | None = None,
):
    """Resolve the MODEL argument ONCE. Returns the plan, or None on refusal.

    Every command starts here, so the trust question, the disk budget, the
    package that gets built and the package that gets loaded all come from the
    same reading of the filesystem.
    """
    from edllm.compiler import CompilerError
    from edllm.local_models import ModelSpecError

    try:
        return _repository_for_args(args).plan(
            args.model if spec is None else spec,
            allow_local_dir=allow_local_dir,
            offline=getattr(args, "offline", False),
        )
    except (CompilerError, ModelSpecError) as error:
        _log.error(f"{error}")
        return None
    except (OSError, ValueError) as error:
        _log.error(f"model repository configuration failed: {error}")
        return None


class TrustDecision(NamedTuple):
    """The outcome of the ONE trust question, and what the loader gets.

    Three states, deliberately not `bool | None`. Collapsing NOT_REQUIRED into
    GRANTED is what turned "this package needs no authorisation" into "remote
    code execution is permitted": a package declaring no module was loaded with
    trust_remote_code=True, so anything that made a module appear between the
    check and the load would have been imported without anyone consenting.
    """

    permitted: bool          # what reaches prepare / Engine / deployment
    proceed: bool            # False means stop; the user said no
    expected_module: str | None  # the module consent was given FOR


TRUST_NOT_REQUIRED = TrustDecision(permitted=False, proceed=True, expected_module=None)
TRUST_DENIED = TrustDecision(permitted=False, proceed=False, expected_module=None)


def _settle_trust(args, plan) -> TrustDecision:
    """Decide ONCE, before anything is prepared. Client Runtime v1 T1-T5.

    Takes the plan rather than re-resolving the MODEL string. Resolving again
    here is what let the question be asked about one directory and the answer
    applied to another.

    Returns the value every consumer receives. An earlier version asked the
    user AFTER preparing and then passed the original flag on regardless; the
    version after that fixed the threading but answered True for packages that
    needed nothing, which is a wider grant than the user ever gave.
    """
    module = plan.trust_module

    if module is None:
        # Nothing to authorise -> nothing is authorised. The loader still
        # refuses custom code, so a module appearing later is refused there.
        return TRUST_NOT_REQUIRED

    # plan.spec, not args.model: generate/chat/deploy address a --package
    # path and have no `model` attribute at all, so reading args here crashed
    # the refusal path with AttributeError instead of refusing. The plan
    # already holds what was resolved, which is the thing consent is about.
    if args.trust_remote_code or _consent_to_remote_code(
        plan.spec, module, granted=False
    ):
        return TrustDecision(permitted=True, proceed=True, expected_module=module)
    return TRUST_DENIED


def _expected_module(decision: TrustDecision) -> object:
    """What the LOADER must verify, or UNSPECIFIED when nothing was granted.

    This replaces a CLI-level re-check that read the package directory after
    preparation. That check could only ever inspect a different thing at a
    different moment than the import: the file transformers actually reads is
    the one in the materialized tokenizer bundle at the instant it reads it.
    Two callers ran the check AFTER the load, so a swapped module had already
    been imported and executed by the time the mismatch was reported -- and the
    runtime leaked on that path.

    The check now lives in TokenizerSpec.from_package, immediately before
    AutoTokenizer, and raises RemoteCodeIdentityError.

    NOT_REQUIRED passes UNSPECIFIED rather than None. None is a real
    expectation -- "this package must declare no custom module" -- and a
    package needing no authorisation is loaded with trust_remote_code=False,
    where the loader refuses custom code outright.
    """
    from edllm.trust_identity import UNSPECIFIED

    if not decision.permitted:
        return UNSPECIFIED
    return decision.expected_module


def _cmd_pull(args: argparse.Namespace) -> int:
    """Prepare a model. A thin adapter over prepare_model.

    Progress on stderr, the result on stdout, so `--json` stays parseable.
    """
    from edllm.compiler import CompilerError
    invalid = _validate_generation_limits(args)
    if invalid is not None:
        _log.error(f"{invalid}")
        return 2

    # allow_local_dir=False: a directory is already local and has nothing to
    # acquire; the refusal names the reason per kind.
    plan = _plan_or_report(args, allow_local_dir=False)
    if plan is None:
        return 1

    # THE SAME trust transaction run and serve use. pull previously had its own
    # consent call and then passed trust_remote_code=True unconditionally, so
    # "one trust decision shared by every command" was not true.
    trust = _settle_trust(args, plan)
    if not trust.proceed:
        return 1

    _log.info("planning")
    _log.info(
        f"  model    {plan.model_id}\n"
        f"  revision {plan.revision}\n"
        f"  licence  {plan.source_license}\n"
        f"  needs    {plan.required_free_bytes / 2**30:.1f} GiB "
        f"({plan.available_free_bytes / 2**30:.1f} GiB free)\n"
        f"  cache    {plan.cache_root}\n"
        f"  mode     {'offline' if args.offline else 'online'}"
    )

    started = time.perf_counter()
    try:
        repository = _repository_for_args(args)
        prepared = repository.prepare_plan(
            plan,
            trust_remote_code=trust.permitted,
            offline=args.offline,
            on_stage=lambda stage: _log.info("%s", stage),
            _planning_emitted=True,
        )
    except CompilerError as error:
        _log.error(str(error))
        return 1
    elapsed = round(time.perf_counter() - started, 3)

    if args.json:
        print(
            json.dumps(
                {
                    "model_id": plan.model_id,
                    "revision": plan.revision,
                    "package_dir": str(prepared.package_dir),
                    "reused_cache": prepared.reused_cache,
                    # Measured at the CALLER: the compiler's own timer excluded
                    # this function's work and so understated the wait.
                    "prepare_seconds": elapsed,
                },
                sort_keys=True,
            )
        )
    else:
        print(prepared.package_dir)
    return 0


def _run_profile_explicit_values(args: argparse.Namespace) -> dict[str, object]:
    resolved_values = getattr(args, "_resolved_run_profile_explicit", None)
    if resolved_values is not None:
        return dict(resolved_values)
    aliases = {"system": "system_prompt"}
    return {
        aliases.get(field, field): getattr(args, field)
        for field in getattr(args, "_run_profile_overrides", ())
    }


def _resolve_run_profile_for_command(args: argparse.Namespace) -> bool:
    """Resolve one saved profile and project it onto the existing CLI seam."""

    profile_name = getattr(args, "run_profile", None)
    if profile_name is None:
        return True
    explicit = _run_profile_explicit_values(args)
    try:
        resolved = _repository_for_args(args).resolve_run_profile(
            profile_name,
            explicit=explicit,
        )
    except (KeyError, OSError, ValueError) as error:
        _log.error("run profile %r is not usable: %s", profile_name, error)
        return False

    args._resolved_run_profile = resolved
    args._resolved_run_profile_explicit = explicit
    args.model = resolved.model
    args.backend = resolved.backend
    args.precision = (
        resolved.precision if resolved.backend == CUDA_NATIVE_BACKEND else None
    )
    args.device_ordinal = resolved.device_ordinal or 0
    args.device_memory_budget_bytes = resolved.device_memory_budget_bytes or 0
    args.offline = bool(resolved.offline)
    args.max_tokens = resolved.max_tokens
    args.context_policy = resolved.context_policy
    args.system = resolved.system_prompt
    return True


def _cmd_run(args: argparse.Namespace) -> int:
    """Compatibility run spelling over the shared model and chat seams."""
    from .result import EmptyGenerationError

    if getattr(args, "run_profile", None) is not None and args.model is not None:
        _log.error("MODEL and --run-profile are mutually exclusive")
        return 2
    if getattr(args, "run_profile", None) is None and args.model is None:
        _log.error("run requires MODEL or --run-profile NAME")
        return 2
    if not _resolve_run_profile_for_command(args):
        return 2
    invalid = _validate_generation_limits(args)
    if invalid is not None:
        _log.error(f"{invalid}")
        return 2

    prompt_file = getattr(args, "prompt_file", None)
    if prompt_file is not None:
        try:
            args.prompt = (
                sys.stdin.read() if prompt_file == "-"
                else Path(prompt_file).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError) as error:
            _log.error("could not read prompt file: %s", error)
            return 2
        if not args.prompt.strip():
            _log.error("prompt file must contain non-whitespace text")
            return 2

    # Argument errors are settled BEFORE anything expensive. This check used to
    # sit after preparation and the Engine load, so a cold cache compiled
    # 28 GiB for minutes and only then reported a bad flag combination.
    if args.max_tokens == 0 and (args.prompt is None or getattr(args, "system", None) is not None):
        _log.error("chat requires --max-tokens greater than 0")
        return 2

    if getattr(args, "json", False) and args.prompt is None:
        _log.error(
            "--json needs --prompt; the chat loop emits a turn at "
            "a time and has no single JSON result"
        )
        return 2

    opened = _open_model_for_command(args, spec=args.model)
    if opened is None:
        return 1
    engine, package = opened

    system_prompt = getattr(args, "system", None)
    chat_session = None

    def run() -> int:
        nonlocal chat_session
        if args.prompt is not None:
            from .result import require_nonempty_generation

            _log.info("Generating; waiting for the complete reply...")
            if system_prompt is None:
                result = engine.generate(
                    args.prompt, max_tokens=args.max_tokens, **_sampling(args)
                )
            else:
                # Engine.generate takes no system prompt -- it is a raw
                # completion. Routing through a single chat turn is the only
                # path on which --system reaches the model at all; wiring it
                # into chat_session() alone left `run --prompt --system` giving
                # byte-identical output with and without the flag.
                result = engine.chat_result(
                    args.prompt,
                    max_tokens=args.max_tokens,
                    system_prompt=system_prompt,
                    **_sampling(args),
                )
            text = result.text
            if args.max_tokens:
                require_nonempty_generation(text)
            if getattr(args, "json", False):
                # stdout stays parseable: every progress line went to stderr.
                print(
                    json.dumps(
                        {
                            **result.to_dict(),
                            "model": str(package),
                            "backend": args.backend,
                            "max_tokens": args.max_tokens,
                            "system": system_prompt,
                            "temperature": getattr(args, "temperature", 0.0),
                            "seed": getattr(args, "seed", None),
                        },
                        sort_keys=True,
                    )
                )
            else:
                print(text)
            return 0
        chat_session = engine.chat_session(system_prompt=system_prompt)

        def new_conversation() -> None:
            nonlocal chat_session
            chat_session = _replace_chat_conversation(
                chat_session, lambda: engine.chat_session(system_prompt=system_prompt),
            )
            print("Started a new conversation.")

        return _chat_loop(
            lambda message: engine.chat(
                message, session=chat_session, max_tokens=args.max_tokens,
                **_sampling(args),
            ),
            new_conversation=new_conversation,
        )

    def close_chat_session() -> None:
        if chat_session is not None:
            chat_session.close()

    def run_or_interrupt() -> int:
        try:
            return run()
        except KeyboardInterrupt:
            return 130
        except EmptyGenerationError as error:
            _log.error("generation failed: %s", error)
            return 1

    return _run_with_checked_cleanup(
        run_or_interrupt,
        ("conversation release", close_chat_session),
        ("runtime release", engine.close),
    )


def _replace_chat_conversation(previous, create):
    """Keep the current session if opening its replacement fails."""
    replacement = create()
    try:
        previous.close()
    except BaseException as primary:
        from .model import _close_preserving_primary

        _close_preserving_primary(replacement.close, primary, label="new conversation release")
        raise
    return replacement


def _chat_loop(turn, *, input_fn=input, output=None, new_conversation=None) -> int:
    """The ONE interactive loop. ``turn(message) -> str`` does the model work.

    There were two, differing in every visible detail: one prompted `you> ` and
    printed the reply bare, the other prompted `user> ` and prefixed
    `assistant> `; one recovered from a failed turn, the other ended the
    session. Two loops meant `run` and `chat` were different products to a
    user, and only one of them was documented.

    This is the documented convention, and the recovery behaviour of the
    better half: a failed turn must not discard a working conversation.
    """
    _log.info("Chat ready. /help for commands; Ctrl-D or /exit to leave.")
    draft: list[str] | None = None
    while True:
        try:
            line = input_fn("...> " if draft is not None else "user> ")
        except EOFError:
            if draft is not None:
                print("Unsent draft discarded.", file=output)
            return 0
        message = line.strip()
        if draft is not None:
            if message == "/cancel":
                draft = None
                print("Draft cancelled.", file=output)
                continue
            if message != "/send":
                draft.append(line[1:] if line.startswith("//") else line)
                continue
            line = "\n".join(draft)
            draft = None
            message = line.strip()
        elif message in {"/exit", "/quit"}:
            return 0
        elif message == "/help":
            print(
                "/multiline: enter a draft; /send submits it; /cancel discards it.\n"
                "/new: start another conversation without deleting history.\n"
                "/exit or /quit: leave. Start a literal slash message with //.",
                file=output,
            )
            continue
        elif message == "/multiline":
            draft = []
            print("Enter multiple lines; /send to submit, /cancel to discard.", file=output)
            continue
        elif message == "/new":
            if new_conversation is None:
                print("Exit and reopen chat with a new --session name.", file=output)
            else:
                try:
                    new_conversation()
                except Exception as error:
                    _log.error("could not start a new conversation: %s", error)
            continue
        elif message.startswith("//"):
            line = line[line.index("//") + 1:]
            message = line.strip()
        elif message.startswith("/") and message not in {"/exit", "/quit"}:
            print("Unknown command. Use /help, or // to send a literal slash.", file=output)
            continue
        if not message:
            continue
        try:
            _log.info("Generating; waiting for the complete reply...")
            print(f"assistant> {turn(line)}", file=output)
        except KeyboardInterrupt:
            raise
        except Exception as error:
            # One bad turn must not end the session. Engine.chat commits the
            # user and assistant turns together and rolls back on failure, so
            # the session is still consistent and the next turn is usable --
            # killing the REPL here would discard a working conversation over
            # a recoverable error.
            _log.error(f"turn failed: {error}")


def _validate_generation_limits(args) -> str | None:
    """Reject bad numbers BEFORE a 28 GiB load. Returns a message or None.

    `run --max-tokens -1` used to load the whole model and then raise a
    traceback out of the generator; `serve --port -1` got as far as a health
    inference before uvicorn refused to bind. Argument validation that costs
    minutes and gigabytes is not validation.
    """
    max_tokens = getattr(args, "max_tokens", None)
    # >= 0, matching runtime_config.validate_generation_length. An earlier
    # version of this guard demanded >= 1 and rejected 0, which the runtime
    # contract defines as a legal empty generation -- fixing negatives by
    # killing a valid value.
    if max_tokens is not None and max_tokens < 0:
        return f"--max-tokens must be 0 or greater; got {max_tokens}"
    sampling = _sampling_problem(args)
    if sampling is not None:
        return sampling
    port = getattr(args, "port", None)
    if port is not None and not (1 <= port <= 65535):
        return f"--port must be between 1 and 65535; got {port}"
    request_timeout = getattr(args, "request_timeout", None)
    if request_timeout is not None and (not math.isfinite(request_timeout) or request_timeout <= 0):
        return "--request-timeout must be a finite positive number"
    backend = getattr(args, "backend", getattr(args, "target", None))
    precision = getattr(args, "precision", None)
    device_ordinal = getattr(args, "device_ordinal", 0)
    device_budget = getattr(args, "device_memory_budget_bytes", 0)
    if device_ordinal < 0:
        return f"--device-ordinal must be 0 or greater; got {device_ordinal}"
    if device_budget < 0:
        return (
            "--device-memory-budget-bytes must be 0 or greater; "
            f"got {device_budget}"
        )
    if backend == CUDA_NATIVE_BACKEND and precision is None:
        return "--precision is required with --backend/--target cuda_native"
    if backend != CUDA_NATIVE_BACKEND and (
        precision is not None or device_ordinal != 0 or device_budget != 0
    ):
        return (
            "Without --run-profile, precision and device options require "
            "--backend/--target cuda_native. For MLX, omit --precision; "
            "the model deployment selects it."
        )
    return None


def _prepare_for_command(args, plan, *, trust: bool) -> "object | None":
    """Turn the resolved plan into a ready package, compiling it if needed.

    Shared by run and serve so the two cannot form different opinions about
    what "prepared" means, and so neither re-verifies a receipt the
    preparation step already verified. Takes the plan the trust decision was
    made from -- resolving again here would prepare a package the user never
    consented to.
    """
    from edllm.compiler import CompilerError
    from edllm.local_models import ModelSpecError

    try:
        return _repository_for_args(args).prepare_plan(
            plan,
            trust_remote_code=trust,
            offline=getattr(args, "offline", False),
            on_stage=lambda stage: _log.info("%s", stage),
        )
    except (CompilerError, ModelSpecError) as error:
        _log.error(f"{error}")
        return None


def _open_model_for_command(args, *, spec: str):
    """Resolve, authorize, prepare, and open one model through one repository.

    The work is `model_opening.open_model`, shared with the chat window. This
    adapter maps arguments, asks for trust on the terminal and logs a refusal.
    """
    from edllm.model_opening import ModelOpenError, OpenRequest, open_model

    from edllm.compiler import CompilerError
    from edllm.local_models import ModelSpecError

    profile_name = getattr(args, "run_profile", None)
    if profile_name is not None and getattr(args, "_resolved_run_profile", None) is None:
        if not _resolve_run_profile_for_command(args):
            return None
    if profile_name is not None:
        repository = _repository_for_args(args)
    else:
        # Built inside _plan_or_report before, where a bad catalog was logged, not raised.
        try:
            repository = _repository_for_args(args)
        except (CompilerError, ModelSpecError) as error:
            _log.error(f"{error}")
            return None
        except (OSError, ValueError) as error:
            _log.error(f"model repository configuration failed: {error}")
            return None
    request = OpenRequest(
        spec=spec,
        backend=args.backend,
        precision=getattr(args, "precision", None),
        device_ordinal=getattr(args, "device_ordinal", 0),
        device_memory_budget_bytes=getattr(args, "device_memory_budget_bytes", 0),
        offline=getattr(args, "offline", False),
        run_profile=profile_name,
        run_profile_explicit=(
            _run_profile_explicit_values(args) if profile_name is not None else {}
        ),
    )

    def consent(model_spec: str, module: str) -> bool:
        return bool(args.trust_remote_code) or _consent_to_remote_code(
            model_spec, module, granted=False
        )

    try:
        opened = open_model(
            repository,
            request,
            consent=consent,
            on_stage=lambda stage: _log.info("%s", stage),
            open_package=lambda package_dir, **options: _open_package_engine(
                package_dir, repository=repository, **options
            ),
        )
    except ModelOpenError as error:
        if error.code != "trust_denied":
            _log.error("%s", error)
        return None
    return opened.engine, opened.package_dir


def _cmd_serve(args: argparse.Namespace) -> int:
    """Foreground HTTP service over the existing controller and ASGI adapter.

    Creates no scheduler, worker, or session of its own: prepare_local_deployment
    loads once, LocalServiceController owns the queue, create_service_app is the
    transport. Ctrl-C and the ASGI lifespan both drain the worker and release
    the runtime.
    """
    invalid = _validate_generation_limits(args)
    if invalid is not None:
        _log.error(f"{invalid}")
        return 2

    # Fail on a missing transport dependency before loading 28 GiB, not after.
    try:
        import starlette  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        _log.error(
            "serve needs the serve extra: %s",
            _extra_install_command("serve"),
        )
        return 2

    plan = _plan_or_report(args)
    if plan is None:
        return 1

    trust = _settle_trust(args, plan)
    if not trust.proceed:
        return 1

    prepared = _prepare_for_command(args, plan, trust=trust.permitted)
    if prepared is None:
        return 1
    package = prepared.package_dir

    from edllm.deployment import DeploymentError
    from edllm.service_asgi import create_service_app, run_service_app
    from edllm.service_runtime import LocalServiceController

    try:
        repository = _repository_for_args(args)
        session = repository.open_deployment(
            package,
            target=args.backend,
            trust_remote_code=trust.permitted,
            expected_module=_expected_module(trust),
            **_cuda_device_kwargs(args, args.backend),
        )
    except DeploymentError as error:
        _log.error(f"{error}")
        return 1

    controller = LocalServiceController(session)

    def serve() -> int:
        try:
            app = create_service_app(controller, request_timeout_s=getattr(args, "request_timeout", 300.0))
            _log.info(
                f"serving {package} on http://{args.host}:{args.port} "
                "(Ctrl-C to stop)"
            )
            run_service_app(app, host=args.host, port=args.port)
            return 0
        except KeyboardInterrupt:
            return 130

    controller_stopped = False

    def shutdown_controller() -> None:
        nonlocal controller_stopped
        if controller.shutdown() is not True:
            raise RuntimeError("service controller did not stop")
        controller_stopped = True

    def release_session() -> None:
        # A timed-out controller may still have a worker executing against this
        # runtime.  Releasing the session in that state turns a cleanup error
        # into a use-after-close race.  The failed controller cleanup already
        # makes the command fail; keep the session alive for process teardown.
        if controller_stopped:
            session.close()

    return _run_with_checked_cleanup(
        serve,
        ("service controller shutdown", shutdown_controller),
        ("deployment runtime release", release_session),
    )


def _cmd_list(args: argparse.Namespace) -> int:
    """List registry-managed cache entries. Imports nothing heavy.

    A machine that has not pulled yet has no MLX and no Torch, and `list` is
    one of the first commands it will run.
    """
    from .local_models import CACHE_MISSING

    if args.verify:
        # --verify re-hashes every artifact of every cached package. For the
        # pinned model that is a 28 GiB read that can run for minutes with no
        # other output, and a silent tool looks hung. Say so BEFORE starting,
        # on stderr so --json stdout stays parseable.
        _log.info(
            "Verifying cached packages: re-reading and hashing every artifact. "
            "This can take minutes."
        )

    try:
        models = _repository_for_args(args).cached_models(verify=args.verify)
    except (OSError, ValueError) as error:
        _log.error(f"model repository configuration failed: {error}")
        return 2
    if args.json:
        print(json.dumps([m.to_dict() for m in models], sort_keys=True))
        return 0

    if not models:
        print("No models in the registry.")
        return 0
    width = max(len(m.model_id) for m in models)
    for model in models:
        size = "" if model.status == CACHE_MISSING else (
            f"  {model.package_bytes / 2**30:.1f} GiB"
        )
        print(
            f"{model.model_id:<{width}}  {model.status:<8}  "
            f"{model.source_license}{size}  {model.package_dir}"
        )
    if not args.verify:
        _log.info(
            "'cached' means the package directory is present, not that it is "
            "intact. Run `diffusorrt list --verify` to check it."
        )
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    """Remove one compiled package through the repository safety boundary."""

    if not args.yes:
        _log.error("refusing removal without --yes")
        return 2
    try:
        removed = _repository_for_args(args).remove(args.model)
    except (OSError, ValueError, RuntimeError) as error:
        _log.error("model removal failed: %s", error)
        return 1
    if args.json:
        print(json.dumps({"removed": str(removed)}, sort_keys=True))
    else:
        print(f"Removed compiled package {removed}.")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    """Launch the optional TUI without importing it for any other command."""

    if (
        getattr(args, "model", None) is not None
        or getattr(args, "list", False) or getattr(args, "json", False)
        or not sys.stdin.isatty() or not sys.stdout.isatty()
    ):
        try:
            repository = _repository_for_args(args)
            if args.model is not None:
                records = [repository.describe(args.model)]
            else:
                records = [
                    {**repository.describe(model.model_id), **model.to_dict()}
                    for model in repository.cached_models()
                ]
        except (OSError, ValueError, RuntimeError) as error:
            _log.error("could not describe models: %s", error)
            return 2
        if args.json:
            print(json.dumps(records[0] if args.model else records, sort_keys=True))
        else:
            for record in records:
                operations = ", ".join(name for name, enabled in record["operations"].items() if enabled)
                print(f"{record['model_id']}  [{record.get('status', 'not checked')}]  {operations}")
            print("Use models MODEL --json for details; models in a terminal opens the optional TUI.")
        return 0
    try:
        from .models_tui import run_models_tui
    except ModuleNotFoundError as error:
        if (error.name or "").split(".", 1)[0] not in {"textual", "rich"}:
            raise
        _log.error(
            "model TUI is not installed. Install the [tui] extra: %s",
            _extra_install_command("tui"),
        )
        return 2

    try:
        repository = _repository_for_args(args)
        run_models_tui(repository)
    except (OSError, ValueError) as error:
        _log.error("model TUI could not start: %s", error)
        return 2
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """ONE doctor, for every MODEL kind. Client Runtime v1 D1-D6.

    There were two. `--offline` took a registry ID and answered about the
    cache; the default took a PACKAGE PATH and answered about backends. So
    `diffusorrt doctor Dream-org/Dream-v0-Instruct-7B` -- the obvious thing to
    type -- resolved the model ID as a directory, found nothing, and reported
    on a path that had never existed.

    `--offline` is now a MODE of the one report: it skips the backend probe
    rather than producing a different document.
    """
    from edllm.compiler import CompilerError
    from edllm.local_models import ModelSpecError, machine_report

    from edllm.backends.kv_capacity import validate_request_bounds
    try:
        device_error = _validate_generation_limits(args)
        if device_error is not None:
            raise ValueError(device_error)
        validate_request_bounds(args.prompt_tokens, args.max_tokens)
        if args.prompt_tokens is not None and not args.package:
            raise ValueError("K/V capacity requires a MODEL or package directory")
    except ValueError as error:
        _log.error(f"{error}")
        return 2

    if not args.package:
        # No model named: report what this machine can run. Same schema, same
        # exit-code rule -- not a second document.
        try:
            report = machine_report(
                requested_backend=args.backend,
                check_backend=True,
                cuda_precision=getattr(args, "precision", None),
                cuda_device_memory_budget_bytes=getattr(
                    args, "device_memory_budget_bytes", 0
                ),
            )
        except ValueError as error:
            _log.error(f"{error}")
            return 2
        if args.json:
            print(json.dumps(report.to_dict(), sort_keys=True))
        else:
            _print_model_report(report)
        return 0 if report.ready else 1

    try:
        report = _repository_for_args(args).report(
            args.package,
            requested_backend=args.backend,
            # doctor NEVER prompts. Without the flag the package is not opened,
            # so a report is produced without executing anyone's code -- the
            # answer is just weaker, and says so.
            trust_remote_code=getattr(args, "trust_remote_code", False),
            # The machine probe runs in BOTH modes -- it is an import and a
            # platform check, not network and not model code. Skipping it made
            # `--offline` report `ready: true` beside `backend_status:
            # not_checked`, a readiness claim covering something nobody looked
            # at.
            check_backend=True,
            # THIS is what --offline suppresses: opening the package. Without
            # the flag the parameter had no branch at all in this handler.
            open_package=not args.offline,
            # And the BUDGET is planned for the mode the user will run in.
            # Every report used to be planned offline, so a machine that could
            # not fit the download was told `fits: true`.
            offline=args.offline,
            prompt_tokens=args.prompt_tokens,
            max_tokens=args.max_tokens,
            cuda_precision=getattr(args, "precision", None),
            # The budget is judged against the package's resolved plan, as it
            # will be when the same package is opened to run.
            cuda_device_memory_budget_bytes=getattr(
                args, "device_memory_budget_bytes", 0
            ),
        )
    except (CompilerError, ModelSpecError) as error:
        _log.error(f"{error}")
        return 1
    except (OSError, ValueError) as error:
        _log.error(f"{error}")
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), sort_keys=True))
        return 0 if report.ready else 1

    _print_model_report(report)
    return 0 if report.ready else 1


def _print_model_report(report) -> None:
    from .local_models import BUDGET_NOT_CHECKED

    if report.kind == "machine":
        print("Model:     none given -- reporting this machine")
    elif report.model_id:
        print(f"Model:     {report.model_id}")
        print(f"Revision:  {report.revision}")
        print(f"Licence:   {report.source_license}")
        # "present", never "valid": Hugging Face does not verify that a cached
        # snapshot is complete, so presence is all that can honestly be
        # reported.
        print(f"Snapshot:  {report.snapshot_cache_status}")
    else:
        # A directory the user handed us. Printing a revision and licence it
        # never claimed is what the second doctor did.
        print(f"Model:     {report.package_dir}  ({report.kind})")

    if report.kind != "machine":
        # Integrity and compiler freshness were checked against the receipt.
        print(f"Package:   {report.package_cache_status}  ({report.package_dir})")
        if report.package_cache_status == "needs_rebuild":
            print("           Intact package from another compiler version; run pull to rebuild.")

    if report.kind == "machine":
        pass
    elif report.trust_required:
        # Whether it was GRANTED, not just whether it is needed. `Ready` now
        # depends on it, so a reader has to be able to see which half is
        # missing.
        granted = "granted" if report.trust_satisfied else "NOT GRANTED"
        print(
            f"Trust:     required ({granted}) -- imports "
            f"{report.custom_tokenizer_module} from the package"
        )
    else:
        print("Trust:     not required -- no custom tokenizer module")

    if report.backend_status == "not_checked":
        print("Backend:   not checked")
    elif report.backend_status == "usable":
        if report.selected_backend:
            print(
                f"Backend:   {report.selected_backend}  "
                f"(usable: {', '.join(report.usable_backends)})"
            )
        else:
            # No package was opened, so backend/package COMPATIBILITY is
            # unanswered. Saying "auto" here would read as a resolved choice.
            print(
                f"Backend:   machine only -- {', '.join(report.usable_backends)}"
                "  (package not opened)"
            )
    else:
        reasons = ", ".join(
            reason for reason in report.backend_reasons
            if not reason.startswith(BUDGET_NOT_CHECKED)
        ) or "no usable backend"
        print(f"Backend:   UNAVAILABLE -- {reasons}")
    if any(reason.startswith(BUDGET_NOT_CHECKED) for reason in report.backend_reasons):
        # Silence beside "Ready: yes" would read as approval of the budget.
        print("Budget:    not checked -- no package plan was resolved; name a package to check it")
    if report.deployment_profile is not None:
        print(f"Profile:   {report.deployment_profile}")
    capacity = report.kv_capacity
    if capacity is not None:
        if capacity["status"] == "available":
            print(
                f"K/V:       {capacity['shared_cache_budget_bytes']} B resident budget"
                f" + {capacity['staging_headroom_bytes']} B staging headroom"
                f" ({capacity['total_rows']} rows, {capacity['storage_dtype']},"
                f" {capacity['stored_heads']} stored heads)"
            )
            print("           Logical limits only; physical peak and device-memory fit are unknown.")
        elif capacity["status"] == "needs_request_bounds":
            print("K/V:       unknown -- supply --prompt-tokens and --max-tokens")
        else:
            print(f"K/V:       {capacity['status']} -- {capacity['reason']}")

    # Three independent facts decide readiness, so all three are printed. The
    # machine having MLX says nothing about whether THIS package opens on it,
    # and conflating them let --offline claim ready for a directory that fails
    # to load.
    if report.kind != "machine":
        if report.package_compatibility == "compatible":
            if report.package_cache_status in ("corrupt", "needs_rebuild"):
                print("Loads:     format accepted; preparation is still required before running")
            else:
                print("Loads:     yes -- the package was opened and accepted")
        elif report.package_compatibility == "incompatible":
            reasons = ", ".join(report.backend_reasons) or "it failed to open"
            print(f"Loads:     NO -- {reasons}")
        else:
            if report.package_cache_status == "needs_rebuild":
                print("Loads:     not checked -- rebuild the package first")
            else:
                print(
                    "Loads:     not checked -- the package was not opened"
                    f"{' (--offline)' if report.backend_status != 'unavailable' else ''}"
                )

    if report.kind == "machine":
        pass
    elif report.package_cache_status in ("valid", "cached"):
        # The package exists; nothing more will be written. Showing a budget
        # here read as a blocker next to "Ready: yes".
        print("Disk:      n/a -- already prepared")
    elif report.model_id:
        mark = "" if report.package_fits else "  INSUFFICIENT"
        print(
            f"Disk:      package needs "
            f"{report.required_free_bytes / 2**30:.1f} GiB on "
            f"{report.package_dir.parent}, "
            f"{report.available_free_bytes / 2**30:.1f} GiB free{mark}"
        )
        if report.snapshot_required_bytes:
            # A SECOND filesystem. Reporting one number for two disks sent
            # users freeing space on the wrong volume.
            mark = "" if report.snapshot_fits else "  INSUFFICIENT"
            print(
                f"           snapshot needs "
                f"{report.snapshot_required_bytes / 2**30:.1f} GiB on "
                f"{report.snapshot_cache_root}, "
                f"{report.snapshot_available_bytes / 2**30:.1f} GiB free{mark}"
            )
    print(f"Ready:     {'yes' if report.ready else 'no'}")
    print(f"Next:      {report.next_action}")


def _cmd_deploy(args: argparse.Namespace) -> int:
    if args.json and args.once is None:
        _log.error("--json requires --once for deploy")
        return 2
    invalid = _validate_generation_limits(args)
    if invalid is not None:
        _log.error("%s", invalid)
        return 2

    prepare = prepare_local_deployment
    if prepare is None:
        from edllm.deployment import prepare_local_deployment as prepare

    package_path = Path(args.package).expanduser().resolve()
    try:
        # Establish the input kind before tokenizer discovery, using the same
        # package guard as generate/chat. Keep this refusal inside the JSON
        # deployment-error boundary as well as the human-readable path.
        if not _package_directory_exists(package_path):
            raise DeploymentError(
                "deployment_package_not_found",
                detail=(
                    f"{package_path} is not a prepared execution-package directory. "
                    "deploy accepts a package path, not a model ID. "
                    "Use 'diffusorrt run MODEL --prompt TEXT' for a model ID, "
                    "or 'diffusorrt pull MODEL' and pass the reported package path."
                ),
            )
        trust = _settle_trust_for_package(args, package_path)
        if trust is None:
            return 1
        session = prepare(
            package_path,
            target=args.target,
            trust_remote_code=trust.permitted,
            expected_module=_expected_module(trust),
            **_cuda_device_kwargs(args, args.target),
        )
    except DeploymentError as exc:
        if args.json:
            payload = {
                "status": "error",
                "error": exc.code,
                "detail": exc.detail,
            }
            report = exc.capability_report
            if report is not None:
                payload["capabilities"] = report.to_dict()
            print(json.dumps(payload, sort_keys=True))
        else:
            _log.error(f"{exc}")
        return 1

    # `deploy` created the session, so it owns checked release on every path.
    return _run_with_checked_cleanup(
        lambda: _deploy_body(args, session),
        ("deployment runtime release", session.close),
    )


def _deploy_body(args: argparse.Namespace, session) -> int:
    if args.once is not None:
        device_options = {}
        if session.backend == CUDA_NATIVE_BACKEND:
            device_options = {
                "precision": session.precision,
                "device_ordinal": session.device_ordinal,
                "device_memory_budget_bytes": session.device_memory_budget_bytes,
            }
        result = session.runtime.generate(
            args.once,
            backend=session.backend,
            max_tokens=args.max_tokens,
            return_debug=args.json,
            **device_options,
        )
        if args.json:
            debug_metrics = (
                result.debug.metrics if result.debug is not None else {}
            )
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "backend": session.backend,
                        "profile": session.profile,
                        "health": session.health.to_dict(),
                        "text": result.text,
                        "finish_reason": result.finish_reason,
                        "debug_metrics": debug_metrics,
                    },
                    sort_keys=True,
                )
            )
        else:
            print(
                "DiffusorRT deploy: READY "
                f"backend={session.backend} profile={session.profile}"
            )
            print(result.text)
        return 0

    print(
        "DiffusorRT deploy: READY "
        f"backend={session.backend} profile={session.profile}"
    )
    from edllm import Engine

    # A thin adapter over the SAME session transaction `run` uses. This used to
    # mutate a ChatSession by hand around a raw RuntimeHandle.generate call,
    # with no rollback and no close path.
    # Borrows the runtime; does NOT own it. Closing this Engine would release
    # buffers the deployment session is still responsible for, so the release
    # stays with the owner in _cmd_deploy's finally.
    engine = Engine(
        session.runtime,
        backend=session.backend,
        **_cuda_device_kwargs(session, session.backend),
    )
    chat_session = engine.chat_session(system_prompt=args.system)
    try:
        return _chat_loop(
            _engine_turn(engine, chat_session, max_tokens=args.max_tokens)
        )
    finally:
        chat_session.close()


def _settle_trust_for_package(args, package_path) -> "TrustDecision | None":
    """The SAME trust transaction, for the package-addressed commands.

    generate/chat/deploy take a --package path rather than a model ID, but the
    question is identical: this directory may declare a tokenizer module that
    gets imported when it loads. They had no trust flow at all, so the flagship
    Dream package could not be used with any of them -- it died at
    `real_tokenizer_load_error: AutoTokenizer failed for non-fixture package`,
    which reads as a broken package rather than a missing authorisation.

    Returns None when the user refused; the caller stops.
    """
    from edllm.local_models import KIND_PACKAGE, ClientModelPlan
    from edllm.package_lifecycle import acquire_package_load_lease
    from edllm.tokenizer import package_declares_custom_tokenizer_code

    # Deliberately NOT plan_client_model: that also decides whether the
    # directory is a valid package, and these commands have always left that
    # verdict to the loader. Adding a trust flow must not quietly narrow what
    # they accept. The only question here is the one they were missing.
    discovery_lease = acquire_package_load_lease(package_path)
    try:
        try:
            module = package_declares_custom_tokenizer_code(
                discovery_lease.package_dir
            )
        except ValueError as error:
            _log.error(f"unreadable tokenizer config: {error}")
            return None
    finally:
        discovery_lease.close()

    plan = ClientModelPlan(
        spec=str(package_path),
        kind=KIND_PACKAGE,
        package_dir=Path(package_path),
        package_present=True,
        trust_module=module,
    )
    decision = _settle_trust(args, plan)
    if not decision.proceed:
        return None
    return decision


def _package_directory_exists(package_path: Path) -> bool:
    """Check package presence under the same guard used by its loaders."""

    from edllm.package_lifecycle import acquire_package_load_lease

    lease = acquire_package_load_lease(package_path)
    try:
        return lease.package_dir.is_dir()
    finally:
        lease.close()


def _cmd_generate(args: argparse.Namespace) -> int:
    invalid = _validate_generation_limits(args)
    if invalid is not None:
        _log.error("%s", invalid)
        return 2
    package_path = Path(args.package)
    if not _package_directory_exists(package_path):
        _log.error(f"package not found: {package_path}")
        return 2

    trust = _settle_trust_for_package(args, package_path)
    if trust is None:
        return 1

    engine = _open_package_engine(
        package_path,
        backend=args.backend,
        trust_remote_code=trust.permitted,
        expected_module=_expected_module(trust),
        device_options=_cuda_device_kwargs(args, args.backend),
    )

    def generate() -> int:
        from .result import EmptyGenerationError, require_nonempty_generation

        try:
            result = engine.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                return_debug=args.json,
                **_sampling(args),
            )
            if args.max_tokens:
                require_nonempty_generation(result.text)
        except KeyboardInterrupt:
            return 130
        except EmptyGenerationError as error:
            _log.error("generation failed: %s", error)
            return 1

        if args.json:
            debug_metrics = result.debug.metrics if result.debug is not None else {}
            print(json.dumps({"text": result.text, "debug_metrics": debug_metrics}))
        else:
            print(result.text)
        return 0

    return _run_with_checked_cleanup(
        generate,
        ("generation runtime release", engine.close),
    )


def _cmd_gui(args: argparse.Namespace) -> int:
    """`diffusorrt gui [MODEL]`: the window, with logs on this terminal."""
    # author:Opus (integration) -- imported as `edllm.gui`, the form the producer
    # pathset check resolves: this file is in a pathset, and the window package is
    # one of its classified imports. `run` opens this platform's window.
    from edllm.gui import run as run_window

    problem = _gui_option_problem(args)
    if problem is not None:
        _log.error("%s", problem)
        return 2
    return run_window(args.model, options=args, log_to_file=False)


def _gui_option_problem(args: argparse.Namespace) -> str | None:
    """What would make every reply in the window fail, said before it opens.

    Not `_validate_generation_limits`: the window decides an "auto" backend
    itself, so `--precision` with it is a request, not a contradiction.
    """
    from .gui.preferences import MAX_TEMPERATURE

    if args.max_tokens < 1:
        return f"--max-tokens must be 1 or greater; got {args.max_tokens}"
    sampling = _sampling_problem(args, maximum=MAX_TEMPERATURE)
    if sampling is not None:
        return sampling
    if sys.platform == "darwin" and getattr(args, "temperature", 0):
        return "The Mac chat window decodes greedily; use --temperature 0."
    if args.device_ordinal < 0:
        return f"--device-ordinal must be 0 or greater; got {args.device_ordinal}"
    if args.device_memory_budget_bytes < 0:
        return ("--device-memory-budget-bytes must be 0 or greater; "
                f"got {args.device_memory_budget_bytes}")
    if args.precision is not None and args.backend not in (AUTO_BACKEND, CUDA_NATIVE_BACKEND):
        return f"--precision applies to cuda_native, not --backend {args.backend}"
    return None


def _cmd_chat(args: argparse.Namespace, *, input_fn=input, output=None) -> int:
    """Primary durable chat command and package-addressed compatibility form."""
    mode_error = _validate_chat_mode(args)
    if mode_error is not None:
        _log.error("%s", mode_error)
        return 2
    if getattr(args, "list_sessions", False) or getattr(
        args, "delete_session", None
    ) is not None:
        return _manage_conversations(args)

    if not _resolve_run_profile_for_command(args):
        return 2
    invalid = _validate_generation_limits(args)
    if invalid is not None:
        _log.error("%s", invalid)
        return 2
    if args.max_tokens == 0:
        _log.error("chat requires --max-tokens greater than 0")
        return 2
    model = getattr(args, "model", None)
    if model is None and not _package_directory_exists(Path(args.package)):
        _log.error("package not found: %s", args.package)
        return 2
    spec = model if model is not None else args.package
    opened = _open_model_for_command(args, spec=spec)
    if opened is None:
        return 1
    engine, _package = opened
    conversation = None
    compatibility_session = None

    def chat() -> int:
        nonlocal conversation, compatibility_session
        try:
            if getattr(args, "session", None) is not None:
                conversation = engine.conversation(
                    session_id=args.session,
                    system_prompt=args.system,
                    context_policy=getattr(args, "context_policy", "recent_only"),
                )
                turn = _conversation_turn(
                    conversation,
                    max_tokens=args.max_tokens,
                    **_sampling(args),
                )
            else:
                compatibility_session = engine.chat_session(
                    system_prompt=args.system
                )
                turn = _engine_turn(
                    engine,
                    compatibility_session,
                    max_tokens=args.max_tokens,
                    **_sampling(args),
                )
        except Exception as error:
            _log.error("could not open conversation: %s", error)
            return 1

        if args.once is not None:
            _log.info("Generating; waiting for the complete reply...")
            try:
                print(turn(args.once))
            except Exception as error:
                _log.error("turn failed: %s", error)
                return 1
            return 0

        def new_conversation() -> None:
            nonlocal conversation, compatibility_session, turn
            if conversation is not None:
                conversation = _replace_chat_conversation(
                    conversation,
                    lambda: engine.conversation(
                        system_prompt=args.system,
                        context_policy=getattr(args, "context_policy", "recent_only"),
                    ),
                )
                turn = _conversation_turn(
                    conversation, max_tokens=args.max_tokens, **_sampling(args)
                )
                print(f"Started session {conversation.session_id}. Previous history is preserved.", file=output)
            else:
                compatibility_session = _replace_chat_conversation(
                    compatibility_session, lambda: engine.chat_session(system_prompt=args.system),
                )
                turn = _engine_turn(
                    engine, compatibility_session, max_tokens=args.max_tokens,
                    **_sampling(args),
                )
                print("Started a new conversation.", file=output)

        return _chat_loop(
            lambda message: turn(message),
            input_fn=input_fn,
            output=output,
            new_conversation=new_conversation,
        )

    def close_conversation() -> None:
        if conversation is not None:
            conversation.close()
        if compatibility_session is not None:
            compatibility_session.close()

    def chat_or_interrupt() -> int:
        try:
            return chat()
        except KeyboardInterrupt:
            return 130

    return _run_with_checked_cleanup(
        chat_or_interrupt,
        ("conversation release", close_conversation),
        ("runtime release", engine.close),
    )


def _validate_chat_mode(args: argparse.Namespace) -> str | None:
    model = getattr(args, "model", None)
    package = getattr(args, "package", None)
    run_profile = getattr(args, "run_profile", None)
    session = getattr(args, "session", None)
    once = getattr(args, "once", None)
    list_sessions = getattr(args, "list_sessions", False)
    delete_session = getattr(args, "delete_session", None)
    expected_revision = getattr(args, "expected_revision", None)
    context_policy = getattr(args, "context_policy", "recent_only")
    managing = list_sessions or delete_session is not None
    selected_models = sum(
        value is not None for value in (model, package, run_profile)
    )
    if selected_models > 1:
        return "MODEL, --package, and --run-profile are mutually exclusive"
    if managing:
        if selected_models:
            return (
                "session management does not load MODEL, --package, "
                "or --run-profile"
            )
        if session is not None or once is not None:
            return "session management cannot start a conversation"
        if delete_session is None and expected_revision is not None:
            return "--expected-revision requires --delete-session"
        if delete_session is not None and expected_revision is None:
            return "--delete-session requires --expected-revision"
        if expected_revision is not None and expected_revision < 0:
            return "--expected-revision must be 0 or greater"
        ignored_execution_options = (
            getattr(args, "backend", AUTO_BACKEND) != AUTO_BACKEND
            or getattr(args, "precision", None) is not None
            or getattr(args, "device_ordinal", 0) != 0
            or getattr(args, "device_memory_budget_bytes", 0) != 0
            or getattr(args, "max_tokens", None) is not None
            or getattr(args, "system", None) is not None
            or context_policy != "recent_only"
            or getattr(args, "offline", False)
            or getattr(args, "trust_remote_code", False)
            or getattr(args, "cache_root", None) is not None
            or getattr(args, "model_catalog", None) is not None
        )
        if ignored_execution_options:
            return "model execution options cannot be used for session management"
        return None
    if selected_models == 0:
        return (
            "chat requires MODEL, --package, --run-profile, "
            "or a session-management option"
        )
    if (model is not None or run_profile is not None) and session is None:
        return "model/profile-addressed chat requires --session NAME"
    if getattr(args, "json", False):
        return "--json is available only for session management"
    if session is None and context_policy != "recent_only":
        return "--context-policy requires --session"
    if expected_revision is not None:
        return "--expected-revision requires --delete-session"
    return None


def _manage_conversations(args: argparse.Namespace) -> int:
    repository = _repository_for_args(args)
    try:
        if getattr(args, "list_sessions", False):
            sessions = repository.list_conversations()
            if getattr(args, "json", False):
                print(
                    json.dumps(
                        [
                            {
                                "session_id": item.session_id,
                                "revision": item.revision,
                                "updated_at_ns": item.updated_at_ns,
                            }
                            for item in sessions
                        ],
                        sort_keys=True,
                    )
                )
            elif not sessions:
                print("No conversations.")
            else:
                for item in sessions:
                    print(
                        f"{item.session_id}\t{item.revision}\t"
                        f"{item.updated_at_ns}"
                    )
            return 0

        repository.delete_conversation(
            args.delete_session,
            expected_revision=args.expected_revision,
        )
    except Exception as error:
        _log.error("conversation management failed: %s", error)
        return 1
    if getattr(args, "json", False):
        print(
            json.dumps(
                {"deleted": args.delete_session},
                sort_keys=True,
            )
        )
    else:
        print(f"Deleted conversation {args.delete_session}.")
    return 0


def _conversation_turn(conversation, *, max_tokens: int, **sampling):
    def _turn(message: str) -> str:
        reply = conversation.send(message, max_tokens=max_tokens, **sampling)
        for warning in reply.extension_warnings:
            _log.warning("%s", warning)
        _warn_on_low_quality(reply.text)
        return reply.text.strip()

    return _turn


def _engine_turn(engine, session, *, max_tokens: int, **sampling):
    """The ONE chat turn. Returns a callable for _chat_loop.

    Goes through Engine.chat, which commits the user and assistant turns
    TOGETHER and rolls the user message back when generation fails. The
    hand-rolled `add_user -> generate -> add_assistant` this replaces left the
    user's message in the session after a failed turn, so the retry sent the
    same question twice -- exactly what client-runtime-v1 L2 forbids, and what
    Engine._chat_locked was written to prevent.
    """

    def _turn(message: str) -> str:
        reply = engine.chat(message, session=session, max_tokens=max_tokens, **sampling)
        _warn_on_low_quality(reply)
        return reply.strip()

    return _turn


def _warn_on_low_quality(text: str) -> None:
    """Report observable repetition, not a guess at answer correctness.

    Short or numeric replies can be complete answers. Empty replies are handled
    by the generation contract, and deterministic generation is never retried.
    """
    from edllm.mask_fixing import evaluate_generation

    if evaluate_generation(text).reason == "repetition":
        _log.warning(
            "reply contains repeated words; this is a surface heuristic, not "
            "an accuracy assessment. Deterministic decode was not retried."
        )


@contextlib.contextmanager
def _console_break_stops_like_ctrl_c():
    """On Windows, make Ctrl-Break stop a command the way Ctrl-C does.

    Its default handler ends the process at once, skipping the cleanup that
    closes a CUDA runtime; a process ended that way was seen to hang during
    DLL teardown. As KeyboardInterrupt it takes the same path as Ctrl-C. The
    caller's handler comes back when the command returns.
    """
    import signal
    import threading

    if not hasattr(signal, "SIGBREAK") or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGBREAK, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGBREAK, signal.SIG_DFL if previous is None else previous)


def main(argv: list[str] | None = None) -> int:
    with _console_break_stops_like_ctrl_c():
        return _main(argv)


def _main(argv: list[str] | None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # -v > DIFFUSORRT_LOG_LEVEL > INFO. The CLI's default is INFO, not the
    # library's WARNING: stage events and progress are INFO, and a WARNING
    # default deletes them from every run that does not pass a flag.
    configure_logging(
        "DEBUG" if getattr(args, "verbose", False) else None, default="INFO"
    )
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return 2
    return func(args)
