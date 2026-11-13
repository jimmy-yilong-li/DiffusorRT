"""Optional Textual view over :class:`edllm.local_models.ModelRepository`.

This module owns presentation only. Model discovery, cache state, run-profile
parsing, and every mutating action remain repository operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)

from .application_state import RunProfile
from .backend_ids import ACCEPTED_BACKEND_IDS, CUDA_PRECISION_IDS

if TYPE_CHECKING:
    from .local_models import CachedModel, ModelRepository


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    amount = float(value)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        amount /= 1024
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
    raise AssertionError("unreachable byte unit")


def _optional_identifier(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _optional_prompt(value: str) -> str | None:
    return None if value == "" else value


def _nonnegative_integer(value: str, *, field: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if number < 0:
        raise ValueError(f"{field} must be 0 or greater")
    return number


def _optional_nonnegative_integer(value: str, *, field: str) -> int | None:
    """An empty field is no limit."""
    return None if not value.strip() else _nonnegative_integer(value, field=field)


@dataclass(frozen=True, slots=True)
class ProfileDraft:
    name: str
    model: str
    backend: str | None
    precision: str | None
    device_ordinal: int
    device_memory_budget_bytes: int
    offline: bool
    max_tokens: int | None
    context_policy: str
    system_prompt: str | None
    deployment_profile_id: str | None


class ConfirmationScreen(ModalScreen[bool]):
    """Small reusable confirmation surface for destructive or trusted actions."""

    CSS = """
    ConfirmationScreen { align: center middle; }
    #confirmation { width: 72; height: auto; border: round $primary; padding: 1 2; }
    #confirmation-buttons { height: auto; align-horizontal: right; }
    #confirmation-buttons Button { margin-left: 1; }
    """

    def __init__(self, message: str, *, confirm_label: str = "Confirm") -> None:
        super().__init__()
        self._message = message
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirmation"):
            yield Label(self._message)
            with Horizontal(id="confirmation-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button(self._confirm_label, id="confirm", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


class ProfileEditorScreen(ModalScreen[ProfileDraft | None]):
    """Edit user choices; identity binding remains a repository operation."""

    CSS = """
    ProfileEditorScreen { align: center middle; }
    #profile-editor { width: 84; height: 90%; border: round $primary; padding: 1 2; }
    #profile-editor Label { margin-top: 1; }
    #profile-error { height: auto; color: $error; margin-top: 1; }
    #profile-buttons { height: auto; align-horizontal: right; margin-top: 1; }
    #profile-buttons Button { margin-left: 1; }
    """

    def __init__(self, profile: RunProfile | None = None) -> None:
        super().__init__()
        self._profile = profile

    def compose(self) -> ComposeResult:
        profile = self._profile
        with VerticalScroll(id="profile-editor"):
            yield Label("Profile name")
            yield Input(
                value="" if profile is None else profile.name,
                disabled=profile is not None,
                id="profile-name",
            )
            yield Label("Model ID or prepared package")
            yield Input(value="" if profile is None else profile.model, id="profile-model")
            yield Label("Backend")
            yield Select(
                [(backend, backend) for backend in ACCEPTED_BACKEND_IDS],
                value=(
                    "auto"
                    if profile is None or profile.backend is None
                    else profile.backend
                ),
                allow_blank=False,
                id="profile-backend",
            )
            yield Label("Precision (leave blank for the model default)")
            yield Select(
                [
                    (precision, precision)
                    for precision in ("fp32", *CUDA_PRECISION_IDS)
                ],
                value=(
                    Select.NULL
                    if profile is None or profile.precision is None
                    else profile.precision
                ),
                allow_blank=True,
                id="profile-precision",
            )
            yield Label("Device ordinal")
            yield Input(
                value=str(0 if profile is None else profile.device_ordinal or 0),
                id="profile-device",
            )
            yield Label("Device memory budget in bytes (0 uses device capacity)")
            yield Input(
                value=str(
                    0
                    if profile is None
                    else profile.device_memory_budget_bytes or 0
                ),
                id="profile-budget",
            )
            yield Label("Maximum generated tokens")
            # author:Opus -- empty is no limit: a reply runs until the model ends it.
            yield Input(
                value=(
                    ""
                    if profile is None or profile.max_tokens is None
                    else str(profile.max_tokens)
                ),
                placeholder="Until the model ends the reply",
                id="profile-max-tokens",
            )
            yield Label("Context policy")
            yield Select(
                [("Recent only", "recent_only"), ("Bounded summary", "bounded_summary")],
                value=(
                    "recent_only"
                    if profile is None or profile.context_policy is None
                    else profile.context_policy
                ),
                allow_blank=False,
                id="profile-context",
            )
            yield Label("System prompt (optional)")
            yield Input(
                value="" if profile is None else profile.system_prompt or "",
                id="profile-system",
            )
            yield Label("Deployment profile ID (optional)")
            yield Input(
                value="" if profile is None else profile.deployment_profile_id or "",
                id="profile-deployment",
            )
            yield Checkbox(
                "Offline only",
                value=False if profile is None else bool(profile.offline),
                id="profile-offline",
            )
            yield Static(id="profile-error")
            with Horizontal(id="profile-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Save", id="save", variant="primary")

    def on_mount(self) -> None:
        target = "#profile-name" if self._profile is None else "#profile-model"
        self.query_one(target, Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        try:
            name = self.query_one("#profile-name", Input).value.strip()
            model = self.query_one("#profile-model", Input).value.strip()
            if not name or not model:
                raise ValueError("profile name and model are required")
            backend_value = self.query_one("#profile-backend", Select).value
            precision_value = self.query_one("#profile-precision", Select).value
            context_value = self.query_one("#profile-context", Select).value
            if backend_value is Select.NULL or context_value is Select.NULL:
                raise ValueError("backend and context policy are required")
            draft = ProfileDraft(
                name=name,
                model=model,
                backend=str(backend_value),
                precision=(
                    None if precision_value is Select.NULL else str(precision_value)
                ),
                device_ordinal=_nonnegative_integer(
                    self.query_one("#profile-device", Input).value,
                    field="device ordinal",
                ),
                device_memory_budget_bytes=_nonnegative_integer(
                    self.query_one("#profile-budget", Input).value,
                    field="memory budget",
                ),
                offline=self.query_one("#profile-offline", Checkbox).value,
                max_tokens=_optional_nonnegative_integer(
                    self.query_one("#profile-max-tokens", Input).value,
                    field="maximum tokens",
                ),
                context_policy=str(context_value),
                system_prompt=_optional_prompt(
                    self.query_one("#profile-system", Input).value
                ),
                deployment_profile_id=_optional_identifier(
                    self.query_one("#profile-deployment", Input).value
                ),
            )
        except ValueError as error:
            self.query_one("#profile-error", Static).update(str(error))
            return
        self.dismiss(draft)


class ModelManagerApp(App[None]):
    """Optional presentation layer over one application-scoped repository."""

    TITLE = "DiffusorRT Models"
    SUB_TITLE = "Repository and run profiles"
    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("n", "new_profile", "New profile"),
        Binding("e", "edit_profile", "Edit profile"),
        Binding("d", "delete_profile", "Delete profile"),
        Binding("p", "prepare_model", "Prepare"),
        Binding("v", "verify_model", "Verify"),
        Binding("x", "remove_model", "Remove"),
        Binding("q", "quit", "Quit"),
    ]
    CSS = """
    Screen {
        layout: vertical;
    }
    #content {
        height: 1fr;
        padding: 0 1;
    }
    .section-title {
        height: 2;
        padding-top: 1;
        text-style: bold;
    }
    DataTable {
        height: 1fr;
        min-height: 5;
        border: round $surface;
    }
    #status {
        height: 2;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, repository: "ModelRepository") -> None:
        super().__init__()
        self._repository = repository
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="content"):
            yield Static("Models", classes="section-title")
            yield DataTable(id="models", cursor_type="row", zebra_stripes=True)
            yield Static("Saved run profiles", classes="section-title")
            yield DataTable(id="profiles", cursor_type="row", zebra_stripes=True)
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        models = self.query_one("#models", DataTable)
        models.add_columns(
            "Model",
            "Cache",
            "Package estimate",
            "License",
            "Revision",
            "Source estimate",
            "Package disk free",
            "Source disk free",
            "Machine backends",
        )
        profiles = self.query_one("#profiles", DataTable)
        profiles.add_columns(
            "Profile",
            "Model or package",
            "Backend",
            "Precision",
            "Context",
        )
        self.action_refresh()

    def action_refresh(self) -> None:
        """Refresh without blocking terminal input on backend capability probes."""

        if not self._start_operation("Refreshing repository state…"):
            return
        self._refresh_worker(None)

    @work(thread=True, exit_on_error=False, group="repository")
    def _refresh_worker(self, completion_message: str | None) -> None:
        try:
            models_snapshot = tuple(self._repository.managed_models())
            profiles_snapshot = self._repository.list_run_profiles()
        except Exception as error:
            self.call_from_thread(
                self._finish_operation,
                f"Refresh failed: {type(error).__name__}: {error}",
                refresh=False,
            )
            return
        self.call_from_thread(
            self._apply_snapshot,
            models_snapshot,
            profiles_snapshot,
            completion_message,
        )

    def _apply_snapshot(
        self,
        models_snapshot: tuple["CachedModel", ...],
        profiles_snapshot: tuple[RunProfile, ...],
        completion_message: str | None,
    ) -> None:
        """Publish one complete snapshot on the UI thread."""

        models = self.query_one("#models", DataTable)
        profiles = self.query_one("#profiles", DataTable)
        models.clear()
        profiles.clear()
        for model in models_snapshot:
            models.add_row(
                model.model_id,
                model.status,
                _format_bytes(model.package_bytes),
                model.source_license,
                model.revision[:12],
                _format_bytes(model.source_bytes),
                _format_bytes(model.package_available_bytes),
                _format_bytes(model.source_available_bytes),
                ", ".join(model.usable_backends) or "none",
                key=model.model_id,
            )
        for profile in profiles_snapshot:
            profiles.add_row(
                profile.name,
                profile.model,
                profile.backend or "default",
                profile.precision or "default",
                profile.context_policy or "recent_only",
                key=profile.name,
            )
        self._busy = False
        summary = (
            f"{len(models_snapshot)} model(s), "
            f"{len(profiles_snapshot)} run profile(s). "
            "Use N/E/D for profiles and P/V/X for model packages."
        )
        self._set_status(
            summary
            if completion_message is None
            else f"{completion_message} {summary}"
        )

    def action_quit(self) -> None:
        if self._busy:
            self._set_status("A repository operation is running; wait for it to finish.")
            return
        self.exit()

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(Text(message))

    def _selected_value(self, table_id: str) -> str | None:
        table = self.query_one(table_id, DataTable)
        if table.row_count == 0:
            return None
        return str(table.get_row_at(table.cursor_row)[0])

    def _action_is_available(self) -> bool:
        if self._busy:
            self._set_status("A repository operation is still running.")
            return False
        return True

    def action_new_profile(self) -> None:
        if not self._action_is_available():
            return
        self.push_screen(ProfileEditorScreen(), self._save_profile)

    def action_edit_profile(self) -> None:
        if not self._action_is_available():
            return
        name = self._selected_value("#profiles")
        if name is None:
            self._set_status("Select a run profile to edit.")
            return
        try:
            profile = self._repository.load_run_profile(name)
        except Exception as error:
            self._set_status(f"Profile load failed: {type(error).__name__}: {error}")
            return
        self.push_screen(ProfileEditorScreen(profile), self._save_profile)

    def action_delete_profile(self) -> None:
        if not self._action_is_available():
            return
        name = self._selected_value("#profiles")
        if name is None:
            self._set_status("Select a run profile to delete.")
            return
        self.push_screen(
            ConfirmationScreen(
                f"Delete run profile {name!r}? Model files are not removed.",
                confirm_label="Delete profile",
            ),
            lambda confirmed: self._delete_profile(name) if confirmed else None,
        )

    def action_prepare_model(self) -> None:
        if not self._action_is_available():
            return
        model = self._selected_value("#models")
        if model is None:
            self._set_status("Select a model to prepare.")
            return
        try:
            plan = self._repository.plan(model, offline=False)
        except Exception as error:
            self._set_status(f"Planning failed: {type(error).__name__}: {error}")
            return
        if plan.trust_required:
            self.push_screen(
                ConfirmationScreen(
                    f"{model} imports custom tokenizer code from "
                    f"{plan.trust_module!r}. Prepare and permit that exact module?",
                    confirm_label="Prepare and trust",
                ),
                lambda confirmed: (
                    self._run_model_operation("prepare", model, True)
                    if confirmed
                    else None
                ),
            )
            return
        self._run_model_operation("prepare", model, False)

    def action_verify_model(self) -> None:
        if not self._action_is_available():
            return
        model = self._selected_value("#models")
        if model is None:
            self._set_status("Select a model to verify.")
            return
        self._run_model_operation("verify", model, False)

    def action_remove_model(self) -> None:
        if not self._action_is_available():
            return
        model = self._selected_value("#models")
        if model is None:
            self._set_status("Select a model to remove.")
            return
        self.push_screen(
            ConfirmationScreen(
                f"Remove the compiled package for {model}? The source cache is kept.",
                confirm_label="Remove package",
            ),
            lambda confirmed: (
                self._run_model_operation("remove", model, False)
                if confirmed
                else None
            ),
        )

    def _start_operation(self, message: str) -> bool:
        if self._busy:
            self._set_status("Another repository operation is still running.")
            return False
        self._busy = True
        self._set_status(message)
        return True

    def _finish_operation(self, message: str, *, refresh: bool) -> None:
        self._busy = False
        if refresh:
            if self._start_operation(f"{message} Refreshing repository state…"):
                self._refresh_worker(message)
            return
        self._set_status(message)

    def _save_profile(self, draft: ProfileDraft | None) -> None:
        if draft is None or not self._start_operation(f"Saving profile {draft.name!r}…"):
            return
        self._save_profile_worker(draft)

    @work(thread=True, exit_on_error=False, group="repository")
    def _save_profile_worker(self, draft: ProfileDraft) -> None:
        try:
            self._repository.configure_run_profile(
                draft.name,
                draft.model,
                backend=draft.backend,
                precision=draft.precision,
                device_ordinal=draft.device_ordinal,
                device_memory_budget_bytes=draft.device_memory_budget_bytes,
                offline=draft.offline,
                max_tokens=draft.max_tokens,
                context_policy=draft.context_policy,
                system_prompt=draft.system_prompt,
                deployment_profile_id=draft.deployment_profile_id,
            )
        except Exception as error:
            self.call_from_thread(
                self._finish_operation,
                f"Profile save failed: {type(error).__name__}: {error}",
                refresh=False,
            )
            return
        self.call_from_thread(
            self._finish_operation,
            f"Saved run profile {draft.name!r}.",
            refresh=True,
        )

    def _delete_profile(self, name: str) -> None:
        if not self._start_operation(f"Deleting profile {name!r}…"):
            return
        self._delete_profile_worker(name)

    @work(thread=True, exit_on_error=False, group="repository")
    def _delete_profile_worker(self, name: str) -> None:
        try:
            self._repository.delete_run_profile(name)
        except Exception as error:
            self.call_from_thread(
                self._finish_operation,
                f"Profile deletion failed: {type(error).__name__}: {error}",
                refresh=False,
            )
            return
        self.call_from_thread(
            self._finish_operation,
            f"Deleted run profile {name!r}.",
            refresh=True,
        )

    def _run_model_operation(self, operation: str, model: str, trust: bool) -> None:
        if not self._start_operation(f"{operation.title()} {model}…"):
            return
        self._model_operation_worker(operation, model, trust)

    @work(thread=True, exit_on_error=False, group="repository")
    def _model_operation_worker(
        self,
        operation: str,
        model: str,
        trust: bool,
    ) -> None:
        try:
            if operation == "prepare":
                self._repository.prepare(
                    model,
                    trust_remote_code=trust,
                    offline=False,
                    on_stage=lambda stage: self.call_from_thread(
                        self._set_status, f"Preparing {model}: {stage}"
                    ),
                )
                message = f"Prepared {model}."
            elif operation == "verify":
                self._repository.verify(model)
                message = f"Verified {model}."
            elif operation == "remove":
                self._repository.remove(model)
                message = f"Removed the compiled package for {model}."
            else:
                raise AssertionError(f"unknown model operation: {operation}")
        except Exception as error:
            self.call_from_thread(
                self._finish_operation,
                f"{operation.title()} failed: {type(error).__name__}: {error}",
                refresh=False,
            )
            return
        self.call_from_thread(
            self._finish_operation,
            message,
            refresh=True,
        )


def run_models_tui(repository: "ModelRepository") -> None:
    """Run the optional model browser against one repository instance."""

    ModelManagerApp(repository).run()


__all__ = ["ModelManagerApp", "run_models_tui"]
