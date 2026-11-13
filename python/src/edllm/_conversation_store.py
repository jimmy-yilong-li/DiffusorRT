"""Versioned SQLite storage for local conversations.

This is the persistence half of :mod:`edllm.conversation`. It does not execute
models, select context, or call memory providers; it publishes complete turns
and stores derived summary records.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
import stat
import threading
import time
from typing import Iterator
import uuid

from .application_state import _ensure_owner_only_directory


CONVERSATION_SCHEMA_VERSION = 1
_DATABASE_FILE = "conversations.sqlite3"
_SUMMARY_POLICY_KEY = "bounded_summary"
_SUMMARY_POLICY_VERSION = 1


class ConversationError(RuntimeError):
    """Base class for conversation storage and context failures."""


class ConversationConflictError(ConversationError):
    """The stored session revision changed while a caller was working."""


class ConversationIdentityError(ConversationError):
    """A stored session belongs to a different executable model identity."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One validated message in a conversation."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"invalid chat role: {self.role!r}")
        if type(self.content) is not str:
            raise TypeError("chat message content must be a string")

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class ConversationInfo:
    session_id: str
    revision: int
    updated_at_ns: int


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    """A conversation an engine can resume, with a one-line title for a list."""

    session_id: str
    revision: int
    updated_at_ns: int
    title: str


_TITLE_LIMIT = 80


def _one_line_title(message: str) -> str:
    line = message.strip().splitlines()[0].strip() if message.strip() else ""
    return line if len(line) <= _TITLE_LIMIT else line[: _TITLE_LIMIT - 1] + "…"


@dataclass(frozen=True, slots=True)
class _ConversationIdentity:
    package_digest: str
    tokenizer_identity: str
    graph_digest: str
    model_definition_digest: str
    source_revision: str

    def values(self) -> tuple[str, str, str, str, str]:
        return (
            self.package_digest,
            self.tokenizer_identity,
            self.graph_digest,
            self.model_definition_digest,
            self.source_revision,
        )


@dataclass(frozen=True, slots=True)
class _StoredSummary:
    content: str
    source_first_ordinal: int
    source_last_ordinal: int
    token_count: int


@dataclass(frozen=True, slots=True)
class _ConversationSnapshot:
    revision: int
    generation: str
    context_policy: str
    messages: tuple[ChatMessage, ...]


def _validate_session_id(session_id: object) -> str:
    if (
        type(session_id) is not str
        or not session_id
        or session_id != session_id.strip()
        or len(session_id) > 128
        or any(ord(character) < 32 for character in session_id)
    ):
        raise ValueError("conversation session id is invalid")
    return session_id


def _create_database_file(path: Path) -> None:
    """Create or validate the database inode with owner-only permissions."""

    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ConversationError("conversation_database_file_unsafe")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


class _ConversationStore:
    """One versioned SQLite store; connections are short-lived per operation."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._memory_connection: sqlite3.Connection | None = None
        if path is None:
            self._memory_connection = sqlite3.connect(
                ":memory:",
                isolation_level=None,
                check_same_thread=False,
            )
            self._configure(self._memory_connection)
            self._ensure_schema(self._memory_connection)
            return

        _ensure_owner_only_directory(path.parent)
        _create_database_file(path)
        with self._connection() as connection:
            self._ensure_schema(connection)

    @classmethod
    def persistent(cls, root: Path) -> "_ConversationStore":
        return cls(root / _DATABASE_FILE)

    @classmethod
    def ephemeral(cls) -> "_ConversationStore":
        return cls(None)

    @staticmethod
    def _configure(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self._memory_connection is not None:
            yield self._memory_connection
            return
        if self._path is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("conversation store path missing")
        connection = sqlite3.connect(self._path, isolation_level=None, timeout=5.0)
        try:
            self._configure(connection)
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > CONVERSATION_SCHEMA_VERSION:
            raise ConversationError(f"conversation_schema_newer: {version}")
        if version == CONVERSATION_SCHEMA_VERSION:
            return
        if version != 0:
            raise ConversationError(f"conversation_schema_unsupported: {version}")
        try:
            connection.execute("BEGIN IMMEDIATE")
            # Re-read after obtaining the writer lock: another process may have
            # completed the initial schema while this connection was waiting.
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == CONVERSATION_SCHEMA_VERSION:
                connection.execute("COMMIT")
                return
            if version != 0:
                raise ConversationError(f"conversation_schema_unsupported: {version}")
            connection.execute(
                """CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL CHECK (revision >= 0),
                    generation TEXT NOT NULL,
                    package_digest TEXT NOT NULL,
                    tokenizer_identity TEXT NOT NULL,
                    graph_digest TEXT NOT NULL,
                    model_definition_digest TEXT NOT NULL,
                    source_revision TEXT NOT NULL,
                    system_prompt TEXT,
                    context_policy TEXT NOT NULL,
                    created_at_ns INTEGER NOT NULL,
                    updated_at_ns INTEGER NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE messages (
                    session_id TEXT NOT NULL REFERENCES sessions(session_id)
                        ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                    revision INTEGER NOT NULL CHECK (revision > 0),
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    PRIMARY KEY (session_id, ordinal)
                )"""
            )
            connection.execute(
                """CREATE TABLE summaries (
                    session_id TEXT NOT NULL REFERENCES sessions(session_id)
                        ON DELETE CASCADE,
                    source_first_ordinal INTEGER NOT NULL,
                    source_last_ordinal INTEGER NOT NULL,
                    package_digest TEXT NOT NULL,
                    tokenizer_identity TEXT NOT NULL,
                    policy_key TEXT NOT NULL,
                    policy_version INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    PRIMARY KEY (
                        session_id, source_first_ordinal, source_last_ordinal,
                        package_digest, tokenizer_identity, policy_key,
                        policy_version
                    )
                )"""
            )
            connection.execute(f"PRAGMA user_version = {CONVERSATION_SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def close(self) -> None:
        with self._lock:
            if self._memory_connection is not None:
                self._memory_connection.close()
                self._memory_connection = None

    def open_session(
        self,
        session_id: str,
        identity: _ConversationIdentity,
        *,
        system_prompt: str | None,
        context_policy: str,
    ) -> _ConversationSnapshot:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT revision, package_digest, tokenizer_identity,
                           graph_digest, model_definition_digest,
                           source_revision, system_prompt, context_policy,
                           generation
                      FROM sessions WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if row is None:
                    now = time.time_ns()
                    connection.execute(
                        """
                        INSERT INTO sessions(
                            session_id, revision, generation, package_digest,
                            tokenizer_identity, graph_digest,
                            model_definition_digest, source_revision,
                            system_prompt, context_policy, created_at_ns,
                            updated_at_ns
                        ) VALUES (?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            str(uuid.uuid4()),
                            *identity.values(),
                            system_prompt,
                            context_policy,
                            now,
                            now,
                        ),
                    )
                else:
                    stored_identity = tuple(str(value) for value in row[1:6])
                    if stored_identity != identity.values() or row[6] != system_prompt:
                        has_messages = connection.execute(
                            "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                            (session_id,),
                        ).fetchone() is not None
                        if row[0] != 0 or has_messages or row[6] != system_prompt:
                            raise ConversationIdentityError(
                                "conversation execution identity or system prompt differs; "
                                "saved history was not changed. Start a new session "
                                "(CLI: choose another --session name), or explicitly "
                                "remove the old history with --delete-session."
                            )
                        # A failed first turn leaves no history to migrate.
                        # Rebinding it is atomic with opening the session; a
                        # new generation invalidates handles still using the
                        # previous package before they can publish a turn.
                        connection.execute(
                            """
                            UPDATE sessions
                               SET generation = ?, package_digest = ?,
                                   tokenizer_identity = ?, graph_digest = ?,
                                   model_definition_digest = ?, source_revision = ?,
                                   updated_at_ns = ?
                             WHERE session_id = ?
                            """,
                            (str(uuid.uuid4()), *identity.values(), time.time_ns(), session_id),
                        )
                        connection.execute(
                            "DELETE FROM summaries WHERE session_id = ?", (session_id,)
                        )
                    if row[7] != context_policy:
                        connection.execute(
                            """
                            UPDATE sessions
                               SET context_policy = ?, updated_at_ns = ?
                             WHERE session_id = ?
                            """,
                            (context_policy, time.time_ns(), session_id),
                        )
                        connection.execute(
                            "DELETE FROM summaries WHERE session_id = ?",
                            (session_id,),
                        )
                snapshot = self._snapshot_at(connection, session_id)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return snapshot

    def snapshot(self, session_id: str) -> _ConversationSnapshot:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN")
            try:
                snapshot = self._snapshot_at(connection, session_id)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return snapshot

    @staticmethod
    def _snapshot_at(
        connection: sqlite3.Connection, session_id: str
    ) -> _ConversationSnapshot:
        """Capture the handle generation inside the caller's transaction."""
        row = connection.execute(
            """
            SELECT revision, generation, context_policy
              FROM sessions WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown conversation: {session_id!r}")
        message_rows = connection.execute(
            """
            SELECT role, content FROM messages
             WHERE session_id = ? ORDER BY ordinal
            """,
            (session_id,),
        ).fetchall()
        messages = tuple(
            ChatMessage(role=role, content=content) for role, content in message_rows
        )
        if len(messages) % 2 or any(
            message.role != ("user" if index % 2 == 0 else "assistant")
            for index, message in enumerate(messages)
        ):
            raise ConversationError("conversation store contains an incomplete turn")
        return _ConversationSnapshot(
            revision=int(row[0]),
            generation=str(row[1]),
            context_policy=str(row[2]),
            messages=messages,
        )

    def commit_turn(
        self,
        session_id: str,
        *,
        expected_revision: int,
        expected_generation: str,
        expected_context_policy: str,
        user_message: str,
        assistant_message: str,
    ) -> int:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT revision, generation, context_policy
                      FROM sessions WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown conversation: {session_id!r}")
                actual_revision = int(row[0])
                if (
                    actual_revision != expected_revision
                    or str(row[1]) != expected_generation
                    or str(row[2]) != expected_context_policy
                ):
                    raise ConversationConflictError(
                        "conversation revision changed; reload before retrying"
                    )
                ordinal = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()[0]
                )
                new_revision = expected_revision + 1
                connection.executemany(
                    """
                    INSERT INTO messages(session_id, ordinal, revision, role, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        (session_id, ordinal, new_revision, "user", user_message),
                        (
                            session_id,
                            ordinal + 1,
                            new_revision,
                            "assistant",
                            assistant_message,
                        ),
                    ),
                )
                updated = connection.execute(
                    """
                    UPDATE sessions SET revision = ?, updated_at_ns = ?
                     WHERE session_id = ? AND revision = ? AND generation = ?
                       AND context_policy = ?
                    """,
                    (
                        new_revision,
                        time.time_ns(),
                        session_id,
                        expected_revision,
                        expected_generation,
                        expected_context_policy,
                    ),
                ).rowcount
                if updated != 1:
                    raise ConversationConflictError(
                        "conversation revision changed; reload before retrying"
                    )
                connection.execute("COMMIT")
                return new_revision
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def latest_summary(
        self,
        session_id: str,
        *,
        expected_generation: str,
        expected_context_policy: str,
        source_last_ordinal: int,
        identity: _ConversationIdentity,
    ) -> _StoredSummary | None:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN")
            try:
                session = connection.execute(
                    """
                    SELECT generation, context_policy
                      FROM sessions WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise KeyError(f"unknown conversation: {session_id!r}")
                if (
                    str(session[0]) != expected_generation
                    or str(session[1]) != expected_context_policy
                ):
                    raise ConversationConflictError(
                        "conversation session changed; open a new handle"
                    )
                row = connection.execute(
                    """
                    SELECT content, source_first_ordinal,
                           source_last_ordinal, token_count
                      FROM summaries
                     WHERE session_id = ? AND source_first_ordinal = 0
                       AND source_last_ordinal <= ? AND package_digest = ?
                       AND tokenizer_identity = ? AND policy_key = ?
                       AND policy_version = ?
                     ORDER BY source_last_ordinal DESC
                     LIMIT 1
                    """,
                    (
                        session_id,
                        source_last_ordinal,
                        identity.package_digest,
                        identity.tokenizer_identity,
                        _SUMMARY_POLICY_KEY,
                        _SUMMARY_POLICY_VERSION,
                    ),
                ).fetchone()
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        if row is None:
            return None
        return _StoredSummary(
            content=str(row[0]),
            source_first_ordinal=int(row[1]),
            source_last_ordinal=int(row[2]),
            token_count=int(row[3]),
        )

    def save_summary(
        self,
        session_id: str,
        *,
        expected_generation: str,
        expected_context_policy: str,
        source_last_ordinal: int,
        identity: _ConversationIdentity,
        content: str,
        token_count: int,
    ) -> _StoredSummary:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                session = connection.execute(
                    """
                    SELECT generation, context_policy
                      FROM sessions WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise KeyError(f"unknown conversation: {session_id!r}")
                if (
                    str(session[0]) != expected_generation
                    or str(session[1]) != expected_context_policy
                ):
                    raise ConversationConflictError(
                        "conversation session changed; open a new handle"
                    )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO summaries(
                        session_id, source_first_ordinal, source_last_ordinal,
                        package_digest, tokenizer_identity, policy_key,
                        policy_version, content, token_count
                    ) VALUES (?, 0, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        source_last_ordinal,
                        identity.package_digest,
                        identity.tokenizer_identity,
                        _SUMMARY_POLICY_KEY,
                        _SUMMARY_POLICY_VERSION,
                        content,
                        token_count,
                    ),
                )
                connection.execute(
                    """
                    DELETE FROM summaries
                     WHERE session_id = ? AND package_digest = ?
                       AND tokenizer_identity = ? AND policy_key = ?
                       AND policy_version = ? AND source_last_ordinal < ?
                    """,
                    (
                        session_id,
                        identity.package_digest,
                        identity.tokenizer_identity,
                        _SUMMARY_POLICY_KEY,
                        _SUMMARY_POLICY_VERSION,
                        source_last_ordinal,
                    ),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return _StoredSummary(content, 0, source_last_ordinal, token_count)

    def list_resumable(
        self,
        identity: _ConversationIdentity,
        *,
        system_prompt: str | None,
        context_policy: str,
    ) -> tuple[ConversationSummary, ...]:
        """Conversations with history that this identity can continue, newest first.

        One read. It opens no session, so nothing is rebound and no summary is dropped,
        which opening a session with a different policy would do.
        """
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT s.session_id, s.revision, s.updated_at_ns,
                       (SELECT m.content FROM messages m
                         WHERE m.session_id = s.session_id AND m.role = 'user'
                         ORDER BY m.ordinal LIMIT 1)
                  FROM sessions s
                 WHERE s.package_digest = ? AND s.tokenizer_identity = ?
                   AND s.graph_digest = ? AND s.model_definition_digest = ?
                   AND s.source_revision = ?
                   AND s.system_prompt IS ? AND s.context_policy = ?
                   AND EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.session_id)
                 ORDER BY s.updated_at_ns DESC, s.session_id
                """,
                (*identity.values(), system_prompt, context_policy),
            ).fetchall()
        return tuple(
            ConversationSummary(str(session_id), int(revision), int(updated_at_ns),
                                _one_line_title(str(first or "")))
            for session_id, revision, updated_at_ns, first in rows
        )

    def list_sessions(self) -> tuple[ConversationInfo, ...]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT session_id, revision, updated_at_ns "
                "FROM sessions ORDER BY session_id"
            ).fetchall()
        return tuple(
            ConversationInfo(str(session_id), int(revision), int(updated_at_ns))
            for session_id, revision, updated_at_ns in rows
        )

    def delete_session(
        self,
        session_id: str,
        *,
        expected_revision: int,
        expected_generation: str,
        expected_context_policy: str,
    ) -> None:
        self._delete_session(
            session_id,
            expected_revision=expected_revision,
            expected_generation=expected_generation,
            expected_context_policy=expected_context_policy,
        )

    def delete_session_at_revision(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> None:
        """Administrative deletion when no live handle supplies a generation."""

        self._delete_session(
            session_id,
            expected_revision=expected_revision,
            expected_generation=None,
            expected_context_policy=None,
        )

    def _delete_session(
        self,
        session_id: str,
        *,
        expected_revision: int,
        expected_generation: str | None,
        expected_context_policy: str | None,
    ) -> None:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT revision, generation, context_policy
                      FROM sessions WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown conversation: {session_id!r}")
                if (
                    int(row[0]) != expected_revision
                    or (
                        expected_generation is not None
                        and str(row[1]) != expected_generation
                    )
                    or (
                        expected_context_policy is not None
                        and str(row[2]) != expected_context_policy
                    )
                ):
                    raise ConversationConflictError(
                        "conversation revision changed; reload before deleting"
                    )
                connection.execute(
                    "DELETE FROM sessions WHERE session_id = ?", (session_id,)
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise


__all__ = ["CONVERSATION_SCHEMA_VERSION"]
