"""Sync manager for cross-device knowledge synchronization."""

import json
import sqlite3
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# File locking - platform specific
if sys.platform != "win32":
    import fcntl

    HAS_FCNTL = True
else:
    HAS_FCNTL = False

if TYPE_CHECKING:
    import chromadb

    from claude_knowledge._config import ConfigManager
    from claude_knowledge._embedding import EmbeddingService

from claude_knowledge.utils import (
    compute_content_hash,
    context_to_json,
    create_brief,
    get_machine_id,
    json_to_context,
    json_to_tags,
    tags_to_json,
)


@dataclass
class SyncResult:
    """Result of a sync operation."""

    pushed: int = 0
    pulled: int = 0
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    deletions_pushed: int = 0
    deletions_pulled: int = 0
    errors: list[str] = field(default_factory=list)


class SyncManager:
    """Manages synchronization of knowledge entries across devices.

    This service handles bidirectional sync with a shared directory,
    including conflict resolution, tombstone-based deletion tracking,
    and file locking for concurrent access.
    """

    # File locking
    DEFAULT_FILE_LOCK_TIMEOUT = 30.0

    def __init__(
        self,
        conn: sqlite3.Connection,
        collection: "chromadb.Collection",
        embedding_service: "EmbeddingService",
        config_manager: "ConfigManager",
        manager: Any,  # KnowledgeManager - using Any to avoid circular import
    ) -> None:
        """Initialize the sync manager.

        Args:
            conn: SQLite database connection.
            collection: ChromaDB collection.
            embedding_service: Service for generating embeddings.
            config_manager: Configuration manager instance.
            manager: KnowledgeManager instance for delegating operations.
        """
        self.conn = conn
        self.collection = collection
        self._embedding = embedding_service
        self._config = config_manager
        self._manager = manager

    @contextmanager
    def _file_lock(
        self, lock_path: Path, timeout: float | None = None
    ) -> Iterator[None]:
        """Acquire an exclusive file lock for sync operations.

        Uses fcntl on Unix systems for proper file locking.
        Falls back to a simple lock file mechanism on Windows.

        Args:
            lock_path: Path to the lock file.
            timeout: Maximum seconds to wait for lock.
                Defaults to DEFAULT_FILE_LOCK_TIMEOUT.

        Yields:
            None when lock is acquired.

        Raises:
            TimeoutError: If lock cannot be acquired within timeout.
        """
        if timeout is None:
            timeout = self.DEFAULT_FILE_LOCK_TIMEOUT
        lock_file = lock_path.with_suffix(".lock")
        start_time = time.time()

        if HAS_FCNTL:
            # Unix: Use fcntl for proper file locking
            lock_fd = open(lock_file, "w")
            try:
                while True:
                    try:
                        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        if time.time() - start_time > timeout:
                            lock_fd.close()
                            raise TimeoutError(
                                f"Could not acquire lock on {lock_file} "
                                f"within {timeout} seconds"
                            ) from None
                        time.sleep(0.1)
                try:
                    yield
                finally:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            finally:
                lock_fd.close()
        else:
            # Windows: Simple lock file mechanism
            while lock_file.exists():
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Could not acquire lock on {lock_file} "
                        f"within {timeout} seconds"
                    )
                time.sleep(0.1)
            try:
                lock_file.write_text(str(datetime.now().isoformat()))
                yield
            finally:
                try:
                    lock_file.unlink()
                except OSError:
                    pass  # Lock file may already be removed

    def init_sync_dir(self, sync_path: str | Path) -> None:
        """Initialize a sync directory structure.

        Args:
            sync_path: Path to the sync directory.

        Raises:
            TimeoutError: If unable to acquire lock within timeout.
        """
        sync_path = Path(sync_path).expanduser()
        sync_path.mkdir(parents=True, exist_ok=True)

        # Use file lock to prevent race conditions during initialization
        with self._file_lock(sync_path / "manifest.json"):
            (sync_path / "entries").mkdir(exist_ok=True)
            (sync_path / "tombstones").mkdir(exist_ok=True)

            # Create manifest if it doesn't exist
            manifest_path = sync_path / "manifest.json"
            if not manifest_path.exists():
                self._save_manifest(
                    sync_path, {"version": 1, "last_sync": {}, "entries": {}}
                )

            # Create tombstones file if it doesn't exist
            tombstones_path = sync_path / "tombstones" / "deleted.json"
            if not tombstones_path.exists():
                self._save_tombstones(sync_path, {"deletions": []})

    def _load_manifest(self, sync_path: Path) -> dict[str, Any]:
        """Load sync manifest from sync directory.

        Args:
            sync_path: Path to sync directory.

        Returns:
            Manifest dictionary.
        """
        manifest_path = sync_path / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"version": 1, "last_sync": {}, "entries": {}}

    def _save_manifest(self, sync_path: Path, manifest: dict[str, Any]) -> None:
        """Save sync manifest to sync directory.

        Args:
            sync_path: Path to sync directory.
            manifest: Manifest dictionary.
        """
        manifest_path = sync_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _load_tombstones(self, sync_path: Path) -> dict[str, dict[str, str]]:
        """Load tombstones (deletion records) from sync directory.

        Args:
            sync_path: Path to sync directory.

        Returns:
            Dictionary mapping entry ID to deletion info.
        """
        tombstones_path = sync_path / "tombstones" / "deleted.json"
        if tombstones_path.exists():
            try:
                with open(tombstones_path) as f:
                    data = json.load(f)
                    # Convert list to dict keyed by ID for fast lookup
                    return {d["id"]: d for d in data.get("deletions", [])}
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_tombstones(
        self, sync_path: Path, tombstones: dict[str, Any] | list[dict[str, Any]]
    ) -> None:
        """Save tombstones to sync directory.

        Args:
            sync_path: Path to sync directory.
            tombstones: Either a dict keyed by ID, or a dict with "deletions" list.
        """
        tombstones_path = sync_path / "tombstones" / "deleted.json"
        tombstones_path.parent.mkdir(exist_ok=True)

        # Handle both formats
        if isinstance(tombstones, dict) and "deletions" in tombstones:
            data = tombstones
        else:
            # Convert dict keyed by ID to list format
            data = {"deletions": list(tombstones.values())}

        with open(tombstones_path, "w") as f:
            json.dump(data, f, indent=2)

    def _export_entry_for_sync(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Convert a database entry to sync format.

        Args:
            entry: Entry from SQLite.

        Returns:
            Entry in sync format with parsed JSON fields.
        """
        return {
            "id": entry["id"],
            "title": entry["title"],
            "description": entry["description"],
            "content": entry["content"],
            "brief": entry.get("brief"),
            "tags": json_to_tags(entry.get("tags")),
            "context": json_to_context(entry.get("context")),
            "created": entry.get("created"),
            "updated_at": entry.get("updated_at"),
            "last_used": entry.get("last_used"),
            "usage_count": entry.get("usage_count", 0),
            "confidence": entry.get("confidence", 1.0),
            "source": entry.get("source"),
            "project": entry.get("project"),
            "content_hash": compute_content_hash(entry),
        }

    def _get_local_state(self, project: str | None = None) -> dict[str, dict[str, Any]]:
        """Get current state of all local entries.

        Args:
            project: Optional project filter.

        Returns:
            Dictionary mapping entry ID to state info (hash, updated_at).
        """
        entries = self._manager.export_all(project=project)
        return {
            entry["id"]: {
                "content_hash": compute_content_hash(entry),
                "updated_at": entry.get("updated_at") or entry.get("created"),
            }
            for entry in entries
        }

    def _get_remote_state(self, sync_path: Path) -> dict[str, dict[str, Any]]:
        """Get current state of all remote entries in sync directory.

        Args:
            sync_path: Path to sync directory.

        Returns:
            Dictionary mapping entry ID to state info (hash, updated_at).
        """
        entries_dir = sync_path / "entries"
        state = {}

        if not entries_dir.exists():
            return state

        for entry_file in entries_dir.glob("*.json"):
            try:
                with open(entry_file) as f:
                    entry = json.load(f)
                    entry_id = entry.get("id")
                    if entry_id:
                        state[entry_id] = {
                            "content_hash": entry.get("content_hash")
                            or compute_content_hash(entry),
                            "updated_at": entry.get("updated_at") or entry.get("created"),
                        }
            except (json.JSONDecodeError, OSError):
                continue

        return state

    def _read_remote_entry(self, sync_path: Path, entry_id: str) -> dict[str, Any] | None:
        """Read a single entry from the sync directory.

        Args:
            sync_path: Path to sync directory.
            entry_id: ID of entry to read.

        Returns:
            Entry dictionary or None if not found.
        """
        entry_path = sync_path / "entries" / f"{entry_id}.json"
        if entry_path.exists():
            try:
                with open(entry_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _write_remote_entry(self, sync_path: Path, entry: dict[str, Any]) -> None:
        """Write an entry to the sync directory.

        Args:
            sync_path: Path to sync directory.
            entry: Entry to write.
        """
        entry_path = sync_path / "entries" / f"{entry['id']}.json"
        with open(entry_path, "w") as f:
            json.dump(entry, f, indent=2)

    def _delete_remote_entry(self, sync_path: Path, entry_id: str) -> None:
        """Delete an entry from the sync directory.

        Args:
            sync_path: Path to sync directory.
            entry_id: ID of entry to delete.
        """
        entry_path = sync_path / "entries" / f"{entry_id}.json"
        if entry_path.exists():
            entry_path.unlink()

    def sync_status(
        self,
        sync_path: str | Path | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:
        """Get sync status without making changes.

        Args:
            sync_path: Path to sync directory (uses saved path if None).
            project: Optional project filter.

        Returns:
            Dictionary with pending changes.
        """
        if sync_path is None:
            sync_path = self._config.get_sync_path()
            if sync_path is None:
                raise ValueError("No sync path configured. Run sync with a path first.")
        else:
            sync_path = Path(sync_path).expanduser()

        if not sync_path.exists():
            return {"error": "Sync directory does not exist"}

        manifest = self._load_manifest(sync_path)
        tombstones = self._load_tombstones(sync_path)
        local_state = self._get_local_state(project=project)
        remote_state = self._get_remote_state(sync_path)
        manifest_entries = manifest.get("entries", {})

        to_push = []
        to_pull = []
        conflicts = []
        to_delete_local = []
        to_delete_remote = []

        all_ids = set(local_state.keys()) | set(remote_state.keys())

        for entry_id in all_ids:
            local = local_state.get(entry_id)
            remote = remote_state.get(entry_id)
            manifest_entry = manifest_entries.get(entry_id)
            tombstone = tombstones.get(entry_id)

            action = self._categorize_entry(entry_id, local, remote, manifest_entry, tombstone)

            if action == "push":
                to_push.append(entry_id)
            elif action == "pull":
                to_pull.append(entry_id)
            elif action == "conflict":
                conflicts.append(entry_id)
            elif action == "delete_local":
                to_delete_local.append(entry_id)
            elif action == "delete_remote":
                to_delete_remote.append(entry_id)

        return {
            "to_push": to_push,
            "to_pull": to_pull,
            "conflicts": conflicts,
            "to_delete_local": to_delete_local,
            "to_delete_remote": to_delete_remote,
            "sync_path": str(sync_path),
        }

    def _categorize_entry(
        self,
        entry_id: str,
        local_state: dict[str, Any] | None,
        remote_state: dict[str, Any] | None,
        manifest_state: dict[str, Any] | None,
        tombstone: dict[str, str] | None,
    ) -> str:
        """Categorize an entry for sync action.

        Returns one of: "no_change", "push", "pull", "conflict",
        "delete_local", "delete_remote", "skip"
        """
        local_exists = local_state is not None
        remote_exists = remote_state is not None
        was_synced = manifest_state is not None
        tombstone_time = tombstone.get("deleted_at") if tombstone else None

        if local_exists and remote_exists:
            local_hash = local_state["content_hash"]
            remote_hash = remote_state["content_hash"]

            if local_hash == remote_hash:
                return "no_change"

            manifest_hash = manifest_state["content_hash"] if was_synced else None
            local_changed = local_hash != manifest_hash
            remote_changed = remote_hash != manifest_hash

            if local_changed and not remote_changed:
                return "push"
            elif remote_changed and not local_changed:
                return "pull"
            else:
                return "conflict"

        elif local_exists and not remote_exists:
            if tombstone_time:
                local_updated = local_state.get("updated_at", "")
                if tombstone_time > local_updated:
                    return "delete_local"
            return "push"

        elif remote_exists and not local_exists:
            if tombstone_time:
                remote_updated = remote_state.get("updated_at", "")
                if tombstone_time > remote_updated:
                    return "delete_remote"
            return "pull"

        return "skip"

    def sync(
        self,
        sync_path: str | Path | None = None,
        strategy: str = "last-write-wins",
        push_only: bool = False,
        pull_only: bool = False,
        dry_run: bool = False,
        project: str | None = None,
    ) -> SyncResult:
        """Synchronize with a sync directory.

        Args:
            sync_path: Path to sync directory (uses saved path if None).
            strategy: Conflict resolution strategy ("last-write-wins", "local-wins",
                     "remote-wins", "manual").
            push_only: Only push local changes to sync directory.
            pull_only: Only pull remote changes from sync directory.
            dry_run: Show what would be synced without making changes.
            project: Optional project filter.

        Returns:
            SyncResult with counts and any conflicts.

        Raises:
            TimeoutError: If unable to acquire lock within timeout.
        """
        result = SyncResult()

        # Resolve sync path
        if sync_path is None:
            sync_path = self._config.get_sync_path()
            if sync_path is None:
                result.errors.append("No sync path configured. Provide a path argument.")
                return result
        else:
            sync_path = Path(sync_path).expanduser()

        # Initialize sync directory if needed
        manifest_exists = (sync_path / "manifest.json").exists()
        if not sync_path.exists() or not manifest_exists:
            if dry_run:
                if not sync_path.exists():
                    result.errors.append(f"Sync directory does not exist: {sync_path}")
                else:
                    result.errors.append(f"Sync directory not initialized: {sync_path}")
                return result
            self.init_sync_dir(sync_path)

        # Save sync path for future use
        if not dry_run:
            self._config.set_sync_path(sync_path)

        # Acquire file lock to prevent concurrent sync operations
        with self._file_lock(sync_path / "manifest.json"):
            manifest = self._load_manifest(sync_path)
            tombstones = self._load_tombstones(sync_path)
            local_state = self._get_local_state(project=project)
            remote_state = self._get_remote_state(sync_path)
            local_sync_state = self._config.get_local_sync_state()  # What this machine last synced
            manifest_entries = manifest.get("entries", {})

            all_ids = set(local_state.keys()) | set(remote_state.keys())

            for entry_id in all_ids:
                local = local_state.get(entry_id)
                remote = remote_state.get(entry_id)
                last_synced = local_sync_state.get(entry_id)  # Use local sync state
                tombstone = tombstones.get(entry_id)

                action = self._categorize_entry(
                    entry_id, local, remote, last_synced, tombstone
                )

                if action == "no_change":
                    continue
                elif action == "push" and not pull_only:
                    self._handle_push(
                        sync_path,
                        entry_id,
                        manifest_entries,
                        local_sync_state,
                        dry_run,
                        result,
                    )
                elif action == "pull" and not push_only:
                    self._handle_pull(
                        sync_path,
                        entry_id,
                        manifest_entries,
                        local_sync_state,
                        dry_run,
                        result,
                    )
                elif action == "conflict":
                    self._handle_conflict(
                        sync_path,
                        entry_id,
                        local,
                        remote,
                        manifest_entries,
                        local_sync_state,
                        strategy,
                        push_only,
                        pull_only,
                        dry_run,
                        result,
                    )
                elif action == "delete_local" and not push_only:
                    self._handle_delete_local(entry_id, local_sync_state, dry_run, result)
                elif action == "delete_remote" and not pull_only:
                    self._handle_delete_remote(
                        sync_path,
                        entry_id,
                        manifest_entries,
                        local_sync_state,
                        dry_run,
                        result,
                    )

            # Update manifest and local sync state
            if not dry_run:
                machine_id = get_machine_id()
                manifest["last_sync"][machine_id] = datetime.now().isoformat()
                manifest["entries"] = manifest_entries
                self._save_manifest(sync_path, manifest)
                self._config.set_local_sync_state(local_sync_state)

        return result

    def _handle_push(
        self,
        sync_path: Path,
        entry_id: str,
        manifest_entries: dict[str, Any],
        local_sync_state: dict[str, Any],
        dry_run: bool,
        result: SyncResult,
    ) -> None:
        """Handle pushing a local entry to sync directory."""
        entry = self._manager.get(entry_id)
        if not entry:
            return

        sync_entry = self._export_entry_for_sync(entry)

        if not dry_run:
            self._write_remote_entry(sync_path, sync_entry)
            state_update = {
                "content_hash": sync_entry["content_hash"],
                "updated_at": sync_entry["updated_at"],
            }
            manifest_entries[entry_id] = state_update
            local_sync_state[entry_id] = state_update

        result.pushed += 1

    def _handle_pull(
        self,
        sync_path: Path,
        entry_id: str,
        manifest_entries: dict[str, Any],
        local_sync_state: dict[str, Any],
        dry_run: bool,
        result: SyncResult,
    ) -> None:
        """Handle pulling a remote entry to local database."""
        remote_entry = self._read_remote_entry(sync_path, entry_id)
        if not remote_entry:
            return

        if not dry_run:
            # Check if entry exists locally (update vs insert)
            existing = self._manager.get(entry_id)
            if existing:
                # Update existing entry
                self._manager.update(
                    entry_id,
                    title=remote_entry["title"],
                    description=remote_entry["description"],
                    content=remote_entry["content"],
                    tags=remote_entry.get("tags"),
                    context=remote_entry.get("context"),
                    project=remote_entry.get("project"),
                    confidence=remote_entry.get("confidence", 1.0),
                )
            else:
                # Import new entry, preserving the original ID
                self._import_entry_with_id(remote_entry)

            state_update = {
                "content_hash": remote_entry.get("content_hash")
                or compute_content_hash(remote_entry),
                "updated_at": remote_entry.get("updated_at"),
            }
            manifest_entries[entry_id] = state_update
            local_sync_state[entry_id] = state_update

        result.pulled += 1

    def _import_entry_with_id(self, entry: dict[str, Any]) -> None:
        """Import an entry preserving its original ID.

        Args:
            entry: Entry dictionary from sync format.
        """
        # Generate embedding
        embedding_text = self._embedding.create_embedding_text(
            entry["title"],
            entry["description"],
            entry["content"],
        )
        embedding = self._embedding.generate_embedding(embedding_text)

        # Store in ChromaDB with original ID
        metadata = {
            "title": entry["title"],
            "project": entry.get("project") or "",
        }
        self.collection.add(
            ids=[entry["id"]],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[embedding_text],
        )

        # Store in SQLite with original ID
        brief = create_brief(entry["content"])
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO knowledge (
                id, title, description, content, brief, tags, context,
                created, updated_at, source, project, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["id"],
                entry["title"],
                entry["description"],
                entry["content"],
                brief,
                tags_to_json(entry.get("tags")),
                context_to_json(entry.get("context")),
                entry.get("created") or datetime.now().isoformat(),
                entry.get("updated_at") or datetime.now().isoformat(),
                entry.get("source", "sync"),
                entry.get("project"),
                entry.get("confidence", 1.0),
            ),
        )
        self.conn.commit()

    def _handle_conflict(
        self,
        sync_path: Path,
        entry_id: str,
        local_state: dict[str, Any],
        remote_state: dict[str, Any],
        manifest_entries: dict[str, Any],
        local_sync_state: dict[str, Any],
        strategy: str,
        push_only: bool,
        pull_only: bool,
        dry_run: bool,
        result: SyncResult,
    ) -> None:
        """Handle a sync conflict."""
        local_entry = self._manager.get(entry_id)
        remote_entry = self._read_remote_entry(sync_path, entry_id)

        if not local_entry or not remote_entry:
            return

        local_updated = local_state.get("updated_at", "")
        remote_updated = remote_state.get("updated_at", "")

        conflict_info = {
            "id": entry_id,
            "title": local_entry.get("title", ""),
            "local_updated": local_updated,
            "remote_updated": remote_updated,
            "resolution": "pending",
        }

        if strategy == "manual":
            conflict_info["resolution"] = "manual"
            result.conflicts.append(conflict_info)
            return

        # Determine winner based on strategy
        if strategy == "local-wins":
            use_local = True
        elif strategy == "remote-wins":
            use_local = False
        else:  # last-write-wins
            use_local = local_updated >= remote_updated

        if use_local and not pull_only:
            conflict_info["resolution"] = "local"
            self._handle_push(
                sync_path, entry_id, manifest_entries, local_sync_state, dry_run, result
            )
        elif not use_local and not push_only:
            conflict_info["resolution"] = "remote"
            self._handle_pull(
                sync_path, entry_id, manifest_entries, local_sync_state, dry_run, result
            )
        else:
            conflict_info["resolution"] = "skipped"

        result.conflicts.append(conflict_info)

    def _handle_delete_local(
        self,
        entry_id: str,
        local_sync_state: dict[str, Any],
        dry_run: bool,
        result: SyncResult,
    ) -> None:
        """Handle deleting a local entry that was deleted remotely."""
        if not dry_run:
            self._manager.delete(entry_id)
            if entry_id in local_sync_state:
                del local_sync_state[entry_id]
        result.deletions_pulled += 1

    def _handle_delete_remote(
        self,
        sync_path: Path,
        entry_id: str,
        manifest_entries: dict[str, Any],
        local_sync_state: dict[str, Any],
        dry_run: bool,
        result: SyncResult,
    ) -> None:
        """Handle deleting a remote entry that was deleted locally."""
        if not dry_run:
            self._delete_remote_entry(sync_path, entry_id)
            if entry_id in manifest_entries:
                del manifest_entries[entry_id]
            if entry_id in local_sync_state:
                del local_sync_state[entry_id]
        result.deletions_pushed += 1
