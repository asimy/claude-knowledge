"""Versioning service for knowledge entry history and rollback."""

import difflib
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any

from claude_knowledge.utils import generate_id

if TYPE_CHECKING:
    import chromadb

    from claude_knowledge._embedding import EmbeddingService


class VersioningService:
    """Handles version history for knowledge entries.

    This service provides methods for creating snapshots, viewing history,
    rolling back to previous versions, and comparing versions.
    """

    # Default maximum versions to retain per entry
    DEFAULT_MAX_VERSIONS = 50

    def __init__(
        self,
        conn: sqlite3.Connection,
        collection: "chromadb.Collection",
        embedding_service: "EmbeddingService",
        manager: Any,  # KnowledgeManager - using Any to avoid circular import
        max_versions: int | None = None,
    ) -> None:
        """Initialize the versioning service.

        Args:
            conn: SQLite database connection.
            collection: ChromaDB collection.
            embedding_service: Service for generating embeddings.
            manager: KnowledgeManager instance for delegating operations.
            max_versions: Maximum versions to retain per entry.
                         Defaults to DEFAULT_MAX_VERSIONS.
        """
        self.conn = conn
        self.collection = collection
        self._embedding = embedding_service
        self._manager = manager
        self.max_versions = max_versions or self.DEFAULT_MAX_VERSIONS

    def create_version(
        self,
        entry_id: str,
        created_by: str | None = None,
        change_summary: str | None = None,
    ) -> str | None:
        """Create a version snapshot of the current entry state.

        Called before an update to preserve the previous state.

        Args:
            entry_id: ID of the entry to version.
            created_by: Optional identifier of who created this version.
            change_summary: Optional description of what changed.

        Returns:
            The version ID if created, None if entry not found.
        """
        # Get current entry state
        entry = self._manager.get(entry_id)
        if not entry:
            return None

        # Get next version number
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT MAX(version_number) FROM entry_versions WHERE entry_id = ?",
            (entry_id,),
        )
        result = cursor.fetchone()
        next_version = (result[0] or 0) + 1

        # Generate version ID
        version_id = generate_id(f"v{next_version}-{entry_id}", datetime.now())

        # Store the snapshot
        cursor.execute(
            """
            INSERT INTO entry_versions (
                id, entry_id, version_number, title, description, content,
                brief, tags, context, confidence, source, project,
                created_at, created_by, change_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                entry_id,
                next_version,
                entry["title"],
                entry["description"],
                entry["content"],
                entry.get("brief"),
                entry.get("tags"),
                entry.get("context"),
                entry.get("confidence", 1.0),
                entry.get("source"),
                entry.get("project"),
                datetime.now().isoformat(),
                created_by,
                change_summary,
            ),
        )
        self.conn.commit()

        # Prune old versions if over limit
        self._prune_old_versions(entry_id)

        return version_id

    def get_version(self, entry_id: str, version_number: int) -> dict[str, Any] | None:
        """Retrieve a specific version of an entry.

        Args:
            entry_id: ID of the entry.
            version_number: Version number to retrieve.

        Returns:
            Version dictionary or None if not found.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM entry_versions
            WHERE entry_id = ? AND version_number = ?
            """,
            (entry_id, version_number),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_history(
        self,
        entry_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get version history for an entry.

        Args:
            entry_id: ID of the entry.
            limit: Maximum number of versions to return.

        Returns:
            List of version summaries, ordered by version number descending
            (newest first).
        """
        cursor = self.conn.cursor()

        query = """
            SELECT id, entry_id, version_number, title, created_at,
                   created_by, change_summary
            FROM entry_versions
            WHERE entry_id = ?
            ORDER BY version_number DESC
        """
        params: list[Any] = [entry_id]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def rollback(
        self,
        entry_id: str,
        version_number: int,
        created_by: str | None = None,
    ) -> bool:
        """Restore an entry to a previous version.

        Creates a new version capturing the current state before rolling back.
        Regenerates embeddings for the restored content.

        Args:
            entry_id: ID of the entry to rollback.
            version_number: Version number to restore to.
            created_by: Optional identifier of who initiated the rollback.

        Returns:
            True if rollback succeeded, False if entry or version not found.
        """
        # Get the target version
        version = self.get_version(entry_id, version_number)
        if not version:
            return False

        # Check entry still exists
        current = self._manager.get(entry_id)
        if not current:
            return False

        # Create a version of current state before rollback
        self.create_version(
            entry_id,
            created_by=created_by,
            change_summary=f"Auto-saved before rollback to version {version_number}",
        )

        # Update entry with version data
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE knowledge SET
                title = ?,
                description = ?,
                content = ?,
                brief = ?,
                tags = ?,
                context = ?,
                confidence = ?,
                source = ?,
                project = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                version["title"],
                version["description"],
                version["content"],
                version.get("brief"),
                version.get("tags"),
                version.get("context"),
                version.get("confidence", 1.0),
                version.get("source"),
                version.get("project"),
                datetime.now().isoformat(),
                entry_id,
            ),
        )
        self.conn.commit()

        # Regenerate embedding for restored content
        embedding_text = self._embedding.create_embedding_text(
            version["title"],
            version["description"],
            version["content"],
        )
        embedding = self._embedding.generate_embedding(embedding_text)

        self.collection.update(
            ids=[entry_id],
            embeddings=[embedding],
            metadatas=[{"title": version["title"], "project": version.get("project", "")}],
            documents=[embedding_text],
        )

        return True

    def diff(
        self,
        entry_id: str,
        version_a: int,
        version_b: int | None = None,
    ) -> dict[str, Any]:
        """Generate a diff between two versions.

        Args:
            entry_id: ID of the entry.
            version_a: First version number.
            version_b: Second version number. If None, compares to current state.

        Returns:
            Dictionary containing:
                - 'version_a': First version info
                - 'version_b': Second version info (or 'current' if None)
                - 'title_diff': Unified diff of title (if changed)
                - 'description_diff': Unified diff of description (if changed)
                - 'content_diff': Unified diff of content (if changed)
                - 'tags_changed': True if tags differ
                - 'project_changed': True if project differs

        Raises:
            ValueError: If entry or version not found.
        """
        # Get first version
        ver_a = self.get_version(entry_id, version_a)
        if not ver_a:
            raise ValueError(f"Version {version_a} not found for entry {entry_id}")

        # Get second version or current state
        if version_b is not None:
            ver_b = self.get_version(entry_id, version_b)
            if not ver_b:
                raise ValueError(f"Version {version_b} not found for entry {entry_id}")
            version_b_label = f"v{version_b}"
        else:
            ver_b = self._manager.get(entry_id)
            if not ver_b:
                raise ValueError(f"Entry {entry_id} not found")
            version_b_label = "current"

        result: dict[str, Any] = {
            "version_a": {
                "number": version_a,
                "created_at": ver_a.get("created_at"),
            },
            "version_b": {
                "number": version_b,
                "label": version_b_label,
            },
        }

        # Generate diffs for text fields
        result["title_diff"] = self._generate_diff(
            ver_a["title"], ver_b["title"], f"v{version_a}", version_b_label
        )
        result["description_diff"] = self._generate_diff(
            ver_a["description"], ver_b["description"], f"v{version_a}", version_b_label
        )
        result["content_diff"] = self._generate_diff(
            ver_a["content"], ver_b["content"], f"v{version_a}", version_b_label
        )

        # Check for simple changes
        result["tags_changed"] = ver_a.get("tags") != ver_b.get("tags")
        result["project_changed"] = ver_a.get("project") != ver_b.get("project")
        result["confidence_changed"] = ver_a.get("confidence") != ver_b.get("confidence")

        return result

    def _generate_diff(
        self,
        text_a: str,
        text_b: str,
        label_a: str,
        label_b: str,
    ) -> str | None:
        """Generate unified diff between two text strings.

        Args:
            text_a: First text.
            text_b: Second text.
            label_a: Label for first text (e.g., "v1").
            label_b: Label for second text (e.g., "v2").

        Returns:
            Unified diff string, or None if texts are identical.
        """
        if text_a == text_b:
            return None

        lines_a = (text_a or "").splitlines(keepends=True)
        lines_b = (text_b or "").splitlines(keepends=True)

        diff = difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=label_a,
            tofile=label_b,
            lineterm="",
        )

        return "".join(diff)

    def _prune_old_versions(self, entry_id: str) -> int:
        """Remove oldest versions if over the retention limit.

        Args:
            entry_id: ID of the entry.

        Returns:
            Number of versions deleted.
        """
        cursor = self.conn.cursor()

        # Count versions
        cursor.execute(
            "SELECT COUNT(*) FROM entry_versions WHERE entry_id = ?",
            (entry_id,),
        )
        count = cursor.fetchone()[0]

        if count <= self.max_versions:
            return 0

        # Get IDs of versions to delete (oldest ones)
        versions_to_delete = count - self.max_versions
        cursor.execute(
            """
            SELECT id FROM entry_versions
            WHERE entry_id = ?
            ORDER BY version_number ASC
            LIMIT ?
            """,
            (entry_id, versions_to_delete),
        )
        ids_to_delete = [row[0] for row in cursor.fetchall()]

        # Delete old versions
        if ids_to_delete:
            placeholders = ",".join("?" * len(ids_to_delete))
            cursor.execute(
                f"DELETE FROM entry_versions WHERE id IN ({placeholders})",
                ids_to_delete,
            )
            self.conn.commit()

        return len(ids_to_delete)

    def get_version_count(self, entry_id: str) -> int:
        """Get the number of versions for an entry.

        Args:
            entry_id: ID of the entry.

        Returns:
            Number of versions.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM entry_versions WHERE entry_id = ?",
            (entry_id,),
        )
        return cursor.fetchone()[0]

    def get_latest_version_number(self, entry_id: str) -> int | None:
        """Get the latest version number for an entry.

        Args:
            entry_id: ID of the entry.

        Returns:
            Latest version number, or None if no versions exist.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT MAX(version_number) FROM entry_versions WHERE entry_id = ?",
            (entry_id,),
        )
        result = cursor.fetchone()[0]
        return result
