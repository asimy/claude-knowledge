"""Relationships service for knowledge base entry linking and collections."""

import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any

from claude_knowledge.utils import generate_id

if TYPE_CHECKING:
    import chromadb

    from claude_knowledge._embedding import EmbeddingService


class RelationshipsService:
    """Handles relationships between knowledge entries and collections.

    This service provides methods for:
    - Linking entries with typed relationships (related, depends-on, supersedes)
    - Managing named collections of entries
    - Querying dependency trees and related entries
    """

    RELATIONSHIP_TYPES = frozenset({"related", "depends-on", "supersedes"})

    # Limits
    MAX_COLLECTION_NAME_LENGTH = 200
    MAX_COLLECTION_DESCRIPTION_LENGTH = 2000
    DEFAULT_LIST_LIMIT = 50
    MAX_DEPENDENCY_DEPTH = 10

    def __init__(
        self,
        conn: sqlite3.Connection,
        collection: "chromadb.Collection",
        embedding_service: "EmbeddingService",
        manager: Any,  # KnowledgeManager - using Any to avoid circular import
    ) -> None:
        """Initialize the relationships service.

        Args:
            conn: SQLite database connection.
            collection: ChromaDB collection.
            embedding_service: Service for generating embeddings.
            manager: KnowledgeManager instance for delegating operations.
        """
        self.conn = conn
        self.collection = collection
        self._embedding = embedding_service
        self._manager = manager

    # =========================================================================
    # Entry Linking
    # =========================================================================

    def link(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str = "related",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a relationship between two entries.

        For 'related' type (bidirectional), the relationship is stored with
        the lexicographically smaller ID as source to ensure uniqueness.

        Args:
            source_id: ID of the source entry.
            target_id: ID of the target entry.
            relationship_type: Type of relationship ('related', 'depends-on', 'supersedes').
            metadata: Optional metadata for the relationship.

        Returns:
            The relationship ID.

        Raises:
            ValueError: If relationship type is invalid, IDs are the same,
                       or either entry doesn't exist.
        """
        # Validate relationship type
        if relationship_type not in self.RELATIONSHIP_TYPES:
            allowed = ", ".join(sorted(self.RELATIONSHIP_TYPES))
            raise ValueError(
                f"Invalid relationship_type '{relationship_type}'. Must be one of: {allowed}"
            )

        # Validate IDs are different
        if source_id == target_id:
            raise ValueError("Cannot create a relationship between an entry and itself")

        # Validate entries exist
        source_entry = self._manager.get(source_id)
        if not source_entry:
            raise ValueError(f"Source entry not found: {source_id}")

        target_entry = self._manager.get(target_id)
        if not target_entry:
            raise ValueError(f"Target entry not found: {target_id}")

        # For bidirectional 'related' type, normalize by using smaller ID as source
        if relationship_type == "related":
            if source_id > target_id:
                source_id, target_id = target_id, source_id

        # Generate relationship ID
        timestamp = datetime.now()
        rel_id = generate_id(f"{source_id}-{target_id}-{relationship_type}", timestamp)

        # Serialize metadata
        metadata_json = json.dumps(metadata) if metadata else None

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO knowledge_relationships (
                    id, source_id, target_id, relationship_type, created, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    rel_id,
                    source_id,
                    target_id,
                    relationship_type,
                    timestamp.isoformat(),
                    metadata_json,
                ),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Relationship already exists
            cursor.execute(
                """
                SELECT id FROM knowledge_relationships
                WHERE source_id = ? AND target_id = ? AND relationship_type = ?
                """,
                (source_id, target_id, relationship_type),
            )
            row = cursor.fetchone()
            if row:
                return row[0]
            raise

        return rel_id

    def unlink(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str | None = None,
    ) -> bool:
        """Remove a relationship between two entries.

        For 'related' type, checks both orderings since it's bidirectional.

        Args:
            source_id: ID of the source entry.
            target_id: ID of the target entry.
            relationship_type: Optional type filter. If None, removes all relationships
                              between the two entries.

        Returns:
            True if any relationship was removed, False if none found.
        """
        cursor = self.conn.cursor()
        deleted = False

        if relationship_type:
            # For 'related' type, normalize order
            if relationship_type == "related":
                if source_id > target_id:
                    source_id, target_id = target_id, source_id

            cursor.execute(
                """
                DELETE FROM knowledge_relationships
                WHERE source_id = ? AND target_id = ? AND relationship_type = ?
                """,
                (source_id, target_id, relationship_type),
            )
            deleted = cursor.rowcount > 0
        else:
            # Remove all relationships between the two entries (both directions)
            cursor.execute(
                """
                DELETE FROM knowledge_relationships
                WHERE (source_id = ? AND target_id = ?)
                   OR (source_id = ? AND target_id = ?)
                """,
                (source_id, target_id, target_id, source_id),
            )
            deleted = cursor.rowcount > 0

        self.conn.commit()
        return deleted

    def get_related(
        self,
        entry_id: str,
        relationship_type: str | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Get entries related to the given entry.

        Args:
            entry_id: ID of the entry to find relationships for.
            relationship_type: Optional filter by relationship type.
            direction: Direction filter ('outgoing', 'incoming', 'both').
                      - 'outgoing': Entry is the source
                      - 'incoming': Entry is the target
                      - 'both': Either direction (default)

        Returns:
            List of related entries with relationship info.
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(f"Invalid direction '{direction}'. Must be: outgoing, incoming, both")

        cursor = self.conn.cursor()
        results = []

        # Build query based on direction
        if direction in ("outgoing", "both"):
            query = """
                SELECT r.id, r.source_id, r.target_id, r.relationship_type,
                       r.created, r.metadata, 'outgoing' as direction
                FROM knowledge_relationships r
                WHERE r.source_id = ?
            """
            params: list[str] = [entry_id]

            if relationship_type:
                query += " AND r.relationship_type = ?"
                params.append(relationship_type)

            cursor.execute(query, params)
            for row in cursor.fetchall():
                rel = dict(row)
                rel["related_id"] = rel["target_id"]
                rel["metadata"] = json.loads(rel["metadata"]) if rel["metadata"] else None
                results.append(rel)

        if direction in ("incoming", "both"):
            query = """
                SELECT r.id, r.source_id, r.target_id, r.relationship_type,
                       r.created, r.metadata, 'incoming' as direction
                FROM knowledge_relationships r
                WHERE r.target_id = ?
            """
            params = [entry_id]

            if relationship_type:
                query += " AND r.relationship_type = ?"
                params.append(relationship_type)

            cursor.execute(query, params)
            for row in cursor.fetchall():
                rel = dict(row)
                rel["related_id"] = rel["source_id"]
                rel["metadata"] = json.loads(rel["metadata"]) if rel["metadata"] else None
                results.append(rel)

        # Fetch entry details for related entries
        related_ids = list({r["related_id"] for r in results})
        if related_ids:
            placeholders = ",".join("?" * len(related_ids))
            cursor.execute(
                f"SELECT id, title, project FROM knowledge WHERE id IN ({placeholders})",
                related_ids,
            )
            entry_info = {row["id"]: dict(row) for row in cursor.fetchall()}

            for rel in results:
                info = entry_info.get(rel["related_id"], {})
                rel["related_title"] = info.get("title")
                rel["related_project"] = info.get("project")

        return results

    def get_dependency_tree(
        self,
        entry_id: str,
        depth: int = 3,
    ) -> dict[str, Any]:
        """Get the dependency tree for an entry.

        Traverses 'depends-on' relationships to build a tree of dependencies.

        Args:
            entry_id: ID of the entry to get dependencies for.
            depth: Maximum depth to traverse (default 3, max 10).

        Returns:
            Dictionary with entry info and nested dependencies.
        """
        if depth < 1:
            depth = 1
        if depth > self.MAX_DEPENDENCY_DEPTH:
            depth = self.MAX_DEPENDENCY_DEPTH

        entry = self._manager.get(entry_id)
        if not entry:
            return {}

        return self._build_dependency_tree(entry_id, entry["title"], depth, set())

    def _build_dependency_tree(
        self,
        entry_id: str,
        title: str,
        remaining_depth: int,
        visited: set[str],
    ) -> dict[str, Any]:
        """Recursively build dependency tree."""
        if entry_id in visited:
            return {"id": entry_id, "title": title, "circular": True}

        visited.add(entry_id)

        node: dict[str, Any] = {
            "id": entry_id,
            "title": title,
            "dependencies": [],
        }

        if remaining_depth <= 0:
            return node

        # Get outgoing 'depends-on' relationships
        deps = self.get_related(entry_id, relationship_type="depends-on", direction="outgoing")

        for dep in deps:
            child = self._build_dependency_tree(
                dep["related_id"],
                dep.get("related_title", ""),
                remaining_depth - 1,
                visited.copy(),
            )
            node["dependencies"].append(child)

        return node

    def get_relationship(self, relationship_id: str) -> dict[str, Any] | None:
        """Get a relationship by ID.

        Args:
            relationship_id: The relationship ID.

        Returns:
            Relationship dictionary or None if not found.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, source_id, target_id, relationship_type, created, metadata
            FROM knowledge_relationships
            WHERE id = ?
            """,
            (relationship_id,),
        )
        row = cursor.fetchone()
        if row:
            rel = dict(row)
            rel["metadata"] = json.loads(rel["metadata"]) if rel["metadata"] else None
            return rel
        return None

    def get_entry_relationships(self, entry_id: str) -> list[dict[str, Any]]:
        """Get all relationships for an entry (both directions).

        Args:
            entry_id: ID of the entry.

        Returns:
            List of all relationships involving this entry.
        """
        return self.get_related(entry_id, direction="both")

    # =========================================================================
    # Collections
    # =========================================================================

    def create_collection(
        self,
        name: str,
        description: str = "",
    ) -> str:
        """Create a new collection.

        Args:
            name: Unique name for the collection.
            description: Optional description.

        Returns:
            The collection ID.

        Raises:
            ValueError: If name is empty, too long, or already exists.
        """
        if not name or not name.strip():
            raise ValueError("Collection name cannot be empty")

        name = name.strip()
        if len(name) > self.MAX_COLLECTION_NAME_LENGTH:
            raise ValueError(
                f"Collection name exceeds maximum length of {self.MAX_COLLECTION_NAME_LENGTH}"
            )

        if len(description) > self.MAX_COLLECTION_DESCRIPTION_LENGTH:
            raise ValueError(
                f"Collection description exceeds maximum length of "
                f"{self.MAX_COLLECTION_DESCRIPTION_LENGTH}"
            )

        timestamp = datetime.now()
        collection_id = generate_id(f"collection-{name}", timestamp)

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO knowledge_collections (id, name, description, created, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    collection_id,
                    name,
                    description,
                    timestamp.isoformat(),
                    timestamp.isoformat(),
                ),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Collection with name '{name}' already exists") from None

        return collection_id

    def delete_collection(self, collection_id_or_name: str) -> bool:
        """Delete a collection.

        Members are automatically removed due to CASCADE.

        Args:
            collection_id_or_name: Collection ID or name.

        Returns:
            True if deleted, False if not found.
        """
        collection = self.get_collection(collection_id_or_name)
        if not collection:
            return False

        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM knowledge_collections WHERE id = ?",
            (collection["id"],),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_collection(self, collection_id_or_name: str) -> dict[str, Any] | None:
        """Get a collection by ID or name.

        Args:
            collection_id_or_name: Collection ID or name.

        Returns:
            Collection dictionary or None if not found.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, name, description, created, updated_at
            FROM knowledge_collections
            WHERE id = ? OR name = ?
            """,
            (collection_id_or_name, collection_id_or_name),
        )
        row = cursor.fetchone()
        if row:
            collection = dict(row)
            # Add member count
            cursor.execute(
                "SELECT COUNT(*) FROM collection_members WHERE collection_id = ?",
                (collection["id"],),
            )
            collection["member_count"] = cursor.fetchone()[0]
            return collection
        return None

    def list_collections(self, limit: int | None = None) -> list[dict[str, Any]]:
        """List all collections.

        Args:
            limit: Maximum number of collections to return.

        Returns:
            List of collections with member counts.
        """
        if limit is None:
            limit = self.DEFAULT_LIST_LIMIT

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT c.id, c.name, c.description, c.created, c.updated_at,
                   COUNT(m.entry_id) as member_count
            FROM knowledge_collections c
            LEFT JOIN collection_members m ON c.id = m.collection_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_collection(
        self,
        collection_id_or_name: str,
        name: str | None = None,
        description: str | None = None,
    ) -> bool:
        """Update a collection's name or description.

        Args:
            collection_id_or_name: Collection ID or name.
            name: New name (optional).
            description: New description (optional).

        Returns:
            True if updated, False if not found.

        Raises:
            ValueError: If new name is invalid or already exists.
        """
        collection = self.get_collection(collection_id_or_name)
        if not collection:
            return False

        updates = {}
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Collection name cannot be empty")
            if len(name) > self.MAX_COLLECTION_NAME_LENGTH:
                raise ValueError(
                    f"Collection name exceeds maximum length of {self.MAX_COLLECTION_NAME_LENGTH}"
                )
            updates["name"] = name

        if description is not None:
            if len(description) > self.MAX_COLLECTION_DESCRIPTION_LENGTH:
                raise ValueError(
                    f"Collection description exceeds maximum length of "
                    f"{self.MAX_COLLECTION_DESCRIPTION_LENGTH}"
                )
            updates["description"] = description

        if not updates:
            return True

        updates["updated_at"] = datetime.now().isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [collection["id"]]

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                f"UPDATE knowledge_collections SET {set_clause} WHERE id = ?",
                values,
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Collection with name '{name}' already exists") from None

        return True

    def add_to_collection(
        self,
        collection_id_or_name: str,
        entry_id: str,
    ) -> bool:
        """Add an entry to a collection.

        Args:
            collection_id_or_name: Collection ID or name.
            entry_id: ID of the entry to add.

        Returns:
            True if added, False if already a member.

        Raises:
            ValueError: If collection or entry doesn't exist.
        """
        collection = self.get_collection(collection_id_or_name)
        if not collection:
            raise ValueError(f"Collection not found: {collection_id_or_name}")

        entry = self._manager.get(entry_id)
        if not entry:
            raise ValueError(f"Entry not found: {entry_id}")

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO collection_members (collection_id, entry_id, added_at)
                VALUES (?, ?, ?)
                """,
                (collection["id"], entry_id, datetime.now().isoformat()),
            )
            # Update collection's updated_at
            cursor.execute(
                "UPDATE knowledge_collections SET updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), collection["id"]),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Already a member
            return False

    def remove_from_collection(
        self,
        collection_id_or_name: str,
        entry_id: str,
    ) -> bool:
        """Remove an entry from a collection.

        Args:
            collection_id_or_name: Collection ID or name.
            entry_id: ID of the entry to remove.

        Returns:
            True if removed, False if not a member.

        Raises:
            ValueError: If collection doesn't exist.
        """
        collection = self.get_collection(collection_id_or_name)
        if not collection:
            raise ValueError(f"Collection not found: {collection_id_or_name}")

        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM collection_members WHERE collection_id = ? AND entry_id = ?",
            (collection["id"], entry_id),
        )
        if cursor.rowcount > 0:
            # Update collection's updated_at
            cursor.execute(
                "UPDATE knowledge_collections SET updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), collection["id"]),
            )
            self.conn.commit()
            return True
        self.conn.commit()
        return False

    def get_collection_members(
        self,
        collection_id_or_name: str,
    ) -> list[dict[str, Any]]:
        """Get all entries in a collection.

        Args:
            collection_id_or_name: Collection ID or name.

        Returns:
            List of entries in the collection.

        Raises:
            ValueError: If collection doesn't exist.
        """
        collection = self.get_collection(collection_id_or_name)
        if not collection:
            raise ValueError(f"Collection not found: {collection_id_or_name}")

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT k.id, k.title, k.description, k.tags, k.project,
                   k.usage_count, k.created, k.last_used, m.added_at
            FROM knowledge k
            JOIN collection_members m ON k.id = m.entry_id
            WHERE m.collection_id = ?
            ORDER BY m.added_at DESC
            """,
            (collection["id"],),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_entry_collections(self, entry_id: str) -> list[dict[str, Any]]:
        """Get all collections that contain an entry.

        Args:
            entry_id: ID of the entry.

        Returns:
            List of collections containing the entry.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT c.id, c.name, c.description, c.created, c.updated_at, m.added_at
            FROM knowledge_collections c
            JOIN collection_members m ON c.id = m.collection_id
            WHERE m.entry_id = ?
            ORDER BY m.added_at DESC
            """,
            (entry_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def has_relationships_or_collections(self, entry_id: str) -> dict[str, int]:
        """Check if an entry has any relationships or collection memberships.

        Useful for warning before deletion.

        Args:
            entry_id: ID of the entry.

        Returns:
            Dictionary with counts: {'relationships': N, 'collections': N}
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) FROM knowledge_relationships
            WHERE source_id = ? OR target_id = ?
            """,
            (entry_id, entry_id),
        )
        rel_count = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM collection_members WHERE entry_id = ?",
            (entry_id,),
        )
        coll_count = cursor.fetchone()[0]

        return {"relationships": rel_count, "collections": coll_count}
