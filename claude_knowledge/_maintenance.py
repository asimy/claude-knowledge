"""Maintenance service for knowledge base quality and upkeep."""

import sqlite3
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from claude_knowledge.utils import json_to_context, json_to_tags

if TYPE_CHECKING:
    import chromadb

    from claude_knowledge._embedding import EmbeddingService


class MaintenanceService:
    """Handles maintenance operations for the knowledge base.

    This service provides methods for finding duplicates, stale entries,
    scoring quality, merging entries, and generating statistics.
    """

    # Duplicate detection
    DEFAULT_DUPLICATE_THRESHOLD = 0.85
    MAX_DUPLICATE_CHECK_ENTRIES = 1000
    DEFAULT_DUPLICATE_COMPARISON_LIMIT = 20

    # Staleness tracking
    DEFAULT_STALE_DAYS = 90

    # Quality scoring (each component contributes this many points out of 100)
    QUALITY_SCORE_WEIGHT = 25
    MIN_DESCRIPTION_LENGTH_FOR_QUALITY = 50
    MIN_CONTENT_LENGTH_FOR_QUALITY = 100

    def __init__(
        self,
        conn: sqlite3.Connection,
        collection: "chromadb.Collection",
        embedding_service: "EmbeddingService",
        manager: Any,  # KnowledgeManager - using Any to avoid circular import
    ) -> None:
        """Initialize the maintenance service.

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

    def stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge base.

        Returns:
            Dictionary with statistics.
        """
        cursor = self.conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) FROM knowledge")
        total = cursor.fetchone()[0]

        # By project
        cursor.execute("""
            SELECT project, COUNT(*) as count
            FROM knowledge
            GROUP BY project
            ORDER BY count DESC
        """)
        by_project = {row[0] or "(no project)": row[1] for row in cursor.fetchall()}

        # Most used
        cursor.execute("""
            SELECT id, title, usage_count
            FROM knowledge
            ORDER BY usage_count DESC
            LIMIT 5
        """)
        most_used = [
            {"id": row[0], "title": row[1], "usage_count": row[2]} for row in cursor.fetchall()
        ]

        # Recently added
        cursor.execute("""
            SELECT id, title, created
            FROM knowledge
            ORDER BY created DESC
            LIMIT 5
        """)
        recent = [{"id": row[0], "title": row[1], "created": row[2]} for row in cursor.fetchall()]

        # Recently used
        cursor.execute("""
            SELECT id, title, last_used
            FROM knowledge
            WHERE last_used IS NOT NULL
            ORDER BY last_used DESC
            LIMIT 5
        """)
        recently_used = [
            {"id": row[0], "title": row[1], "last_used": row[2]} for row in cursor.fetchall()
        ]

        return {
            "total_entries": total,
            "by_project": by_project,
            "most_used": most_used,
            "recently_added": recent,
            "recently_used": recently_used,
        }

    def find_duplicates(
        self,
        threshold: float | None = None,
        project: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Find potential duplicate entries based on semantic similarity.

        Args:
            threshold: Minimum similarity score (0.0-1.0) to consider as duplicate.
                Defaults to DEFAULT_DUPLICATE_THRESHOLD.
            project: Optional project filter.

        Returns:
            List of duplicate groups, where each group is a list of similar entries
            with their similarity scores.
        """
        if threshold is None:
            threshold = self.DEFAULT_DUPLICATE_THRESHOLD
        entries = self._manager.list_all(project=project, limit=self.MAX_DUPLICATE_CHECK_ENTRIES)
        if len(entries) < 2:
            return []

        # Track which entries have been grouped
        grouped_ids: set[str] = set()
        duplicate_groups: list[list[dict[str, Any]]] = []

        for entry in entries:
            if entry["id"] in grouped_ids:
                continue

            # Query ChromaDB for similar entries
            entry_full = self._manager.get(entry["id"])
            if not entry_full:
                continue

            embedding_text = self._embedding.create_embedding_text(
                entry_full["title"],
                entry_full["description"],
                entry_full["content"],
            )
            query_embedding = self._embedding.generate_embedding(embedding_text)

            # Get more results than needed to find all duplicates
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(len(entries), self.DEFAULT_DUPLICATE_COMPARISON_LIMIT),
                include=["distances"],
            )

            if not results["ids"] or not results["ids"][0]:
                continue

            # Find entries above threshold (excluding self)
            group = [{"id": entry["id"], "title": entry["title"], "similarity": 1.0}]
            ids = results["ids"][0]
            distances = results["distances"][0] if results["distances"] else []

            for i, kid in enumerate(ids):
                if kid == entry["id"] or kid in grouped_ids:
                    continue

                similarity = 1 - distances[i] if i < len(distances) else 0
                if similarity >= threshold:
                    other = self._manager.get(kid)
                    if other:
                        # Apply project filter if specified
                        if project and other.get("project") != project:
                            continue
                        group.append(
                            {
                                "id": kid,
                                "title": other["title"],
                                "similarity": round(similarity, 3),
                            }
                        )

            # Only include groups with actual duplicates
            if len(group) > 1:
                # Mark all in group as processed
                for item in group:
                    grouped_ids.add(item["id"])
                duplicate_groups.append(group)

        return duplicate_groups

    def find_stale(
        self,
        days: int | None = None,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find entries that haven't been used or updated recently.

        Args:
            days: Number of days to consider an entry stale.
                Defaults to DEFAULT_STALE_DAYS.
            project: Optional project filter.

        Returns:
            List of stale entries with staleness info.
        """
        if days is None:
            days = self.DEFAULT_STALE_DAYS
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        cursor = self.conn.cursor()

        # Find entries where both last_used and updated_at are older than cutoff
        # or where last_used is NULL and updated_at (or created) is older than cutoff
        if project:
            cursor.execute(
                """
                SELECT id, title, description, created, updated_at, last_used, usage_count
                FROM knowledge
                WHERE project = ?
                AND (
                    (last_used IS NOT NULL AND last_used < ? AND updated_at < ?)
                    OR (last_used IS NULL AND COALESCE(updated_at, created) < ?)
                )
                ORDER BY COALESCE(last_used, updated_at, created) ASC
                """,
                (project, cutoff, cutoff, cutoff),
            )
        else:
            cursor.execute(
                """
                SELECT id, title, description, created, updated_at, last_used, usage_count
                FROM knowledge
                WHERE (
                    (last_used IS NOT NULL AND last_used < ? AND updated_at < ?)
                    OR (last_used IS NULL AND COALESCE(updated_at, created) < ?)
                )
                ORDER BY COALESCE(last_used, updated_at, created) ASC
                """,
                (cutoff, cutoff, cutoff),
            )

        rows = cursor.fetchall()
        stale_entries = []

        for row in rows:
            entry = dict(row)
            # Calculate days since last activity
            last_activity = (
                entry.get("last_used") or entry.get("updated_at") or entry.get("created")
            )
            if last_activity:
                try:
                    last_dt = datetime.fromisoformat(last_activity)
                    days_stale = (datetime.now() - last_dt).days
                    entry["days_stale"] = days_stale
                except ValueError:
                    entry["days_stale"] = None
            else:
                entry["days_stale"] = None
            stale_entries.append(entry)

        return stale_entries

    def merge_entries(
        self,
        target_id: str,
        source_id: str,
        delete_source: bool = True,
    ) -> bool:
        """Merge two entries, combining their content.

        The source entry's content is appended to the target entry.
        Tags and context are merged. Usage counts are summed.

        Args:
            target_id: ID of the entry to merge into (kept).
            source_id: ID of the entry to merge from (optionally deleted).
            delete_source: Whether to delete the source entry after merging.

        Returns:
            True if merge succeeded, False if either entry not found.
        """
        target = self._manager.get(target_id)
        source = self._manager.get(source_id)

        if not target or not source:
            return False

        # Merge content
        merged_content = f"{target['content']}\n\n---\n\n{source['content']}"

        # Merge tags
        target_tags = json_to_tags(target.get("tags"))
        source_tags = json_to_tags(source.get("tags"))
        merged_tags = list(set(target_tags + source_tags))

        # Merge context
        target_context = json_to_context(target.get("context"))
        source_context = json_to_context(source.get("context"))
        merged_context = list(set(target_context + source_context))

        # Update target
        self._manager.update(
            target_id,
            content=merged_content,
            tags=merged_tags,
            context=merged_context,
        )

        # Update usage count (sum both)
        cursor = self.conn.cursor()
        new_usage = (target.get("usage_count") or 0) + (source.get("usage_count") or 0)
        cursor.execute(
            "UPDATE knowledge SET usage_count = ? WHERE id = ?",
            (new_usage, target_id),
        )
        self.conn.commit()

        # Delete source if requested
        if delete_source:
            self._manager.delete(source_id)

        return True

    def score_quality(
        self,
        project: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Score entries by quality based on completeness metrics.

        Quality score (0-100) is calculated from (QUALITY_SCORE_WEIGHT points each):
        - Tags present
        - Description length >= MIN_DESCRIPTION_LENGTH_FOR_QUALITY chars
        - Content length >= MIN_CONTENT_LENGTH_FOR_QUALITY chars
        - Usage count > 0

        Args:
            project: Optional project filter.
            min_score: Optional minimum score filter (inclusive).
            max_score: Optional maximum score filter (inclusive).

        Returns:
            List of entries with quality_score field, sorted by score ascending.
        """
        cursor = self.conn.cursor()
        if project:
            cursor.execute(
                """
                SELECT id, title, description, content, tags, usage_count, created,
                       last_used, project
                FROM knowledge
                WHERE project = ?
                """,
                (project,),
            )
        else:
            cursor.execute(
                """
                SELECT id, title, description, content, tags, usage_count, created,
                       last_used, project
                FROM knowledge
                """
            )

        rows = cursor.fetchall()
        scored_entries = []

        for row in rows:
            entry = dict(row)
            score = 0

            # Tags present: QUALITY_SCORE_WEIGHT points
            tags_json = entry.get("tags") or "[]"
            tags_list = json_to_tags(tags_json)
            if tags_list:
                score += self.QUALITY_SCORE_WEIGHT

            # Description length >= MIN_DESCRIPTION_LENGTH_FOR_QUALITY: QUALITY_SCORE_WEIGHT points
            description = entry.get("description") or ""
            if len(description) >= self.MIN_DESCRIPTION_LENGTH_FOR_QUALITY:
                score += self.QUALITY_SCORE_WEIGHT

            # Content length >= MIN_CONTENT_LENGTH_FOR_QUALITY: QUALITY_SCORE_WEIGHT points
            content = entry.get("content") or ""
            if len(content) >= self.MIN_CONTENT_LENGTH_FOR_QUALITY:
                score += self.QUALITY_SCORE_WEIGHT

            # Usage count > 0: QUALITY_SCORE_WEIGHT points
            usage_count = entry.get("usage_count") or 0
            if usage_count > 0:
                score += self.QUALITY_SCORE_WEIGHT

            entry["quality_score"] = score

            # Apply score filters
            if min_score is not None and score < min_score:
                continue
            if max_score is not None and score > max_score:
                continue

            scored_entries.append(entry)

        # Sort by score ascending (lowest quality first for review)
        scored_entries.sort(key=lambda e: e["quality_score"])

        return scored_entries
