"""Core knowledge management functionality."""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from claude_knowledge.utils import (
    compute_content_hash,
    context_to_json,
    create_brief,
    escape_like_pattern,
    estimate_tokens,
    format_knowledge_item,
    generate_id,
    get_machine_id,
    json_to_context,
    json_to_tags,
    sanitize_for_embedding,
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


class KnowledgeManager:
    """Manages knowledge storage and retrieval using ChromaDB and SQLite."""

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "knowledge"

    def __init__(self, base_path: str = "~/.claude_knowledge"):
        """Initialize the knowledge manager.

        Args:
            base_path: Base directory for all data storage.
        """
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.chroma_path = self.base_path / "chroma_db"
        self.sqlite_path = self.base_path / "knowledge.db"

        # Initialize components
        self._init_chroma()
        self._init_sqlite()
        self._init_embedding_model()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB client and collection."""
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _init_sqlite(self) -> None:
        """Initialize SQLite database and create tables if needed."""
        self.conn = sqlite3.connect(str(self.sqlite_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                content TEXT NOT NULL,
                brief TEXT,
                tags TEXT,
                context TEXT,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                last_used TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                project TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_project ON knowledge(project)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_last_used ON knowledge(last_used DESC)
        """)
        self.conn.commit()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Apply schema migrations for existing databases."""
        cursor = self.conn.cursor()

        # Check if updated_at column exists
        cursor.execute("PRAGMA table_info(knowledge)")
        columns = {row[1] for row in cursor.fetchall()}

        if "updated_at" not in columns:
            # Add updated_at column and backfill with created timestamp
            cursor.execute("ALTER TABLE knowledge ADD COLUMN updated_at TIMESTAMP")
            cursor.execute("UPDATE knowledge SET updated_at = created WHERE updated_at IS NULL")
            self.conn.commit()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from config.json.

        Returns:
            Configuration dictionary.
        """
        config_path = self.base_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_config(self, config: dict[str, Any]) -> None:
        """Save configuration to config.json.

        Args:
            config: Configuration dictionary to save.
        """
        config_path = self.base_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def get_sync_path(self) -> Path | None:
        """Get the saved sync path from config.

        Returns:
            Path to sync directory, or None if not configured.
        """
        config = self._load_config()
        sync_path = config.get("sync_path")
        if sync_path:
            return Path(sync_path).expanduser()
        return None

    def set_sync_path(self, path: Path) -> None:
        """Save the sync path to config.

        Args:
            path: Path to sync directory.
        """
        config = self._load_config()
        config["sync_path"] = str(path)
        self._save_config(config)

    def _get_local_sync_state(self) -> dict[str, dict[str, Any]]:
        """Get the local record of what was last synced.

        Returns:
            Dictionary mapping entry ID to {content_hash, updated_at}.
        """
        config = self._load_config()
        return config.get("sync_state", {})

    def _set_local_sync_state(self, state: dict[str, dict[str, Any]]) -> None:
        """Save the local record of what was last synced.

        Args:
            state: Dictionary mapping entry ID to sync state.
        """
        config = self._load_config()
        config["sync_state"] = state
        self._save_config(config)

    def _init_embedding_model(self) -> None:
        """Initialize the sentence-transformers embedding model."""
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.EMBEDDING_MODEL)
        return self._model

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        clean_text = sanitize_for_embedding(text)
        embedding = self.model.encode(clean_text, convert_to_numpy=True)
        return embedding.tolist()

    def _create_embedding_text(self, title: str, description: str, content: str) -> str:
        """Create combined text for embedding generation.

        Args:
            title: Knowledge entry title.
            description: Knowledge entry description.
            content: Knowledge entry content.

        Returns:
            Combined text for embedding.
        """
        return f"{title}. {description}. {content}"

    def capture(
        self,
        title: str,
        description: str,
        content: str,
        tags: str | list[str] | None = None,
        context: list[str] | None = None,
        project: str | None = None,
        source: str = "user",
        confidence: float = 1.0,
    ) -> str:
        """Capture new knowledge.

        Args:
            title: Short title for the knowledge entry.
            description: Description of what this knowledge covers.
            content: Full content/details of the knowledge.
            tags: Optional tags (comma-separated string or list).
            context: Optional context list (e.g., ["backend", "python"]).
            project: Optional project identifier.
            source: Source of knowledge ("user", "auto", "inferred").
            confidence: Confidence score (0.0 to 1.0).

        Returns:
            The generated knowledge ID.

        Raises:
            ValueError: If title, description, or content is empty, or confidence is out of range.
        """
        # Validate required fields
        if not title or not title.strip():
            raise ValueError("title cannot be empty")
        if not description or not description.strip():
            raise ValueError("description cannot be empty")
        if not content or not content.strip():
            raise ValueError("content cannot be empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        timestamp = datetime.now()
        knowledge_id = generate_id(title, timestamp)
        brief = create_brief(content)

        # Generate embedding
        embedding_text = self._create_embedding_text(title, description, content)
        embedding = self._generate_embedding(embedding_text)

        # Store in ChromaDB
        metadata = {
            "title": title,
            "project": project or "",
        }
        self.collection.add(
            ids=[knowledge_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[embedding_text],
        )

        # Store in SQLite
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO knowledge (
                id, title, description, content, brief, tags, context,
                created, updated_at, source, project, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                knowledge_id,
                title,
                description,
                content,
                brief,
                tags_to_json(tags),
                context_to_json(context),
                timestamp.isoformat(),
                timestamp.isoformat(),  # updated_at = created for new entries
                source,
                project,
                confidence,
            ),
        )
        self.conn.commit()

        return knowledge_id

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        token_budget: int = 2000,
        project: str | None = None,
        min_score: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant knowledge based on query.

        Args:
            query: Search query text.
            n_results: Maximum number of results to return.
            token_budget: Maximum total tokens for returned content.
            project: Optional project filter.
            min_score: Minimum relevance score (0.0 to 1.0).

        Returns:
            List of knowledge items with metadata and scores.
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Build ChromaDB query
        where_filter = None
        if project:
            where_filter = {"project": project}

        # Query more results than needed to allow filtering
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results * 3, 50),
            where=where_filter,
            include=["distances", "metadatas"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        # Process results
        items = []
        ids = results["ids"][0]
        distances = results["distances"][0] if results["distances"] else []

        for i, knowledge_id in enumerate(ids):
            # Calculate relevance score (1 - cosine distance)
            distance = distances[i] if i < len(distances) else 0.5
            score = 1 - distance

            if score < min_score:
                continue

            # Fetch full metadata from SQLite
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,))
            row = cursor.fetchone()

            if row:
                item = dict(row)
                item["score"] = score
                items.append(item)

        # Sort by score
        items.sort(key=lambda x: x["score"], reverse=True)

        # Apply token budget
        selected = []
        total_tokens = 0

        for item in items:
            item_tokens = estimate_tokens(item.get("content", ""))
            if total_tokens + item_tokens <= token_budget:
                selected.append(item)
                total_tokens += item_tokens
            elif not selected:
                # Always include at least one result
                selected.append(item)
                break

            if len(selected) >= n_results:
                break

        # Update usage statistics
        self._update_usage(selected)

        return selected

    def _update_usage(self, items: list[dict[str, Any]]) -> None:
        """Update usage count and last_used for retrieved items.

        Args:
            items: List of retrieved knowledge items.
        """
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()

        for item in items:
            cursor.execute(
                """
                UPDATE knowledge
                SET usage_count = usage_count + 1, last_used = ?
                WHERE id = ?
                """,
                (timestamp, item["id"]),
            )
        self.conn.commit()

    def format_for_context(self, items: list[dict[str, Any]]) -> str:
        """Format retrieved items for inclusion in Claude context.

        Args:
            items: List of knowledge items.

        Returns:
            Formatted markdown string.
        """
        if not items:
            return "No relevant knowledge found."

        lines = ["## Retrieved Knowledge\n"]
        for item in items:
            lines.append(format_knowledge_item(item, include_content=True, include_score=True))

        return "\n".join(lines)

    def list_all(
        self,
        project: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List all knowledge entries.

        Args:
            project: Optional project filter.
            limit: Maximum number of results.

        Returns:
            List of knowledge items with basic metadata.
        """
        cursor = self.conn.cursor()

        if project:
            cursor.execute(
                """
                SELECT id, title, description, tags, usage_count, created, last_used, project
                FROM knowledge
                WHERE project = ?
                ORDER BY last_used DESC NULLS LAST, created DESC
                LIMIT ?
                """,
                (project, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, title, description, tags, usage_count, created, last_used, project
                FROM knowledge
                ORDER BY last_used DESC NULLS LAST, created DESC
                LIMIT ?
                """,
                (limit,),
            )

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get(self, knowledge_id: str) -> dict[str, Any] | None:
        """Get a single knowledge entry by ID.

        Args:
            knowledge_id: The knowledge entry ID.

        Returns:
            Knowledge item dict or None if not found.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete(self, knowledge_id: str) -> bool:
        """Delete a knowledge entry.

        Args:
            knowledge_id: The knowledge entry ID to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            Exception: If ChromaDB deletion fails for reasons other than missing ID.
        """
        # Check if exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM knowledge WHERE id = ?", (knowledge_id,))
        if not cursor.fetchone():
            return False

        # Delete from ChromaDB
        # ChromaDB's delete doesn't raise an error if the ID doesn't exist,
        # so we don't need special handling for that case. Any exception here
        # indicates a real problem that should be surfaced.
        self.collection.delete(ids=[knowledge_id])

        # Delete from SQLite
        cursor.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
        self.conn.commit()

        return True

    def update(self, knowledge_id: str, **kwargs: Any) -> bool:
        """Update a knowledge entry.

        Args:
            knowledge_id: The knowledge entry ID to update.
            **kwargs: Fields to update (title, description, content, tags, context, project).

        Returns:
            True if updated, False if not found.
        """
        # Check if exists
        existing = self.get(knowledge_id)
        if not existing:
            return False

        # Prepare update values
        allowed_fields = {
            "title",
            "description",
            "content",
            "tags",
            "context",
            "project",
            "confidence",
        }
        updates = {}

        for field_name, value in kwargs.items():
            if field_name not in allowed_fields:
                continue
            if field_name == "tags":
                updates["tags"] = tags_to_json(value)
            elif field_name == "context":
                updates["context"] = context_to_json(value)
            else:
                updates[field_name] = value

        if not updates:
            return True  # Nothing to update

        # Update brief if content changed
        if "content" in updates:
            updates["brief"] = create_brief(updates["content"])

        # Always update the updated_at timestamp
        updates["updated_at"] = datetime.now().isoformat()

        # Build and execute SQL
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [knowledge_id]

        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE knowledge SET {set_clause} WHERE id = ?", values)
        self.conn.commit()

        # Regenerate embedding if relevant fields changed
        if any(f in updates for f in ("title", "description", "content")):
            # Get updated record
            updated = self.get(knowledge_id)
            if updated:
                embedding_text = self._create_embedding_text(
                    updated["title"],
                    updated["description"],
                    updated["content"],
                )
                embedding = self._generate_embedding(embedding_text)

                # Update ChromaDB
                self.collection.update(
                    ids=[knowledge_id],
                    embeddings=[embedding],
                    metadatas=[{"title": updated["title"], "project": updated.get("project", "")}],
                    documents=[embedding_text],
                )

        return True

    def search(
        self,
        text: str,
        project: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Text search in titles and descriptions.

        Args:
            text: Search text.
            project: Optional project filter.
            limit: Maximum results.

        Returns:
            List of matching knowledge items.
        """
        cursor = self.conn.cursor()
        # Escape LIKE wildcards to prevent pattern injection
        escaped_text = escape_like_pattern(text)
        search_pattern = f"%{escaped_text}%"

        if project:
            cursor.execute(
                """
                SELECT id, title, description, tags, usage_count, created, last_used, project
                FROM knowledge
                WHERE (title LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\'
                       OR content LIKE ? ESCAPE '\\')
                AND project = ?
                ORDER BY usage_count DESC, last_used DESC NULLS LAST
                LIMIT ?
                """,
                (search_pattern, search_pattern, search_pattern, project, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, title, description, tags, usage_count, created, last_used, project
                FROM knowledge
                WHERE title LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\'
                      OR content LIKE ? ESCAPE '\\'
                ORDER BY usage_count DESC, last_used DESC NULLS LAST
                LIMIT ?
                """,
                (search_pattern, search_pattern, search_pattern, limit),
            )

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

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
        threshold: float = 0.85,
        project: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Find potential duplicate entries based on semantic similarity.

        Args:
            threshold: Minimum similarity score (0.0-1.0) to consider as duplicate.
            project: Optional project filter.

        Returns:
            List of duplicate groups, where each group is a list of similar entries
            with their similarity scores.
        """
        entries = self.list_all(project=project, limit=1000)
        if len(entries) < 2:
            return []

        # Track which entries have been grouped
        grouped_ids: set[str] = set()
        duplicate_groups: list[list[dict[str, Any]]] = []

        for entry in entries:
            if entry["id"] in grouped_ids:
                continue

            # Query ChromaDB for similar entries
            entry_full = self.get(entry["id"])
            if not entry_full:
                continue

            embedding_text = self._create_embedding_text(
                entry_full["title"],
                entry_full["description"],
                entry_full["content"],
            )
            query_embedding = self._generate_embedding(embedding_text)

            # Get more results than needed to find all duplicates
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(len(entries), 20),
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
                    other = self.get(kid)
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
        days: int = 90,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find entries that haven't been used or updated recently.

        Args:
            days: Number of days to consider an entry stale.
            project: Optional project filter.

        Returns:
            List of stale entries with staleness info.
        """
        from datetime import timedelta

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
        target = self.get(target_id)
        source = self.get(source_id)

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
        self.update(
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
            self.delete(source_id)

        return True

    def score_quality(
        self,
        project: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Score entries by quality based on completeness metrics.

        Quality score (0-100) is calculated from:
        - Tags present (25 points)
        - Description length >= 50 chars (25 points)
        - Content length >= 100 chars (25 points)
        - Usage count > 0 (25 points)

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

            # Tags present: 25 points
            tags_json = entry.get("tags") or "[]"
            tags_list = json_to_tags(tags_json)
            if tags_list:
                score += 25

            # Description length >= 50 chars: 25 points
            description = entry.get("description") or ""
            if len(description) >= 50:
                score += 25

            # Content length >= 100 chars: 25 points
            content = entry.get("content") or ""
            if len(content) >= 100:
                score += 25

            # Usage count > 0: 25 points
            usage_count = entry.get("usage_count") or 0
            if usage_count > 0:
                score += 25

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

    def export_all(self, project: str | None = None) -> list[dict[str, Any]]:
        """Export all knowledge entries as a list of dictionaries.

        Args:
            project: Optional project filter.

        Returns:
            List of knowledge entries with all fields.
        """
        cursor = self.conn.cursor()

        if project:
            cursor.execute("SELECT * FROM knowledge WHERE project = ?", (project,))
        else:
            cursor.execute("SELECT * FROM knowledge")

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def import_data(
        self,
        entries: list[dict[str, Any]],
        skip_duplicates: bool = True,
    ) -> dict[str, int]:
        """Import knowledge entries from a list of dictionaries.

        Args:
            entries: List of knowledge entry dictionaries.
            skip_duplicates: If True, skip entries with existing IDs. If False, raise error.

        Returns:
            Dictionary with counts: {"imported": n, "skipped": n, "errors": n}
        """
        imported = 0
        skipped = 0
        errors = 0

        for entry in entries:
            try:
                # Check for required fields
                required = ["title", "description", "content"]
                if not all(entry.get(f) for f in required):
                    errors += 1
                    continue

                # Check if ID already exists
                existing_id = entry.get("id")
                if existing_id:
                    existing = self.get(existing_id)
                    if existing:
                        if skip_duplicates:
                            skipped += 1
                            continue
                        else:
                            raise ValueError(f"Entry with ID {existing_id} already exists")

                # Capture the entry (generates new ID if not provided or if provided ID exists)
                self.capture(
                    title=entry["title"],
                    description=entry["description"],
                    content=entry["content"],
                    tags=entry.get("tags"),
                    context=entry.get("context"),
                    project=entry.get("project"),
                    source=entry.get("source", "import"),
                    confidence=entry.get("confidence", 1.0),
                )
                imported += 1

            except Exception:
                errors += 1

        return {"imported": imported, "skipped": skipped, "errors": errors}

    def purge(self, project: str | None = None) -> int:
        """Delete all knowledge entries, optionally filtered by project.

        Args:
            project: If specified, only delete entries for this project.
                    If None, delete ALL entries.

        Returns:
            Number of entries deleted.
        """
        cursor = self.conn.cursor()

        # Get IDs to delete
        if project:
            cursor.execute("SELECT id FROM knowledge WHERE project = ?", (project,))
        else:
            cursor.execute("SELECT id FROM knowledge")

        ids = [row[0] for row in cursor.fetchall()]

        if not ids:
            return 0

        # Delete from ChromaDB
        self.collection.delete(ids=ids)

        # Delete from SQLite
        if project:
            cursor.execute("DELETE FROM knowledge WHERE project = ?", (project,))
        else:
            cursor.execute("DELETE FROM knowledge")

        self.conn.commit()
        return len(ids)

    # Sync methods

    def init_sync_dir(self, sync_path: str | Path) -> None:
        """Initialize a sync directory structure.

        Args:
            sync_path: Path to the sync directory.
        """
        sync_path = Path(sync_path).expanduser()
        sync_path.mkdir(parents=True, exist_ok=True)
        (sync_path / "entries").mkdir(exist_ok=True)
        (sync_path / "tombstones").mkdir(exist_ok=True)

        # Create manifest if it doesn't exist
        manifest_path = sync_path / "manifest.json"
        if not manifest_path.exists():
            self._save_manifest(sync_path, {"version": 1, "last_sync": {}, "entries": {}})

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
        entries = self.export_all(project=project)
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
            sync_path = self.get_sync_path()
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
        """
        result = SyncResult()

        # Resolve sync path
        if sync_path is None:
            sync_path = self.get_sync_path()
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
            self.set_sync_path(sync_path)

        manifest = self._load_manifest(sync_path)
        tombstones = self._load_tombstones(sync_path)
        local_state = self._get_local_state(project=project)
        remote_state = self._get_remote_state(sync_path)
        local_sync_state = self._get_local_sync_state()  # What this machine last synced
        manifest_entries = manifest.get("entries", {})

        all_ids = set(local_state.keys()) | set(remote_state.keys())

        for entry_id in all_ids:
            local = local_state.get(entry_id)
            remote = remote_state.get(entry_id)
            last_synced = local_sync_state.get(entry_id)  # Use local sync state
            tombstone = tombstones.get(entry_id)

            action = self._categorize_entry(entry_id, local, remote, last_synced, tombstone)

            if action == "no_change":
                continue
            elif action == "push" and not pull_only:
                self._handle_push(
                    sync_path, entry_id, manifest_entries, local_sync_state, dry_run, result
                )
            elif action == "pull" and not push_only:
                self._handle_pull(
                    sync_path, entry_id, manifest_entries, local_sync_state, dry_run, result
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
                    sync_path, entry_id, manifest_entries, local_sync_state, dry_run, result
                )

        # Update manifest and local sync state
        if not dry_run:
            machine_id = get_machine_id()
            manifest["last_sync"][machine_id] = datetime.now().isoformat()
            manifest["entries"] = manifest_entries
            self._save_manifest(sync_path, manifest)
            self._set_local_sync_state(local_sync_state)

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
        entry = self.get(entry_id)
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
            existing = self.get(entry_id)
            if existing:
                # Update existing entry
                self.update(
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
        embedding_text = self._create_embedding_text(
            entry["title"],
            entry["description"],
            entry["content"],
        )
        embedding = self._generate_embedding(embedding_text)

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
        local_entry = self.get(entry_id)
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
            self.delete(entry_id)
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

    def close(self) -> None:
        """Close database connections."""
        if hasattr(self, "conn"):
            self.conn.close()

    def __enter__(self) -> "KnowledgeManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
