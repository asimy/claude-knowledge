"""Core knowledge management functionality."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from claude_knowledge._config import ConfigManager
from claude_knowledge._embedding import EmbeddingError, EmbeddingService
from claude_knowledge._maintenance import MaintenanceService
from claude_knowledge._relationships import RelationshipsService
from claude_knowledge._sync import SyncManager, SyncResult
from claude_knowledge._tracking import ProcessingTracker
from claude_knowledge._versioning import VersioningService
from claude_knowledge.utils import (
    context_to_json,
    create_brief,
    escape_like_pattern,
    estimate_tokens,
    format_knowledge_item,
    fuzzy_match_tags,
    generate_id,
    json_to_tags,
    tags_to_json,
)

# Re-export EmbeddingError for backward compatibility
__all__ = ["KnowledgeManager", "EmbeddingError", "SyncResult"]


class KnowledgeManager:
    """Manages knowledge storage and retrieval using ChromaDB and SQLite."""

    # Core configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "knowledge"

    # SQL injection prevention
    ALLOWED_DATE_FIELDS = frozenset({"created", "last_used"})

    # Input size limits for capture() to prevent resource exhaustion
    MAX_TITLE_LENGTH = 500
    MAX_DESCRIPTION_LENGTH = 10_000
    MAX_CONTENT_LENGTH = 1_000_000  # ~1MB of text

    # Default values for retrieval and search
    DEFAULT_TOKEN_BUDGET = 2000
    DEFAULT_RETRIEVE_RESULTS = 5
    DEFAULT_MIN_SCORE = 0.3
    DEFAULT_LIST_LIMIT = 50
    DEFAULT_SEARCH_LIMIT = 20

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

    # File locking
    DEFAULT_FILE_LOCK_TIMEOUT = 30.0

    def __init__(
        self,
        base_path: str = "~/.claude_knowledge",
        embedding_service: "EmbeddingService | None" = None,
    ):
        """Initialize the knowledge manager.

        Args:
            base_path: Base directory for all data storage.
            embedding_service: Optional pre-configured EmbeddingService instance.
                              If None, a new service will be created.
                              Useful for sharing a model across multiple instances in tests.
        """
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.chroma_path = self.base_path / "chroma_db"
        self.sqlite_path = self.base_path / "knowledge.db"

        # Store provided embedding service for later initialization
        self._provided_embedding_service = embedding_service

        # Initialize components
        self._init_chroma()
        self._init_sqlite()
        self._init_config()
        self._init_embedding_model()
        self._init_tracker()
        self._init_maintenance()
        self._init_relationships()
        self._init_versioning()
        self._init_sync()

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
        # Enable foreign key support for cascade deletes
        self.conn.execute("PRAGMA foreign_keys = ON")
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
        # Table for tracking processed sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_sessions (
                session_id TEXT PRIMARY KEY,
                project_path TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entries_created INTEGER DEFAULT 0
            )
        """)
        # Table for tracking processed git commits
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_commits (
                commit_sha TEXT NOT NULL,
                repo_path TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entries_created INTEGER DEFAULT 0,
                PRIMARY KEY (commit_sha, repo_path)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processed_commits_repo
            ON processed_commits(repo_path)
        """)
        # Table for tracking processed files (for code analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                file_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                repo_path TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entries_created INTEGER DEFAULT 0,
                PRIMARY KEY (file_path, repo_path)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processed_files_repo
            ON processed_files(repo_path)
        """)
        # Table for entry-to-entry relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (source_id) REFERENCES knowledge(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES knowledge(id) ON DELETE CASCADE,
                UNIQUE(source_id, target_id, relationship_type)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relationships_source
            ON knowledge_relationships(source_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relationships_target
            ON knowledge_relationships(target_id)
        """)
        # Table for named collections
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)
        # Table for collection membership
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_members (
                collection_id TEXT NOT NULL,
                entry_id TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (collection_id) REFERENCES knowledge_collections(id) ON DELETE CASCADE,
                FOREIGN KEY (entry_id) REFERENCES knowledge(id) ON DELETE CASCADE,
                PRIMARY KEY (collection_id, entry_id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_collection_members_entry
            ON collection_members(entry_id)
        """)
        # Table for entry version history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entry_versions (
                id TEXT PRIMARY KEY,
                entry_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                content TEXT NOT NULL,
                brief TEXT,
                tags TEXT,
                context TEXT,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                project TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                change_summary TEXT,
                FOREIGN KEY (entry_id) REFERENCES knowledge(id) ON DELETE CASCADE,
                UNIQUE(entry_id, version_number)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entry_versions_entry
            ON entry_versions(entry_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entry_versions_created
            ON entry_versions(created_at DESC)
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

    def _init_config(self) -> None:
        """Initialize the configuration manager."""
        self._config = ConfigManager(self.base_path)

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from config.json.

        Returns:
            Configuration dictionary.
        """
        return self._config.load_config()

    def _save_config(self, config: dict[str, Any]) -> None:
        """Save configuration to config.json.

        Args:
            config: Configuration dictionary to save.
        """
        self._config.save_config(config)

    def get_sync_path(self) -> Path | None:
        """Get the saved sync path from config.

        Returns:
            Path to sync directory, or None if not configured.
        """
        return self._config.get_sync_path()

    def set_sync_path(self, path: Path) -> None:
        """Save the sync path to config.

        Args:
            path: Path to sync directory.
        """
        self._config.set_sync_path(path)

    def _get_local_sync_state(self) -> dict[str, dict[str, Any]]:
        """Get the local record of what was last synced.

        Returns:
            Dictionary mapping entry ID to {content_hash, updated_at}.
        """
        return self._config.get_local_sync_state()

    def _set_local_sync_state(self, state: dict[str, dict[str, Any]]) -> None:
        """Save the local record of what was last synced.

        Args:
            state: Dictionary mapping entry ID to sync state.
        """
        self._config.set_local_sync_state(state)

    def _init_embedding_model(self) -> None:
        """Initialize the embedding service."""
        if self._provided_embedding_service is not None:
            self._embedding = self._provided_embedding_service
        else:
            self._embedding = EmbeddingService()

    def _init_tracker(self) -> None:
        """Initialize the processing tracker."""
        self._tracker = ProcessingTracker(self.conn)

    def _init_maintenance(self) -> None:
        """Initialize the maintenance service."""
        self._maintenance = MaintenanceService(self.conn, self.collection, self._embedding, self)

    def _init_relationships(self) -> None:
        """Initialize the relationships service."""
        self._relationships = RelationshipsService(
            self.conn, self.collection, self._embedding, self
        )

    def _init_versioning(self) -> None:
        """Initialize the versioning service."""
        self._versioning = VersioningService(
            self.conn, self.collection, self._embedding, self
        )

    def _init_sync(self) -> None:
        """Initialize the sync manager."""
        self._sync = SyncManager(self.conn, self.collection, self._embedding, self._config, self)

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model on first use.

        Returns:
            The initialized SentenceTransformer model.

        Raises:
            EmbeddingError: If the model fails to load.

        Note:
            This property is kept for backward compatibility.
            The model is managed by the EmbeddingService.
        """
        return self._embedding.model

    def _validate_date_field(self, date_field: str) -> None:
        """Validate that date_field is in the allowed whitelist.

        Args:
            date_field: The date field name to validate.

        Raises:
            ValueError: If date_field is not in ALLOWED_DATE_FIELDS.
        """
        if date_field not in self.ALLOWED_DATE_FIELDS:
            allowed = ", ".join(sorted(self.ALLOWED_DATE_FIELDS))
            raise ValueError(f"Invalid date_field '{date_field}'. Must be one of: {allowed}")

    @contextmanager
    def _file_lock(self, lock_path: Path, timeout: float | None = None) -> Iterator[None]:
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
        with self._sync._file_lock(lock_path, timeout):
            yield

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If text is empty after sanitization or encoding fails.
        """
        return self._embedding.generate_embedding(text)

    def _create_embedding_text(self, title: str, description: str, content: str) -> str:
        """Create combined text for embedding generation.

        Args:
            title: Knowledge entry title.
            description: Knowledge entry description.
            content: Knowledge entry content.

        Returns:
            Combined text for embedding.
        """
        return self._embedding.create_embedding_text(title, description, content)

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

        # Validate input size limits
        if len(title) > self.MAX_TITLE_LENGTH:
            raise ValueError(f"title exceeds maximum length of {self.MAX_TITLE_LENGTH} characters")
        if len(description) > self.MAX_DESCRIPTION_LENGTH:
            raise ValueError(
                f"description exceeds maximum length of {self.MAX_DESCRIPTION_LENGTH} characters"
            )
        if len(content) > self.MAX_CONTENT_LENGTH:
            raise ValueError(
                f"content exceeds maximum length of {self.MAX_CONTENT_LENGTH} characters"
            )

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
        tags: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        date_field: str = "created",
        fuzzy: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant knowledge based on query.

        Args:
            query: Search query text.
            n_results: Maximum number of results to return.
            token_budget: Maximum total tokens for returned content.
            project: Optional project filter.
            min_score: Minimum relevance score (0.0 to 1.0).
            tags: Optional list of tags to filter by (AND logic).
            since: Optional start date (ISO format) for date filtering.
            until: Optional end date (ISO format) for date filtering.
            date_field: Field to use for date filtering ("created" or "last_used").
            fuzzy: Enable fuzzy tag matching (edit distance <= 2).

        Returns:
            List of knowledge items with metadata and scores.

        Raises:
            ValueError: If date_field is not a valid field name.
        """
        # Validate date_field to prevent SQL injection
        self._validate_date_field(date_field)

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

        # Build score map and filter by min_score
        id_scores: dict[str, float] = {}
        for i, knowledge_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 0.5
            score = 1 - distance
            if score >= min_score:
                id_scores[knowledge_id] = score

        if not id_scores:
            return []

        # Batch fetch all metadata from SQLite (fixes N+1 query problem)
        cursor = self.conn.cursor()
        placeholders = ",".join("?" * len(id_scores))
        cursor.execute(
            f"SELECT * FROM knowledge WHERE id IN ({placeholders})",
            list(id_scores.keys()),
        )
        rows = {row["id"]: dict(row) for row in cursor.fetchall()}

        # Process rows with scores and apply filters
        for knowledge_id, score in id_scores.items():
            row = rows.get(knowledge_id)
            if not row:
                continue

            item = row
            item["score"] = score

            # Apply tag filter
            if tags:
                item_tags = json_to_tags(item.get("tags"))
                if fuzzy:
                    if not fuzzy_match_tags(tags, item_tags):
                        continue
                elif not all(tag in item_tags for tag in tags):
                    continue

            # Apply date filter
            if since or until:
                date_value = item.get(date_field)
                if date_value:
                    if since and date_value < since:
                        continue
                    if until and date_value > until:
                        continue

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
        tags: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        date_field: str = "created",
        fuzzy: bool = False,
    ) -> list[dict[str, Any]]:
        """List all knowledge entries.

        Args:
            project: Optional project filter.
            limit: Maximum number of results.
            tags: Optional list of tags to filter by (AND logic).
            since: Optional start date (ISO format) for date filtering.
            until: Optional end date (ISO format) for date filtering.
            date_field: Field to use for date filtering ("created" or "last_used").
            fuzzy: Enable fuzzy tag matching (edit distance <= 2).

        Returns:
            List of knowledge items with basic metadata.

        Raises:
            ValueError: If date_field is not a valid field name.
        """
        # Validate date_field to prevent SQL injection
        self._validate_date_field(date_field)

        cursor = self.conn.cursor()

        # Build query dynamically based on filters
        query = """
            SELECT id, title, description, tags, usage_count, created, last_used, project
            FROM knowledge
            WHERE 1=1
        """
        params: list[str | int] = []

        if project:
            query += " AND project = ?"
            params.append(project)

        if since:
            query += f" AND {date_field} >= ?"
            params.append(since)

        if until:
            query += f" AND {date_field} <= ?"
            params.append(until)

        query += " ORDER BY last_used DESC NULLS LAST, created DESC"

        # Fetch more results if we need to filter by tags (post-query)
        fetch_limit = limit * 3 if tags else limit
        query += " LIMIT ?"
        params.append(fetch_limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        items = [dict(row) for row in rows]

        # Apply tag filter (requires JSON parsing)
        if tags:
            filtered = []
            for item in items:
                item_tags = json_to_tags(item.get("tags"))
                if fuzzy:
                    matches = fuzzy_match_tags(tags, item_tags)
                else:
                    matches = all(tag in item_tags for tag in tags)
                if matches:
                    filtered.append(item)
                    if len(filtered) >= limit:
                        break
            return filtered

        return items[:limit]

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

    def update(
        self,
        knowledge_id: str,
        create_version: bool = True,
        **kwargs: Any,
    ) -> bool:
        """Update a knowledge entry.

        Args:
            knowledge_id: The knowledge entry ID to update.
            create_version: Whether to create a version snapshot before updating.
                           Defaults to True. Set to False for internal updates
                           that shouldn't be versioned (e.g., usage count updates).
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

        # Create version snapshot before making changes
        if create_version:
            # Generate change summary from updated fields
            changed_fields = list(updates.keys())
            change_summary = f"Updated: {', '.join(changed_fields)}"
            self._versioning.create_version(knowledge_id, change_summary=change_summary)

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
        tags: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        date_field: str = "created",
        fuzzy: bool = False,
    ) -> list[dict[str, Any]]:
        """Text search in titles and descriptions.

        Args:
            text: Search text.
            project: Optional project filter.
            limit: Maximum results.
            tags: Optional list of tags to filter by (AND logic).
            since: Optional start date (ISO format) for date filtering.
            until: Optional end date (ISO format) for date filtering.
            date_field: Field to use for date filtering ("created" or "last_used").
            fuzzy: Enable fuzzy tag matching (edit distance <= 2).

        Returns:
            List of matching knowledge items.

        Raises:
            ValueError: If date_field is not a valid field name.
        """
        # Validate date_field to prevent SQL injection
        self._validate_date_field(date_field)

        cursor = self.conn.cursor()
        # Escape LIKE wildcards to prevent pattern injection
        escaped_text = escape_like_pattern(text)
        search_pattern = f"%{escaped_text}%"

        # Build query dynamically
        query = """
            SELECT id, title, description, tags, usage_count, created, last_used, project
            FROM knowledge
            WHERE (title LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\'
                   OR content LIKE ? ESCAPE '\\')
        """
        params: list[str | int] = [search_pattern, search_pattern, search_pattern]

        if project:
            query += " AND project = ?"
            params.append(project)

        if since:
            query += f" AND {date_field} >= ?"
            params.append(since)

        if until:
            query += f" AND {date_field} <= ?"
            params.append(until)

        query += " ORDER BY usage_count DESC, last_used DESC NULLS LAST"

        # Fetch more results if we need to filter by tags (post-query)
        fetch_limit = limit * 3 if tags else limit
        query += " LIMIT ?"
        params.append(fetch_limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        items = [dict(row) for row in rows]

        # Apply tag filter (requires JSON parsing)
        if tags:
            filtered = []
            for item in items:
                item_tags = json_to_tags(item.get("tags"))
                if fuzzy:
                    matches = fuzzy_match_tags(tags, item_tags)
                else:
                    matches = all(tag in item_tags for tag in tags)
                if matches:
                    filtered.append(item)
                    if len(filtered) >= limit:
                        break
            return filtered

        return items[:limit]

    def get_distinct_projects(self) -> list[str]:
        """Get list of distinct project names.

        Returns:
            List of project names, sorted alphabetically.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT DISTINCT project FROM knowledge "
            "WHERE project IS NOT NULL AND project != '' "
            "ORDER BY project"
        )
        return [row[0] for row in cursor.fetchall()]

    def stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge base.

        Returns:
            Dictionary with statistics.
        """
        return self._maintenance.stats()

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
        return self._maintenance.find_duplicates(threshold, project)

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
        return self._maintenance.find_stale(days, project)

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
        return self._maintenance.merge_entries(target_id, source_id, delete_source)

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
        return self._maintenance.score_quality(project, min_score, max_score)

    # Relationship methods (delegated to RelationshipsService)

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
        return self._relationships.link(source_id, target_id, relationship_type, metadata)

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
        return self._relationships.unlink(source_id, target_id, relationship_type)

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

        Returns:
            List of related entries with relationship info.
        """
        return self._relationships.get_related(entry_id, relationship_type, direction)

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
        return self._relationships.get_dependency_tree(entry_id, depth)

    def get_entry_relationships(self, entry_id: str) -> list[dict[str, Any]]:
        """Get all relationships for an entry (both directions).

        Args:
            entry_id: ID of the entry.

        Returns:
            List of all relationships involving this entry.
        """
        return self._relationships.get_entry_relationships(entry_id)

    def has_relationships_or_collections(self, entry_id: str) -> dict[str, int]:
        """Check if an entry has any relationships or collection memberships.

        Args:
            entry_id: ID of the entry.

        Returns:
            Dictionary with counts: {'relationships': N, 'collections': N}
        """
        return self._relationships.has_relationships_or_collections(entry_id)

    # Collection methods (delegated to RelationshipsService)

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
        return self._relationships.create_collection(name, description)

    def delete_collection(self, collection_id_or_name: str) -> bool:
        """Delete a collection.

        Args:
            collection_id_or_name: Collection ID or name.

        Returns:
            True if deleted, False if not found.
        """
        return self._relationships.delete_collection(collection_id_or_name)

    def get_collection(self, collection_id_or_name: str) -> dict[str, Any] | None:
        """Get a collection by ID or name.

        Args:
            collection_id_or_name: Collection ID or name.

        Returns:
            Collection dictionary or None if not found.
        """
        return self._relationships.get_collection(collection_id_or_name)

    def list_collections(self, limit: int | None = None) -> list[dict[str, Any]]:
        """List all collections.

        Args:
            limit: Maximum number of collections to return.

        Returns:
            List of collections with member counts.
        """
        return self._relationships.list_collections(limit)

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
        return self._relationships.add_to_collection(collection_id_or_name, entry_id)

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
        return self._relationships.remove_from_collection(collection_id_or_name, entry_id)

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
        return self._relationships.get_collection_members(collection_id_or_name)

    def get_entry_collections(self, entry_id: str) -> list[dict[str, Any]]:
        """Get all collections that contain an entry.

        Args:
            entry_id: ID of the entry.

        Returns:
            List of collections containing the entry.
        """
        return self._relationships.get_entry_collections(entry_id)

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

    # Sync methods (delegated to SyncManager)

    def init_sync_dir(self, sync_path: str | Path) -> None:
        """Initialize a sync directory structure.

        Args:
            sync_path: Path to the sync directory.

        Raises:
            TimeoutError: If unable to acquire lock within timeout.
        """
        self._sync.init_sync_dir(sync_path)

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
        return self._sync.sync_status(sync_path, project)

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
        return self._sync.sync(sync_path, strategy, push_only, pull_only, dry_run, project)

    # Session processing methods (delegated to ProcessingTracker)

    def is_session_processed(self, session_id: str) -> bool:
        """Check if a session has been processed.

        Args:
            session_id: Session UUID.

        Returns:
            True if the session has been processed.
        """
        return self._tracker.is_session_processed(session_id)

    def mark_session_processed(
        self,
        session_id: str,
        project_path: str,
        entries_created: int = 0,
    ) -> None:
        """Mark a session as processed.

        Args:
            session_id: Session UUID.
            project_path: Project path.
            entries_created: Number of knowledge entries created from this session.
        """
        self._tracker.mark_session_processed(session_id, project_path, entries_created)

    def get_processed_sessions(
        self,
        project_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get list of processed sessions.

        Args:
            project_path: Optional filter by project path.

        Returns:
            List of processed session records.
        """
        return self._tracker.get_processed_sessions(project_path)

    # Git commit processing methods (delegated to ProcessingTracker)

    def is_commit_processed(self, commit_sha: str, repo_path: str) -> bool:
        """Check if a commit has been processed.

        Args:
            commit_sha: Git commit SHA.
            repo_path: Path to the repository.

        Returns:
            True if the commit has been processed.
        """
        return self._tracker.is_commit_processed(commit_sha, repo_path)

    def mark_commit_processed(
        self,
        commit_sha: str,
        repo_path: str,
        entries_created: int = 0,
    ) -> None:
        """Mark a commit as processed.

        Args:
            commit_sha: Git commit SHA.
            repo_path: Path to the repository.
            entries_created: Number of knowledge entries created from this commit.
        """
        self._tracker.mark_commit_processed(commit_sha, repo_path, entries_created)

    def get_processed_commits(
        self,
        repo_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get list of processed commits.

        Args:
            repo_path: Optional filter by repository path.

        Returns:
            List of processed commit records.
        """
        return self._tracker.get_processed_commits(repo_path)

    # File processing methods (delegated to ProcessingTracker)

    def is_file_processed(
        self,
        file_path: str,
        repo_path: str,
        content_hash: str | None = None,
    ) -> bool:
        """Check if a file has been processed.

        Args:
            file_path: Path to the file.
            repo_path: Path to the repository.
            content_hash: Optional content hash. If provided, also checks
                         if the hash matches (file hasn't changed).

        Returns:
            True if the file has been processed (and hasn't changed if hash provided).
        """
        return self._tracker.is_file_processed(file_path, repo_path, content_hash)

    def mark_file_processed(
        self,
        file_path: str,
        content_hash: str,
        repo_path: str,
        entries_created: int = 0,
    ) -> None:
        """Mark a file as processed.

        Args:
            file_path: Path to the file.
            content_hash: Content hash of the file.
            repo_path: Path to the repository.
            entries_created: Number of knowledge entries created from this file.
        """
        self._tracker.mark_file_processed(file_path, content_hash, repo_path, entries_created)

    def get_processed_files(
        self,
        repo_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get list of processed files.

        Args:
            repo_path: Optional filter by repository path.

        Returns:
            List of processed file records.
        """
        return self._tracker.get_processed_files(repo_path)

    # Versioning methods (delegated to VersioningService)

    def get_version(self, entry_id: str, version_number: int) -> dict[str, Any] | None:
        """Retrieve a specific version of an entry.

        Args:
            entry_id: ID of the entry.
            version_number: Version number to retrieve.

        Returns:
            Version dictionary or None if not found.
        """
        return self._versioning.get_version(entry_id, version_number)

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
        return self._versioning.get_history(entry_id, limit)

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
        return self._versioning.rollback(entry_id, version_number, created_by)

    def diff_versions(
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
            Dictionary containing diff information for title, description,
            content, and flags for other changed fields.

        Raises:
            ValueError: If entry or version not found.
        """
        return self._versioning.diff(entry_id, version_a, version_b)

    def get_version_count(self, entry_id: str) -> int:
        """Get the number of versions for an entry.

        Args:
            entry_id: ID of the entry.

        Returns:
            Number of versions.
        """
        return self._versioning.get_version_count(entry_id)

    def close(self) -> None:
        """Close database connections."""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
            self.conn = None  # Prevent double-close

    def __del__(self) -> None:
        """Ensure connections are closed when object is garbage collected."""
        self.close()

    def __enter__(self) -> "KnowledgeManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
