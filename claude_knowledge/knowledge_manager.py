"""Core knowledge management functionality."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from claude_knowledge.utils import (
    context_to_json,
    create_brief,
    estimate_tokens,
    format_knowledge_item,
    generate_id,
    sanitize_for_embedding,
    tags_to_json,
)


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
        self.config_path = self.base_path / "config.json"

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
        """
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
                created, source, project, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        """
        # Check if exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM knowledge WHERE id = ?", (knowledge_id,))
        if not cursor.fetchone():
            return False

        # Delete from ChromaDB
        try:
            self.collection.delete(ids=[knowledge_id])
        except Exception:
            pass  # May not exist in ChromaDB

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

        for field, value in kwargs.items():
            if field not in allowed_fields:
                continue
            if field == "tags":
                updates["tags"] = tags_to_json(value)
            elif field == "context":
                updates["context"] = context_to_json(value)
            else:
                updates[field] = value

        if not updates:
            return True  # Nothing to update

        # Update brief if content changed
        if "content" in updates:
            updates["brief"] = create_brief(updates["content"])

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
        search_pattern = f"%{text}%"

        if project:
            cursor.execute(
                """
                SELECT id, title, description, tags, usage_count, created, last_used, project
                FROM knowledge
                WHERE (title LIKE ? OR description LIKE ? OR content LIKE ?)
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
                WHERE title LIKE ? OR description LIKE ? OR content LIKE ?
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
