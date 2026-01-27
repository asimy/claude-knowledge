"""Processing tracker for sessions, commits, and files."""

import sqlite3
from typing import Any


class ProcessingTracker:
    """Tracks which sessions, commits, and files have been processed.

    This service manages the tracking tables that record what content
    has already been processed to avoid duplicate processing.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the processing tracker.

        Args:
            conn: SQLite database connection.
        """
        self.conn = conn

    # Session processing methods

    def is_session_processed(self, session_id: str) -> bool:
        """Check if a session has been processed.

        Args:
            session_id: Session UUID.

        Returns:
            True if the session has been processed.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT 1 FROM processed_sessions WHERE session_id = ?",
            (session_id,),
        )
        return cursor.fetchone() is not None

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
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO processed_sessions
            (session_id, project_path, processed_at, entries_created)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?)
            """,
            (session_id, project_path, entries_created),
        )
        self.conn.commit()

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
        cursor = self.conn.cursor()
        if project_path:
            cursor.execute(
                """
                SELECT session_id, project_path, processed_at, entries_created
                FROM processed_sessions
                WHERE project_path = ?
                ORDER BY processed_at DESC
                """,
                (project_path,),
            )
        else:
            cursor.execute(
                """
                SELECT session_id, project_path, processed_at, entries_created
                FROM processed_sessions
                ORDER BY processed_at DESC
                """
            )
        return [dict(row) for row in cursor.fetchall()]

    # Git commit processing methods

    def is_commit_processed(self, commit_sha: str, repo_path: str) -> bool:
        """Check if a commit has been processed.

        Args:
            commit_sha: Git commit SHA.
            repo_path: Path to the repository.

        Returns:
            True if the commit has been processed.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT 1 FROM processed_commits WHERE commit_sha = ? AND repo_path = ?",
            (commit_sha, repo_path),
        )
        return cursor.fetchone() is not None

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
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO processed_commits
            (commit_sha, repo_path, processed_at, entries_created)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?)
            """,
            (commit_sha, repo_path, entries_created),
        )
        self.conn.commit()

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
        cursor = self.conn.cursor()
        if repo_path:
            cursor.execute(
                """
                SELECT commit_sha, repo_path, processed_at, entries_created
                FROM processed_commits
                WHERE repo_path = ?
                ORDER BY processed_at DESC
                """,
                (repo_path,),
            )
        else:
            cursor.execute(
                """
                SELECT commit_sha, repo_path, processed_at, entries_created
                FROM processed_commits
                ORDER BY processed_at DESC
                """
            )
        return [dict(row) for row in cursor.fetchall()]

    # File processing methods (for code analysis)

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
        cursor = self.conn.cursor()
        if content_hash:
            cursor.execute(
                """
                SELECT 1 FROM processed_files
                WHERE file_path = ? AND repo_path = ? AND content_hash = ?
                """,
                (file_path, repo_path, content_hash),
            )
        else:
            cursor.execute(
                "SELECT 1 FROM processed_files WHERE file_path = ? AND repo_path = ?",
                (file_path, repo_path),
            )
        return cursor.fetchone() is not None

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
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO processed_files
            (file_path, content_hash, repo_path, processed_at, entries_created)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
            """,
            (file_path, content_hash, repo_path, entries_created),
        )
        self.conn.commit()

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
        cursor = self.conn.cursor()
        if repo_path:
            cursor.execute(
                """
                SELECT file_path, content_hash, repo_path, processed_at, entries_created
                FROM processed_files
                WHERE repo_path = ?
                ORDER BY processed_at DESC
                """,
                (repo_path,),
            )
        else:
            cursor.execute(
                """
                SELECT file_path, content_hash, repo_path, processed_at, entries_created
                FROM processed_files
                ORDER BY processed_at DESC
                """
            )
        return [dict(row) for row in cursor.fetchall()]
