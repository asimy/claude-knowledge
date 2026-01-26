"""Parse Claude Code session transcripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SessionMessage:
    """A single message from a session transcript."""

    uuid: str
    role: str  # "user" or "assistant"
    content: list[dict[str, Any]]
    timestamp: datetime
    parent_uuid: str | None = None
    message_type: str = "message"  # "user", "assistant", "file-history-snapshot", etc.

    @property
    def text_content(self) -> str:
        """Extract plain text content from the message."""
        # Handle string content directly
        if isinstance(self.content, str):
            return self.content

        text_parts = []
        for block in self.content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    # Include tool result content
                    result_content = block.get("content", [])
                    if isinstance(result_content, str):
                        text_parts.append(result_content)
                    elif isinstance(result_content, list):
                        for item in result_content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts)

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        """Extract tool use blocks from the message."""
        tools = []
        for block in self.content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tools.append(block)
        return tools

    @property
    def tool_results(self) -> list[dict[str, Any]]:
        """Extract tool result blocks from the message."""
        results = []
        for block in self.content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results.append(block)
        return results


@dataclass
class SessionTranscript:
    """A parsed session transcript."""

    session_id: str
    project_path: str
    messages: list[SessionMessage] = field(default_factory=list)
    summary: str | None = None
    first_prompt: str | None = None
    created: datetime | None = None
    modified: datetime | None = None
    git_branch: str | None = None
    message_count: int = 0

    def get_conversation_pairs(self) -> list[tuple[SessionMessage, SessionMessage]]:
        """Get user-assistant message pairs in order.

        Returns:
            List of (user_message, assistant_message) tuples.
        """
        pairs = []
        user_messages = [m for m in self.messages if m.role == "user"]
        assistant_messages = [m for m in self.messages if m.role == "assistant"]

        # Match based on parent_uuid relationship or sequential order
        for user_msg in user_messages:
            # Find assistant response that follows this user message
            for asst_msg in assistant_messages:
                if asst_msg.parent_uuid == user_msg.uuid:
                    pairs.append((user_msg, asst_msg))
                    break

        return pairs


class SessionParser:
    """Parse Claude Code session transcripts from JSONL files."""

    CLAUDE_DIR = Path.home() / ".claude" / "projects"

    def __init__(self, claude_dir: Path | None = None):
        """Initialize the session parser.

        Args:
            claude_dir: Optional override for Claude projects directory.
        """
        self.claude_dir = claude_dir or self.CLAUDE_DIR

    def _path_to_project_dir(self, project_path: str) -> str:
        """Convert a project path to the Claude directory name format.

        Claude Code stores projects with paths like:
        /Users/glen/code/myproject -> -Users-glen-code-myproject
        """
        # Replace path separators with hyphens
        # Claude uses the full path with hyphens, starting with a hyphen
        return project_path.replace("/", "-")

    def _project_dir_to_path(self, dir_name: str) -> str:
        """Convert a Claude directory name back to a project path.

        -Users-glen-code-myproject -> /Users/glen/code/myproject
        """
        # Replace hyphens with path separators
        return "/" + dir_name.replace("-", "/")

    def list_projects(self) -> list[dict[str, Any]]:
        """List all Claude Code projects with sessions.

        Returns:
            List of project info dicts with 'dir_name', 'path', and 'session_count'.
        """
        projects = []

        if not self.claude_dir.exists():
            return projects

        for project_dir in self.claude_dir.iterdir():
            if not project_dir.is_dir():
                continue

            # Count JSONL files (sessions)
            jsonl_files = list(project_dir.glob("*.jsonl"))
            if not jsonl_files:
                continue

            # Try to get project path from sessions-index.json
            index_path = project_dir / "sessions-index.json"
            project_path = None
            if index_path.exists():
                try:
                    with open(index_path) as f:
                        index_data = json.load(f)
                        project_path = index_data.get("originalPath")
                except (json.JSONDecodeError, OSError):
                    pass

            if not project_path:
                project_path = self._project_dir_to_path(project_dir.name)

            projects.append(
                {
                    "dir_name": project_dir.name,
                    "path": project_path,
                    "session_count": len(jsonl_files),
                }
            )

        return sorted(projects, key=lambda p: p["path"])

    def list_sessions(
        self,
        project_path: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List available sessions.

        Args:
            project_path: Optional project path to filter by.
            limit: Maximum number of sessions to return (newest first).

        Returns:
            List of session info dicts sorted by modified time (newest first).
        """
        sessions = []

        if project_path:
            # Find the specific project directory
            project_dir_name = self._path_to_project_dir(project_path)
            project_dirs = [self.claude_dir / project_dir_name]
        else:
            # Search all project directories
            project_dirs = [d for d in self.claude_dir.iterdir() if d.is_dir() and d.name != "."]

        for project_dir in project_dirs:
            if not project_dir.exists():
                continue

            index_path = project_dir / "sessions-index.json"
            if not index_path.exists():
                continue

            try:
                with open(index_path) as f:
                    index_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            original_path = index_data.get("originalPath", "")

            for entry in index_data.get("entries", []):
                sessions.append(
                    {
                        "session_id": entry.get("sessionId"),
                        "project_path": entry.get("projectPath", original_path),
                        "project_dir": project_dir.name,
                        "first_prompt": entry.get("firstPrompt"),
                        "summary": entry.get("summary"),
                        "message_count": entry.get("messageCount", 0),
                        "created": entry.get("created"),
                        "modified": entry.get("modified"),
                        "git_branch": entry.get("gitBranch"),
                        "full_path": entry.get("fullPath"),
                    }
                )

        # Sort by modified time (newest first)
        sessions.sort(key=lambda s: s.get("modified") or "", reverse=True)

        if limit:
            sessions = sessions[:limit]

        return sessions

    def get_session_path(self, session_id: str, project_path: str | None = None) -> Path | None:
        """Get the file path for a session.

        Args:
            session_id: Session UUID.
            project_path: Optional project path to narrow search.

        Returns:
            Path to the session JSONL file, or None if not found.
        """
        sessions = self.list_sessions(project_path=project_path)
        for session in sessions:
            if session["session_id"] == session_id:
                full_path = session.get("full_path")
                if full_path:
                    return Path(full_path)
                # Fallback: construct path from project_dir
                project_dir = session.get("project_dir")
                if project_dir:
                    return self.claude_dir / project_dir / f"{session_id}.jsonl"
        return None

    def parse_session(
        self,
        session_id: str,
        project_path: str | None = None,
    ) -> SessionTranscript | None:
        """Parse a session transcript from its JSONL file.

        Args:
            session_id: Session UUID.
            project_path: Optional project path to narrow search.

        Returns:
            Parsed SessionTranscript, or None if not found.
        """
        session_path = self.get_session_path(session_id, project_path)
        if not session_path or not session_path.exists():
            return None

        # Get session metadata from index
        sessions = self.list_sessions(project_path=project_path)
        session_info = next((s for s in sessions if s["session_id"] == session_id), None)

        transcript = SessionTranscript(
            session_id=session_id,
            project_path=session_info.get("project_path", "") if session_info else "",
            summary=session_info.get("summary") if session_info else None,
            first_prompt=session_info.get("first_prompt") if session_info else None,
            git_branch=session_info.get("git_branch") if session_info else None,
            message_count=session_info.get("message_count", 0) if session_info else 0,
        )

        # Parse created/modified timestamps
        if session_info:
            if session_info.get("created"):
                try:
                    transcript.created = datetime.fromisoformat(
                        session_info["created"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
            if session_info.get("modified"):
                try:
                    transcript.modified = datetime.fromisoformat(
                        session_info["modified"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

        # Parse JSONL file
        messages = []
        with open(session_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip non-message entries
                entry_type = entry.get("type")
                if entry_type not in ("user", "assistant"):
                    continue

                # Extract message content
                message_data = entry.get("message", {})
                content = message_data.get("content", [])

                # Handle string content (convert to list format)
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]

                # Parse timestamp
                timestamp_str = entry.get("timestamp")
                timestamp = datetime.now()
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    except ValueError:
                        pass

                msg = SessionMessage(
                    uuid=entry.get("uuid", ""),
                    role=message_data.get("role", entry_type),
                    content=content,
                    timestamp=timestamp,
                    parent_uuid=entry.get("parentUuid"),
                    message_type=entry_type,
                )
                messages.append(msg)

        transcript.messages = messages
        return transcript

    def get_session_index(self, project_path: str) -> dict[str, Any]:
        """Get the sessions-index.json content for a project.

        Args:
            project_path: Project path.

        Returns:
            Index data dictionary, or empty dict if not found.
        """
        project_dir_name = self._path_to_project_dir(project_path)
        index_path = self.claude_dir / project_dir_name / "sessions-index.json"

        if not index_path.exists():
            return {}

        try:
            with open(index_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def get_sessions_since(
        self,
        since: datetime,
        project_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get sessions modified since a given time.

        Args:
            since: Datetime threshold (can be timezone-aware or naive).
            project_path: Optional project path filter.

        Returns:
            List of session info dicts for sessions modified after 'since'.
        """
        sessions = self.list_sessions(project_path=project_path)
        filtered = []

        for session in sessions:
            modified_str = session.get("modified")
            if not modified_str:
                continue

            try:
                modified = datetime.fromisoformat(modified_str.replace("Z", "+00:00"))
                # Make comparison work for both naive and aware datetimes
                # by comparing in UTC or stripping timezone
                if modified.tzinfo is not None and since.tzinfo is None:
                    # Convert modified to naive by removing tzinfo
                    modified = modified.replace(tzinfo=None)
                elif modified.tzinfo is None and since.tzinfo is not None:
                    # Convert since to naive
                    since = since.replace(tzinfo=None)
                if modified > since:
                    filtered.append(session)
            except ValueError:
                continue

        return filtered
