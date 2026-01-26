"""Tests for the session parser module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from claude_knowledge.session_parser import (
    SessionMessage,
    SessionParser,
    SessionTranscript,
)


@pytest.fixture
def temp_claude_dir():
    """Create a temporary Claude projects directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        claude_dir = Path(tmpdir) / ".claude" / "projects"
        claude_dir.mkdir(parents=True)
        yield claude_dir


@pytest.fixture
def sample_session_data():
    """Sample session JSONL data."""
    return [
        {
            "type": "user",
            "uuid": "user-1",
            "parentUuid": None,
            "timestamp": "2026-01-26T10:00:00.000Z",
            "message": {
                "role": "user",
                "content": "How do I fix this authentication error?",
            },
        },
        {
            "type": "assistant",
            "uuid": "assistant-1",
            "parentUuid": "user-1",
            "timestamp": "2026-01-26T10:00:05.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll help you fix the authentication error."},
                    {
                        "type": "tool_use",
                        "name": "Edit",
                        "input": {
                            "file_path": "/app/auth.py",
                            "old_string": "token = None",
                            "new_string": "token = get_token()",
                        },
                    },
                ],
            },
        },
        {
            "type": "user",
            "uuid": "user-2",
            "parentUuid": "assistant-1",
            "timestamp": "2026-01-26T10:01:00.000Z",
            "message": {
                "role": "user",
                "content": "Thanks, that works!",
            },
        },
    ]


@pytest.fixture
def sample_index_data():
    """Sample sessions-index.json data."""
    return {
        "version": 1,
        "entries": [
            {
                "sessionId": "test-session-123",
                "fullPath": "/path/to/session.jsonl",
                "firstPrompt": "How do I fix this authentication error?",
                "summary": "Fixed authentication error",
                "messageCount": 3,
                "created": "2026-01-26T10:00:00.000Z",
                "modified": "2026-01-26T10:01:00.000Z",
                "gitBranch": "main",
                "projectPath": "/Users/test/myproject",
            }
        ],
        "originalPath": "/Users/test/myproject",
    }


class TestSessionMessage:
    """Tests for SessionMessage dataclass."""

    def test_text_content_from_string(self):
        """Test extracting text content from string message."""
        msg = SessionMessage(
            uuid="test-1",
            role="user",
            content="Hello, world!",
            timestamp=datetime.now(),
        )
        # String content should be handled
        assert "Hello" in msg.text_content or msg.text_content == ""

    def test_text_content_from_list(self):
        """Test extracting text content from list message."""
        msg = SessionMessage(
            uuid="test-1",
            role="assistant",
            content=[
                {"type": "text", "text": "Here is the solution."},
                {"type": "tool_use", "name": "Edit", "input": {}},
            ],
            timestamp=datetime.now(),
        )
        assert "Here is the solution" in msg.text_content

    def test_tool_uses_extraction(self):
        """Test extracting tool use blocks."""
        msg = SessionMessage(
            uuid="test-1",
            role="assistant",
            content=[
                {"type": "text", "text": "Let me help."},
                {"type": "tool_use", "name": "Edit", "input": {"file": "test.py"}},
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            ],
            timestamp=datetime.now(),
        )
        tools = msg.tool_uses
        assert len(tools) == 2
        assert tools[0]["name"] == "Edit"
        assert tools[1]["name"] == "Bash"

    def test_tool_results_extraction(self):
        """Test extracting tool result blocks."""
        msg = SessionMessage(
            uuid="test-1",
            role="user",
            content=[
                {"type": "tool_result", "content": "Success"},
            ],
            timestamp=datetime.now(),
        )
        results = msg.tool_results
        assert len(results) == 1


class TestSessionTranscript:
    """Tests for SessionTranscript dataclass."""

    def test_get_conversation_pairs(self):
        """Test getting user-assistant pairs."""
        user_msg = SessionMessage(
            uuid="user-1",
            role="user",
            content=[{"type": "text", "text": "Question"}],
            timestamp=datetime.now(),
        )
        assistant_msg = SessionMessage(
            uuid="assistant-1",
            role="assistant",
            content=[{"type": "text", "text": "Answer"}],
            timestamp=datetime.now(),
            parent_uuid="user-1",
        )
        transcript = SessionTranscript(
            session_id="test-123",
            project_path="/test/project",
            messages=[user_msg, assistant_msg],
        )
        pairs = transcript.get_conversation_pairs()
        assert len(pairs) == 1
        assert pairs[0][0].uuid == "user-1"
        assert pairs[0][1].uuid == "assistant-1"


class TestSessionParser:
    """Tests for SessionParser class."""

    def test_path_to_project_dir(self, temp_claude_dir):
        """Test converting path to project directory name."""
        parser = SessionParser(claude_dir=temp_claude_dir)
        result = parser._path_to_project_dir("/Users/glen/code/myproject")
        assert result == "-Users-glen-code-myproject"

    def test_project_dir_to_path(self, temp_claude_dir):
        """Test converting project directory name back to path."""
        parser = SessionParser(claude_dir=temp_claude_dir)
        result = parser._project_dir_to_path("Users-glen-code-myproject")
        assert result == "/Users/glen/code/myproject"

    def test_list_projects_empty(self, temp_claude_dir):
        """Test listing projects with empty directory."""
        parser = SessionParser(claude_dir=temp_claude_dir)
        projects = parser.list_projects()
        assert projects == []

    def test_list_projects(self, temp_claude_dir, sample_index_data, sample_session_data):
        """Test listing projects with sessions."""
        # Create a project directory with session files
        project_dir = temp_claude_dir / "-Users-test-myproject"
        project_dir.mkdir()

        # Create sessions-index.json
        index_path = project_dir / "sessions-index.json"
        with open(index_path, "w") as f:
            json.dump(sample_index_data, f)

        # Create a session JSONL file
        session_path = project_dir / "test-session-123.jsonl"
        with open(session_path, "w") as f:
            for entry in sample_session_data:
                f.write(json.dumps(entry) + "\n")

        parser = SessionParser(claude_dir=temp_claude_dir)
        projects = parser.list_projects()

        assert len(projects) == 1
        assert projects[0]["path"] == "/Users/test/myproject"
        assert projects[0]["session_count"] == 1

    def test_list_sessions(self, temp_claude_dir, sample_index_data, sample_session_data):
        """Test listing sessions."""
        # Setup project directory
        project_dir = temp_claude_dir / "-Users-test-myproject"
        project_dir.mkdir()

        index_path = project_dir / "sessions-index.json"
        with open(index_path, "w") as f:
            json.dump(sample_index_data, f)

        session_path = project_dir / "test-session-123.jsonl"
        with open(session_path, "w") as f:
            for entry in sample_session_data:
                f.write(json.dumps(entry) + "\n")

        parser = SessionParser(claude_dir=temp_claude_dir)
        sessions = parser.list_sessions()

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "test-session-123"
        assert sessions[0]["first_prompt"] == "How do I fix this authentication error?"
        assert sessions[0]["summary"] == "Fixed authentication error"

    def test_parse_session(self, temp_claude_dir, sample_index_data, sample_session_data):
        """Test parsing a session transcript."""
        # Setup project directory
        project_dir = temp_claude_dir / "-Users-test-myproject"
        project_dir.mkdir()

        # Update sample_index_data with correct path
        sample_index_data["entries"][0]["fullPath"] = str(project_dir / "test-session-123.jsonl")

        index_path = project_dir / "sessions-index.json"
        with open(index_path, "w") as f:
            json.dump(sample_index_data, f)

        session_path = project_dir / "test-session-123.jsonl"
        with open(session_path, "w") as f:
            for entry in sample_session_data:
                f.write(json.dumps(entry) + "\n")

        parser = SessionParser(claude_dir=temp_claude_dir)
        transcript = parser.parse_session("test-session-123")

        assert transcript is not None
        assert transcript.session_id == "test-session-123"
        assert transcript.summary == "Fixed authentication error"
        assert len(transcript.messages) == 3

        # Check message roles
        roles = [m.role for m in transcript.messages]
        assert roles == ["user", "assistant", "user"]

    def test_get_sessions_since(self, temp_claude_dir, sample_index_data, sample_session_data):
        """Test getting sessions since a given time."""
        # Setup project directory
        project_dir = temp_claude_dir / "-Users-test-myproject"
        project_dir.mkdir()

        index_path = project_dir / "sessions-index.json"
        with open(index_path, "w") as f:
            json.dump(sample_index_data, f)

        session_path = project_dir / "test-session-123.jsonl"
        with open(session_path, "w") as f:
            for entry in sample_session_data:
                f.write(json.dumps(entry) + "\n")

        parser = SessionParser(claude_dir=temp_claude_dir)

        # Sessions since a date before the session
        from datetime import datetime

        old_date = datetime(2026, 1, 1)
        sessions = parser.get_sessions_since(old_date)
        assert len(sessions) == 1

        # Sessions since a date after the session
        future_date = datetime(2027, 1, 1)
        sessions = parser.get_sessions_since(future_date)
        assert len(sessions) == 0

    def test_get_session_index(self, temp_claude_dir, sample_index_data):
        """Test getting session index data."""
        # Setup project directory
        project_dir = temp_claude_dir / "-Users-test-myproject"
        project_dir.mkdir()

        index_path = project_dir / "sessions-index.json"
        with open(index_path, "w") as f:
            json.dump(sample_index_data, f)

        parser = SessionParser(claude_dir=temp_claude_dir)
        index = parser.get_session_index("/Users/test/myproject")

        assert index["version"] == 1
        assert len(index["entries"]) == 1

    def test_get_session_index_not_found(self, temp_claude_dir):
        """Test getting session index for non-existent project."""
        parser = SessionParser(claude_dir=temp_claude_dir)
        index = parser.get_session_index("/nonexistent/project")
        assert index == {}
