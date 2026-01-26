"""Tests for the session extractor module."""

from datetime import datetime

import pytest

from claude_knowledge.session_extractor import ExtractedKnowledge, SessionExtractor
from claude_knowledge.session_parser import SessionMessage, SessionTranscript


@pytest.fixture
def extractor():
    """Create a SessionExtractor instance."""
    return SessionExtractor()


@pytest.fixture
def sample_transcript():
    """Create a sample transcript for testing."""
    user_msg = SessionMessage(
        uuid="user-1",
        role="user",
        content=[{"type": "text", "text": "How do I fix this authentication error?"}],
        timestamp=datetime.now(),
    )
    assistant_msg = SessionMessage(
        uuid="assistant-1",
        role="assistant",
        content=[
            {
                "type": "text",
                "text": "I'll help you fix the auth error by updating the token handling.",
            },
            {
                "type": "tool_use",
                "id": "tool-1",
                "name": "Edit",
                "input": {
                    "file_path": "/app/auth.py",
                    "old_string": "token = None",
                    "new_string": "token = get_token()",
                },
            },
        ],
        timestamp=datetime.now(),
        parent_uuid="user-1",
    )
    followup_msg = SessionMessage(
        uuid="user-2",
        role="user",
        content=[{"type": "text", "text": "Thanks, that works!"}],
        timestamp=datetime.now(),
        parent_uuid="assistant-1",
    )
    return SessionTranscript(
        session_id="test-123",
        project_path="/Users/test/myproject",
        messages=[user_msg, assistant_msg, followup_msg],
        summary="Fixed authentication error",
    )


class TestExtractedKnowledge:
    """Tests for ExtractedKnowledge dataclass."""

    def test_creation(self):
        """Test creating an ExtractedKnowledge instance."""
        knowledge = ExtractedKnowledge(
            title="Fix authentication error",
            description="Update token handling",
            content="token = get_token()",
            tags=["auth", "python"],
            confidence=0.8,
            extraction_type="problem_solution",
        )
        assert knowledge.title == "Fix authentication error"
        assert knowledge.confidence == 0.8
        assert "auth" in knowledge.tags

    def test_default_values(self):
        """Test default values."""
        knowledge = ExtractedKnowledge(
            title="Test",
            description="Test description",
            content="Test content",
        )
        assert knowledge.tags == []
        assert knowledge.confidence == 0.5
        assert knowledge.extraction_type == "general"


class TestSessionExtractor:
    """Tests for SessionExtractor class."""

    def test_is_question_or_problem_question_mark(self, extractor):
        """Test detecting questions with question mark."""
        assert extractor._is_question_or_problem("How do I fix this?")
        assert extractor._is_question_or_problem("What is the error?")

    def test_is_question_or_problem_patterns(self, extractor):
        """Test detecting questions with keyword patterns."""
        assert extractor._is_question_or_problem("How do I implement this")
        assert extractor._is_question_or_problem("Can you help me with this")
        assert extractor._is_question_or_problem("Fix the authentication bug")
        assert extractor._is_question_or_problem("Error in the login flow")

    def test_is_question_or_problem_negative(self, extractor):
        """Test that non-questions are not detected."""
        assert not extractor._is_question_or_problem("Thanks for the help")
        assert not extractor._is_question_or_problem("Looks good")

    def test_calculate_confidence_base(self, extractor):
        """Test base confidence calculation."""
        user_msg = SessionMessage(
            uuid="user-1",
            role="user",
            content=[{"type": "text", "text": "How to fix this?"}],
            timestamp=datetime.now(),
        )
        assistant_msg = SessionMessage(
            uuid="assistant-1",
            role="assistant",
            content=[{"type": "text", "text": "Here is the fix."}],
            timestamp=datetime.now(),
        )
        confidence = extractor._calculate_confidence(
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            has_code=False,
            success_indicators={},
        )
        # Base score is 0.4
        assert 0.35 <= confidence <= 0.5

    def test_calculate_confidence_with_code(self, extractor):
        """Test confidence boost with code."""
        user_msg = SessionMessage(
            uuid="user-1",
            role="user",
            content=[{"type": "text", "text": "How to fix this?"}],
            timestamp=datetime.now(),
        )
        assistant_msg = SessionMessage(
            uuid="assistant-1",
            role="assistant",
            content=[
                {"type": "tool_use", "name": "Edit", "input": {}},
            ],
            timestamp=datetime.now(),
        )
        confidence = extractor._calculate_confidence(
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            has_code=True,
            success_indicators={},
        )
        # Base (0.4) + code (0.15) + tool use (0.1) = 0.65
        assert confidence >= 0.6

    def test_calculate_confidence_with_acknowledgment(self, extractor):
        """Test confidence boost with user acknowledgment."""
        user_msg = SessionMessage(
            uuid="user-1",
            role="user",
            content=[{"type": "text", "text": "How to fix this?"}],
            timestamp=datetime.now(),
        )
        assistant_msg = SessionMessage(
            uuid="assistant-1",
            role="assistant",
            content=[{"type": "text", "text": "Fixed."}],
            timestamp=datetime.now(),
        )
        confidence = extractor._calculate_confidence(
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            has_code=False,
            success_indicators={"user_acknowledged": True},
        )
        # Base (0.4) + acknowledgment (0.2) = 0.6
        assert confidence >= 0.55

    def test_calculate_confidence_with_corrections(self, extractor):
        """Test confidence penalty with follow-up corrections."""
        user_msg = SessionMessage(
            uuid="user-1",
            role="user",
            content=[{"type": "text", "text": "How to fix this?"}],
            timestamp=datetime.now(),
        )
        assistant_msg = SessionMessage(
            uuid="assistant-1",
            role="assistant",
            content=[{"type": "text", "text": "Here is the fix."}],
            timestamp=datetime.now(),
        )
        confidence = extractor._calculate_confidence(
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            has_code=False,
            success_indicators={"has_follow_up_corrections": True},
        )
        # Base (0.4) - corrections (0.15) = 0.25
        assert confidence <= 0.3

    def test_generate_title_from_question(self, extractor):
        """Test title generation from user question."""
        title = extractor._generate_title("How do I fix this authentication error?")
        assert title
        assert len(title) <= 83  # 80 + "..."

    def test_generate_title_removes_prefix(self, extractor):
        """Test title removes common prefixes."""
        title = extractor._generate_title("Can you please help me fix the bug?")
        assert not title.lower().startswith("can you")

    def test_generate_title_truncates_long(self, extractor):
        """Test title truncation for long questions."""
        long_question = "A" * 200
        title = extractor._generate_title(long_question)
        assert len(title) <= 83

    def test_derive_tags_project(self, extractor):
        """Test deriving tags from project name."""
        transcript = SessionTranscript(
            session_id="test-1",
            project_path="/Users/test/myproject",
            messages=[],
        )
        assistant_msg = SessionMessage(
            uuid="a-1",
            role="assistant",
            content=[],
            timestamp=datetime.now(),
        )
        tags = extractor._derive_tags(transcript, assistant_msg, "test content")
        assert "myproject" in tags

    def test_derive_tags_language(self, extractor):
        """Test deriving tags from detected language."""
        transcript = SessionTranscript(
            session_id="test-1",
            project_path="/test",
            messages=[],
        )
        assistant_msg = SessionMessage(
            uuid="a-1",
            role="assistant",
            content=[],
            timestamp=datetime.now(),
        )
        content = """
def authenticate(token):
    import jwt
    return jwt.decode(token)
"""
        tags = extractor._derive_tags(transcript, assistant_msg, content)
        assert "python" in tags

    def test_derive_tags_topics(self, extractor):
        """Test deriving tags from topic patterns."""
        transcript = SessionTranscript(
            session_id="test-1",
            project_path="/test",
            messages=[],
        )
        assistant_msg = SessionMessage(
            uuid="a-1",
            role="assistant",
            content=[],
            timestamp=datetime.now(),
        )
        content = "Use this API endpoint for authentication with JWT tokens"
        tags = extractor._derive_tags(transcript, assistant_msg, content)
        assert "api" in tags or "auth" in tags

    def test_detect_language_python(self, extractor):
        """Test detecting Python language."""
        content = """
def hello():
    print("Hello, world!")
"""
        assert extractor._detect_language(content) == "python"

    def test_detect_language_javascript(self, extractor):
        """Test detecting JavaScript language."""
        content = """
const greet = (name) => {
    console.log(`Hello, ${name}!`);
};
"""
        assert extractor._detect_language(content) == "javascript"

    def test_detect_language_go(self, extractor):
        """Test detecting Go language."""
        content = """
package main

func main() {
    fmt.Println("Hello")
}
"""
        assert extractor._detect_language(content) == "go"

    def test_extract_basic(self, extractor, sample_transcript):
        """Test basic extraction from transcript."""
        extractions = extractor.extract(sample_transcript)
        # Should extract at least one entry
        assert len(extractions) >= 1

        # Check the extraction
        ext = extractions[0]
        assert ext.title
        assert ext.description
        assert ext.content
        assert ext.confidence > 0

    def test_extract_confidence_threshold(self, extractor):
        """Test that low-confidence extractions are filtered."""
        # Create a transcript with minimal information
        user_msg = SessionMessage(
            uuid="user-1",
            role="user",
            content=[{"type": "text", "text": "ok"}],
            timestamp=datetime.now(),
        )
        assistant_msg = SessionMessage(
            uuid="assistant-1",
            role="assistant",
            content=[{"type": "text", "text": "Sure"}],
            timestamp=datetime.now(),
            parent_uuid="user-1",
        )
        transcript = SessionTranscript(
            session_id="test-1",
            project_path="/test",
            messages=[user_msg, assistant_msg],
        )
        extractions = extractor.extract(transcript)
        # Should not extract anything (not a question)
        assert len(extractions) == 0

    def test_identify_problem_solutions(self, extractor, sample_transcript):
        """Test identifying problem-solution pairs."""
        extractions = extractor.identify_problem_solutions(sample_transcript)
        # Should find problem-solution type extractions
        for ext in extractions:
            assert ext.extraction_type == "problem_solution"

    def test_extract_content_with_edit(self, extractor):
        """Test extracting content from Edit tool use."""
        assistant_msg = SessionMessage(
            uuid="a-1",
            role="assistant",
            content=[
                {"type": "text", "text": "Here is the fix."},
                {
                    "type": "tool_use",
                    "name": "Edit",
                    "input": {
                        "file_path": "/app/test.py",
                        "old_string": "old_code",
                        "new_string": "new_code",
                    },
                },
            ],
            timestamp=datetime.now(),
        )
        content = extractor._extract_content(assistant_msg)
        assert "Here is the fix" in content
        assert "/app/test.py" in content
        assert "new_code" in content

    def test_extract_content_with_write(self, extractor):
        """Test extracting content from Write tool use."""
        assistant_msg = SessionMessage(
            uuid="a-1",
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "name": "Write",
                    "input": {
                        "file_path": "/app/new_file.py",
                        "content": "print('hello')",
                    },
                },
            ],
            timestamp=datetime.now(),
        )
        content = extractor._extract_content(assistant_msg)
        assert "/app/new_file.py" in content
        assert "print('hello')" in content

    def test_check_success_indicators_acknowledgment(self, extractor):
        """Test detecting user acknowledgment."""
        user_msg = SessionMessage(
            uuid="u-1",
            role="user",
            content=[{"type": "text", "text": "Thanks, that works!"}],
            timestamp=datetime.now(),
        )
        assistant_msg = SessionMessage(
            uuid="a-1",
            role="assistant",
            content=[{"type": "text", "text": "Great!"}],
            timestamp=datetime.now(),
        )
        indicators = extractor._check_success_indicators([(user_msg, assistant_msg)])
        assert indicators["user_acknowledged"]

    def test_check_success_indicators_correction(self, extractor):
        """Test detecting follow-up corrections."""
        user_msg = SessionMessage(
            uuid="u-1",
            role="user",
            content=[{"type": "text", "text": "Actually that still doesn't work"}],
            timestamp=datetime.now(),
        )
        assistant_msg = SessionMessage(
            uuid="a-1",
            role="assistant",
            content=[{"type": "text", "text": "Let me try again"}],
            timestamp=datetime.now(),
        )
        indicators = extractor._check_success_indicators([(user_msg, assistant_msg)])
        assert indicators["has_follow_up_corrections"]
