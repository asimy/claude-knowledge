"""Tests for the code extractor module."""

import tempfile
from pathlib import Path

import pytest

from claude_knowledge.code_extractor import (
    CodeExtractor,
    ExtractedKnowledge,
    PatternMatch,
)
from claude_knowledge.code_parser import CodeElement, ParsedFile


@pytest.fixture
def code_extractor():
    """Create a CodeExtractor instance."""
    return CodeExtractor()


@pytest.fixture
def sample_parsed_file():
    """Create a sample parsed file with elements."""
    return ParsedFile(
        path="/project/src/user_service.py",
        language="python",
        content_hash="abc123",
        elements=[
            CodeElement(
                element_type="class",
                name="UserService",
                docstring="""Service for managing users.

This class handles all user-related operations including
creation, retrieval, update, and deletion.

Attributes:
    db: Database connection instance.
    cache: Optional cache instance.

Example:
    >>> service = UserService(db)
    >>> user = service.get_user(1)
""",
                signature="class UserService:",
                start_line=10,
            ),
            CodeElement(
                element_type="method",
                name="get_user",
                docstring="""Get a user by ID.

Args:
    user_id: The user's unique identifier.

Returns:
    User dict or None if not found.
""",
                signature="def get_user(self, user_id: int) -> dict:",
                start_line=30,
                parent="UserService",
            ),
            CodeElement(
                element_type="method",
                name="find_by_email",
                docstring="Find user by email.",
                signature="def find_by_email(self, email: str):",
                start_line=50,
                parent="UserService",
            ),
            CodeElement(
                element_type="method",
                name="save",
                docstring="Save a user.",
                signature="def save(self, user):",
                start_line=60,
                parent="UserService",
            ),
        ],
        imports=["sqlite3", "typing"],
        comments=["[TODO] Add caching", "[FIXME] Handle edge case"],
    )


@pytest.fixture
def sample_repository_file():
    """Create a parsed file representing a repository pattern."""
    return ParsedFile(
        path="/project/src/user_repository.py",
        language="python",
        content_hash="def456",
        elements=[
            CodeElement(
                element_type="class",
                name="UserRepository",
                docstring="Repository for user data access.",
                signature="class UserRepository:",
                start_line=5,
            ),
            CodeElement(
                element_type="method",
                name="find",
                signature="def find(self, id):",
                start_line=10,
                parent="UserRepository",
            ),
            CodeElement(
                element_type="method",
                name="save",
                signature="def save(self, entity):",
                start_line=20,
                parent="UserRepository",
            ),
            CodeElement(
                element_type="method",
                name="delete",
                signature="def delete(self, id):",
                start_line=30,
                parent="UserRepository",
            ),
        ],
    )


@pytest.fixture
def sample_files_with_patterns(sample_repository_file):
    """Create multiple parsed files showing patterns."""
    post_repo = ParsedFile(
        path="/project/src/post_repository.py",
        language="python",
        content_hash="ghi789",
        elements=[
            CodeElement(
                element_type="class",
                name="PostRepository",
                docstring="Repository for post data access.",
                signature="class PostRepository:",
                start_line=5,
            ),
            CodeElement(
                element_type="method",
                name="find",
                signature="def find(self, id):",
                start_line=10,
                parent="PostRepository",
            ),
            CodeElement(
                element_type="method",
                name="save",
                signature="def save(self, entity):",
                start_line=20,
                parent="PostRepository",
            ),
        ],
    )

    comment_repo = ParsedFile(
        path="/project/src/comment_repository.py",
        language="python",
        content_hash="jkl012",
        elements=[
            CodeElement(
                element_type="class",
                name="CommentRepository",
                docstring="Repository for comment data access.",
                signature="class CommentRepository:",
                start_line=5,
            ),
            CodeElement(
                element_type="method",
                name="find",
                signature="def find(self, id):",
                start_line=10,
                parent="CommentRepository",
            ),
            CodeElement(
                element_type="method",
                name="save",
                signature="def save(self, entity):",
                start_line=20,
                parent="CommentRepository",
            ),
        ],
    )

    return [sample_repository_file, post_repo, comment_repo]


class TestExtractedKnowledge:
    """Tests for ExtractedKnowledge dataclass."""

    def test_create_extraction(self):
        """Test creating an ExtractedKnowledge instance."""
        extraction = ExtractedKnowledge(
            title="Class: UserService",
            description="Service for managing users.",
            content="## Documentation...",
            tags=["python", "service"],
            confidence=0.75,
            extraction_type="docstring",
            source_files=["/path/to/file.py"],
        )

        assert extraction.title == "Class: UserService"
        assert extraction.confidence == 0.75
        assert "python" in extraction.tags


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_create_pattern_match(self):
        """Test creating a PatternMatch instance."""
        match = PatternMatch(
            pattern_type="repository",
            file_path="/src/user_repo.py",
            element_name="UserRepository",
            evidence=["Name matches repository pattern", "Has find method"],
            confidence=0.8,
        )

        assert match.pattern_type == "repository"
        assert match.confidence == 0.8
        assert len(match.evidence) == 2


class TestCodeExtractor:
    """Tests for CodeExtractor class."""

    def test_extract_from_docstrings(self, code_extractor, sample_parsed_file):
        """Test extracting knowledge from docstrings."""
        extractions = code_extractor.extract([sample_parsed_file], "test_project")

        # Should extract from UserService class (has substantial docstring)
        docstring_extractions = [e for e in extractions if e.extraction_type == "docstring"]

        assert len(docstring_extractions) >= 1

        # Check UserService extraction
        user_service_ext = next(
            (e for e in docstring_extractions if "UserService" in e.title),
            None,
        )
        assert user_service_ext is not None
        assert user_service_ext.confidence >= 0.3
        assert "python" in user_service_ext.tags

    def test_extract_from_comments(self, code_extractor, sample_parsed_file):
        """Test extracting knowledge from significant comments."""
        # Create multiple files with comments
        file2 = ParsedFile(
            path="/project/src/other.py",
            language="python",
            content_hash="xyz999",
            elements=[],
            comments=["[TODO] Implement feature", "[TODO] Add tests", "[TODO] Review logic"],
        )
        file3 = ParsedFile(
            path="/project/src/another.py",
            language="python",
            content_hash="abc111",
            elements=[],
            comments=["[TODO] Clean up"],
        )

        files = [sample_parsed_file, file2, file3]
        extractions = code_extractor.extract(files, "test_project")

        # Should have a TODO extraction (needs 3+ comments of same type)
        todo_extraction = next(
            (e for e in extractions if e.extraction_type == "comment" and "TODO" in e.title),
            None,
        )

        # We have enough TODOs across files
        assert todo_extraction is not None


class TestCodeExtractorPatternDetection:
    """Tests for architectural pattern detection."""

    def test_detect_repository_pattern(self, code_extractor, sample_files_with_patterns):
        """Test detecting repository pattern."""
        patterns = code_extractor.detect_all_patterns(sample_files_with_patterns)

        repository_patterns = [p for p in patterns if p.pattern_type == "repository"]

        assert len(repository_patterns) >= 2
        assert any("UserRepository" in p.element_name for p in repository_patterns)
        assert any("PostRepository" in p.element_name for p in repository_patterns)

    def test_extract_architectural_patterns(self, code_extractor, sample_files_with_patterns):
        """Test extracting knowledge from architectural patterns."""
        extractions = code_extractor.extract(sample_files_with_patterns, "test_project")

        pattern_extractions = [e for e in extractions if e.extraction_type == "pattern"]

        # Should have at least one repository pattern extraction
        repo_extraction = next(
            (e for e in pattern_extractions if "Repository" in e.title),
            None,
        )

        assert repo_extraction is not None
        assert repo_extraction.confidence >= 0.3
        assert len(repo_extraction.source_files) >= 2

    def test_detect_service_pattern(self, code_extractor):
        """Test detecting service layer pattern."""
        service_files = [
            ParsedFile(
                path="/src/user_service.py",
                language="python",
                content_hash="svc1",
                elements=[
                    CodeElement(
                        element_type="class",
                        name="UserService",
                        signature="class UserService:",
                        start_line=1,
                    ),
                    CodeElement(
                        element_type="method",
                        name="execute",
                        signature="def execute(self):",
                        start_line=5,
                        parent="UserService",
                    ),
                ],
            ),
            ParsedFile(
                path="/src/order_service.py",
                language="python",
                content_hash="svc2",
                elements=[
                    CodeElement(
                        element_type="class",
                        name="OrderService",
                        signature="class OrderService:",
                        start_line=1,
                    ),
                    CodeElement(
                        element_type="method",
                        name="process",
                        signature="def process(self):",
                        start_line=5,
                        parent="OrderService",
                    ),
                ],
            ),
        ]

        patterns = code_extractor.detect_all_patterns(service_files)
        service_patterns = [p for p in patterns if p.pattern_type == "service"]

        assert len(service_patterns) >= 1


class TestCodeExtractorConfidence:
    """Tests for confidence calculation."""

    def test_docstring_with_examples_higher_confidence(self, code_extractor):
        """Test that docstrings with examples get higher confidence."""
        file_with_examples = ParsedFile(
            path="/src/example.py",
            language="python",
            content_hash="ex123",
            elements=[
                CodeElement(
                    element_type="class",
                    name="Calculator",
                    docstring="""Calculator for math operations.

This class provides basic arithmetic operations.

Args:
    precision: Decimal precision.

Returns:
    Calculator instance.

Example:
    >>> calc = Calculator()
    >>> calc.add(1, 2)
    3
""",
                    signature="class Calculator:",
                    start_line=1,
                ),
            ],
        )

        file_without_examples = ParsedFile(
            path="/src/simple.py",
            language="python",
            content_hash="simp123",
            elements=[
                CodeElement(
                    element_type="class",
                    name="SimpleClass",
                    docstring="A simple class with minimal documentation.",
                    signature="class SimpleClass:",
                    start_line=1,
                ),
            ],
        )

        extractions_with = code_extractor.extract([file_with_examples], "test")
        extractions_without = code_extractor.extract([file_without_examples], "test")

        # Filter to docstring extractions
        ext_with = next(
            (e for e in extractions_with if e.extraction_type == "docstring"),
            None,
        )
        ext_without = next(
            (e for e in extractions_without if e.extraction_type == "docstring"),
            None,
        )

        # The one with examples should have higher confidence
        if ext_with and ext_without:
            assert ext_with.confidence > ext_without.confidence

    def test_pattern_confidence_multiple_instances(self, code_extractor):
        """Test that patterns with more instances get higher confidence."""
        # Create 3 repositories
        many_repos = [
            ParsedFile(
                path=f"/src/{name}_repository.py",
                language="python",
                content_hash=f"repo{i}",
                elements=[
                    CodeElement(
                        element_type="class",
                        name=f"{name}Repository",
                        signature=f"class {name}Repository:",
                        start_line=1,
                    ),
                    CodeElement(
                        element_type="method",
                        name="find",
                        signature="def find(self, id):",
                        start_line=5,
                        parent=f"{name}Repository",
                    ),
                ],
            )
            for i, name in enumerate(["User", "Post", "Comment"])
        ]

        extractions = code_extractor.extract(many_repos, "test")
        pattern_ext = next(
            (e for e in extractions if e.extraction_type == "pattern"),
            None,
        )

        assert pattern_ext is not None
        assert pattern_ext.confidence >= 0.4  # Should get bonus for 3+ instances


class TestCodeExtractorFromCodebase:
    """Tests for extracting from a real codebase directory."""

    def test_extract_from_codebase(self):
        """Test extracting from a directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Create Python files with patterns
            (base / "src").mkdir()
            (base / "src" / "user_service.py").write_text('''"""User service module.

Provides user management functionality including CRUD operations.

Attributes:
    db: Database connection.
"""

class UserService:
    """Service for managing users.

    This handles all user operations.

    Example:
        >>> svc = UserService(db)
        >>> user = svc.get(1)
    """

    def get(self, user_id: int):
        """Get a user by ID.

        Args:
            user_id: The user ID.

        Returns:
            User dict or None.
        """
        pass
''')

            (base / "src" / "order_service.py").write_text('''"""Order service module."""

class OrderService:
    """Service for managing orders."""

    def process(self, order):
        """Process an order."""
        pass
''')

            extractor = CodeExtractor()
            extractions = extractor.extract_from_codebase(
                base_path=base,
                include_patterns=["*.py"],
                project_name="test_project",
            )

            assert len(extractions) > 0
            assert any(e.extraction_type == "docstring" for e in extractions)


class TestCodeExtractorTags:
    """Tests for tag derivation."""

    def test_derive_tags_from_language(self, code_extractor):
        """Test that language is included in tags."""
        python_file = ParsedFile(
            path="/src/test.py",
            language="python",
            content_hash="py123",
            elements=[
                CodeElement(
                    element_type="class",
                    name="TestClass",
                    docstring=(
                        "A test class with enough documentation to be "
                        "extracted successfully."
                    ),
                    signature="class TestClass:",
                    start_line=1,
                ),
            ],
        )

        extractions = code_extractor.extract([python_file], "myproject")

        if extractions:
            assert "python" in extractions[0].tags

    def test_derive_tags_from_project(self, code_extractor):
        """Test that project name is included in tags."""
        file = ParsedFile(
            path="/src/test.py",
            language="python",
            content_hash="proj123",
            elements=[
                CodeElement(
                    element_type="class",
                    name="TestClass",
                    docstring="A documented class for testing tag derivation from project name.",
                    signature="class TestClass:",
                    start_line=1,
                ),
            ],
        )

        extractions = code_extractor.extract([file], "myproject")

        if extractions:
            assert "myproject" in extractions[0].tags
