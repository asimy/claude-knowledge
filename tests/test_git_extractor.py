"""Tests for the git extractor module."""

from datetime import datetime

import pytest

from claude_knowledge.git_extractor import ExtractedKnowledge, GitExtractor
from claude_knowledge.git_parser import GitCommit, GitDiff


@pytest.fixture
def git_extractor():
    """Create a GitExtractor instance."""
    return GitExtractor()


@pytest.fixture
def bug_fix_commit():
    """Create a sample bug fix commit."""
    return GitCommit(
        sha="abc123def456",
        message=(
            "fix: resolve authentication error\n\n"
            "This fixes the issue where users couldn't login\n"
            "after the session expired."
        ),
        author="Test User",
        author_email="test@example.com",
        timestamp=datetime(2026, 1, 15, 10, 30),
        diffs=[
            GitDiff(
                file_path="src/auth.py",
                change_type="modified",
                additions=15,
                deletions=5,
                hunks=[],
            ),
            GitDiff(
                file_path="tests/test_auth.py",
                change_type="modified",
                additions=20,
                deletions=0,
                hunks=[],
            ),
        ],
    )


@pytest.fixture
def feature_commit():
    """Create a sample feature commit."""
    return GitCommit(
        sha="def456abc789",
        message="feat(api): add user profile endpoint\n\nImplements GET /api/users/{id}/profile",
        author="Test User",
        author_email="test@example.com",
        timestamp=datetime(2026, 1, 16, 14, 0),
        diffs=[
            GitDiff(
                file_path="src/api/routes.py",
                change_type="modified",
                additions=30,
                deletions=2,
                hunks=[],
            ),
            GitDiff(
                file_path="src/api/handlers.py",
                change_type="modified",
                additions=25,
                deletions=0,
                hunks=[],
            ),
        ],
    )


@pytest.fixture
def refactor_commit():
    """Create a sample refactor commit."""
    return GitCommit(
        sha="789xyz123abc",
        message="refactor: extract database connection to separate module",
        author="Test User",
        author_email="test@example.com",
        timestamp=datetime(2026, 1, 17, 9, 15),
        diffs=[
            GitDiff(
                file_path="src/database.py",
                change_type="added",
                additions=50,
                deletions=0,
                hunks=[],
            ),
            GitDiff(
                file_path="src/main.py",
                change_type="modified",
                additions=5,
                deletions=40,
                hunks=[],
            ),
        ],
    )


@pytest.fixture
def generic_commit():
    """Create a generic/low-value commit."""
    return GitCommit(
        sha="generic123",
        message="update",
        author="Test User",
        author_email="test@example.com",
        timestamp=datetime(2026, 1, 18, 11, 0),
        diffs=[
            GitDiff(
                file_path="config.json",
                change_type="modified",
                additions=1,
                deletions=1,
                hunks=[],
            ),
        ],
    )


@pytest.fixture
def merge_commit():
    """Create a merge commit."""
    return GitCommit(
        sha="merge123456",
        message="Merge branch 'feature/xyz' into main",
        author="Test User",
        author_email="test@example.com",
        timestamp=datetime(2026, 1, 19, 16, 30),
        diffs=[],
    )


class TestGitExtractor:
    """Tests for GitExtractor class."""

    def test_classify_bug_fix(self, git_extractor, bug_fix_commit):
        """Test classifying a bug fix commit."""
        commit_type = git_extractor._classify_commit(bug_fix_commit)
        assert commit_type == "bug_fix"

    def test_classify_feature(self, git_extractor, feature_commit):
        """Test classifying a feature commit."""
        commit_type = git_extractor._classify_commit(feature_commit)
        assert commit_type == "feature"

    def test_classify_refactor(self, git_extractor, refactor_commit):
        """Test classifying a refactor commit."""
        commit_type = git_extractor._classify_commit(refactor_commit)
        assert commit_type == "refactor"

    def test_skip_merge_commits(self, git_extractor, merge_commit):
        """Test that merge commits are skipped."""
        extraction = git_extractor.extract(merge_commit)
        assert extraction is None

    def test_skip_generic_commits(self, git_extractor, generic_commit):
        """Test that generic commits are skipped."""
        extraction = git_extractor.extract(generic_commit)
        assert extraction is None

    def test_extract_bug_fix(self, git_extractor, bug_fix_commit):
        """Test extracting knowledge from bug fix commit."""
        extraction = git_extractor.extract(bug_fix_commit)

        assert extraction is not None
        assert extraction.extraction_type == "bug_fix"
        assert "Fix:" in extraction.title or "authentication" in extraction.title.lower()
        assert extraction.source_sha == bug_fix_commit.sha
        assert extraction.confidence >= 0.3

    def test_extract_feature(self, git_extractor, feature_commit):
        """Test extracting knowledge from feature commit."""
        extraction = git_extractor.extract(feature_commit)

        assert extraction is not None
        assert extraction.extraction_type == "feature"
        assert extraction.confidence >= 0.3

    def test_extract_from_commits(
        self, git_extractor, bug_fix_commit, feature_commit, merge_commit
    ):
        """Test extracting from multiple commits."""
        commits = [bug_fix_commit, feature_commit, merge_commit]
        extractions = git_extractor.extract_from_commits(commits)

        # Should extract from bug_fix and feature, skip merge
        assert len(extractions) == 2

    def test_confidence_with_tests(self, git_extractor, bug_fix_commit):
        """Test that commits with tests get higher confidence."""
        extraction = git_extractor.extract(bug_fix_commit)

        # bug_fix_commit has test changes, so should get bonus
        assert extraction is not None
        assert extraction.confidence >= 0.4  # Base + type + tests

    def test_confidence_without_tests(self, git_extractor, refactor_commit):
        """Test confidence for commits without tests."""
        extraction = git_extractor.extract(refactor_commit)

        assert extraction is not None
        # Should still have reasonable confidence
        assert extraction.confidence >= 0.3

    def test_tags_derivation(self, git_extractor, bug_fix_commit):
        """Test that tags are derived correctly."""
        extraction = git_extractor.extract(bug_fix_commit)

        assert extraction is not None
        assert "bug-fix" in extraction.tags or "python" in extraction.tags


class TestGitExtractorConfidence:
    """Tests for confidence calculation."""

    def test_high_confidence_commit(self, git_extractor):
        """Test that well-documented commits get high confidence."""
        commit = GitCommit(
            sha="high123",
            message=(
                "fix(auth): resolve session timeout issue\n\n"
                "This fixes the bug where user sessions were timing out prematurely.\n\n"
                "Closes #456"
            ),
            author="Test",
            author_email="test@example.com",
            timestamp=datetime.now(),
            diffs=[
                GitDiff(
                    file_path="src/auth.py",
                    change_type="modified",
                    additions=20,
                    deletions=5,
                ),
                GitDiff(
                    file_path="tests/test_auth.py",
                    change_type="modified",
                    additions=30,
                    deletions=0,
                ),
            ],
        )

        extraction = git_extractor.extract(commit)
        assert extraction is not None
        # Should have high confidence: type + body + tests + small focused change
        assert extraction.confidence >= 0.6

    def test_low_confidence_large_commit(self, git_extractor):
        """Test that large unfocused commits get lower confidence."""
        commit = GitCommit(
            sha="large123",
            message="updates",
            author="Test",
            author_email="test@example.com",
            timestamp=datetime.now(),
            diffs=[
                GitDiff(
                    file_path=f"file{i}.py",
                    change_type="modified",
                    additions=100,
                    deletions=50,
                )
                for i in range(15)
            ],
        )

        extraction = git_extractor.extract(commit)
        # Either None (skipped) or low confidence
        assert extraction is None or extraction.confidence < 0.4


class TestGitExtractorPatternMatching:
    """Tests for commit type pattern matching."""

    def test_fix_patterns(self, git_extractor):
        """Test detection of fix-related patterns."""
        messages = [
            "fix: resolve database connection issue",
            "bugfix: handle null pointer exception",
            "fixed the login bug",
            "resolves issue with file upload",
        ]

        for msg in messages:
            commit = GitCommit(
                sha="test123",
                message=msg,
                author="Test",
                author_email="test@example.com",
                timestamp=datetime.now(),
                diffs=[
                    GitDiff(file_path="test.py", change_type="modified", additions=10, deletions=5)
                ],
            )
            commit_type = git_extractor._classify_commit(commit)
            assert commit_type == "bug_fix", f"Expected bug_fix for: {msg}"

    def test_feature_patterns(self, git_extractor):
        """Test detection of feature-related patterns."""
        messages = [
            "feat: add user dashboard",
            "feature: implement search functionality",
            "add support for OAuth",
            "implement new caching layer",
        ]

        for msg in messages:
            commit = GitCommit(
                sha="test123",
                message=msg,
                author="Test",
                author_email="test@example.com",
                timestamp=datetime.now(),
                diffs=[
                    GitDiff(file_path="test.py", change_type="modified", additions=50, deletions=0)
                ],
            )
            commit_type = git_extractor._classify_commit(commit)
            assert commit_type == "feature", f"Expected feature for: {msg}"

    def test_refactor_patterns(self, git_extractor):
        """Test detection of refactor-related patterns."""
        messages = [
            "refactor: restructure auth module",
            "refactored the database layer",
            "clean up legacy code",
            "simplify the request handler",
        ]

        for msg in messages:
            commit = GitCommit(
                sha="test123",
                message=msg,
                author="Test",
                author_email="test@example.com",
                timestamp=datetime.now(),
                diffs=[
                    GitDiff(file_path="test.py", change_type="modified", additions=30, deletions=30)
                ],
            )
            commit_type = git_extractor._classify_commit(commit)
            assert commit_type == "refactor", f"Expected refactor for: {msg}"


class TestExtractedKnowledge:
    """Tests for ExtractedKnowledge dataclass."""

    def test_create_extraction(self):
        """Test creating an ExtractedKnowledge instance."""
        extraction = ExtractedKnowledge(
            title="Fix: Authentication bug",
            description="Resolved the session timeout issue",
            content="## Commit details...",
            tags=["bug-fix", "python", "auth"],
            confidence=0.75,
            extraction_type="bug_fix",
            source_sha="abc123",
        )

        assert extraction.title == "Fix: Authentication bug"
        assert extraction.confidence == 0.75
        assert "bug-fix" in extraction.tags
        assert extraction.source_sha == "abc123"


class TestGitExtractorHelpers:
    """Tests for helper methods."""

    def test_is_generic_commit(self, git_extractor):
        """Test detection of generic/low-value commits."""
        generic_messages = [
            "update",
            "wip",
            "changes",
            "stuff",
            "minor fix",
            "small change",
        ]

        for msg in generic_messages:
            assert git_extractor._is_generic_commit(msg) is True, f"Expected generic: {msg}"

    def test_is_not_generic_commit(self, git_extractor):
        """Test that good commits are not flagged as generic."""
        good_messages = [
            "fix: resolve authentication bug",
            "feat: add user profile endpoint",
            "refactor: extract database module",
        ]

        for msg in good_messages:
            assert git_extractor._is_generic_commit(msg) is False, f"Expected not generic: {msg}"

    def test_identify_bug_fixes(
        self, git_extractor, bug_fix_commit, feature_commit, refactor_commit
    ):
        """Test filtering to only bug fixes."""
        commits = [bug_fix_commit, feature_commit, refactor_commit]
        bug_fixes = git_extractor.identify_bug_fixes(commits)

        assert len(bug_fixes) == 1
        assert bug_fixes[0].extraction_type == "bug_fix"

    def test_identify_features(
        self, git_extractor, bug_fix_commit, feature_commit, refactor_commit
    ):
        """Test filtering to only features."""
        commits = [bug_fix_commit, feature_commit, refactor_commit]
        features = git_extractor.identify_features(commits)

        assert len(features) == 1
        assert features[0].extraction_type == "feature"
