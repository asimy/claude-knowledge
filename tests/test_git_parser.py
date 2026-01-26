"""Tests for the git parser module."""

import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from claude_knowledge.git_parser import (
    DiffHunk,
    GitCommit,
    GitDiff,
    GitParser,
)


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "test_repo"
        repo_path.mkdir()

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        yield repo_path


@pytest.fixture
def repo_with_commits(temp_git_repo):
    """Create a repo with some commits."""
    repo_path = temp_git_repo

    # Create initial file and commit
    (repo_path / "main.py").write_text("# Initial file\n")
    subprocess.run(
        ["git", "add", "main.py"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "feat: initial commit"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    # Add second commit with bug fix
    (repo_path / "main.py").write_text("# Fixed file\ndef main():\n    pass\n")
    subprocess.run(
        ["git", "add", "main.py"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "fix: resolve issue with main function\n\nThis fixes #123"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    # Add test file
    (repo_path / "test_main.py").write_text("def test_main():\n    assert True\n")
    subprocess.run(
        ["git", "add", "test_main.py"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "test: add tests for main"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    yield repo_path


class TestDiffHunk:
    """Tests for DiffHunk dataclass."""

    def test_create_diff_hunk(self):
        """Test creating a DiffHunk."""
        hunk = DiffHunk(
            old_start=10,
            old_count=5,
            new_start=10,
            new_count=8,
            content="+added line\n-removed line",
            header="function_name",
        )

        assert hunk.old_start == 10
        assert hunk.old_count == 5
        assert hunk.new_start == 10
        assert hunk.new_count == 8
        assert "+added line" in hunk.content
        assert hunk.header == "function_name"


class TestGitDiff:
    """Tests for GitDiff dataclass."""

    def test_is_test_file_patterns(self):
        """Test test file detection."""
        test_diff = GitDiff(file_path="tests/test_main.py", change_type="added")
        assert test_diff.is_test_file is True

        test_diff2 = GitDiff(file_path="spec/main_spec.rb", change_type="added")
        assert test_diff2.is_test_file is True

        non_test = GitDiff(file_path="src/main.py", change_type="modified")
        assert non_test.is_test_file is False

    def test_language_detection(self):
        """Test language detection from file extension."""
        py_diff = GitDiff(file_path="main.py", change_type="modified")
        assert py_diff.language == "python"

        js_diff = GitDiff(file_path="app.js", change_type="added")
        assert js_diff.language == "javascript"

        go_diff = GitDiff(file_path="main.go", change_type="modified")
        assert go_diff.language == "go"

        unknown_diff = GitDiff(file_path="data.txt", change_type="added")
        assert unknown_diff.language is None


class TestGitCommit:
    """Tests for GitCommit dataclass."""

    def test_subject_and_body(self):
        """Test extracting subject and body from message."""
        commit = GitCommit(
            sha="abc123",
            message=(
                "fix: resolve authentication bug\n\n"
                "This fixes the login issue\n"
                "where users couldn't authenticate."
            ),
            author="Test",
            author_email="test@example.com",
            timestamp=datetime.now(),
        )

        assert commit.subject == "fix: resolve authentication bug"
        assert "This fixes the login issue" in commit.body

    def test_subject_only_message(self):
        """Test commit with subject only."""
        commit = GitCommit(
            sha="abc123",
            message="quick fix",
            author="Test",
            author_email="test@example.com",
            timestamp=datetime.now(),
        )

        assert commit.subject == "quick fix"
        assert commit.body == ""

    def test_has_tests(self):
        """Test detecting if commit has test changes."""
        commit = GitCommit(
            sha="abc123",
            message="feat: add feature",
            author="Test",
            author_email="test@example.com",
            timestamp=datetime.now(),
            diffs=[
                GitDiff(file_path="main.py", change_type="modified"),
                GitDiff(file_path="test_main.py", change_type="added"),
            ],
        )

        assert commit.has_tests is True

        commit_no_tests = GitCommit(
            sha="def456",
            message="feat: add feature",
            author="Test",
            author_email="test@example.com",
            timestamp=datetime.now(),
            diffs=[
                GitDiff(file_path="main.py", change_type="modified"),
            ],
        )

        assert commit_no_tests.has_tests is False

    def test_languages(self):
        """Test getting languages from commit."""
        commit = GitCommit(
            sha="abc123",
            message="feat: multi-language commit",
            author="Test",
            author_email="test@example.com",
            timestamp=datetime.now(),
            diffs=[
                GitDiff(file_path="main.py", change_type="modified"),
                GitDiff(file_path="app.js", change_type="added"),
                GitDiff(file_path="README.md", change_type="modified"),
            ],
        )

        assert "python" in commit.languages
        assert "javascript" in commit.languages
        assert len(commit.languages) == 2  # README.md has no language

    def test_total_changes(self):
        """Test calculating total additions and deletions."""
        commit = GitCommit(
            sha="abc123",
            message="refactor",
            author="Test",
            author_email="test@example.com",
            timestamp=datetime.now(),
            diffs=[
                GitDiff(file_path="a.py", change_type="modified", additions=10, deletions=5),
                GitDiff(file_path="b.py", change_type="modified", additions=20, deletions=15),
            ],
        )

        assert commit.total_additions == 30
        assert commit.total_deletions == 20


class TestGitParser:
    """Tests for GitParser class."""

    def test_is_git_repo(self, temp_git_repo):
        """Test detecting if path is a git repo."""
        parser = GitParser(temp_git_repo)
        assert parser.is_git_repo() is True

    def test_is_not_git_repo(self):
        """Test detecting non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = GitParser(tmpdir)
            assert parser.is_git_repo() is False

    def test_get_repo_name(self, temp_git_repo):
        """Test getting repository name."""
        parser = GitParser(temp_git_repo)
        assert parser.get_repo_name() == "test_repo"

    def test_get_commits(self, repo_with_commits):
        """Test getting commits from repository."""
        parser = GitParser(repo_with_commits)
        commits = parser.get_commits()

        assert len(commits) == 3
        assert commits[0].subject == "test: add tests for main"
        assert commits[1].subject == "fix: resolve issue with main function"
        assert commits[2].subject == "feat: initial commit"

    def test_get_commits_with_limit(self, repo_with_commits):
        """Test limiting number of commits."""
        parser = GitParser(repo_with_commits)
        commits = parser.get_commits(limit=2)

        assert len(commits) == 2

    def test_get_branch_name(self, repo_with_commits):
        """Test getting current branch name."""
        parser = GitParser(repo_with_commits)
        # Default branch could be 'main' or 'master' depending on git config
        branch = parser.get_branch_name()
        assert branch in ("main", "master")

    def test_get_diff(self, repo_with_commits):
        """Test getting diff for a commit."""
        parser = GitParser(repo_with_commits)
        commits = parser.get_commits(limit=1)

        diffs = parser.get_diff(commits[0].sha)

        assert len(diffs) == 1
        assert diffs[0].file_path == "test_main.py"
        assert diffs[0].change_type == "added"

    def test_get_commits_with_diffs(self, repo_with_commits):
        """Test getting commits with their diffs populated."""
        parser = GitParser(repo_with_commits)
        commits = parser.get_commits_with_diffs(limit=2)

        assert len(commits) == 2
        for commit in commits:
            assert commit.diffs is not None


class TestGitParserDiffParsing:
    """Tests for diff parsing functionality."""

    def test_parse_diff_output_modified(self):
        """Test parsing diff output for modified file."""
        parser = GitParser()

        diff_output = """diff --git a/main.py b/main.py
index abc123..def456 100644
--- a/main.py
+++ b/main.py
@@ -1,3 +1,5 @@
 # Header
+# New comment
 def main():
-    pass
+    print("hello")
+    return 0
"""
        diffs = parser._parse_diff_output(diff_output)

        assert len(diffs) == 1
        assert diffs[0].file_path == "main.py"
        assert diffs[0].change_type == "modified"
        assert diffs[0].additions == 3
        assert diffs[0].deletions == 1
        assert len(diffs[0].hunks) == 1

    def test_parse_diff_output_added(self):
        """Test parsing diff output for new file."""
        parser = GitParser()

        diff_output = """diff --git a/new_file.py b/new_file.py
new file mode 100644
index 0000000..abc123
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+# New file
+def new_function():
+    pass
"""
        diffs = parser._parse_diff_output(diff_output)

        assert len(diffs) == 1
        assert diffs[0].file_path == "new_file.py"
        assert diffs[0].change_type == "added"
        assert diffs[0].additions == 3

    def test_parse_diff_output_deleted(self):
        """Test parsing diff output for deleted file."""
        parser = GitParser()

        diff_output = """diff --git a/old_file.py b/old_file.py
deleted file mode 100644
index abc123..0000000
--- a/old_file.py
+++ /dev/null
@@ -1,2 +0,0 @@
-# Old file
-def old_function():
"""
        diffs = parser._parse_diff_output(diff_output)

        assert len(diffs) == 1
        assert diffs[0].file_path == "old_file.py"
        assert diffs[0].change_type == "deleted"
        assert diffs[0].deletions == 2
