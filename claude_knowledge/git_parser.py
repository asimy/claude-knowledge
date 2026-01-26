"""Parse git commit history and diffs."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DiffHunk:
    """A single hunk from a diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    content: str
    header: str = ""


@dataclass
class GitDiff:
    """A diff for a single file."""

    file_path: str
    change_type: str  # "added", "modified", "deleted", "renamed"
    old_path: str | None = None  # For renames
    hunks: list[DiffHunk] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0

    @property
    def is_test_file(self) -> bool:
        """Check if this is a test file."""
        path_lower = self.file_path.lower()
        return (
            "test" in path_lower
            or "spec" in path_lower
            or path_lower.startswith("tests/")
            or "/__tests__/" in path_lower
        )

    @property
    def language(self) -> str | None:
        """Detect language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".go": "go",
            ".rb": "ruby",
            ".rs": "rust",
            ".java": "java",
            ".kt": "kotlin",
            ".swift": "swift",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".php": "php",
            ".sql": "sql",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "bash",
        }
        ext = Path(self.file_path).suffix.lower()
        return ext_map.get(ext)


@dataclass
class GitCommit:
    """A parsed git commit."""

    sha: str
    message: str
    author: str
    author_email: str
    timestamp: datetime
    files_changed: list[str] = field(default_factory=list)
    diffs: list[GitDiff] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def subject(self) -> str:
        """Get the commit subject (first line of message)."""
        return self.message.split("\n")[0].strip()

    @property
    def body(self) -> str:
        """Get the commit body (message without subject)."""
        lines = self.message.split("\n")
        if len(lines) > 1:
            # Skip subject and blank line after it
            body_lines = lines[1:]
            while body_lines and not body_lines[0].strip():
                body_lines = body_lines[1:]
            return "\n".join(body_lines).strip()
        return ""

    @property
    def total_additions(self) -> int:
        """Total lines added across all files."""
        return sum(d.additions for d in self.diffs)

    @property
    def total_deletions(self) -> int:
        """Total lines deleted across all files."""
        return sum(d.deletions for d in self.diffs)

    @property
    def has_tests(self) -> bool:
        """Check if this commit includes test changes."""
        return any(d.is_test_file for d in self.diffs)

    @property
    def languages(self) -> set[str]:
        """Get the set of languages affected by this commit."""
        return {d.language for d in self.diffs if d.language}


class GitParser:
    """Parse git commit history and diffs from a repository."""

    def __init__(self, repo_path: str | Path | None = None):
        """Initialize the git parser.

        Args:
            repo_path: Path to the git repository. Defaults to current directory.
        """
        self.repo_path = Path(repo_path).resolve() if repo_path else Path.cwd()

    def _run_git(
        self,
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command in the repository.

        Args:
            *args: Git command arguments.
            check: Whether to raise on non-zero exit.

        Returns:
            Completed process result.

        Raises:
            subprocess.CalledProcessError: If check=True and command fails.
            FileNotFoundError: If git is not installed.
        """
        cmd = ["git", "-C", str(self.repo_path)] + list(args)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
        )

    def is_git_repo(self) -> bool:
        """Check if the path is a git repository.

        Returns:
            True if the path is inside a git repository.
        """
        result = self._run_git("rev-parse", "--git-dir", check=False)
        return result.returncode == 0

    def get_repo_name(self) -> str:
        """Get the repository name from remote or directory.

        Returns:
            Repository name.
        """
        # Try to get from remote origin
        result = self._run_git("remote", "get-url", "origin", check=False)
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract repo name from URL
            # Handle: git@github.com:user/repo.git or https://github.com/user/repo.git
            match = re.search(r"[:/]([^/]+)\.git$", url)
            if match:
                return match.group(1)
            match = re.search(r"[:/]([^/]+)$", url)
            if match:
                return match.group(1)

        # Fall back to directory name
        return self.repo_path.name

    def get_commits(
        self,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        limit: int | None = None,
        author: str | None = None,
        path: str | None = None,
        branch: str | None = None,
    ) -> list[GitCommit]:
        """Get commits from the repository.

        Args:
            since: Only commits after this date.
            until: Only commits before this date.
            limit: Maximum number of commits to return.
            author: Filter by author name or email.
            path: Only commits affecting this path.
            branch: Branch to get commits from (default: current branch).

        Returns:
            List of GitCommit objects, newest first.
        """
        # Build git log command
        # Use %x00 (null byte) as record separator for reliable parsing
        # Format: sha<NUL>author<NUL>email<NUL>timestamp<NUL>subject<NUL>body<NUL>
        format_str = "%H%x00%an%x00%ae%x00%aI%x00%s%x00%b%x00"
        args = [
            "log",
            f"--format={format_str}",
            "-z",  # Use NUL as commit separator
        ]

        if since:
            if isinstance(since, datetime):
                since = since.isoformat()
            args.append(f"--since={since}")

        if until:
            if isinstance(until, datetime):
                until = until.isoformat()
            args.append(f"--until={until}")

        if limit:
            args.append(f"-n{limit}")

        if author:
            args.append(f"--author={author}")

        if branch:
            args.append(branch)

        if path:
            args.extend(["--", path])

        result = self._run_git(*args)
        return self._parse_log_output(result.stdout)

    def _parse_log_output(self, output: str) -> list[GitCommit]:
        """Parse git log output into GitCommit objects.

        Args:
            output: Raw git log output (NUL-separated fields).

        Returns:
            List of GitCommit objects.
        """
        commits = []

        if not output.strip():
            return commits

        # Split by NUL character
        parts = output.split("\x00")

        # Each commit has 6 fields: sha, author, email, timestamp, subject, body
        # The -z flag adds an extra NUL between commits
        i = 0
        while i + 5 < len(parts):
            sha = parts[i].strip()
            if not sha:
                i += 1
                continue

            author = parts[i + 1]
            email = parts[i + 2]
            timestamp_str = parts[i + 3]
            subject = parts[i + 4]
            body = parts[i + 5].strip()

            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.now()

            message = subject
            if body:
                message = f"{subject}\n\n{body}"

            commits.append(
                GitCommit(
                    sha=sha,
                    message=message,
                    author=author,
                    author_email=email,
                    timestamp=timestamp,
                    files_changed=[],
                )
            )

            i += 6

        return commits

    def get_commit(self, sha: str) -> GitCommit | None:
        """Get a single commit by SHA.

        Args:
            sha: Commit SHA (full or abbreviated).

        Returns:
            GitCommit object or None if not found.
        """
        # Use show for specific commit
        format_str = "%H|%an|%ae|%aI%n%B"
        result = self._run_git("show", f"--format={format_str}", "--name-only", sha, check=False)

        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split("\n")
        if not lines:
            return None

        # First line: sha|author|email|timestamp
        header_line = lines[0]
        parts = header_line.split("|")
        if len(parts) < 4:
            return None

        commit_sha = parts[0]
        author = parts[1]
        email = parts[2]
        timestamp_str = parts[3]

        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except ValueError:
            timestamp = datetime.now()

        # Message is everything until empty line followed by file list
        message_lines = []
        files_started = False
        files_changed = []

        for line in lines[1:]:
            if not files_started:
                if line == "":
                    # Could be end of message or blank line in message
                    # Check if next non-empty line looks like a file
                    continue
                elif "/" in line or "." in line:
                    # Likely a file path
                    files_started = True
                    files_changed.append(line)
                else:
                    message_lines.append(line)
            else:
                if line:
                    files_changed.append(line)

        message = "\n".join(message_lines).strip()

        return GitCommit(
            sha=commit_sha,
            message=message,
            author=author,
            author_email=email,
            timestamp=timestamp,
            files_changed=files_changed,
        )

    def get_diff(
        self,
        sha: str,
        context_lines: int = 3,
    ) -> list[GitDiff]:
        """Get the diff for a commit.

        Args:
            sha: Commit SHA.
            context_lines: Number of context lines in diff.

        Returns:
            List of GitDiff objects for each file changed.
        """
        # Get diff with stat and patch
        args = [
            "show",
            f"-U{context_lines}",
            "--stat",
            "--patch",
            sha,
        ]
        result = self._run_git(*args, check=False)

        if result.returncode != 0:
            return []

        return self._parse_diff_output(result.stdout)

    def get_diff_between(
        self,
        base: str,
        head: str,
        context_lines: int = 3,
    ) -> list[GitDiff]:
        """Get the diff between two commits.

        Args:
            base: Base commit SHA or ref.
            head: Head commit SHA or ref.
            context_lines: Number of context lines in diff.

        Returns:
            List of GitDiff objects.
        """
        args = [
            "diff",
            f"-U{context_lines}",
            "--stat",
            "--patch",
            f"{base}..{head}",
        ]
        result = self._run_git(*args, check=False)

        if result.returncode != 0:
            return []

        return self._parse_diff_output(result.stdout)

    def _parse_diff_output(self, output: str) -> list[GitDiff]:
        """Parse diff output into GitDiff objects.

        Args:
            output: Raw diff output.

        Returns:
            List of GitDiff objects.
        """
        diffs = []

        # Split into file diffs
        # Each file diff starts with "diff --git a/path b/path"
        file_diffs = re.split(r"(?=^diff --git )", output, flags=re.MULTILINE)

        for file_diff in file_diffs:
            file_diff = file_diff.strip()
            if not file_diff or not file_diff.startswith("diff --git"):
                continue

            # Parse file paths
            header_match = re.match(r"diff --git a/(.+) b/(.+)", file_diff)
            if not header_match:
                continue

            old_path = header_match.group(1)
            new_path = header_match.group(2)

            # Determine change type
            change_type = "modified"
            if "new file mode" in file_diff:
                change_type = "added"
            elif "deleted file mode" in file_diff:
                change_type = "deleted"
            elif "rename from" in file_diff:
                change_type = "renamed"

            # Parse hunks
            hunks = []
            hunk_pattern = r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
            hunk_matches = list(re.finditer(hunk_pattern, file_diff, re.MULTILINE))

            for i, match in enumerate(hunk_matches):
                old_start = int(match.group(1))
                old_count = int(match.group(2)) if match.group(2) else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) else 1
                header = match.group(5).strip()

                # Get hunk content (until next hunk or end)
                start_pos = match.end()
                if i + 1 < len(hunk_matches):
                    end_pos = hunk_matches[i + 1].start()
                else:
                    end_pos = len(file_diff)

                content = file_diff[start_pos:end_pos].strip()

                hunks.append(
                    DiffHunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        content=content,
                        header=header,
                    )
                )

            # Count additions and deletions
            additions = 0
            deletions = 0
            for hunk in hunks:
                for line in hunk.content.split("\n"):
                    if line.startswith("+") and not line.startswith("+++"):
                        additions += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        deletions += 1

            diff = GitDiff(
                file_path=new_path,
                change_type=change_type,
                old_path=old_path if change_type == "renamed" else None,
                hunks=hunks,
                additions=additions,
                deletions=deletions,
            )
            diffs.append(diff)

        return diffs

    def get_commits_with_diffs(
        self,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        limit: int | None = None,
        author: str | None = None,
        path: str | None = None,
        branch: str | None = None,
    ) -> list[GitCommit]:
        """Get commits with their diffs populated.

        This is a convenience method that fetches commits and their diffs
        in a single pass.

        Args:
            since: Only commits after this date.
            until: Only commits before this date.
            limit: Maximum number of commits to return.
            author: Filter by author name or email.
            path: Only commits affecting this path.
            branch: Branch to get commits from.

        Returns:
            List of GitCommit objects with diffs populated.
        """
        commits = self.get_commits(
            since=since,
            until=until,
            limit=limit,
            author=author,
            path=path,
            branch=branch,
        )

        for commit in commits:
            commit.diffs = self.get_diff(commit.sha)

        return commits

    def get_branch_name(self) -> str | None:
        """Get the current branch name.

        Returns:
            Branch name or None if in detached HEAD state.
        """
        result = self._run_git("rev-parse", "--abbrev-ref", "HEAD", check=False)
        if result.returncode != 0:
            return None

        branch = result.stdout.strip()
        return branch if branch != "HEAD" else None

    def get_commit_info(self, sha: str) -> dict[str, Any] | None:
        """Get basic info about a commit without parsing diffs.

        Args:
            sha: Commit SHA.

        Returns:
            Dict with commit info or None if not found.
        """
        format_str = "%H|%an|%ae|%aI|%s"
        result = self._run_git("show", f"--format={format_str}", "-s", sha, check=False)

        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split("|")
        if len(parts) < 5:
            return None

        return {
            "sha": parts[0],
            "author": parts[1],
            "email": parts[2],
            "timestamp": parts[3],
            "subject": parts[4],
        }

    def get_file_history(
        self,
        file_path: str,
        limit: int | None = None,
    ) -> list[GitCommit]:
        """Get commit history for a specific file.

        Args:
            file_path: Path to the file.
            limit: Maximum number of commits.

        Returns:
            List of commits that modified the file.
        """
        return self.get_commits(path=file_path, limit=limit)
