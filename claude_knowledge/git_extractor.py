"""Extract knowledge from git commit history."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from claude_knowledge.git_parser import GitCommit


@dataclass
class ExtractedKnowledge:
    """A piece of knowledge extracted from git history."""

    title: str
    description: str
    content: str
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    extraction_type: str = "general"  # "bug_fix", "feature", "refactor", "performance"
    source_sha: str = ""


class GitExtractor:
    """Extract knowledge entries from git commits."""

    # Conventional commit prefixes and their types
    CONVENTIONAL_PREFIXES = {
        "fix": "bug_fix",
        "bugfix": "bug_fix",
        "hotfix": "bug_fix",
        "feat": "feature",
        "feature": "feature",
        "refactor": "refactor",
        "perf": "performance",
        "docs": "documentation",
        "test": "testing",
        "chore": "maintenance",
        "style": "style",
        "build": "build",
        "ci": "ci",
    }

    # Patterns that indicate commit types
    TYPE_PATTERNS = {
        "bug_fix": [
            r"\bfix(?:e[sd])?\b",
            r"\bbug\b",
            r"\bissue\b",
            r"\bresolve[sd]?\b",
            r"\bcorrect(?:s|ed)?\b",
            r"\bpatch\b",
            r"\brepair\b",
            r"#\d+",  # Issue references
        ],
        "feature": [
            r"\badd(?:s|ed|ing)?\b",
            r"\bimplement(?:s|ed|ing)?\b",
            r"\bintroduce[sd]?\b",
            r"\bnew\b",
            r"\bcreate[sd]?\b",
            r"\bsupport\b",
            r"\benable[sd]?\b",
        ],
        "refactor": [
            r"\brefactor(?:ed|ing|s)?\b",
            r"\brestructure[sd]?\b",
            r"\breorganize[sd]?\b",
            r"\bclean\s*up\b",
            r"\bsimplify\b",
            r"\bextract(?:s|ed|ing)?\b",
            r"\brename[sd]?\b",
            r"\bmove[sd]?\b",
        ],
        "performance": [
            r"\bperformance\b",
            r"\boptimiz(?:e|ation)\b",
            r"\bspeed\s*up\b",
            r"\bfaster\b",
            r"\bcache[sd]?\b",
            r"\breduce\b.*\b(?:time|memory|latency)\b",
            r"\bimprove[sd]?\b.*\bperformance\b",
        ],
    }

    # Generic/low-value commit message patterns
    GENERIC_PATTERNS = [
        r"^(?:wip|work in progress)$",
        r"^(?:update|updates)$",
        r"^(?:change|changes)$",
        r"^(?:fix|fixes)$",
        r"^(?:stuff|misc|various)$",
        r"^(?:todo|temp|tmp)$",
        r"^(?:commit|save|checkpoint)$",
        r"^(?:minor|small|quick)(?:\s+(?:fix|change|update))?$",
    ]

    def __init__(self) -> None:
        """Initialize the git extractor."""
        self._type_patterns: dict[str, list[re.Pattern[str]]] = {}
        for commit_type, patterns in self.TYPE_PATTERNS.items():
            self._type_patterns[commit_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

        self._generic_patterns = [re.compile(p, re.IGNORECASE) for p in self.GENERIC_PATTERNS]

    def extract(self, commit: GitCommit) -> ExtractedKnowledge | None:
        """Extract knowledge from a single commit.

        Args:
            commit: GitCommit object with diffs populated.

        Returns:
            ExtractedKnowledge if extraction succeeds, None otherwise.
        """
        # Skip merge commits (usually not interesting)
        if commit.subject.lower().startswith("merge "):
            return None

        # Skip generic/low-value commits
        if self._is_generic_commit(commit.subject):
            return None

        # Classify commit type
        commit_type = self._classify_commit(commit)

        # Calculate confidence
        confidence = self._calculate_confidence(commit, commit_type)

        # Skip low confidence commits
        if confidence < 0.3:
            return None

        # Generate title
        title = self._generate_title(commit, commit_type)

        # Generate description
        description = self._generate_description(commit, commit_type)

        # Generate content
        content = self._generate_content(commit)

        # Derive tags
        tags = self._derive_tags(commit, commit_type)

        return ExtractedKnowledge(
            title=title,
            description=description,
            content=content,
            tags=tags,
            confidence=confidence,
            extraction_type=commit_type,
            source_sha=commit.sha,
        )

    def extract_from_commits(
        self,
        commits: list[GitCommit],
        min_confidence: float = 0.3,
    ) -> list[ExtractedKnowledge]:
        """Extract knowledge from multiple commits.

        Args:
            commits: List of GitCommit objects.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of ExtractedKnowledge entries.
        """
        extractions = []

        for commit in commits:
            extraction = self.extract(commit)
            if extraction and extraction.confidence >= min_confidence:
                extractions.append(extraction)

        return extractions

    def _is_generic_commit(self, subject: str) -> bool:
        """Check if commit message is too generic to be useful.

        Args:
            subject: Commit subject line.

        Returns:
            True if the commit is generic/low-value.
        """
        subject_clean = subject.strip().lower()

        for pattern in self._generic_patterns:
            if pattern.match(subject_clean):
                return True

        return False

    def _classify_commit(self, commit: GitCommit) -> str:
        """Classify the commit type.

        Args:
            commit: GitCommit object.

        Returns:
            Commit type string.
        """
        subject_lower = commit.subject.lower()

        # Check for conventional commit prefix
        if ":" in subject_lower:
            prefix = subject_lower.split(":")[0].strip()
            # Handle scoped prefixes like "fix(auth):"
            if "(" in prefix:
                prefix = prefix.split("(")[0].strip()

            if prefix in self.CONVENTIONAL_PREFIXES:
                return self.CONVENTIONAL_PREFIXES[prefix]

        # Check message content against patterns
        full_message = f"{commit.subject} {commit.body}".lower()

        # Count matches for each type
        type_scores: dict[str, int] = {}

        for commit_type, patterns in self._type_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(full_message):
                    score += 1
            if score > 0:
                type_scores[commit_type] = score

        # Return type with highest score, default to general
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]

        return "general"

    def _calculate_confidence(self, commit: GitCommit, commit_type: str) -> float:
        """Calculate confidence score for a commit extraction.

        Confidence scoring:
        - Base: 0.3
        - +0.15 clear commit type (conventional or pattern match)
        - +0.1 multi-line message with explanation
        - +0.1 small focused change (< 100 lines, < 5 files)
        - +0.1 includes test changes
        - -0.1 generic message
        - -0.15 large unfocused change (> 500 lines or > 10 files)

        Args:
            commit: GitCommit object.
            commit_type: Classified commit type.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        score = 0.3  # Base score

        # Clear commit type: +0.15
        if commit_type != "general":
            score += 0.15

        # Conventional commit format: +0.05 bonus
        if ":" in commit.subject.lower():
            prefix = commit.subject.lower().split(":")[0].strip()
            if "(" in prefix:
                prefix = prefix.split("(")[0].strip()
            if prefix in self.CONVENTIONAL_PREFIXES:
                score += 0.05

        # Multi-line message with explanation: +0.1
        if commit.body and len(commit.body) > 20:
            score += 0.1

        # Change size scoring
        total_changes = commit.total_additions + commit.total_deletions
        num_files = len(commit.diffs)

        # Small focused change: +0.1
        if total_changes < 100 and num_files < 5:
            score += 0.1

        # Large unfocused change: -0.15
        if total_changes > 500 or num_files > 10:
            score -= 0.15

        # Includes test changes: +0.1
        if commit.has_tests:
            score += 0.1

        # Generic keywords in message: -0.1
        subject_lower = commit.subject.lower()
        generic_keywords = ["minor", "small", "quick", "temp", "wip"]
        if any(kw in subject_lower for kw in generic_keywords):
            score -= 0.1

        # Clamp to valid range
        return max(0.0, min(1.0, score))

    def _generate_title(self, commit: GitCommit, commit_type: str) -> str:
        """Generate a title for the knowledge entry.

        Args:
            commit: GitCommit object.
            commit_type: Classified commit type.

        Returns:
            Generated title.
        """
        subject = commit.subject

        # Remove conventional commit prefix for cleaner title
        if ":" in subject:
            parts = subject.split(":", 1)
            if len(parts) > 1:
                subject = parts[1].strip()

        # Capitalize first letter
        if subject:
            subject = subject[0].upper() + subject[1:]

        # Truncate if too long
        if len(subject) > 80:
            truncated = subject[:77]
            last_space = truncated.rfind(" ")
            if last_space > 40:
                truncated = truncated[:last_space]
            subject = truncated + "..."

        # Add type prefix if not obvious
        type_prefixes = {
            "bug_fix": "Fix:",
            "feature": "Feature:",
            "refactor": "Refactor:",
            "performance": "Performance:",
        }

        prefix = type_prefixes.get(commit_type, "")
        if prefix and not subject.lower().startswith(prefix.lower().rstrip(":")):
            subject = f"{prefix} {subject}"

        return subject

    def _generate_description(self, commit: GitCommit, commit_type: str) -> str:
        """Generate a description for the knowledge entry.

        Args:
            commit: GitCommit object.
            commit_type: Classified commit type.

        Returns:
            Generated description.
        """
        # Use commit body if available
        if commit.body:
            body = commit.body.strip()
            # Take first paragraph
            paragraphs = body.split("\n\n")
            first_para = paragraphs[0].strip()

            # Truncate if too long
            if len(first_para) > 200:
                first_para = first_para[:197] + "..."

            return first_para

        # Generate description from commit metadata
        parts = []

        # Type description
        type_descriptions = {
            "bug_fix": "Bug fix",
            "feature": "New feature",
            "refactor": "Code refactoring",
            "performance": "Performance improvement",
            "general": "Code change",
        }
        parts.append(type_descriptions.get(commit_type, "Code change"))

        # Files changed summary
        if commit.diffs:
            files = [d.file_path for d in commit.diffs[:3]]
            if len(commit.diffs) > 3:
                files_str = f"{', '.join(files)}, and {len(commit.diffs) - 3} more"
            else:
                files_str = ", ".join(files)
            parts.append(f"affecting {files_str}")

        # Change stats
        if commit.total_additions or commit.total_deletions:
            parts.append(f"(+{commit.total_additions}/-{commit.total_deletions} lines)")

        return " ".join(parts) + "."

    def _generate_content(self, commit: GitCommit) -> str:
        """Generate content for the knowledge entry.

        Args:
            commit: GitCommit object.

        Returns:
            Generated content with commit details and diffs.
        """
        content_parts = []

        # Commit message
        content_parts.append(f"## Commit Message\n\n{commit.message}")

        # Author and date
        content_parts.append(
            f"\n\n**Author:** {commit.author} ({commit.author_email})\n"
            f"**Date:** {commit.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
            f"**SHA:** {commit.sha[:8]}"
        )

        # Files changed
        if commit.diffs:
            content_parts.append("\n\n## Files Changed\n")
            for diff in commit.diffs:
                change_symbol = {
                    "added": "+",
                    "deleted": "-",
                    "modified": "M",
                    "renamed": "R",
                }.get(diff.change_type, "?")
                content_parts.append(
                    f"- [{change_symbol}] {diff.file_path} (+{diff.additions}/-{diff.deletions})"
                )

        # Include significant diffs
        if commit.diffs:
            content_parts.append("\n\n## Key Changes\n")
            for diff in commit.diffs[:5]:  # Limit to 5 files
                if not diff.hunks:
                    continue

                content_parts.append(f"\n### {diff.file_path}\n")

                # Include first few hunks
                for hunk in diff.hunks[:3]:  # Limit to 3 hunks per file
                    # Truncate very long hunks
                    hunk_content = hunk.content
                    if len(hunk_content) > 500:
                        hunk_content = hunk_content[:500] + "\n... (truncated)"

                    lang = diff.language or ""
                    content_parts.append(f"```{lang}\n{hunk_content}\n```\n")

        return "\n".join(content_parts)

    def _derive_tags(self, commit: GitCommit, commit_type: str) -> list[str]:
        """Derive tags from commit metadata.

        Args:
            commit: GitCommit object.
            commit_type: Classified commit type.

        Returns:
            List of derived tags.
        """
        tags = set()

        # Add commit type as tag
        if commit_type != "general":
            tags.add(commit_type.replace("_", "-"))

        # Add languages
        for lang in commit.languages:
            tags.add(lang)

        # Check for scope in conventional commit
        subject_lower = commit.subject.lower()
        if "(" in subject_lower and ")" in subject_lower:
            scope_match = re.search(r"\(([^)]+)\)", subject_lower)
            if scope_match:
                scope = scope_match.group(1).strip()
                if scope and len(scope) < 20:  # Sanity check
                    tags.add(scope)

        # Detect common areas from file paths
        area_patterns = {
            "api": r"(?:^|/)api/",
            "auth": r"(?:^|/)(?:auth|login|session)",
            "database": r"(?:^|/)(?:db|database|models?|migrations?)",
            "config": r"(?:^|/)(?:config|settings)",
            "tests": r"(?:^|/)tests?/",
            "cli": r"(?:^|/)(?:cli|commands?)/",
            "ui": r"(?:^|/)(?:ui|views?|templates?|components?)/",
        }

        for diff in commit.diffs:
            for area, pattern in area_patterns.items():
                if re.search(pattern, diff.file_path, re.IGNORECASE):
                    tags.add(area)

        return sorted(tags)[:5]  # Limit to 5 tags

    def identify_bug_fixes(self, commits: list[GitCommit]) -> list[ExtractedKnowledge]:
        """Extract knowledge specifically from bug fix commits.

        Args:
            commits: List of GitCommit objects.

        Returns:
            List of bug fix extractions.
        """
        all_extractions = self.extract_from_commits(commits)
        return [e for e in all_extractions if e.extraction_type == "bug_fix"]

    def identify_features(self, commits: list[GitCommit]) -> list[ExtractedKnowledge]:
        """Extract knowledge specifically from feature commits.

        Args:
            commits: List of GitCommit objects.

        Returns:
            List of feature extractions.
        """
        all_extractions = self.extract_from_commits(commits)
        return [e for e in all_extractions if e.extraction_type == "feature"]

    def identify_refactors(self, commits: list[GitCommit]) -> list[ExtractedKnowledge]:
        """Extract knowledge specifically from refactoring commits.

        Args:
            commits: List of GitCommit objects.

        Returns:
            List of refactor extractions.
        """
        all_extractions = self.extract_from_commits(commits)
        return [e for e in all_extractions if e.extraction_type == "refactor"]
