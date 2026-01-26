"""Tests for interactive capture functionality."""

import os
from argparse import Namespace
from unittest.mock import MagicMock, patch

from claude_knowledge.interactive import (
    get_editor,
    prompt_line,
    prompt_optional,
    show_preview,
    truncate_text,
)


class TestGetEditor:
    """Tests for get_editor function."""

    def test_uses_visual_env(self):
        """Uses $VISUAL when set."""
        with patch.dict(os.environ, {"VISUAL": "code", "EDITOR": "vim"}, clear=False):
            assert get_editor() == "code"

    def test_uses_editor_env(self):
        """Uses $EDITOR when $VISUAL is not set."""
        with patch.dict(os.environ, {"EDITOR": "nano"}, clear=False):
            env = os.environ.copy()
            env.pop("VISUAL", None)
            with patch.dict(os.environ, env, clear=True):
                assert get_editor() == "nano"

    def test_fallback_unix(self):
        """Falls back to vi on Unix when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("claude_knowledge.interactive.sys.platform", "darwin"):
                assert get_editor() == "vi"

    def test_fallback_windows(self):
        """Falls back to notepad on Windows when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("claude_knowledge.interactive.sys.platform", "win32"):
                assert get_editor() == "notepad"


class TestPromptLine:
    """Tests for prompt_line function."""

    def test_returns_input(self):
        """Returns user input stripped."""
        with patch("builtins.input", return_value="  test value  "):
            result = prompt_line("Prompt: ")
            assert result == "test value"

    def test_allows_empty_when_not_required(self):
        """Allows empty input when not required."""
        with patch("builtins.input", return_value=""):
            result = prompt_line("Prompt: ", required=False)
            assert result == ""

    def test_reprompts_when_required(self):
        """Re-prompts when required field is empty."""
        inputs = iter(["", "  ", "valid"])
        with patch("builtins.input", side_effect=lambda _: next(inputs)):
            with patch("builtins.print"):
                result = prompt_line("Prompt: ", required=True)
                assert result == "valid"


class TestPromptOptional:
    """Tests for prompt_optional function."""

    def test_returns_input(self):
        """Returns user input."""
        with patch("builtins.input", return_value="my-project"):
            result = prompt_optional("Project: ")
            assert result == "my-project"

    def test_shows_suggestions(self, capsys):
        """Shows suggestions when provided."""
        with patch("builtins.input", return_value="proj"):
            prompt_optional("Project: ", suggestions=["proj1", "proj2"])
            captured = capsys.readouterr()
            assert "proj1" in captured.out
            assert "proj2" in captured.out

    def test_truncates_many_suggestions(self, capsys):
        """Truncates suggestions list when too many."""
        suggestions = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        with patch("builtins.input", return_value=""):
            prompt_optional("Project: ", suggestions=suggestions)
            captured = capsys.readouterr()
            assert "and 2 more" in captured.out


class TestTruncateText:
    """Tests for truncate_text function."""

    def test_short_text_unchanged(self):
        """Short text is not truncated."""
        text = "short"
        result, truncated = truncate_text(text, 100)
        assert result == "short"
        assert truncated is False

    def test_long_text_truncated(self):
        """Long text is truncated with ellipsis."""
        text = "a" * 150
        result, truncated = truncate_text(text, 100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")
        assert truncated is True


class TestShowPreview:
    """Tests for show_preview function."""

    def test_shows_all_fields(self, capsys):
        """Preview shows all fields."""
        with patch("builtins.input", return_value="y"):
            show_preview(
                title="Test Title",
                description="Test Description",
                content="Test Content",
                tags="tag1,tag2",
                project="my-project",
            )
        captured = capsys.readouterr()
        assert "Test Title" in captured.out
        assert "Test Description" in captured.out
        assert "Test Content" in captured.out
        assert "tag1,tag2" in captured.out
        assert "my-project" in captured.out

    def test_shows_none_for_empty_optional(self, capsys):
        """Shows (none) for empty optional fields."""
        with patch("builtins.input", return_value="y"):
            show_preview(
                title="Test",
                description="Desc",
                content="Content",
                tags="",
                project="",
            )
        captured = capsys.readouterr()
        assert "(none)" in captured.out

    def test_returns_true_on_yes(self):
        """Returns True when user confirms."""
        with patch("builtins.input", return_value="y"):
            result = show_preview("T", "D", "C", "", "")
            assert result is True

    def test_returns_true_on_empty(self):
        """Returns True when user presses Enter (default yes)."""
        with patch("builtins.input", return_value=""):
            result = show_preview("T", "D", "C", "", "")
            assert result is True

    def test_returns_false_on_no(self):
        """Returns False when user declines."""
        with patch("builtins.input", return_value="n"):
            result = show_preview("T", "D", "C", "", "")
            assert result is False


class TestGetDistinctProjects:
    """Tests for KnowledgeManager.get_distinct_projects."""

    def test_returns_project_list(self, tmp_path):
        """Returns list of distinct projects."""
        from claude_knowledge.knowledge_manager import KnowledgeManager

        km = KnowledgeManager(base_path=str(tmp_path))
        km.capture(
            title="Test 1",
            description="Desc",
            content="Content",
            project="project-a",
        )
        km.capture(
            title="Test 2",
            description="Desc",
            content="Content",
            project="project-b",
        )
        km.capture(
            title="Test 3",
            description="Desc",
            content="Content",
            project="project-a",
        )

        projects = km.get_distinct_projects()

        assert "project-a" in projects
        assert "project-b" in projects
        assert len(projects) == 2
        km.close()

    def test_excludes_empty_projects(self, tmp_path):
        """Excludes empty and null projects."""
        from claude_knowledge.knowledge_manager import KnowledgeManager

        km = KnowledgeManager(base_path=str(tmp_path))
        km.capture(
            title="Test 1",
            description="Desc",
            content="Content",
            project="project-a",
        )
        km.capture(
            title="Test 2",
            description="Desc",
            content="Content",
        )  # No project

        projects = km.get_distinct_projects()

        assert projects == ["project-a"]
        km.close()


class TestCaptureInteractiveFlag:
    """Tests for capture command with -i flag."""

    def test_interactive_flag_in_parser(self):
        """Parser accepts -i/--interactive flag."""
        from claude_knowledge.cli import create_parser

        parser = create_parser()

        # Test short flag
        args = parser.parse_args(["capture", "-i"])
        assert args.interactive is True

        # Test long flag
        args = parser.parse_args(["capture", "--interactive"])
        assert args.interactive is True

    def test_non_interactive_requires_fields(self, capsys):
        """Non-interactive mode requires title, description, content."""
        from claude_knowledge.cli import cmd_capture

        args = Namespace(
            interactive=False,
            title=None,
            description=None,
            content=None,
            tags=None,
            context=None,
            project=None,
        )

        km = MagicMock()
        result = cmd_capture(args, km)

        assert result == 1
        captured = capsys.readouterr()
        assert "--title" in captured.out
        assert "--description" in captured.out
        assert "--content" in captured.out
