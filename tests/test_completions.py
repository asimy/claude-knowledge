"""Tests for shell completion functionality."""

from argparse import Namespace
from unittest.mock import MagicMock, patch


class TestProjectCompleter:
    """Tests for project name completion."""

    def test_returns_matching_projects(self):
        """Project completer returns projects starting with prefix."""
        from claude_knowledge.completions import get_project_completer

        completer = get_project_completer()

        with patch("claude_knowledge.knowledge_manager.KnowledgeManager") as mock_km_class:
            mock_km = MagicMock()
            mock_km_class.return_value = mock_km
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ("project-a",),
                ("project-b",),
                ("other",),
            ]
            mock_km.conn.cursor.return_value = mock_cursor

            results = completer("proj", Namespace())

            assert "project-a" in results
            assert "project-b" in results
            assert "other" not in results

    def test_returns_all_when_empty_prefix(self):
        """Project completer returns all projects for empty prefix."""
        from claude_knowledge.completions import get_project_completer

        completer = get_project_completer()

        with patch("claude_knowledge.knowledge_manager.KnowledgeManager") as mock_km_class:
            mock_km = MagicMock()
            mock_km_class.return_value = mock_km
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ("project-a",),
                ("project-b",),
            ]
            mock_km.conn.cursor.return_value = mock_cursor

            results = completer("", Namespace())

            assert "project-a" in results
            assert "project-b" in results

    def test_handles_database_errors(self):
        """Project completer returns empty list on database errors."""
        from claude_knowledge.completions import get_project_completer

        completer = get_project_completer()

        with patch("claude_knowledge.knowledge_manager.KnowledgeManager") as mock_km_class:
            mock_km_class.side_effect = Exception("Database error")

            results = completer("proj", Namespace())

            assert results == []


class TestEntryIdCompleter:
    """Tests for entry ID completion."""

    def test_returns_matching_ids(self):
        """Entry ID completer returns IDs starting with prefix."""
        from claude_knowledge.completions import get_entry_id_completer

        completer = get_entry_id_completer()

        with patch("claude_knowledge.knowledge_manager.KnowledgeManager") as mock_km_class:
            mock_km = MagicMock()
            mock_km_class.return_value = mock_km
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ("abc123",),
                ("abc456",),
                ("xyz789",),
            ]
            mock_km.conn.cursor.return_value = mock_cursor

            results = completer("abc", Namespace())

            assert "abc123" in results
            assert "abc456" in results
            assert "xyz789" not in results

    def test_handles_database_errors(self):
        """Entry ID completer returns empty list on database errors."""
        from claude_knowledge.completions import get_entry_id_completer

        completer = get_entry_id_completer()

        with patch("claude_knowledge.knowledge_manager.KnowledgeManager") as mock_km_class:
            mock_km_class.side_effect = Exception("Database error")

            results = completer("abc", Namespace())

            assert results == []


class TestSyncPathCompleter:
    """Tests for sync path completion."""

    def test_returns_saved_path(self):
        """Sync path completer returns saved sync path."""
        from claude_knowledge.completions import get_sync_path_completer

        completer = get_sync_path_completer()

        with patch("claude_knowledge.knowledge_manager.KnowledgeManager") as mock_km_class:
            mock_km = MagicMock()
            mock_km_class.return_value = mock_km
            mock_km.get_sync_path.return_value = "/Users/test/sync"

            results = completer("/Users", Namespace())

            assert "/Users/test/sync" in results

    def test_filters_by_prefix(self):
        """Sync path completer filters by prefix."""
        from claude_knowledge.completions import get_sync_path_completer

        completer = get_sync_path_completer()

        with patch("claude_knowledge.knowledge_manager.KnowledgeManager") as mock_km_class:
            mock_km = MagicMock()
            mock_km_class.return_value = mock_km
            mock_km.get_sync_path.return_value = "/Users/test/sync"

            results = completer("/var", Namespace())

            assert "/Users/test/sync" not in results

    def test_handles_no_saved_path(self):
        """Sync path completer handles no saved path."""
        from claude_knowledge.completions import get_sync_path_completer

        completer = get_sync_path_completer()

        with patch("claude_knowledge.knowledge_manager.KnowledgeManager") as mock_km_class:
            mock_km = MagicMock()
            mock_km_class.return_value = mock_km
            mock_km.get_sync_path.return_value = None

            results = completer("/Users", Namespace())

            assert results == []

    def test_handles_errors(self):
        """Sync path completer returns empty list on errors."""
        from claude_knowledge.completions import get_sync_path_completer

        completer = get_sync_path_completer()

        with patch("claude_knowledge.knowledge_manager.KnowledgeManager") as mock_km_class:
            mock_km_class.side_effect = Exception("Error")

            results = completer("/Users", Namespace())

            assert results == []


class TestCompletionsCommand:
    """Tests for the completions CLI command."""

    def test_completions_command_bash(self, capsys):
        """Completions command outputs bash instructions."""
        from claude_knowledge.cli import cmd_completions

        args = Namespace(shell="bash")

        with patch("claude_knowledge.cli.ARGCOMPLETE_AVAILABLE", True):
            result = cmd_completions(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "bashrc" in captured.out
        assert "register-python-argcomplete" in captured.out

    def test_completions_command_zsh(self, capsys):
        """Completions command outputs zsh instructions."""
        from claude_knowledge.cli import cmd_completions

        args = Namespace(shell="zsh")

        with patch("claude_knowledge.cli.ARGCOMPLETE_AVAILABLE", True):
            result = cmd_completions(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "zshrc" in captured.out
        assert "bashcompinit" in captured.out

    def test_completions_command_fish(self, capsys):
        """Completions command outputs fish instructions."""
        from claude_knowledge.cli import cmd_completions

        args = Namespace(shell="fish")

        with patch("claude_knowledge.cli.ARGCOMPLETE_AVAILABLE", True):
            result = cmd_completions(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "fish" in captured.out
        assert "completions/claude-kb.fish" in captured.out

    def test_completions_command_without_argcomplete(self, capsys):
        """Completions command fails without argcomplete installed."""
        from claude_knowledge.cli import cmd_completions

        args = Namespace(shell="bash")

        with patch("claude_knowledge.cli.ARGCOMPLETE_AVAILABLE", False):
            result = cmd_completions(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "argcomplete is not installed" in captured.out
