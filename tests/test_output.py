"""Tests for the output formatting module."""

import io

from rich.console import Console
from rich.text import Text

from claude_knowledge.output import (
    create_stats_table,
    detect_language,
    format_quality_score,
    format_score,
    print_code_block,
    print_error,
    print_knowledge_item,
    print_list_item,
    print_search_item,
    print_success,
    print_warning,
)


class TestDetectLanguage:
    """Tests for language detection."""

    def test_detect_python_shebang(self):
        """Detect Python from shebang."""
        content = "#!/usr/bin/env python3\nprint('hello')"
        assert detect_language(content) == "python"

    def test_detect_bash_shebang(self):
        """Detect bash from shebang."""
        content = "#!/bin/bash\necho hello"
        assert detect_language(content) == "bash"

    def test_detect_python_patterns(self):
        """Detect Python from code patterns."""
        content = "def hello():\n    return 'world'"
        assert detect_language(content) == "python"

        content = "import os\nfrom pathlib import Path"
        assert detect_language(content) == "python"

    def test_detect_go_patterns(self):
        """Detect Go from code patterns."""
        content = "package main\n\nfunc main() {}"
        assert detect_language(content) == "go"

    def test_detect_javascript_patterns(self):
        """Detect JavaScript from code patterns."""
        content = "const foo = () => { return 'bar'; }"
        assert detect_language(content) == "javascript"

    def test_detect_sql_patterns(self):
        """Detect SQL from keywords."""
        content = "SELECT * FROM users WHERE id = 1"
        assert detect_language(content) == "sql"

    def test_detect_json(self):
        """Detect JSON from structure."""
        content = '{"key": "value"}'
        assert detect_language(content) == "json"

        content = '["a", "b", "c"]'
        assert detect_language(content) == "json"

    def test_detect_yaml(self):
        """Detect YAML from structure."""
        content = "---\nkey: value\n  nested: item"
        assert detect_language(content) == "yaml"

    def test_fallback_to_text(self):
        """Fall back to text for unknown content."""
        content = "Just some plain text"
        assert detect_language(content) == "text"

    def test_empty_content(self):
        """Handle empty content."""
        assert detect_language("") == "text"
        assert detect_language(None) == "text"


class TestFormatScore:
    """Tests for score formatting."""

    def test_high_score_green(self):
        """High scores should be green."""
        text = format_score(0.85)
        assert isinstance(text, Text)
        assert text.plain == "85%"
        assert text.style == "green"

    def test_medium_score_yellow(self):
        """Medium scores should be yellow."""
        text = format_score(0.55)
        assert text.plain == "55%"
        assert text.style == "yellow"

    def test_low_score_red(self):
        """Low scores should be red."""
        text = format_score(0.25)
        assert text.plain == "25%"
        assert text.style == "red"

    def test_custom_max_score(self):
        """Test with custom max score."""
        text = format_score(70, max_score=100)
        assert text.plain == "70/100"
        assert text.style == "green"

    def test_zero_max_score(self):
        """Handle zero max score gracefully."""
        text = format_score(0.5, max_score=0)
        assert text.style == "red"


class TestFormatQualityScore:
    """Tests for quality score formatting."""

    def test_high_quality_green(self):
        """High quality scores should be green."""
        text = format_quality_score(85)
        assert text.plain == "85/100"
        assert text.style == "green"

    def test_medium_quality_yellow(self):
        """Medium quality scores should be yellow."""
        text = format_quality_score(55)
        assert text.plain == "55/100"
        assert text.style == "yellow"

    def test_low_quality_red(self):
        """Low quality scores should be red."""
        text = format_quality_score(25)
        assert text.plain == "25/100"
        assert text.style == "red"


class TestPrintFunctions:
    """Tests for print helper functions."""

    def test_print_error(self, capsys):
        """Test error message formatting."""
        # Create a test console that writes to a string buffer
        test_console = Console(force_terminal=True, width=80)
        buffer = io.StringIO()
        test_console = Console(file=buffer, force_terminal=True, width=80)

        # Temporarily replace the global console
        import claude_knowledge.output as output_module

        original_console = output_module.console
        output_module.console = test_console

        try:
            print_error("Something went wrong")
            output = buffer.getvalue()
            assert "Error:" in output
            assert "Something went wrong" in output
        finally:
            output_module.console = original_console

    def test_print_success(self):
        """Test success message formatting."""
        buffer = io.StringIO()
        test_console = Console(file=buffer, force_terminal=True, width=80)

        import claude_knowledge.output as output_module

        original_console = output_module.console
        output_module.console = test_console

        try:
            print_success("Operation completed")
            output = buffer.getvalue()
            assert "Operation completed" in output
        finally:
            output_module.console = original_console

    def test_print_warning(self):
        """Test warning message formatting."""
        buffer = io.StringIO()
        test_console = Console(file=buffer, force_terminal=True, width=80)

        import claude_knowledge.output as output_module

        original_console = output_module.console
        output_module.console = test_console

        try:
            print_warning("Be careful")
            output = buffer.getvalue()
            assert "Be careful" in output
        finally:
            output_module.console = original_console


class TestCreateStatsTable:
    """Tests for stats table creation."""

    def test_creates_table_with_title(self):
        """Test table creation with custom title."""
        table = create_stats_table("My Stats")
        assert table.title == "My Stats"

    def test_creates_table_with_columns(self):
        """Test table has correct columns."""
        table = create_stats_table()
        assert len(table.columns) == 2


class TestPrintKnowledgeItem:
    """Tests for knowledge item formatting."""

    def test_print_knowledge_item_with_score(self):
        """Test printing knowledge item with score."""
        buffer = io.StringIO()
        test_console = Console(file=buffer, force_terminal=True, width=80)

        import claude_knowledge.output as output_module

        original_console = output_module.console
        output_module.console = test_console

        try:
            item = {
                "id": "abc123",
                "title": "Test Entry",
                "description": "A test description",
                "score": 0.85,
                "tags": '["python", "test"]',
            }
            print_knowledge_item(item, include_score=True)
            output = buffer.getvalue()
            assert "Test Entry" in output
            assert "abc123" in output
            assert "A test description" in output
        finally:
            output_module.console = original_console

    def test_print_knowledge_item_without_score(self):
        """Test printing knowledge item without score."""
        buffer = io.StringIO()
        test_console = Console(file=buffer, force_terminal=True, width=80)

        import claude_knowledge.output as output_module

        original_console = output_module.console
        output_module.console = test_console

        try:
            item = {
                "id": "abc123",
                "title": "Test Entry",
                "description": "A test description",
            }
            print_knowledge_item(item, include_score=False)
            output = buffer.getvalue()
            assert "Test Entry" in output
            assert "85%" not in output
        finally:
            output_module.console = original_console


class TestPrintListItem:
    """Tests for list item formatting."""

    def test_print_list_item(self):
        """Test list item formatting."""
        buffer = io.StringIO()
        test_console = Console(file=buffer, force_terminal=True, width=80)

        import claude_knowledge.output as output_module

        original_console = output_module.console
        output_module.console = test_console

        try:
            item = {
                "id": "abc123",
                "title": "Test Entry",
                "tags": '["python"]',
                "project": "myproject",
                "usage_count": 5,
            }
            print_list_item(item)
            output = buffer.getvalue()
            assert "abc123" in output
            assert "Test Entry" in output
            assert "python" in output
            assert "myproject" in output
            assert "5x" in output
        finally:
            output_module.console = original_console


class TestPrintSearchItem:
    """Tests for search item formatting."""

    def test_print_search_item(self):
        """Test search item formatting."""
        buffer = io.StringIO()
        test_console = Console(file=buffer, force_terminal=True, width=80)

        import claude_knowledge.output as output_module

        original_console = output_module.console
        output_module.console = test_console

        try:
            item = {
                "id": "abc123",
                "title": "Test Entry",
                "description": "A description",
                "tags": '["tag1"]',
                "project": "proj",
            }
            print_search_item(item)
            output = buffer.getvalue()
            assert "abc123" in output
            assert "Test Entry" in output
            assert "A description" in output
        finally:
            output_module.console = original_console


class TestPrintCodeBlock:
    """Tests for code block printing."""

    def test_print_code_block_with_language(self):
        """Test code block with explicit language."""
        buffer = io.StringIO()
        test_console = Console(file=buffer, force_terminal=True, width=80)

        import claude_knowledge.output as output_module

        original_console = output_module.console
        output_module.console = test_console

        try:
            print_code_block("def hello():\n    pass", language="python")
            output = buffer.getvalue()
            assert "def" in output
            assert "hello" in output
        finally:
            output_module.console = original_console

    def test_print_code_block_auto_detect(self):
        """Test code block with auto-detected language."""
        buffer = io.StringIO()
        test_console = Console(file=buffer, force_terminal=True, width=80)

        import claude_knowledge.output as output_module

        original_console = output_module.console
        output_module.console = test_console

        try:
            print_code_block("package main\n\nfunc main() {}")
            output = buffer.getvalue()
            assert "func" in output
            assert "main" in output
        finally:
            output_module.console = original_console
