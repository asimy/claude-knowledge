"""Interactive input helpers for claude-kb CLI."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def get_editor() -> str:
    """Get the user's preferred editor.

    Checks in order: $VISUAL, $EDITOR, then falls back to 'vi' (Unix) or 'notepad' (Windows).
    """
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if editor:
        return editor

    # Platform-specific fallback
    if sys.platform == "win32":
        return "notepad"
    return "vi"


def prompt_line(prompt: str, required: bool = False) -> str:
    """Prompt for single-line input.

    Args:
        prompt: The prompt text to display.
        required: If True, re-prompt until non-empty input is provided.

    Returns:
        The user's input, stripped of leading/trailing whitespace.
    """
    while True:
        try:
            value = input(prompt).strip()
            if value or not required:
                return value
            print("This field is required. Please enter a value.")
        except EOFError:
            # Handle piped input ending
            if required:
                raise ValueError("Required input not provided") from None
            return ""


def prompt_optional(prompt: str, suggestions: list[str] | None = None) -> str:
    """Prompt for optional input with suggestions.

    Args:
        prompt: The prompt text to display.
        suggestions: Optional list of suggestions to show.

    Returns:
        The user's input, or empty string if skipped.
    """
    if suggestions:
        print(f"  Suggestions: {', '.join(suggestions[:5])}")
        if len(suggestions) > 5:
            print(f"  ... and {len(suggestions) - 5} more")
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def prompt_editor(initial_content: str = "") -> str:
    """Open the user's editor for multi-line input.

    Args:
        initial_content: Optional initial content to populate the editor.

    Returns:
        The content entered by the user (comment lines stripped).

    Raises:
        RuntimeError: If the editor cannot be opened or exits with an error.
    """
    editor = get_editor()

    # Create tempfile with .md extension for syntax highlighting
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write("# Enter content below. Lines starting with # are ignored.\n")
        f.write("# Save and close the editor when done.\n")
        f.write("# Leave empty and save to cancel.\n")
        f.write("\n")
        if initial_content:
            f.write(initial_content)
        tmpfile = f.name

    try:
        # Handle editors that may need shell parsing (e.g., "code --wait")
        editor_parts = editor.split()
        result = subprocess.run(
            [*editor_parts, tmpfile],
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Editor '{editor}' exited with code {result.returncode}")

        with open(tmpfile, encoding="utf-8") as f:
            lines = f.readlines()

        # Filter out comment lines (starting with #)
        content_lines = [line for line in lines if not line.lstrip().startswith("#")]
        return "".join(content_lines).strip()

    finally:
        # Clean up tempfile
        try:
            os.unlink(tmpfile)
        except OSError:
            pass


def truncate_text(text: str, max_length: int = 200) -> tuple[str, bool]:
    """Truncate text to a maximum length.

    Args:
        text: The text to truncate.
        max_length: Maximum length before truncation.

    Returns:
        Tuple of (truncated_text, was_truncated).
    """
    if len(text) <= max_length:
        return text, False
    return text[:max_length] + "...", True


def show_preview(
    title: str,
    description: str,
    content: str,
    tags: str,
    project: str,
) -> bool:
    """Display a preview of the entry and ask for confirmation.

    Args:
        title: Entry title.
        description: Entry description.
        content: Entry content.
        tags: Comma-separated tags.
        project: Project name.

    Returns:
        True if user confirms, False otherwise.
    """
    separator = "\u2500" * 50  # Box drawing character

    print()
    print(separator)
    print("Preview")
    print(separator)
    print()

    print(f"Title:       {title}")
    print(f"Description: {description}")
    print(f"Tags:        {tags if tags else '(none)'}")
    print(f"Project:     {project if project else '(none)'}")
    print()

    # Show content with optional truncation
    content_preview, was_truncated = truncate_text(content, 300)
    print("Content:")
    # Indent content lines for readability
    for line in content_preview.split("\n"):
        print(f"  {line}")
    if was_truncated:
        print(f"  ... (truncated, {len(content)} chars total)")

    print()
    print(separator)
    print()

    try:
        response = input("Save this entry? [Y/n] ").strip().lower()
        # Default to yes if empty
        return response in ("", "y", "yes")
    except EOFError:
        return False
