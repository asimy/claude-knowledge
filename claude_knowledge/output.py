"""Terminal output formatting with rich."""

from __future__ import annotations

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Global console instance - auto-detects TTY
console = Console()


def print_error(message: str) -> None:
    """Print error message in red."""
    console.print(f"[red]Error:[/red] {message}")


def print_success(message: str) -> None:
    """Print success message in green."""
    console.print(f"[green]{message}[/green]")


def print_warning(message: str) -> None:
    """Print warning message in yellow."""
    console.print(f"[yellow]{message}[/yellow]")


def format_score(score: float, max_score: float = 1.0) -> Text:
    """Format score with color based on value.

    Args:
        score: The score value
        max_score: Maximum possible score (default 1.0)

    Returns:
        Rich Text object with colored score
    """
    percentage = score / max_score if max_score > 0 else 0
    if percentage >= 0.7:
        color = "green"
    elif percentage >= 0.4:
        color = "yellow"
    else:
        color = "red"

    # Format as percentage if max_score is 1.0, otherwise as raw value
    if max_score == 1.0:
        text = f"{score:.0%}"
    else:
        text = f"{score:.0f}/{max_score:.0f}"

    return Text(text, style=color)


def format_quality_score(score: float) -> Text:
    """Format quality score (0-100) with color based on value.

    Args:
        score: Quality score from 0 to 100

    Returns:
        Rich Text object with colored score
    """
    if score >= 70:
        color = "green"
    elif score >= 40:
        color = "yellow"
    else:
        color = "red"

    return Text(f"{score:.0f}/100", style=color)


def detect_language(content: str) -> str:
    """Attempt to detect programming language from content.

    Args:
        content: Code content to analyze

    Returns:
        Language identifier string for syntax highlighting
    """
    if not content:
        return "text"

    first_line = content.split("\n")[0].strip()

    # Check shebang
    if first_line.startswith("#!"):
        if "python" in first_line:
            return "python"
        if "bash" in first_line or "sh" in first_line:
            return "bash"
        if "ruby" in first_line:
            return "ruby"
        if "node" in first_line:
            return "javascript"
        if "perl" in first_line:
            return "perl"

    # Pattern matching for common languages
    content_lower = content.lower()

    # Python patterns
    if "def " in content or "import " in content or "class " in content:
        if "from " in content or "self" in content or ":" in first_line:
            return "python"

    # Go patterns
    if "func " in content or "package " in content:
        return "go"

    # JavaScript/TypeScript patterns
    if "function " in content or "const " in content or "let " in content:
        if "=>" in content or "require(" in content or "export " in content:
            return "javascript"

    # Ruby patterns
    if "def " in content and "end" in content:
        if "do " in content or "|" in content:
            return "ruby"

    # Rust patterns
    if "fn " in content and "let " in content:
        if "::" in content or "mut " in content:
            return "rust"

    # SQL patterns
    if any(
        kw in content_lower for kw in ["select ", "insert ", "update ", "delete ", "create table"]
    ):
        return "sql"

    # YAML patterns
    if content.strip().startswith("---") or (": " in content and "  " in content):
        if "{" not in content and ";" not in content:
            return "yaml"

    # JSON patterns
    if content.strip().startswith("{") or content.strip().startswith("["):
        return "json"

    # Shell patterns
    if first_line.startswith("$") or "#!/" in content or "export " in content:
        return "bash"

    return "text"


def print_code_block(content: str, language: str = "") -> None:
    """Print syntax-highlighted code block.

    Args:
        content: Code content to display
        language: Language for syntax highlighting (auto-detect if empty)
    """
    if not language:
        language = detect_language(content)

    syntax = Syntax(content, language, theme="monokai", line_numbers=False)
    console.print(syntax)


def create_stats_table(title: str = "Knowledge Base Statistics") -> Table:
    """Create a styled table for statistics.

    Args:
        title: Table title

    Returns:
        Rich Table object
    """
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    return table


def print_knowledge_item(
    item: dict,
    include_content: bool = False,
    include_score: bool = True,
) -> None:
    """Print a formatted knowledge item.

    Args:
        item: Knowledge item dictionary with id, title, description, content, tags, score
        include_content: Whether to display the full content
        include_score: Whether to display the relevance score
    """
    from claude_knowledge.utils import json_to_tags

    # Build title line
    title_parts = []

    if include_score and "score" in item:
        score_text = format_score(item["score"])
        title_parts.append(score_text)
        title_parts.append(Text(" "))

    title_parts.append(Text(item["title"], style="bold"))

    # Add tags
    tags = json_to_tags(item.get("tags"))
    if tags:
        # Use escaped brackets for display
        tags_str = f" [{', '.join(tags)}]"
        title_parts.append(Text(tags_str, style="cyan"))

    # Combine and print title line
    title_line = Text()
    for part in title_parts:
        title_line.append(part)
    console.print(title_line)

    # Print ID (dimmed)
    console.print(f"  ID: [dim]{item['id']}[/dim]")

    # Print description
    if item.get("description"):
        console.print(f"  {item['description']}")

    # Print content if requested
    if include_content and item.get("content"):
        console.print()
        print_code_block(item["content"])

    console.print()


def print_list_item(item: dict) -> None:
    """Print a formatted list item (for list command).

    Args:
        item: Knowledge item dictionary
    """
    from claude_knowledge.utils import json_to_tags

    tags = json_to_tags(item.get("tags"))
    # Escape brackets for Rich markup - use \[ to display literal [
    tag_str = f" [cyan]\\[{', '.join(tags)}][/cyan]" if tags else ""
    project_str = f" [magenta]({item['project']})[/magenta]" if item.get("project") else ""
    usage = item.get("usage_count", 0)

    console.print(
        f"[dim]{item['id']}[/dim]: [bold]{item['title']}[/bold]{tag_str}{project_str} "
        f"[dim](used {usage}x)[/dim]"
    )


def print_search_item(item: dict) -> None:
    """Print a formatted search result item.

    Args:
        item: Knowledge item dictionary
    """
    from claude_knowledge.utils import json_to_tags

    tags = json_to_tags(item.get("tags"))
    # Escape brackets for Rich markup - use \[ to display literal [
    tag_str = f" [cyan]\\[{', '.join(tags)}][/cyan]" if tags else ""
    project_str = f" [magenta]({item['project']})[/magenta]" if item.get("project") else ""

    console.print(f"[dim]{item['id']}[/dim]: [bold]{item['title']}[/bold]{tag_str}{project_str}")
    if item.get("description"):
        console.print(f"  {item['description']}")
    console.print()


def print_relationship(rel: dict, show_direction: bool = True) -> None:
    """Print a formatted relationship.

    Args:
        rel: Relationship dictionary with relationship_type, related_id, related_title, direction
        show_direction: Whether to show direction indicator
    """
    rel_type = rel.get("relationship_type", "related")
    related_id = rel.get("related_id", "")
    related_title = rel.get("related_title", "(unknown)")
    direction = rel.get("direction", "")

    # Format relationship type with color
    if rel_type == "depends-on":
        type_str = "[yellow]depends-on[/yellow]"
        if direction == "outgoing":
            arrow = "->"
        else:
            arrow = "<-"
    elif rel_type == "supersedes":
        type_str = "[red]supersedes[/red]"
        if direction == "outgoing":
            arrow = "->"
        else:
            arrow = "<-"
    else:  # related
        type_str = "[cyan]related[/cyan]"
        arrow = "<->"

    if show_direction and direction:
        console.print(f"  {arrow} {type_str} [dim]{related_id[:8]}[/dim] {related_title}")
    else:
        console.print(f"  {type_str} [dim]{related_id[:8]}[/dim] {related_title}")


def print_collection(collection: dict, show_description: bool = True) -> None:
    """Print a formatted collection.

    Args:
        collection: Collection dictionary with name, id, description, member_count
        show_description: Whether to show description
    """
    name = collection.get("name", "")
    coll_id = collection.get("id", "")
    description = collection.get("description", "")
    member_count = collection.get("member_count", 0)

    console.print(
        f"[bold]{name}[/bold] [dim]({coll_id[:8]})[/dim] [cyan]{member_count} entries[/cyan]"
    )
    if show_description and description:
        console.print(f"  {description}")


def print_collection_member(member: dict) -> None:
    """Print a formatted collection member.

    Args:
        member: Member dictionary with id, title, project, added_at
    """
    from claude_knowledge.utils import json_to_tags

    tags = json_to_tags(member.get("tags"))
    tag_str = f" [cyan]\\[{', '.join(tags)}][/cyan]" if tags else ""
    project_str = f" [magenta]({member['project']})[/magenta]" if member.get("project") else ""
    added = member.get("added_at", "")[:10] if member.get("added_at") else ""
    added_str = f" [dim](added {added})[/dim]" if added else ""

    console.print(
        f"  [dim]{member['id']}[/dim]: [bold]{member['title']}[/bold]"
        f"{tag_str}{project_str}{added_str}"
    )


def print_version_summary(version: dict, show_full: bool = False) -> None:
    """Print a formatted version summary.

    Args:
        version: Version dictionary with version_number, title, created_at, etc.
        show_full: Whether to show additional details.
    """
    version_num = version.get("version_number", "?")
    title = version.get("title", "(untitled)")
    created_at = version.get("created_at", "")
    created_by = version.get("created_by", "")
    change_summary = version.get("change_summary", "")

    # Format timestamp
    if created_at:
        # Show date and time
        timestamp = created_at[:19].replace("T", " ")
    else:
        timestamp = "(unknown)"

    console.print(f"  [bold cyan]v{version_num}[/bold cyan] {title}")
    console.print(f"    Created: [dim]{timestamp}[/dim]", end="")
    if created_by:
        console.print(f" by [dim]{created_by}[/dim]", end="")
    console.print()

    if change_summary:
        console.print(f"    [dim]{change_summary}[/dim]")

    if show_full:
        version_id = version.get("id", "")
        console.print(f"    ID: [dim]{version_id}[/dim]")


def print_version_diff(diff_result: dict) -> None:
    """Print a formatted diff between versions.

    Args:
        diff_result: Dictionary from VersioningService.diff() containing
                    version info and diff strings.
    """
    version_a = diff_result.get("version_a", {})
    version_b = diff_result.get("version_b", {})

    # Print version comparison header
    ver_a_num = version_a.get("number", "?")
    ver_b_label = version_b.get("label", f"v{version_b.get('number', '?')}")

    console.print(f"[bold]Comparing v{ver_a_num} -> {ver_b_label}[/bold]")
    console.print()

    # Track if any changes
    has_changes = False

    # Title diff
    title_diff = diff_result.get("title_diff")
    if title_diff:
        has_changes = True
        console.print("[bold]Title changes:[/bold]")
        _print_unified_diff(title_diff)
        console.print()

    # Description diff
    desc_diff = diff_result.get("description_diff")
    if desc_diff:
        has_changes = True
        console.print("[bold]Description changes:[/bold]")
        _print_unified_diff(desc_diff)
        console.print()

    # Content diff
    content_diff = diff_result.get("content_diff")
    if content_diff:
        has_changes = True
        console.print("[bold]Content changes:[/bold]")
        _print_unified_diff(content_diff)
        console.print()

    # Other changes
    other_changes = []
    if diff_result.get("tags_changed"):
        other_changes.append("tags")
    if diff_result.get("project_changed"):
        other_changes.append("project")
    if diff_result.get("confidence_changed"):
        other_changes.append("confidence")

    if other_changes:
        has_changes = True
        console.print(f"[bold]Also changed:[/bold] {', '.join(other_changes)}")

    if not has_changes:
        console.print("[dim]No differences found[/dim]")


def _print_unified_diff(diff_text: str) -> None:
    """Print unified diff with syntax highlighting.

    Args:
        diff_text: Unified diff string.
    """
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            console.print(f"[bold]{line}[/bold]")
        elif line.startswith("@@"):
            console.print(f"[cyan]{line}[/cyan]")
        elif line.startswith("+"):
            console.print(f"[green]{line}[/green]")
        elif line.startswith("-"):
            console.print(f"[red]{line}[/red]")
        else:
            console.print(line)
