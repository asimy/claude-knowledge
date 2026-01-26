"""Command-line interface for the knowledge management system."""

import argparse
import json
import os
import sys

from claude_knowledge.knowledge_manager import KnowledgeManager
from claude_knowledge.output import (
    console,
    create_stats_table,
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
from claude_knowledge.utils import json_to_tags, parse_relative_date

try:
    import argcomplete

    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="claude-kb",
        description="Knowledge management system for Claude Code",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Import completers if argcomplete is available
    project_completer = None
    entry_id_completer = None
    sync_path_completer = None

    if ARGCOMPLETE_AVAILABLE:
        from claude_knowledge.completions import (
            get_entry_id_completer,
            get_project_completer,
            get_sync_path_completer,
        )

        project_completer = get_project_completer()
        entry_id_completer = get_entry_id_completer()
        sync_path_completer = get_sync_path_completer()

    # capture command
    capture_parser = subparsers.add_parser(
        "capture",
        help="Capture new knowledge",
    )
    capture_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive mode with prompts for each field",
    )
    capture_parser.add_argument(
        "--title",
        help="Short title for the knowledge entry (required unless -i)",
    )
    capture_parser.add_argument(
        "--description",
        help="Description of what this knowledge covers (required unless -i)",
    )
    capture_parser.add_argument(
        "--content",
        help="Full content/details of the knowledge (required unless -i)",
    )
    capture_parser.add_argument(
        "--tags",
        help="Comma-separated tags (e.g., 'auth,oauth,python')",
    )
    capture_parser.add_argument(
        "--context",
        help="Comma-separated context (e.g., 'backend,python')",
    )
    capture_project = capture_parser.add_argument(
        "--project",
        help="Project name/identifier",
    )
    if project_completer:
        capture_project.completer = project_completer

    # retrieve command
    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Retrieve relevant knowledge",
    )
    retrieve_parser.add_argument(
        "--query",
        required=True,
        help="Search query text",
    )
    retrieve_project = retrieve_parser.add_argument(
        "--project",
        help="Filter by project",
    )
    if project_completer:
        retrieve_project.completer = project_completer
    retrieve_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results (default: 5)",
    )
    retrieve_parser.add_argument(
        "--budget",
        type=int,
        default=2000,
        help="Token budget for results (default: 2000)",
    )
    retrieve_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    retrieve_parser.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="Minimum relevance score 0.0-1.0 (default: 0.3)",
    )
    retrieve_parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        metavar="TAG",
        help="Filter by tag (can be repeated for AND logic)",
    )
    retrieve_parser.add_argument(
        "--since",
        help="Filter entries after date (ISO format or relative: 7d, 2w, 1m)",
    )
    retrieve_parser.add_argument(
        "--until",
        help="Filter entries before date (ISO format or relative: 7d, 2w, 1m)",
    )
    retrieve_parser.add_argument(
        "--date-field",
        choices=["created", "last_used"],
        default="created",
        help="Date field to filter on (default: created)",
    )
    retrieve_parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Enable fuzzy tag matching (matches with minor typos)",
    )

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List knowledge entries",
    )
    list_project = list_parser.add_argument(
        "--project",
        help="Filter by project",
    )
    if project_completer:
        list_project.completer = project_completer
    list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of results (default: 20)",
    )
    list_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    list_parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        metavar="TAG",
        help="Filter by tag (can be repeated for AND logic)",
    )
    list_parser.add_argument(
        "--since",
        help="Filter entries after date (ISO format or relative: 7d, 2w, 1m)",
    )
    list_parser.add_argument(
        "--until",
        help="Filter entries before date (ISO format or relative: 7d, 2w, 1m)",
    )
    list_parser.add_argument(
        "--date-field",
        choices=["created", "last_used"],
        default="created",
        help="Date field to filter on (default: created)",
    )
    list_parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Enable fuzzy tag matching (matches with minor typos)",
    )

    # delete command
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a knowledge entry",
    )
    delete_id = delete_parser.add_argument(
        "id",
        help="Knowledge entry ID to delete",
    )
    if entry_id_completer:
        delete_id.completer = entry_id_completer

    # search command
    search_parser = subparsers.add_parser(
        "search",
        help="Text search in knowledge",
    )
    search_parser.add_argument(
        "text",
        help="Search text",
    )
    search_project = search_parser.add_argument(
        "--project",
        help="Filter by project",
    )
    if project_completer:
        search_project.completer = project_completer
    search_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of results (default: 20)",
    )
    search_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    search_parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        metavar="TAG",
        help="Filter by tag (can be repeated for AND logic)",
    )
    search_parser.add_argument(
        "--since",
        help="Filter entries after date (ISO format or relative: 7d, 2w, 1m)",
    )
    search_parser.add_argument(
        "--until",
        help="Filter entries before date (ISO format or relative: 7d, 2w, 1m)",
    )
    search_parser.add_argument(
        "--date-field",
        choices=["created", "last_used"],
        default="created",
        help="Date field to filter on (default: created)",
    )
    search_parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Enable fuzzy tag matching (matches with minor typos)",
    )

    # stats command
    subparsers.add_parser(
        "stats",
        help="Show knowledge base statistics",
    )

    # get command
    get_parser = subparsers.add_parser(
        "get",
        help="Get a specific knowledge entry",
    )
    get_id = get_parser.add_argument(
        "id",
        help="Knowledge entry ID",
    )
    if entry_id_completer:
        get_id.completer = entry_id_completer
    get_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    # update command
    update_parser = subparsers.add_parser(
        "update",
        help="Update a knowledge entry",
    )
    update_id = update_parser.add_argument(
        "id",
        help="Knowledge entry ID to update",
    )
    if entry_id_completer:
        update_id.completer = entry_id_completer
    update_parser.add_argument(
        "--title",
        help="New title",
    )
    update_parser.add_argument(
        "--description",
        help="New description",
    )
    update_parser.add_argument(
        "--content",
        help="New content",
    )
    update_parser.add_argument(
        "--tags",
        help="New tags (comma-separated)",
    )
    update_project = update_parser.add_argument(
        "--project",
        help="New project",
    )
    if project_completer:
        update_project.completer = project_completer

    # export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export knowledge to JSON file",
    )
    export_parser.add_argument(
        "file",
        help="Output file path (use - for stdout)",
    )
    export_project = export_parser.add_argument(
        "--project",
        help="Only export entries for this project",
    )
    if project_completer:
        export_project.completer = project_completer

    # import command
    import_parser = subparsers.add_parser(
        "import",
        help="Import knowledge from JSON file",
    )
    import_parser.add_argument(
        "file",
        help="Input file path (use - for stdin)",
    )
    import_parser.add_argument(
        "--no-skip-duplicates",
        action="store_true",
        help="Raise error on duplicate IDs instead of skipping",
    )

    # purge command
    purge_parser = subparsers.add_parser(
        "purge",
        help="Delete all knowledge entries",
    )
    purge_project = purge_parser.add_argument(
        "--project",
        help="Only purge entries for this project",
    )
    if project_completer:
        purge_project.completer = project_completer
    purge_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # sync command
    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync knowledge with a directory",
    )
    sync_path = sync_parser.add_argument(
        "path",
        nargs="?",
        help="Path to sync directory (uses saved path if omitted)",
    )
    if sync_path_completer:
        sync_path.completer = sync_path_completer
    sync_parser.add_argument(
        "--push-only",
        action="store_true",
        help="Only push local changes to sync directory",
    )
    sync_parser.add_argument(
        "--pull-only",
        action="store_true",
        help="Only pull remote changes from sync directory",
    )
    sync_parser.add_argument(
        "--strategy",
        choices=["last-write-wins", "local-wins", "remote-wins", "manual"],
        default="last-write-wins",
        help="Conflict resolution strategy (default: last-write-wins)",
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without making changes",
    )
    sync_parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize sync directory structure",
    )
    sync_parser.add_argument(
        "--status",
        action="store_true",
        help="Show sync status without syncing",
    )
    sync_project = sync_parser.add_argument(
        "--project",
        help="Only sync entries for this project",
    )
    if project_completer:
        sync_project.completer = project_completer

    # duplicates command
    duplicates_parser = subparsers.add_parser(
        "duplicates",
        help="Find potential duplicate entries",
    )
    duplicates_parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold 0.0-1.0 (default: 0.85)",
    )
    duplicates_project = duplicates_parser.add_argument(
        "--project",
        help="Only check entries for this project",
    )
    if project_completer:
        duplicates_project.completer = project_completer
    duplicates_merge = duplicates_parser.add_argument(
        "--merge",
        nargs=2,
        metavar=("TARGET_ID", "SOURCE_ID"),
        help="Merge SOURCE_ID into TARGET_ID",
    )
    if entry_id_completer:
        duplicates_merge.completer = entry_id_completer

    # stale command
    stale_parser = subparsers.add_parser(
        "stale",
        help="Find entries that haven't been used recently",
    )
    stale_parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Days of inactivity to consider stale (default: 90)",
    )
    stale_project = stale_parser.add_argument(
        "--project",
        help="Only check entries for this project",
    )
    if project_completer:
        stale_project.completer = project_completer
    stale_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # quality command
    quality_parser = subparsers.add_parser(
        "quality",
        help="Score entries by quality metrics",
    )
    quality_project = quality_parser.add_argument(
        "--project",
        help="Only check entries for this project",
    )
    if project_completer:
        quality_project.completer = project_completer
    quality_parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum quality score to include (0-100)",
    )
    quality_parser.add_argument(
        "--max-score",
        type=float,
        help="Maximum quality score to include (0-100)",
    )
    quality_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # completions command
    completions_parser = subparsers.add_parser(
        "completions",
        help="Generate shell completion setup instructions",
    )
    completions_parser.add_argument(
        "shell",
        choices=["bash", "zsh", "fish"],
        help="Shell to generate completions for",
    )

    # summarize command
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Extract knowledge from Claude Code sessions",
    )
    summarize_parser.add_argument(
        "--list",
        action="store_true",
        dest="list_sessions",
        help="List available sessions",
    )
    summarize_parser.add_argument(
        "--session",
        help="Session ID to summarize",
    )
    summarize_parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview extraction without capturing",
    )
    summarize_parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-capture entries meeting confidence threshold",
    )
    summarize_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for auto-capture (default: 0.5)",
    )
    summarize_parser.add_argument(
        "--since",
        help="Summarize sessions from the last N days (e.g., '7d')",
    )
    summarize_parser.add_argument(
        "--mark-processed",
        action="store_true",
        help="Mark session as processed without extracting",
    )
    summarize_project = summarize_parser.add_argument(
        "--project",
        help="Filter by project path",
    )
    if project_completer:
        summarize_project.completer = project_completer
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum sessions to list (default: 20)",
    )

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Extract knowledge from git commits and code patterns",
    )
    analyze_parser.add_argument(
        "--commits",
        action="store_true",
        help="Analyze git commit history",
    )
    analyze_parser.add_argument(
        "--patterns",
        action="store_true",
        help="Analyze code patterns and architecture",
    )
    analyze_parser.add_argument(
        "--since",
        help="Analyze commits from the last N days (e.g., '30d')",
    )
    analyze_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum commits to analyze (default: 50)",
    )
    analyze_parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview extraction without capturing",
    )
    analyze_parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-capture entries meeting confidence threshold",
    )
    analyze_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for auto-capture (default: 0.5)",
    )
    analyze_parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Include file patterns (e.g., '*.py'). Can be specified multiple times.",
    )
    analyze_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude file patterns (e.g., 'test_*'). Can be specified multiple times.",
    )
    analyze_project = analyze_parser.add_argument(
        "--project",
        help="Project path to analyze (default: current directory)",
    )
    if sync_path_completer:
        analyze_project.completer = sync_path_completer

    return parser


def cmd_capture(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the capture command."""
    if args.interactive:
        return cmd_capture_interactive(args, km)

    # Non-interactive mode: validate required fields
    missing = []
    if not args.title:
        missing.append("--title")
    if not args.description:
        missing.append("--description")
    if not args.content:
        missing.append("--content")

    if missing:
        print_error(f"Missing required arguments: {', '.join(missing)}")
        console.print(
            "Use -i/--interactive for interactive mode, or provide all required arguments."
        )
        return 1

    context = None
    if args.context:
        context = [c.strip() for c in args.context.split(",") if c.strip()]

    knowledge_id = km.capture(
        title=args.title,
        description=args.description,
        content=args.content,
        tags=args.tags,
        context=context,
        project=args.project,
    )

    print_success(f"Knowledge captured with ID: {knowledge_id}")
    return 0


def cmd_capture_interactive(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle interactive capture mode."""
    from claude_knowledge.interactive import (
        prompt_editor,
        prompt_line,
        prompt_optional,
        show_preview,
    )

    console.print("Interactive capture mode. Press Ctrl+C to cancel at any time.\n")

    try:
        # Get existing projects for suggestions
        projects = km.get_distinct_projects()

        # Prompt for required fields
        title = prompt_line("Title: ", required=True)
        description = prompt_line("Description: ", required=True)

        # Open editor for content
        console.print("Opening editor for content...")
        content = prompt_editor()

        if not content:
            print_error("Content cannot be empty.")
            return 1

        # Optional fields
        tags = prompt_optional("Tags (comma-separated, optional): ")
        project = prompt_optional("Project (optional): ", suggestions=projects)

        # Show preview and confirm
        if not show_preview(title, description, content, tags, project):
            print_warning("Capture cancelled.")
            return 0

        # Parse context from args if provided (can be combined with interactive)
        context = None
        if args.context:
            context = [c.strip() for c in args.context.split(",") if c.strip()]

        # Capture the entry
        knowledge_id = km.capture(
            title=title,
            description=description,
            content=content,
            tags=tags if tags else None,
            context=context,
            project=project if project else None,
        )

        console.print()
        print_success(f"Knowledge captured with ID: {knowledge_id}")
        return 0

    except KeyboardInterrupt:
        console.print("\n")
        print_warning("Capture cancelled.")
        return 0
    except RuntimeError as e:
        console.print()
        print_error(str(e))
        return 1


def cmd_retrieve(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the retrieve command."""
    # Parse date filters
    since = parse_relative_date(args.since) if args.since else None
    until = parse_relative_date(args.until) if args.until else None

    items = km.retrieve(
        query=args.query,
        n_results=args.limit,
        token_budget=args.budget,
        project=args.project,
        min_score=args.min_score,
        tags=args.tags,
        since=since,
        until=until,
        date_field=args.date_field,
        fuzzy=args.fuzzy,
    )

    if not items:
        console.print("No relevant knowledge found.")
        return 0

    if args.format == "json":
        # Clean up items for JSON output
        output = []
        for item in items:
            clean_item = {
                "id": item["id"],
                "title": item["title"],
                "description": item["description"],
                "content": item["content"],
                "tags": json_to_tags(item.get("tags")),
                "project": item.get("project"),
                "score": round(item["score"], 3),
                "usage_count": item.get("usage_count", 0),
            }
            output.append(clean_item)
        print(json.dumps(output, indent=2))
    elif args.format == "markdown":
        print(km.format_for_context(items))
    else:  # text
        for item in items:
            print_knowledge_item(item, include_content=False, include_score=True)

    return 0


def cmd_list(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the list command."""
    # Parse date filters
    since = parse_relative_date(args.since) if args.since else None
    until = parse_relative_date(args.until) if args.until else None

    items = km.list_all(
        project=args.project,
        limit=args.limit,
        tags=args.tags,
        since=since,
        until=until,
        date_field=args.date_field,
        fuzzy=args.fuzzy,
    )

    if not items:
        console.print("No knowledge entries found.")
        return 0

    if args.format == "json":
        output = []
        for item in items:
            clean_item = {
                "id": item["id"],
                "title": item["title"],
                "tags": json_to_tags(item.get("tags")),
                "project": item.get("project"),
                "usage_count": item.get("usage_count", 0),
            }
            output.append(clean_item)
        print(json.dumps(output, indent=2))
    else:  # text
        for item in items:
            print_list_item(item)

    return 0


def cmd_delete(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the delete command."""
    if km.delete(args.id):
        print_success(f"Deleted knowledge entry: {args.id}")
        return 0
    else:
        print_error(f"Knowledge entry not found: {args.id}")
        return 1


def cmd_search(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the search command."""
    # Parse date filters
    since = parse_relative_date(args.since) if args.since else None
    until = parse_relative_date(args.until) if args.until else None

    items = km.search(
        args.text,
        project=args.project,
        limit=args.limit,
        tags=args.tags,
        since=since,
        until=until,
        date_field=args.date_field,
        fuzzy=args.fuzzy,
    )

    if not items:
        console.print("No matching entries found.")
        return 0

    if args.format == "json":
        output = []
        for item in items:
            clean_item = {
                "id": item["id"],
                "title": item["title"],
                "description": item["description"],
                "tags": json_to_tags(item.get("tags")),
                "project": item.get("project"),
                "usage_count": item.get("usage_count", 0),
            }
            output.append(clean_item)
        print(json.dumps(output, indent=2))
    else:  # text
        for item in items:
            print_search_item(item)

    return 0


def cmd_stats(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the stats command."""
    stats = km.stats()

    # Main stats table
    table = create_stats_table()
    table.add_row("Total entries", str(stats["total_entries"]))
    console.print(table)

    if stats["by_project"]:
        console.print("\n[bold]By project:[/bold]")
        for project, count in stats["by_project"].items():
            console.print(f"  [magenta]{project}[/magenta]: {count}")

    if stats["most_used"]:
        console.print("\n[bold]Most used:[/bold]")
        for item in stats["most_used"]:
            console.print(f"  {item['title']} [dim]({item['usage_count']}x)[/dim]")

    if stats["recently_added"]:
        console.print("\n[bold]Recently added:[/bold]")
        for item in stats["recently_added"]:
            console.print(f"  {item['title']} [dim]({item['created'][:10]})[/dim]")

    if stats["recently_used"]:
        console.print("\n[bold]Recently used:[/bold]")
        for item in stats["recently_used"]:
            last_used = item["last_used"][:10] if item["last_used"] else "never"
            console.print(f"  {item['title']} [dim]({last_used})[/dim]")

    return 0


def cmd_get(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the get command."""
    item = km.get(args.id)

    if not item:
        print_error(f"Knowledge entry not found: {args.id}")
        return 1

    if args.format == "json":
        clean_item = {
            "id": item["id"],
            "title": item["title"],
            "description": item["description"],
            "content": item["content"],
            "tags": json_to_tags(item.get("tags")),
            "project": item.get("project"),
            "usage_count": item.get("usage_count", 0),
            "created": item.get("created"),
            "last_used": item.get("last_used"),
        }
        print(json.dumps(clean_item, indent=2))
    elif args.format == "markdown":
        from claude_knowledge.utils import format_knowledge_item

        print(format_knowledge_item(item, include_content=True, include_score=False))
    else:  # text
        tags = json_to_tags(item.get("tags"))
        console.print(f"[bold]Title:[/bold] {item['title']}")
        console.print(f"[bold]ID:[/bold] [dim]{item['id']}[/dim]")
        project = item.get("project")
        project_str = f"[magenta]{project}[/magenta]" if project else "[dim](none)[/dim]"
        console.print(f"[bold]Project:[/bold] {project_str}")
        tags_str = f"[cyan]{', '.join(tags)}[/cyan]" if tags else "[dim](none)[/dim]"
        console.print(f"[bold]Tags:[/bold] {tags_str}")
        console.print(f"[bold]Usage count:[/bold] {item.get('usage_count', 0)}")
        console.print(f"[bold]Created:[/bold] [dim]{item.get('created')}[/dim]")
        last_used = item.get("last_used") or "never"
        console.print(f"[bold]Last used:[/bold] [dim]{last_used}[/dim]")
        console.print()
        console.print("[bold]Description:[/bold]")
        console.print(f"  {item['description']}")
        console.print()
        console.print("[bold]Content:[/bold]")
        print_code_block(item["content"])

    return 0


def cmd_update(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the update command."""
    updates = {}
    if args.title:
        updates["title"] = args.title
    if args.description:
        updates["description"] = args.description
    if args.content:
        updates["content"] = args.content
    if args.tags:
        updates["tags"] = args.tags
    if args.project:
        updates["project"] = args.project

    if not updates:
        print_error("No updates specified")
        return 1

    if km.update(args.id, **updates):
        print_success(f"Updated knowledge entry: {args.id}")
        return 0
    else:
        print_error(f"Knowledge entry not found: {args.id}")
        return 1


def cmd_export(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the export command."""
    entries = km.export_all(project=args.project)

    if not entries:
        console.print("No entries to export.")
        return 0

    output = json.dumps(entries, indent=2, default=str)

    if args.file == "-":
        print(output)
    else:
        with open(args.file, "w") as f:
            f.write(output)
        print_success(f"Exported {len(entries)} entries to {args.file}")

    return 0


def cmd_import(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the import command."""
    if args.file == "-":
        data = sys.stdin.read()
    else:
        with open(args.file) as f:
            data = f.read()

    try:
        entries = json.loads(data)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        return 1

    if not isinstance(entries, list):
        print_error("JSON must be a list of entries")
        return 1

    result = km.import_data(entries, skip_duplicates=not args.no_skip_duplicates)

    print_success(f"Imported: {result['imported']}")
    if result["skipped"]:
        print_warning(f"Skipped (duplicates): {result['skipped']}")
    if result["errors"]:
        print_error(f"Errors: {result['errors']}")

    return 0 if result["errors"] == 0 else 1


def cmd_purge(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the purge command."""
    if not args.force:
        if args.project:
            prompt = f"Delete all entries for project '[magenta]{args.project}[/magenta]'? [y/N] "
        else:
            prompt = "[bold red]Delete ALL knowledge entries?[/bold red] [y/N] "

        console.print(prompt, end="")
        response = input().strip().lower()
        if response != "y":
            print_warning("Aborted.")
            return 0

    count = km.purge(project=args.project)
    print_success(f"Deleted {count} entries.")
    return 0


def cmd_sync(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the sync command."""
    sync_path = args.path

    # Handle --init flag
    if args.init:
        if not sync_path:
            print_error("Path required for --init")
            return 1
        km.init_sync_dir(sync_path)
        km.set_sync_path(sync_path)
        print_success(f"Initialized sync directory: {sync_path}")
        return 0

    # Handle --status flag
    if args.status:
        try:
            status = km.sync_status(sync_path=sync_path, project=args.project)
        except ValueError as e:
            print_error(str(e))
            return 1

        if "error" in status:
            print_error(status["error"])
            return 1

        console.print(f"[bold]Sync status for:[/bold] {status['sync_path']}")
        console.rule()
        console.print(f"To push:         [green]{len(status['to_push'])}[/green]")
        console.print(f"To pull:         [cyan]{len(status['to_pull'])}[/cyan]")
        console.print(f"Conflicts:       [yellow]{len(status['conflicts'])}[/yellow]")
        console.print(f"Delete local:    [red]{len(status['to_delete_local'])}[/red]")
        console.print(f"Delete remote:   [red]{len(status['to_delete_remote'])}[/red]")

        if status["to_push"]:
            console.print("\n[bold]Entries to push:[/bold]")
            for entry_id in status["to_push"][:5]:
                entry = km.get(entry_id)
                if entry:
                    console.print(f"  [dim]{entry_id}[/dim]: {entry['title']}")
            if len(status["to_push"]) > 5:
                console.print(f"  [dim]... and {len(status['to_push']) - 5} more[/dim]")

        if status["to_pull"]:
            console.print("\n[bold]Entries to pull:[/bold]")
            for entry_id in status["to_pull"][:5]:
                console.print(f"  [dim]{entry_id}[/dim]")
            if len(status["to_pull"]) > 5:
                console.print(f"  [dim]... and {len(status['to_pull']) - 5} more[/dim]")

        return 0

    # Perform sync
    try:
        result = km.sync(
            sync_path=sync_path,
            strategy=args.strategy,
            push_only=args.push_only,
            pull_only=args.pull_only,
            dry_run=args.dry_run,
            project=args.project,
        )
    except ValueError as e:
        print_error(str(e))
        return 1

    if result.errors:
        for error in result.errors:
            print_error(error)
        return 1

    prefix = "[yellow][DRY RUN][/yellow] " if args.dry_run else ""

    print_success(f"{prefix}Sync complete:")
    console.print(f"  Pushed:           [green]{result.pushed}[/green]")
    console.print(f"  Pulled:           [cyan]{result.pulled}[/cyan]")
    console.print(f"  Deletions pushed: [red]{result.deletions_pushed}[/red]")
    console.print(f"  Deletions pulled: [red]{result.deletions_pulled}[/red]")

    if result.conflicts:
        console.print(f"\n[yellow]Conflicts ({len(result.conflicts)}):[/yellow]")
        for conflict in result.conflicts:
            resolution = conflict.get("resolution", "unknown")
            cid = conflict["id"]
            title = conflict["title"]
            console.print(f"  [dim]{cid}[/dim]: {title} -> [yellow]{resolution}[/yellow]")

    saved_path = km.get_sync_path()
    if saved_path and not args.dry_run:
        console.print(f"\n[dim]Sync path saved: {saved_path}[/dim]")

    return 0


def cmd_duplicates(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the duplicates command."""
    # Handle merge operation
    if args.merge:
        target_id, source_id = args.merge
        if km.merge_entries(target_id, source_id):
            print_success(f"Merged {source_id} into {target_id}")
            return 0
        else:
            print_error("One or both entries not found")
            return 1

    # Find duplicates
    groups = km.find_duplicates(threshold=args.threshold, project=args.project)

    if not groups:
        console.print("No potential duplicates found.")
        return 0

    console.print(f"Found [yellow]{len(groups)}[/yellow] potential duplicate group(s):\n")

    for i, group in enumerate(groups, 1):
        console.print(f"[bold]Group {i}:[/bold]")
        for entry in group:
            sim = entry["similarity"]
            if sim < 1.0:
                sim_text = format_score(sim)
                console.print(f"  [dim]{entry['id']}[/dim]: {entry['title']} ", end="")
                console.print(sim_text)
            else:
                console.print(f"  [dim]{entry['id']}[/dim]: {entry['title']}")
        console.print()

    console.print("[dim]To merge duplicates, use:[/dim]")
    console.print("  [cyan]claude-kb duplicates --merge <target_id> <source_id>[/cyan]")

    return 0


def cmd_stale(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the stale command."""
    entries = km.find_stale(days=args.days, project=args.project)

    if not entries:
        console.print(f"No entries inactive for more than {args.days} days.")
        return 0

    if args.format == "json":
        output = []
        for entry in entries:
            output.append(
                {
                    "id": entry["id"],
                    "title": entry["title"],
                    "days_stale": entry.get("days_stale"),
                    "last_used": entry.get("last_used"),
                    "usage_count": entry.get("usage_count", 0),
                }
            )
        print(json.dumps(output, indent=2))
    else:
        console.print(
            f"Found [yellow]{len(entries)}[/yellow] stale entries (>{args.days} days inactive):\n"
        )
        for entry in entries:
            days = entry.get("days_stale")
            days_str = f"{days} days" if days else "unknown"
            usage = entry.get("usage_count", 0)
            console.print(f"  [dim]{entry['id']}[/dim]: [bold]{entry['title']}[/bold]")
            console.print(f"    Inactive: [yellow]{days_str}[/yellow], Used: {usage}x")

    return 0


def cmd_quality(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the quality command."""
    entries = km.score_quality(
        project=args.project,
        min_score=args.min_score,
        max_score=args.max_score,
    )

    if not entries:
        console.print("No entries found matching the criteria.")
        return 0

    if args.format == "json":
        output = []
        for entry in entries:
            output.append(
                {
                    "id": entry["id"],
                    "title": entry["title"],
                    "quality_score": entry["quality_score"],
                    "usage_count": entry.get("usage_count", 0),
                    "has_tags": bool(entry.get("tags", "").strip()),
                    "description_length": len(entry.get("description") or ""),
                    "content_length": len(entry.get("content") or ""),
                }
            )
        print(json.dumps(output, indent=2))
    else:
        num_entries = len(entries)
        console.print(
            f"Quality scores for [cyan]{num_entries}[/cyan] entries (sorted low to high):\n"
        )
        for entry in entries:
            score = entry["quality_score"]
            score_text = format_quality_score(score)
            tags = "[green]yes[/green]" if entry.get("tags", "").strip() else "[dim]no[/dim]"
            desc_len = len(entry.get("description") or "")
            content_len = len(entry.get("content") or "")
            usage = entry.get("usage_count", 0)
            console.print(f"  [dim]{entry['id']}[/dim]: [bold]{entry['title']}[/bold]")
            console.print("    Score: ", end="")
            console.print(score_text, end="")
            console.print(
                f" | Tags: {tags} | Desc: {desc_len} chars | "
                f"Content: {content_len} chars | Used: {usage}x"
            )

    return 0


def cmd_completions(args: argparse.Namespace) -> int:
    """Handle the completions command."""
    if not ARGCOMPLETE_AVAILABLE:
        print_error("argcomplete is not installed.")
        console.print("Install with: [cyan]pip install 'claude-knowledge[completions]'[/cyan]")
        return 1

    shell = args.shell

    if shell == "bash":
        print("""# Add this to your ~/.bashrc:
eval "$(register-python-argcomplete claude-kb)"
""")
    elif shell == "zsh":
        print("""# Add this to your ~/.zshrc:
autoload -U bashcompinit
bashcompinit
eval "$(register-python-argcomplete claude-kb)"
""")
    elif shell == "fish":
        print("""# Run this command once:
register-python-argcomplete --shell fish claude-kb > ~/.config/fish/completions/claude-kb.fish
""")

    return 0


def cmd_summarize(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the summarize command."""
    from datetime import datetime, timedelta

    from claude_knowledge.session_extractor import SessionExtractor
    from claude_knowledge.session_parser import SessionParser

    parser = SessionParser()
    extractor = SessionExtractor()

    # Handle --list flag
    if args.list_sessions:
        sessions = parser.list_sessions(project_path=args.project, limit=args.limit)
        if not sessions:
            console.print("No sessions found.")
            return 0

        console.print(f"Found [cyan]{len(sessions)}[/cyan] session(s):\n")
        for session in sessions:
            session_id = session.get("session_id", "")[:8]
            first_prompt = session.get("first_prompt", "")[:60]
            if len(session.get("first_prompt", "")) > 60:
                first_prompt += "..."
            msg_count = session.get("message_count", 0)
            modified = session.get("modified", "")[:10] if session.get("modified") else ""
            project = session.get("project_path", "").split("/")[-1] or "(unknown)"

            # Check if processed
            full_id = session.get("session_id", "")
            processed = km.is_session_processed(full_id)
            status = "[green]processed[/green]" if processed else "[dim]pending[/dim]"

            console.print(f"[dim]{session_id}[/dim] {status} [magenta]{project}[/magenta]")
            console.print(f"  [bold]{first_prompt}[/bold]")
            console.print(f"  [dim]{msg_count} messages, modified {modified}[/dim]")
            console.print()
        return 0

    # Handle --since flag
    if args.since:
        # Parse duration (e.g., "7d", "24h")
        since_str = args.since.lower()
        if since_str.endswith("d"):
            days = int(since_str[:-1])
            since_time = datetime.now() - timedelta(days=days)
        elif since_str.endswith("h"):
            hours = int(since_str[:-1])
            since_time = datetime.now() - timedelta(hours=hours)
        else:
            print_error(f"Invalid --since format: {args.since}. Use format like '7d' or '24h'.")
            return 1

        sessions = parser.get_sessions_since(since_time, project_path=args.project)
        if not sessions:
            console.print(f"No sessions modified since {since_time.strftime('%Y-%m-%d %H:%M')}.")
            return 0

        console.print(
            f"Found [cyan]{len(sessions)}[/cyan] session(s) since "
            f"{since_time.strftime('%Y-%m-%d %H:%M')}:\n"
        )

        total_extracted = 0
        total_captured = 0

        for session_info in sessions:
            session_id = session_info.get("session_id", "")
            if km.is_session_processed(session_id):
                continue

            transcript = parser.parse_session(session_id)
            if not transcript:
                continue

            extractions = extractor.extract(transcript)
            if not extractions:
                continue

            console.print(f"[bold]{session_info.get('first_prompt', '')[:60]}[/bold]")
            console.print(f"  [dim]Session: {session_id[:8]}[/dim]")

            for ext in extractions:
                if ext.confidence >= args.min_confidence:
                    total_extracted += 1
                    confidence_text = format_score(ext.confidence)
                    console.print("  ", end="")
                    console.print(confidence_text, end="")
                    console.print(f" {ext.title}")

                    if args.auto:
                        # Auto-capture the entry
                        knowledge_id = km.capture(
                            title=ext.title,
                            description=ext.description,
                            content=ext.content,
                            tags=",".join(ext.tags),
                            project=transcript.project_path.split("/")[-1],
                            source="session",
                            confidence=ext.confidence,
                        )
                        total_captured += 1
                        console.print(f"    [green]Captured: {knowledge_id}[/green]")

            if args.auto:
                above_threshold = [e for e in extractions if e.confidence >= args.min_confidence]
                km.mark_session_processed(
                    session_id,
                    transcript.project_path,
                    entries_created=len(above_threshold),
                )

            console.print()

        console.print(f"\nTotal: [cyan]{total_extracted}[/cyan] extractions found")
        if args.auto:
            console.print(f"Captured: [green]{total_captured}[/green] entries")
        return 0

    # Handle --session flag
    if args.session:
        session_id = args.session

        # Handle --mark-processed without extraction
        if args.mark_processed:
            sessions = parser.list_sessions()
            session_info = next(
                (s for s in sessions if s.get("session_id") == session_id),
                None,
            )
            if not session_info:
                print_error(f"Session not found: {session_id}")
                return 1

            km.mark_session_processed(
                session_id,
                session_info.get("project_path", ""),
                entries_created=0,
            )
            print_success(f"Marked session {session_id[:8]} as processed")
            return 0

        # Parse and extract from the session
        transcript = parser.parse_session(session_id, project_path=args.project)
        if not transcript:
            print_error(f"Session not found or could not be parsed: {session_id}")
            return 1

        extractions = extractor.extract(transcript)
        if not extractions:
            console.print("No knowledge entries could be extracted from this session.")
            if not args.preview:
                km.mark_session_processed(session_id, transcript.project_path, entries_created=0)
            return 0

        console.print(f"Session: [dim]{session_id}[/dim]")
        console.print(f"Project: [magenta]{transcript.project_path}[/magenta]")
        if transcript.summary:
            console.print(f"Summary: {transcript.summary}")
        console.print()
        console.print(f"Found [cyan]{len(extractions)}[/cyan] potential knowledge entries:\n")

        captured_count = 0
        for i, ext in enumerate(extractions, 1):
            confidence_text = format_score(ext.confidence)
            meets_threshold = ext.confidence >= args.min_confidence

            console.print(f"[bold]{i}. {ext.title}[/bold]")
            console.print("   Confidence: ", end="")
            console.print(confidence_text)
            console.print(f"   Type: [cyan]{ext.extraction_type}[/cyan]")
            if ext.tags:
                console.print(f"   Tags: [cyan]{', '.join(ext.tags)}[/cyan]")
            console.print(f"   Description: {ext.description[:100]}...")
            console.print()

            # In preview mode, just show the extraction
            if args.preview:
                if len(ext.content) > 200:
                    console.print("   Content preview:")
                    print_code_block(ext.content[:200] + "\n...")
                else:
                    console.print("   Content:")
                    print_code_block(ext.content)
                console.print()
                continue

            # In auto mode, capture entries meeting threshold
            if args.auto and meets_threshold:
                knowledge_id = km.capture(
                    title=ext.title,
                    description=ext.description,
                    content=ext.content,
                    tags=",".join(ext.tags),
                    project=transcript.project_path.split("/")[-1],
                    source="session",
                    confidence=ext.confidence,
                )
                captured_count += 1
                console.print(f"   [green]Captured: {knowledge_id}[/green]")
                console.print()

            # In interactive mode (not preview, not auto), prompt user
            elif not args.auto:
                console.print("   Content preview:")
                preview_content = ext.content[:300] if len(ext.content) > 300 else ext.content
                print_code_block(preview_content)
                if len(ext.content) > 300:
                    console.print("   [dim]... (content truncated)[/dim]")
                console.print()

                # Prompt for capture
                console.print("   Capture this entry? [y/N/q] ", end="")
                try:
                    response = input().strip().lower()
                except EOFError:
                    response = "n"

                if response == "q":
                    print_warning("Aborted.")
                    return 0
                elif response == "y":
                    knowledge_id = km.capture(
                        title=ext.title,
                        description=ext.description,
                        content=ext.content,
                        tags=",".join(ext.tags),
                        project=transcript.project_path.split("/")[-1],
                        source="session",
                        confidence=ext.confidence,
                    )
                    captured_count += 1
                    print_success(f"Captured: {knowledge_id}")
                console.print()

        # Mark session as processed
        if not args.preview:
            km.mark_session_processed(
                session_id,
                transcript.project_path,
                entries_created=captured_count,
            )

        if captured_count > 0:
            print_success(f"Captured {captured_count} entries from session")
        return 0

    # No specific action - show help
    console.print("Usage: claude-kb summarize [--list] [--session ID] [--since 7d]")
    console.print()
    console.print("Options:")
    console.print("  --list              List available sessions")
    console.print("  --session ID        Summarize a specific session")
    console.print("  --preview           Preview without capturing")
    console.print("  --auto              Auto-capture entries meeting threshold")
    console.print("  --min-confidence N  Minimum confidence (default: 0.5)")
    console.print("  --since Nd          Process sessions from last N days")
    console.print("  --project PATH      Filter by project path")
    return 0


def cmd_analyze(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the analyze command."""
    from datetime import datetime, timedelta
    from pathlib import Path

    from claude_knowledge.code_extractor import CodeExtractor
    from claude_knowledge.code_parser import CodeParser
    from claude_knowledge.git_extractor import GitExtractor
    from claude_knowledge.git_parser import GitParser

    # Determine project path
    project_path = Path(args.project).resolve() if args.project else Path.cwd()
    project_name = project_path.name

    if not args.commits and not args.patterns:
        # Default: show help
        console.print("Usage: claude-kb analyze [--commits] [--patterns] [options]")
        console.print()
        console.print("Modes:")
        console.print("  --commits           Analyze git commit history")
        console.print("  --patterns          Analyze code patterns and architecture")
        console.print()
        console.print("Options:")
        console.print("  --since Nd          Analyze commits from last N days (e.g., '30d')")
        console.print("  --limit N           Maximum commits to analyze (default: 50)")
        console.print("  --preview           Preview without capturing")
        console.print("  --auto              Auto-capture entries meeting threshold")
        console.print("  --min-confidence N  Minimum confidence (default: 0.5)")
        console.print("  --include PATTERN   Include file patterns (can repeat)")
        console.print("  --exclude PATTERN   Exclude file patterns (can repeat)")
        console.print("  --project PATH      Project path (default: current directory)")
        return 0

    total_extracted = 0
    total_captured = 0

    # Handle --commits
    if args.commits:
        console.print(f"[bold]Analyzing git commits in:[/bold] {project_path}")

        git_parser = GitParser(project_path)

        if not git_parser.is_git_repo():
            print_error(f"Not a git repository: {project_path}")
            return 1

        # Parse --since argument
        since = None
        if args.since:
            since_str = args.since.lower()
            if since_str.endswith("d"):
                days = int(since_str[:-1])
                since = datetime.now() - timedelta(days=days)
            elif since_str.endswith("h"):
                hours = int(since_str[:-1])
                since = datetime.now() - timedelta(hours=hours)
            else:
                print_error(f"Invalid --since format: {args.since}. Use '30d' or '24h'.")
                return 1

        # Get commits
        commits = git_parser.get_commits_with_diffs(
            since=since,
            limit=args.limit,
        )

        if not commits:
            console.print("No commits found to analyze.")
        else:
            console.print(f"Found [cyan]{len(commits)}[/cyan] commits to analyze.\n")

            # Filter out already processed commits
            unprocessed = []
            for commit in commits:
                if not km.is_commit_processed(commit.sha, str(project_path)):
                    unprocessed.append(commit)

            if not unprocessed:
                console.print("All commits have already been processed.")
            else:
                console.print(
                    f"[cyan]{len(unprocessed)}[/cyan] unprocessed commits "
                    f"(skipping {len(commits) - len(unprocessed)} already processed).\n"
                )

                git_extractor = GitExtractor()
                extractions = git_extractor.extract_from_commits(
                    unprocessed,
                    min_confidence=args.min_confidence,
                )

                for ext in extractions:
                    total_extracted += 1
                    confidence_text = format_score(ext.confidence)

                    console.print(f"[bold]{ext.title}[/bold]")
                    console.print("  Confidence: ", end="")
                    console.print(confidence_text)
                    console.print(f"  Type: [cyan]{ext.extraction_type}[/cyan]")
                    if ext.tags:
                        console.print(f"  Tags: [cyan]{', '.join(ext.tags)}[/cyan]")
                    console.print(f"  SHA: [dim]{ext.source_sha[:8]}[/dim]")

                    if args.preview:
                        # Show content preview
                        preview = ext.content[:200] if len(ext.content) > 200 else ext.content
                        console.print(f"  Preview: {preview}...")
                        console.print()
                        continue

                    if args.auto and ext.confidence >= args.min_confidence:
                        knowledge_id = km.capture(
                            title=ext.title,
                            description=ext.description,
                            content=ext.content,
                            tags=",".join(ext.tags),
                            project=project_name,
                            source="git",
                            confidence=ext.confidence,
                        )
                        total_captured += 1
                        console.print(f"  [green]Captured: {knowledge_id}[/green]")

                        # Mark commit as processed
                        km.mark_commit_processed(ext.source_sha, str(project_path), 1)

                    console.print()

                # Mark remaining commits as processed (even if no extraction)
                if args.auto and not args.preview:
                    processed_shas = {ext.source_sha for ext in extractions}
                    for commit in unprocessed:
                        if commit.sha not in processed_shas:
                            km.mark_commit_processed(commit.sha, str(project_path), 0)

    # Handle --patterns
    if args.patterns:
        console.print(f"\n[bold]Analyzing code patterns in:[/bold] {project_path}")

        default_includes = ["*.py", "*.js", "*.ts", "*.go", "*.rb"]
        include_patterns = args.include if args.include else default_includes
        exclude_patterns = args.exclude if args.exclude else None

        code_parser = CodeParser(
            base_path=project_path,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

        files = code_parser.scan_files()
        if not files:
            console.print("No source files found to analyze.")
        else:
            console.print(f"Found [cyan]{len(files)}[/cyan] source files.\n")

            # Filter out already processed files
            unprocessed_files = []
            for file_path in files:
                content_hash = code_parser.get_file_hash(file_path)
                if content_hash and not km.is_file_processed(
                    str(file_path), str(project_path), content_hash
                ):
                    unprocessed_files.append(file_path)

            if not unprocessed_files:
                console.print("All files have already been processed.")
            else:
                console.print(
                    f"[cyan]{len(unprocessed_files)}[/cyan] files to process "
                    f"(skipping {len(files) - len(unprocessed_files)} unchanged).\n"
                )

                parsed_files = code_parser.parse_files(unprocessed_files)
                code_extractor = CodeExtractor(code_parser)
                extractions = code_extractor.extract(parsed_files, project_name)

                for ext in extractions:
                    if ext.confidence < args.min_confidence:
                        continue

                    total_extracted += 1
                    confidence_text = format_score(ext.confidence)

                    console.print(f"[bold]{ext.title}[/bold]")
                    console.print("  Confidence: ", end="")
                    console.print(confidence_text)
                    console.print(f"  Type: [cyan]{ext.extraction_type}[/cyan]")
                    if ext.tags:
                        console.print(f"  Tags: [cyan]{', '.join(ext.tags)}[/cyan]")
                    if ext.source_files:
                        files_preview = ", ".join(Path(f).name for f in ext.source_files[:3])
                        if len(ext.source_files) > 3:
                            files_preview += f", +{len(ext.source_files) - 3} more"
                        console.print(f"  Files: [dim]{files_preview}[/dim]")

                    if args.preview:
                        # Show description
                        console.print(f"  Description: {ext.description[:100]}...")
                        console.print()
                        continue

                    if args.auto and ext.confidence >= args.min_confidence:
                        knowledge_id = km.capture(
                            title=ext.title,
                            description=ext.description,
                            content=ext.content,
                            tags=",".join(ext.tags),
                            project=project_name,
                            source="code",
                            confidence=ext.confidence,
                        )
                        total_captured += 1
                        console.print(f"  [green]Captured: {knowledge_id}[/green]")

                    console.print()

                # Mark files as processed
                if args.auto and not args.preview:
                    for parsed_file in parsed_files:
                        km.mark_file_processed(
                            parsed_file.path,
                            parsed_file.content_hash,
                            str(project_path),
                            0,  # We don't track per-file entry counts
                        )

    # Summary
    console.print()
    console.print(f"Total extractions found: [cyan]{total_extracted}[/cyan]")
    if args.auto:
        console.print(f"Total entries captured: [green]{total_captured}[/green]")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()

    # Enable argcomplete if available
    if ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)

    args = parser.parse_args(argv)

    # Handle --no-color flag and NO_COLOR environment variable
    if args.no_color or os.environ.get("NO_COLOR"):
        console.no_color = True

    if not args.command:
        parser.print_help()
        return 0

    # Handle completions command separately (doesn't need KnowledgeManager)
    if args.command == "completions":
        return cmd_completions(args)

    try:
        km = KnowledgeManager()
    except Exception as e:
        print_error(f"Initializing knowledge manager: {e}")
        return 1

    try:
        if args.command == "capture":
            return cmd_capture(args, km)
        elif args.command == "retrieve":
            return cmd_retrieve(args, km)
        elif args.command == "list":
            return cmd_list(args, km)
        elif args.command == "delete":
            return cmd_delete(args, km)
        elif args.command == "search":
            return cmd_search(args, km)
        elif args.command == "stats":
            return cmd_stats(args, km)
        elif args.command == "get":
            return cmd_get(args, km)
        elif args.command == "update":
            return cmd_update(args, km)
        elif args.command == "export":
            return cmd_export(args, km)
        elif args.command == "import":
            return cmd_import(args, km)
        elif args.command == "purge":
            return cmd_purge(args, km)
        elif args.command == "sync":
            return cmd_sync(args, km)
        elif args.command == "duplicates":
            return cmd_duplicates(args, km)
        elif args.command == "stale":
            return cmd_stale(args, km)
        elif args.command == "quality":
            return cmd_quality(args, km)
        elif args.command == "summarize":
            return cmd_summarize(args, km)
        elif args.command == "analyze":
            return cmd_analyze(args, km)
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130
    except Exception as e:
        print_error(str(e))
        return 1
    finally:
        km.close()


if __name__ == "__main__":
    sys.exit(main())
