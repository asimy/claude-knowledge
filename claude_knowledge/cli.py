"""Command-line interface for the knowledge management system."""

import argparse
import json
import sys

from claude_knowledge.knowledge_manager import KnowledgeManager
from claude_knowledge.utils import json_to_tags


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="claude-kb",
        description="Knowledge management system for Claude Code",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # capture command
    capture_parser = subparsers.add_parser(
        "capture",
        help="Capture new knowledge",
    )
    capture_parser.add_argument(
        "--title",
        required=True,
        help="Short title for the knowledge entry",
    )
    capture_parser.add_argument(
        "--description",
        required=True,
        help="Description of what this knowledge covers",
    )
    capture_parser.add_argument(
        "--content",
        required=True,
        help="Full content/details of the knowledge",
    )
    capture_parser.add_argument(
        "--tags",
        help="Comma-separated tags (e.g., 'auth,oauth,python')",
    )
    capture_parser.add_argument(
        "--context",
        help="Comma-separated context (e.g., 'backend,python')",
    )
    capture_parser.add_argument(
        "--project",
        help="Project name/identifier",
    )

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
    retrieve_parser.add_argument(
        "--project",
        help="Filter by project",
    )
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

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List knowledge entries",
    )
    list_parser.add_argument(
        "--project",
        help="Filter by project",
    )
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

    # delete command
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a knowledge entry",
    )
    delete_parser.add_argument(
        "id",
        help="Knowledge entry ID to delete",
    )

    # search command
    search_parser = subparsers.add_parser(
        "search",
        help="Text search in knowledge",
    )
    search_parser.add_argument(
        "text",
        help="Search text",
    )
    search_parser.add_argument(
        "--project",
        help="Filter by project",
    )
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
    get_parser.add_argument(
        "id",
        help="Knowledge entry ID",
    )
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
    update_parser.add_argument(
        "id",
        help="Knowledge entry ID to update",
    )
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
    update_parser.add_argument(
        "--project",
        help="New project",
    )

    # export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export knowledge to JSON file",
    )
    export_parser.add_argument(
        "file",
        help="Output file path (use - for stdout)",
    )
    export_parser.add_argument(
        "--project",
        help="Only export entries for this project",
    )

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
    purge_parser.add_argument(
        "--project",
        help="Only purge entries for this project",
    )
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
    sync_parser.add_argument(
        "path",
        nargs="?",
        help="Path to sync directory (uses saved path if omitted)",
    )
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
    sync_parser.add_argument(
        "--project",
        help="Only sync entries for this project",
    )

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
    duplicates_parser.add_argument(
        "--project",
        help="Only check entries for this project",
    )
    duplicates_parser.add_argument(
        "--merge",
        nargs=2,
        metavar=("TARGET_ID", "SOURCE_ID"),
        help="Merge SOURCE_ID into TARGET_ID",
    )

    return parser


def cmd_capture(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the capture command."""
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

    print(f"Knowledge captured with ID: {knowledge_id}")
    return 0


def cmd_retrieve(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the retrieve command."""
    items = km.retrieve(
        query=args.query,
        n_results=args.limit,
        token_budget=args.budget,
        project=args.project,
        min_score=args.min_score,
    )

    if not items:
        print("No relevant knowledge found.")
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
            score = item.get("score", 0)
            tags = json_to_tags(item.get("tags"))
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            print(f"[{score:.0%}] {item['title']}{tag_str}")
            print(f"  ID: {item['id']}")
            print(f"  {item['description']}")
            print()

    return 0


def cmd_list(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the list command."""
    items = km.list_all(project=args.project, limit=args.limit)

    if not items:
        print("No knowledge entries found.")
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
            tags = json_to_tags(item.get("tags"))
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            project_str = f" ({item['project']})" if item.get("project") else ""
            usage = item.get("usage_count", 0)
            print(f"{item['id']}: {item['title']}{tag_str}{project_str} (used {usage}x)")

    return 0


def cmd_delete(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the delete command."""
    if km.delete(args.id):
        print(f"Deleted knowledge entry: {args.id}")
        return 0
    else:
        print(f"Error: Knowledge entry not found: {args.id}")
        return 1


def cmd_search(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the search command."""
    items = km.search(args.text, project=args.project, limit=args.limit)

    if not items:
        print("No matching entries found.")
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
            tags = json_to_tags(item.get("tags"))
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            project_str = f" ({item['project']})" if item.get("project") else ""
            print(f"{item['id']}: {item['title']}{tag_str}{project_str}")
            print(f"  {item['description']}")
            print()

    return 0


def cmd_stats(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the stats command."""
    stats = km.stats()

    print("Knowledge Base Statistics")
    print("=" * 40)
    print(f"\nTotal entries: {stats['total_entries']}")

    if stats["by_project"]:
        print("\nBy project:")
        for project, count in stats["by_project"].items():
            print(f"  {project}: {count}")

    if stats["most_used"]:
        print("\nMost used:")
        for item in stats["most_used"]:
            print(f"  {item['title']} ({item['usage_count']}x)")

    if stats["recently_added"]:
        print("\nRecently added:")
        for item in stats["recently_added"]:
            print(f"  {item['title']} ({item['created'][:10]})")

    if stats["recently_used"]:
        print("\nRecently used:")
        for item in stats["recently_used"]:
            print(f"  {item['title']} ({item['last_used'][:10] if item['last_used'] else 'never'})")

    return 0


def cmd_get(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the get command."""
    item = km.get(args.id)

    if not item:
        print(f"Error: Knowledge entry not found: {args.id}")
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
        print(f"Title: {item['title']}")
        print(f"ID: {item['id']}")
        print(f"Project: {item.get('project') or '(none)'}")
        print(f"Tags: {', '.join(tags) if tags else '(none)'}")
        print(f"Usage count: {item.get('usage_count', 0)}")
        print(f"Created: {item.get('created')}")
        print(f"Last used: {item.get('last_used') or 'never'}")
        print()
        print("Description:")
        print(f"  {item['description']}")
        print()
        print("Content:")
        print(item["content"])

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
        print("Error: No updates specified")
        return 1

    if km.update(args.id, **updates):
        print(f"Updated knowledge entry: {args.id}")
        return 0
    else:
        print(f"Error: Knowledge entry not found: {args.id}")
        return 1


def cmd_export(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the export command."""
    entries = km.export_all(project=args.project)

    if not entries:
        print("No entries to export.")
        return 0

    output = json.dumps(entries, indent=2, default=str)

    if args.file == "-":
        print(output)
    else:
        with open(args.file, "w") as f:
            f.write(output)
        print(f"Exported {len(entries)} entries to {args.file}")

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
        print(f"Error: Invalid JSON: {e}")
        return 1

    if not isinstance(entries, list):
        print("Error: JSON must be a list of entries")
        return 1

    result = km.import_data(entries, skip_duplicates=not args.no_skip_duplicates)

    print(f"Imported: {result['imported']}")
    if result["skipped"]:
        print(f"Skipped (duplicates): {result['skipped']}")
    if result["errors"]:
        print(f"Errors: {result['errors']}")

    return 0 if result["errors"] == 0 else 1


def cmd_purge(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the purge command."""
    if not args.force:
        if args.project:
            prompt = f"Delete all entries for project '{args.project}'? [y/N] "
        else:
            prompt = "Delete ALL knowledge entries? [y/N] "

        response = input(prompt).strip().lower()
        if response != "y":
            print("Aborted.")
            return 0

    count = km.purge(project=args.project)
    print(f"Deleted {count} entries.")
    return 0


def cmd_sync(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the sync command."""
    sync_path = args.path

    # Handle --init flag
    if args.init:
        if not sync_path:
            print("Error: Path required for --init")
            return 1
        km.init_sync_dir(sync_path)
        km.set_sync_path(sync_path)
        print(f"Initialized sync directory: {sync_path}")
        return 0

    # Handle --status flag
    if args.status:
        try:
            status = km.sync_status(sync_path=sync_path, project=args.project)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        if "error" in status:
            print(f"Error: {status['error']}")
            return 1

        print(f"Sync status for: {status['sync_path']}")
        print("=" * 40)
        print(f"To push:         {len(status['to_push'])}")
        print(f"To pull:         {len(status['to_pull'])}")
        print(f"Conflicts:       {len(status['conflicts'])}")
        print(f"Delete local:    {len(status['to_delete_local'])}")
        print(f"Delete remote:   {len(status['to_delete_remote'])}")

        if status["to_push"]:
            print("\nEntries to push:")
            for entry_id in status["to_push"][:5]:
                entry = km.get(entry_id)
                if entry:
                    print(f"  {entry_id}: {entry['title']}")
            if len(status["to_push"]) > 5:
                print(f"  ... and {len(status['to_push']) - 5} more")

        if status["to_pull"]:
            print("\nEntries to pull:")
            for entry_id in status["to_pull"][:5]:
                print(f"  {entry_id}")
            if len(status["to_pull"]) > 5:
                print(f"  ... and {len(status['to_pull']) - 5} more")

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
        print(f"Error: {e}")
        return 1

    if result.errors:
        for error in result.errors:
            print(f"Error: {error}")
        return 1

    prefix = "[DRY RUN] " if args.dry_run else ""

    print(f"{prefix}Sync complete:")
    print(f"  Pushed:           {result.pushed}")
    print(f"  Pulled:           {result.pulled}")
    print(f"  Deletions pushed: {result.deletions_pushed}")
    print(f"  Deletions pulled: {result.deletions_pulled}")

    if result.conflicts:
        print(f"\nConflicts ({len(result.conflicts)}):")
        for conflict in result.conflicts:
            resolution = conflict.get("resolution", "unknown")
            print(f"  {conflict['id']}: {conflict['title']} -> {resolution}")

    saved_path = km.get_sync_path()
    if saved_path and not args.dry_run:
        print(f"\nSync path saved: {saved_path}")

    return 0


def cmd_duplicates(args: argparse.Namespace, km: KnowledgeManager) -> int:
    """Handle the duplicates command."""
    # Handle merge operation
    if args.merge:
        target_id, source_id = args.merge
        if km.merge_entries(target_id, source_id):
            print(f"Merged {source_id} into {target_id}")
            return 0
        else:
            print("Error: One or both entries not found")
            return 1

    # Find duplicates
    groups = km.find_duplicates(threshold=args.threshold, project=args.project)

    if not groups:
        print("No potential duplicates found.")
        return 0

    print(f"Found {len(groups)} potential duplicate group(s):\n")

    for i, group in enumerate(groups, 1):
        print(f"Group {i}:")
        for entry in group:
            sim_str = f" ({entry['similarity']:.0%})" if entry["similarity"] < 1.0 else ""
            print(f"  {entry['id']}: {entry['title']}{sim_str}")
        print()

    print("To merge duplicates, use:")
    print("  claude-kb duplicates --merge <target_id> <source_id>")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    try:
        km = KnowledgeManager()
    except Exception as e:
        print(f"Error initializing knowledge manager: {e}")
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
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        km.close()


if __name__ == "__main__":
    sys.exit(main())
