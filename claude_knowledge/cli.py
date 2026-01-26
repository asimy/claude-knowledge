"""Command-line interface for the knowledge management system."""

import argparse
import json
import sys
from typing import NoReturn

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


def main(argv: list[str] | None = None) -> int | NoReturn:
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
