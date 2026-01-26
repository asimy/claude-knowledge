"""Utility functions for the knowledge management system."""

import hashlib
import json
import re
from datetime import datetime
from typing import Any


def generate_id(title: str, timestamp: datetime | None = None) -> str:
    """Generate a unique ID from title and timestamp.

    Args:
        title: The knowledge entry title.
        timestamp: Optional timestamp (defaults to current time).

    Returns:
        A 12-character hex ID.
    """
    if timestamp is None:
        timestamp = datetime.now()
    content = f"{title}{timestamp.isoformat()}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def create_brief(content: str, max_length: int = 200) -> str:
    """Create a brief version of content.

    Args:
        content: The full content text.
        max_length: Maximum length of the brief.

    Returns:
        Truncated content with ellipsis if needed.
    """
    if len(content) <= max_length:
        return content
    # Try to break at a word boundary
    truncated = content[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]
    return truncated.rstrip() + "..."


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple heuristic of ~4 characters per token for English text.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return len(text) // 4


def parse_tags(tags: str | list[str] | None) -> list[str]:
    """Parse tags from various input formats.

    Args:
        tags: Tags as comma-separated string, list, or None.

    Returns:
        List of normalized tag strings.
    """
    if tags is None:
        return []
    if isinstance(tags, list):
        return [t.strip().lower() for t in tags if t.strip()]
    return [t.strip().lower() for t in tags.split(",") if t.strip()]


def tags_to_json(tags: str | list[str] | None) -> str:
    """Convert tags to JSON string for storage.

    Args:
        tags: Tags in various formats.

    Returns:
        JSON array string.
    """
    return json.dumps(parse_tags(tags))


def json_to_tags(json_str: str | None) -> list[str]:
    """Parse JSON string back to tag list.

    Args:
        json_str: JSON array string or None.

    Returns:
        List of tags.
    """
    if not json_str:
        return []
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return []


def context_to_json(context: list[str] | None) -> str:
    """Convert context list to JSON string.

    Args:
        context: List of context strings or None.

    Returns:
        JSON array string.
    """
    if context is None:
        return json.dumps([])
    return json.dumps(context)


def json_to_context(json_str: str | None) -> list[str]:
    """Parse JSON string back to context list.

    Args:
        json_str: JSON array string or None.

    Returns:
        List of context strings.
    """
    if not json_str:
        return []
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return []


def format_knowledge_item(
    item: dict[str, Any],
    include_content: bool = True,
    include_score: bool = True,
) -> str:
    """Format a knowledge item as markdown.

    Args:
        item: Knowledge item dictionary.
        include_content: Whether to include full content.
        include_score: Whether to include relevance score.

    Returns:
        Formatted markdown string.
    """
    lines = []

    title = item.get("title", "Untitled")
    lines.append(f"### {title}")

    if include_score and "score" in item:
        score = item["score"]
        lines.append(f"*Relevance: {score:.0%}*")

    if item.get("description"):
        lines.append(f"\n{item['description']}")

    tags = json_to_tags(item.get("tags"))
    if tags:
        tag_str = ", ".join(f"`{t}`" for t in tags)
        lines.append(f"\n**Tags:** {tag_str}")

    if include_content:
        content = item.get("content", item.get("brief", ""))
        if content:
            lines.append(f"\n```\n{content}\n```")
    elif item.get("brief"):
        lines.append(f"\n> {item['brief']}")

    lines.append("")
    return "\n".join(lines)


def sanitize_for_embedding(text: str) -> str:
    """Clean text for embedding generation.

    Args:
        text: Raw text input.

    Returns:
        Cleaned text suitable for embedding.
    """
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters that might confuse embeddings
    text = re.sub(r"[^\w\s.,!?;:\-()]", " ", text)
    return text.strip()
