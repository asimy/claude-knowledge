"""Utility functions for the knowledge management system."""

import hashlib
import json
import re
import socket
from datetime import datetime, timedelta
from typing import Any


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein (edit) distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        The minimum number of single-character edits needed to transform s1 into s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def fuzzy_match_tag(query_tag: str, item_tags: list[str], max_distance: int = 2) -> bool:
    """Check if a query tag fuzzy-matches any item tag.

    Args:
        query_tag: The tag to search for.
        item_tags: List of tags to match against.
        max_distance: Maximum edit distance for a match (default: 2).

    Returns:
        True if query_tag matches any item tag within the edit distance.
    """
    query_lower = query_tag.lower()
    for tag in item_tags:
        tag_lower = tag.lower()
        # Exact match
        if query_lower == tag_lower:
            return True
        # Fuzzy match with edit distance
        if levenshtein_distance(query_lower, tag_lower) <= max_distance:
            return True
    return False


def fuzzy_match_tags(query_tags: list[str], item_tags: list[str], max_distance: int = 2) -> bool:
    """Check if all query tags fuzzy-match item tags (AND logic).

    Args:
        query_tags: List of tags to search for.
        item_tags: List of tags to match against.
        max_distance: Maximum edit distance for a match (default: 2).

    Returns:
        True if all query tags match within the edit distance.
    """
    return all(fuzzy_match_tag(qt, item_tags, max_distance) for qt in query_tags)


def parse_relative_date(date_str: str) -> str | None:
    """Parse a date string that can be ISO format or relative (e.g., '7d', '2w', '1m').

    Args:
        date_str: Date string in ISO format or relative format (7d, 2w, 1m, 1y).

    Returns:
        ISO format date string, or None if parsing fails.
    """
    if not date_str:
        return None

    # Try ISO format first
    try:
        datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return date_str
    except ValueError:
        pass

    # Try relative format (7d, 2w, 1m, 1y)
    match = re.match(r"^(\d+)([dwmy])$", date_str.lower())
    if match:
        amount = int(match.group(1))
        unit = match.group(2)

        now = datetime.now()
        if unit == "d":
            delta = timedelta(days=amount)
        elif unit == "w":
            delta = timedelta(weeks=amount)
        elif unit == "m":
            delta = timedelta(days=amount * 30)  # Approximate month
        elif unit == "y":
            delta = timedelta(days=amount * 365)  # Approximate year
        else:
            return None

        result = now - delta
        return result.isoformat()

    return None


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


def escape_like_pattern(text: str) -> str:
    """Escape special characters for SQLite LIKE patterns.

    Args:
        text: Raw search text.

    Returns:
        Text with %, _, and \\ escaped for use in LIKE queries.
    """
    # Escape backslash first, then the LIKE wildcards
    text = text.replace("\\", "\\\\")
    text = text.replace("%", "\\%")
    text = text.replace("_", "\\_")
    return text


def compute_content_hash(entry: dict[str, Any]) -> str:
    """Compute SHA-256 hash of entry content fields for sync change detection.

    Args:
        entry: Knowledge entry dictionary.

    Returns:
        SHA-256 hash as hex string.
    """
    # Hash the fields that represent the entry's content
    title = entry.get("title", "")
    description = entry.get("description", "")
    body = entry.get("content", "")
    content = f"{title}\n{description}\n{body}"
    return hashlib.sha256(content.encode()).hexdigest()


def get_machine_id() -> str:
    """Get a machine identifier for sync tracking.

    Returns:
        Hostname of the current machine.
    """
    return socket.gethostname()
