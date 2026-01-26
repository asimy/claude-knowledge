"""Shell completion helpers for claude-kb CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace


def get_project_completer():
    """Return a completer function for project names.

    Returns a closure that queries the database for distinct project names.
    The completer is lazy-loaded to avoid importing KnowledgeManager at module load.
    """

    def completer(prefix: str, parsed_args: Namespace, **kwargs) -> list[str]:
        try:
            from claude_knowledge.knowledge_manager import KnowledgeManager

            km = KnowledgeManager()
            cursor = km.conn.cursor()
            cursor.execute(
                "SELECT DISTINCT project FROM knowledge WHERE project IS NOT NULL AND project != ''"
            )
            projects = [row[0] for row in cursor.fetchall()]
            km.close()
            return [p for p in projects if p.startswith(prefix)]
        except Exception:
            return []

    return completer


def get_entry_id_completer():
    """Return a completer function for knowledge entry IDs.

    Returns recent entry IDs (limited to avoid slow completions).
    """

    def completer(prefix: str, parsed_args: Namespace, **kwargs) -> list[str]:
        try:
            from claude_knowledge.knowledge_manager import KnowledgeManager

            km = KnowledgeManager()
            cursor = km.conn.cursor()
            # Limit to 50 most recently used/created entries
            cursor.execute(
                """
                SELECT id FROM knowledge
                ORDER BY COALESCE(last_used, created) DESC
                LIMIT 50
            """
            )
            results = [row[0] for row in cursor.fetchall() if row[0].startswith(prefix)]
            km.close()
            return results
        except Exception:
            return []

    return completer


def get_sync_path_completer():
    """Return a completer for sync paths.

    Returns the saved sync path if configured.
    """

    def completer(prefix: str, parsed_args: Namespace, **kwargs) -> list[str]:
        results = []

        # Add saved sync path if configured
        try:
            from claude_knowledge.knowledge_manager import KnowledgeManager

            km = KnowledgeManager()
            saved_path = km.get_sync_path()
            km.close()
            if saved_path and str(saved_path).startswith(prefix):
                results.append(str(saved_path))
        except Exception:
            pass

        return results

    return completer
