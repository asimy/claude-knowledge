# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A persistent knowledge base for Claude Code that captures and retrieves information from previous sessions using semantic search. Works entirely offline with local storage.

## Commands

```bash
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/
uv run pytest tests/test_knowledge_manager.py::TestCapture::test_capture_basic  # single test

# Lint and format
uv run ruff check claude_knowledge/ tests/
uv run ruff check claude_knowledge/ tests/ --fix  # auto-fix
uv run ruff format claude_knowledge/ tests/

# Run CLI
uv run claude-kb --help
uv run claude-kb stats
uv run claude-kb list
```

## Architecture

**Dual-storage pattern**: SQLite stores structured metadata (title, tags, timestamps, usage counts); ChromaDB stores vector embeddings for semantic search. Both use the same knowledge ID.

**Data flow**:
1. `capture`: Generate embedding via sentence-transformers → store in ChromaDB + SQLite
2. `retrieve`: Query embedding → ChromaDB similarity search → fetch metadata from SQLite → filter/rank → update usage stats

**Key classes**:
- `KnowledgeManager` (`knowledge_manager.py`): Core logic, manages both databases, lazy-loads embedding model
- `cli.py`: Argparse-based CLI with subcommands, delegates to KnowledgeManager

**Storage location**: `~/.claude_knowledge/` (chroma_db/, knowledge.db)

**Embedding model**: `all-MiniLM-L6-v2` (~80MB, downloaded on first use)
