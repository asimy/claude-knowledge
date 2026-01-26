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
uv run claude-kb sync /path/to/sync/dir  # sync across devices

# Interactive capture
uv run claude-kb capture -i

# Quality and maintenance
uv run claude-kb quality              # score entries
uv run claude-kb stale --days 90      # find unused entries
uv run claude-kb duplicates           # find similar entries

# Session summarization
uv run claude-kb summarize --list                  # list available sessions
uv run claude-kb summarize --session <id>          # extract from session
uv run claude-kb summarize --session <id> --preview  # preview only
uv run claude-kb summarize --since 7d --auto       # auto-capture recent

# Shell completions
uv run claude-kb completions bash     # setup instructions
```

## Architecture

**Dual-storage pattern**: SQLite stores structured metadata (title, tags, timestamps, usage counts); ChromaDB stores vector embeddings for semantic search. Both use the same knowledge ID.

**Data flow**:
1. `capture`: Generate embedding via sentence-transformers → store in ChromaDB + SQLite
2. `retrieve`: Query embedding → ChromaDB similarity search → fetch metadata from SQLite → filter/rank → update usage stats
3. `summarize`: Parse session JSONL → extract problem-solution pairs → calculate confidence → optionally capture

**Key modules**:
- `KnowledgeManager` (`knowledge_manager.py`): Core logic, manages both databases, lazy-loads embedding model
- `cli.py`: Argparse-based CLI with subcommands, delegates to KnowledgeManager
- `output.py`: Rich terminal formatting with syntax highlighting
- `interactive.py`: Interactive capture mode with editor integration
- `completions.py`: Shell completion support
- `session_parser.py`: Parse Claude Code session transcripts
- `session_extractor.py`: Extract knowledge from parsed sessions

**Storage location**: `~/.claude_knowledge/` (chroma_db/, knowledge.db, config.json)

**Sync**: Bidirectional sync to shared directory (git, Dropbox, iCloud). One JSON file per entry with tombstone-based deletion tracking.

**Embedding model**: `all-MiniLM-L6-v2` (~80MB, downloaded on first use)

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for planned features and release history.
