# Claude Knowledge Management System

A persistent knowledge base for Claude Code that remembers solutions, patterns, and decisions across sessions.

## Features

- **Semantic Search**: Find relevant knowledge using natural language queries
- **Project Isolation**: Keep knowledge organized by project
- **Smart Retrieval**: Automatically ranks and filters by relevance
- **Token-Aware**: Respects context window limits
- **Offline**: Works completely locally with no API calls
- **Claude Code Integration**: Seamless integration via skills

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Quick Install with uv

```bash
git clone <repository-url>
cd claude-knowledge
chmod +x install.sh
./install.sh
```

### Manual Install with uv

```bash
uv sync --extra dev
```

### Manual Install with pip

```bash
pip install -e .
```

### First Run

The first time you run the system, it will download the sentence-transformers model (~80MB). This only happens once.

## Usage

With uv, prefix commands with `uv run`, or activate the virtual environment first:

```bash
source .venv/bin/activate
```

### Capture Knowledge

```bash
claude-kb capture \
  --title "OAuth Implementation" \
  --description "How we handle OAuth in our Python backend" \
  --content "Use authlib with rotating tokens. Store in Redis. Config: OAUTH_TIMEOUT=3600" \
  --tags "auth,oauth,python" \
  --project "myapp"
```

### Retrieve Knowledge

```bash
# Basic retrieval
claude-kb retrieve --query "authentication" --project "myapp"

# With formatting
claude-kb retrieve --query "auth setup" --format json

# Filter by minimum relevance score (0.0-1.0)
claude-kb retrieve --query "oauth" --min-score 0.5
```

### List Knowledge

```bash
# List all
claude-kb list

# Filter by project
claude-kb list --project "myapp"

# Limit results
claude-kb list --limit 10
```

### View Statistics

```bash
claude-kb stats
```

### Delete Knowledge

```bash
claude-kb delete <knowledge-id>
```

### Get Specific Entry

```bash
claude-kb get <knowledge-id>
```

### Update Entry

```bash
claude-kb update <knowledge-id> --title "New Title" --tags "new,tags"
```

### Text Search

```bash
claude-kb search "oauth" --project "myapp"
```

### Export Knowledge

```bash
# Export all knowledge to a file
claude-kb export backup.json

# Export specific project
claude-kb export myapp-backup.json --project "myapp"

# Export to stdout
claude-kb export -
```

### Import Knowledge

```bash
# Import from file
claude-kb import backup.json

# Import from stdin
cat backup.json | claude-kb import -

# Error on duplicates instead of skipping
claude-kb import backup.json --no-skip-duplicates
```

### Purge Knowledge

```bash
# Delete all entries (with confirmation)
claude-kb purge

# Delete entries for specific project
claude-kb purge --project "myapp"

# Skip confirmation prompt
claude-kb purge --force
```

## Claude Code Integration

### Installing the Skill

Copy the skill file to your Claude Code skills directory:

**For all projects (personal skills):**
```bash
mkdir -p ~/.claude/skills/knowledge
cp skill/SKILL.md ~/.claude/skills/knowledge/SKILL.md
```

**For a specific project only:**
```bash
mkdir -p .claude/skills/knowledge
cp skill/SKILL.md .claude/skills/knowledge/SKILL.md
```

Claude Code will automatically discover and use the skill.

### Usage

Before starting a task, Claude Code can check for relevant knowledge:
```bash
claude-kb retrieve --query "describe your task"
```

## Best Practices

### What to Capture

Good candidates:
- Solutions to tricky problems
- API integration patterns
- Configuration that works
- Debugging solutions
- Architecture decisions
- Common pitfalls

Avoid capturing:
- Sensitive data (passwords, keys)
- Temporary solutions
- Standard library usage
- Overly generic information

### Writing Good Descriptions

Good description:
> "OAuth implementation using authlib with Redis token storage"

Bad description:
> "Auth stuff"

### Tagging Strategy

Use consistent, specific tags:
- Language: `python`, `javascript`, `rust`
- Domain: `auth`, `api`, `database`, `frontend`
- Framework: `flask`, `react`, `django`
- Concept: `caching`, `retry-logic`, `rate-limiting`

## Troubleshooting

### Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'chromadb'`
**Solution**: Run `uv sync` or `pip install -e .`

**Problem**: Slow first run
**Solution**: Expected - downloading embedding model (~80MB)

### Usage Issues

**Problem**: No results for query
**Solution**:
- Check spelling
- Try broader search terms
- Verify project name is correct
- Use `claude-kb list` to see what's stored

**Problem**: Wrong results returned
**Solution**:
- Make knowledge descriptions more specific
- Add more relevant tags
- Capture more examples

## Architecture

```
~/.claude_knowledge/
├── chroma_db/           # Vector embeddings
└── knowledge.db         # SQLite metadata
```

Components:
- **KnowledgeManager**: Core logic
- **ChromaDB**: Semantic search via vector embeddings
- **SQLite**: Structured metadata storage
- **sentence-transformers**: Local embedding generation

## Development

### Setup

```bash
uv sync --extra dev
```

### Running Tests

```bash
uv run pytest tests/
```

### Linting and Formatting

```bash
uv run ruff check claude_knowledge/ tests/
uv run ruff format claude_knowledge/ tests/
```

### Type Checking

```bash
uv run ty check claude_knowledge/
```

### Code Structure

- `knowledge_manager.py`: Core functionality
- `cli.py`: Command-line interface
- `utils.py`: Helper functions

## License

MIT
