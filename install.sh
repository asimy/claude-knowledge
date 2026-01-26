#!/bin/bash
# Install the knowledge management system using uv

set -e

echo "Installing Claude Knowledge Management System..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "Please restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
    echo "Then run this script again."
    exit 1
fi

echo "Found uv: $(uv --version)"

# Create base data directory
mkdir -p ~/.claude_knowledge

# Sync dependencies and install package
echo "Installing dependencies..."
uv sync

# Install in editable mode
echo "Installing claude-knowledge package..."
uv pip install -e .

# Test installation
echo ""
echo "Testing installation..."
if uv run claude-kb stats 2>/dev/null; then
    echo ""
    echo "Installation complete!"
else
    echo "Installation complete! (First run will download embedding model)"
fi

echo ""
echo "Usage:"
echo "  uv run claude-kb --help"
echo "  uv run claude-kb capture --help"
echo "  uv run claude-kb retrieve --query 'your query'"
echo ""
echo "To run without 'uv run' prefix, activate the venv:"
echo "  source .venv/bin/activate"
echo ""
echo "To integrate with Claude Code skills:"
echo "  mkdir -p ~/.claude/skills/knowledge"
echo "  cp skill/SKILL.md ~/.claude/skills/knowledge/SKILL.md"
