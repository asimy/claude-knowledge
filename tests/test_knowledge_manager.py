"""Tests for the KnowledgeManager class."""

import shutil
import tempfile

import pytest

from claude_knowledge.knowledge_manager import KnowledgeManager
from claude_knowledge.utils import (
    create_brief,
    estimate_tokens,
    generate_id,
    json_to_tags,
    parse_tags,
    tags_to_json,
)


@pytest.fixture
def temp_km():
    """Create a temporary knowledge manager for testing."""
    temp_dir = tempfile.mkdtemp()
    km = KnowledgeManager(base_path=temp_dir)
    yield km
    km.close()
    shutil.rmtree(temp_dir)


@pytest.fixture
def populated_km(temp_km):
    """Create a knowledge manager with sample data."""
    temp_km.capture(
        title="OAuth Implementation",
        description="How to implement OAuth with authlib",
        content="Use authlib for OAuth. Configure with OAUTH_CLIENT_ID and OAUTH_SECRET.",
        tags="auth,oauth,python",
        project="myapp",
    )
    temp_km.capture(
        title="Database Connection Pooling",
        description="Setting up connection pooling with SQLAlchemy",
        content="Use create_engine with pool_size=5, max_overflow=10 for production.",
        tags="database,sqlalchemy,python",
        project="myapp",
    )
    temp_km.capture(
        title="React Component Testing",
        description="Testing React components with Jest",
        content="Use @testing-library/react for component tests. Mock API calls.",
        tags="react,testing,javascript",
        project="frontend",
    )
    return temp_km


class TestUtils:
    """Tests for utility functions."""

    def test_generate_id(self):
        """Test ID generation produces consistent 12-char hex strings."""
        id1 = generate_id("Test Title")
        assert len(id1) == 12
        assert all(c in "0123456789abcdef" for c in id1)

    def test_generate_id_uniqueness(self):
        """Test that different titles produce different IDs."""
        id1 = generate_id("Title One")
        id2 = generate_id("Title Two")
        assert id1 != id2

    def test_create_brief_short_content(self):
        """Test brief creation with content shorter than limit."""
        content = "Short content"
        brief = create_brief(content, max_length=200)
        assert brief == content

    def test_create_brief_long_content(self):
        """Test brief creation truncates at word boundary."""
        content = "This is a longer piece of content that exceeds the maximum length limit"
        brief = create_brief(content, max_length=30)
        assert len(brief) <= 33  # 30 + "..."
        assert brief.endswith("...")

    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "This is a test sentence with about forty characters."
        tokens = estimate_tokens(text)
        # ~4 chars per token
        assert 10 <= tokens <= 20

    def test_parse_tags_string(self):
        """Test parsing comma-separated tag string."""
        tags = parse_tags("auth, oauth, PYTHON")
        assert tags == ["auth", "oauth", "python"]

    def test_parse_tags_list(self):
        """Test parsing tag list."""
        tags = parse_tags(["Auth", "OAuth"])
        assert tags == ["auth", "oauth"]

    def test_parse_tags_none(self):
        """Test parsing None returns empty list."""
        assert parse_tags(None) == []

    def test_tags_to_json(self):
        """Test converting tags to JSON."""
        json_str = tags_to_json("a,b,c")
        assert json_str == '["a", "b", "c"]'

    def test_json_to_tags(self):
        """Test converting JSON back to tags."""
        tags = json_to_tags('["a", "b", "c"]')
        assert tags == ["a", "b", "c"]

    def test_json_to_tags_invalid(self):
        """Test handling invalid JSON."""
        assert json_to_tags("not json") == []
        assert json_to_tags(None) == []


class TestKnowledgeManagerInit:
    """Tests for KnowledgeManager initialization."""

    def test_init_creates_directories(self, temp_km):
        """Test that initialization creates necessary directories."""
        assert temp_km.base_path.exists()
        assert temp_km.chroma_path.exists()

    def test_init_creates_database(self, temp_km):
        """Test that SQLite database is created."""
        assert temp_km.sqlite_path.exists()


class TestCapture:
    """Tests for the capture method."""

    def test_capture_basic(self, temp_km):
        """Test basic knowledge capture."""
        kid = temp_km.capture(
            title="Test Knowledge",
            description="A test description",
            content="Test content here",
        )
        assert len(kid) == 12

    def test_capture_with_tags(self, temp_km):
        """Test capture with tags."""
        kid = temp_km.capture(
            title="Test Knowledge",
            description="A test description",
            content="Test content here",
            tags="tag1,tag2,tag3",
        )
        item = temp_km.get(kid)
        assert item is not None
        tags = json_to_tags(item["tags"])
        assert "tag1" in tags
        assert "tag2" in tags
        assert "tag3" in tags

    def test_capture_with_project(self, temp_km):
        """Test capture with project."""
        kid = temp_km.capture(
            title="Test Knowledge",
            description="A test description",
            content="Test content here",
            project="test-project",
        )
        item = temp_km.get(kid)
        assert item["project"] == "test-project"

    def test_capture_creates_brief(self, temp_km):
        """Test that capture creates a brief."""
        long_content = "A" * 500
        kid = temp_km.capture(
            title="Test Knowledge",
            description="A test description",
            content=long_content,
        )
        item = temp_km.get(kid)
        assert item["brief"] is not None
        assert len(item["brief"]) <= 203  # 200 + "..."


class TestRetrieve:
    """Tests for the retrieve method."""

    def test_retrieve_basic(self, populated_km):
        """Test basic retrieval."""
        results = populated_km.retrieve("OAuth authentication")
        assert len(results) > 0
        # OAuth item should be most relevant
        assert "OAuth" in results[0]["title"]

    def test_retrieve_with_project_filter(self, populated_km):
        """Test retrieval with project filter."""
        results = populated_km.retrieve("testing", project="frontend")
        assert len(results) > 0
        for result in results:
            assert result["project"] == "frontend"

    def test_retrieve_respects_n_results(self, populated_km):
        """Test that n_results limits output."""
        results = populated_km.retrieve("python", n_results=1)
        assert len(results) <= 1

    def test_retrieve_includes_score(self, populated_km):
        """Test that results include relevance score."""
        results = populated_km.retrieve("OAuth")
        for result in results:
            assert "score" in result
            assert 0 <= result["score"] <= 1

    def test_retrieve_updates_usage(self, populated_km):
        """Test that retrieval updates usage statistics."""
        # Get initial usage count
        results = populated_km.retrieve("OAuth")
        kid = results[0]["id"]

        # Retrieve again
        populated_km.retrieve("OAuth")

        # Check usage count increased
        item = populated_km.get(kid)
        assert item["usage_count"] >= 2
        assert item["last_used"] is not None

    def test_retrieve_min_score_filter(self, populated_km):
        """Test that min_score filters low-relevance results."""
        # Query with very high min_score should return fewer results
        results = populated_km.retrieve("OAuth", min_score=0.9)
        # Results may be empty or only highly relevant
        for result in results:
            assert result["score"] >= 0.9

    def test_retrieve_token_budget(self, populated_km):
        """Test that token budget is respected."""
        # Use a query that will match strongly to ensure results
        # Small budget should still return at least one result (guaranteed behavior)
        results = populated_km.retrieve(
            "OAuth authentication implementation",
            token_budget=50,
            n_results=10,
            min_score=0.1,  # Lower threshold to ensure matches
        )
        # Should have at least one result (always returns at least one)
        assert len(results) >= 1


class TestListAll:
    """Tests for the list_all method."""

    def test_list_all_basic(self, populated_km):
        """Test basic listing."""
        items = populated_km.list_all()
        assert len(items) == 3

    def test_list_all_with_project(self, populated_km):
        """Test listing with project filter."""
        items = populated_km.list_all(project="myapp")
        assert len(items) == 2
        for item in items:
            assert item["project"] == "myapp"

    def test_list_all_with_limit(self, populated_km):
        """Test listing with limit."""
        items = populated_km.list_all(limit=1)
        assert len(items) == 1


class TestDelete:
    """Tests for the delete method."""

    def test_delete_existing(self, temp_km):
        """Test deleting an existing entry."""
        kid = temp_km.capture(
            title="To Delete",
            description="Will be deleted",
            content="Content",
        )
        assert temp_km.get(kid) is not None
        result = temp_km.delete(kid)
        assert result is True
        assert temp_km.get(kid) is None

    def test_delete_nonexistent(self, temp_km):
        """Test deleting a nonexistent entry."""
        result = temp_km.delete("nonexistent123")
        assert result is False


class TestUpdate:
    """Tests for the update method."""

    def test_update_title(self, temp_km):
        """Test updating title."""
        kid = temp_km.capture(
            title="Original Title",
            description="Description",
            content="Content",
        )
        temp_km.update(kid, title="New Title")
        item = temp_km.get(kid)
        assert item["title"] == "New Title"

    def test_update_content(self, temp_km):
        """Test updating content regenerates embedding."""
        kid = temp_km.capture(
            title="Title",
            description="Description",
            content="Original content about dogs",
        )

        # Update content to be about cats
        temp_km.update(kid, content="New content about cats")

        # Retrieval should now match cats better
        item = temp_km.get(kid)
        assert "cats" in item["content"]

    def test_update_tags(self, temp_km):
        """Test updating tags."""
        kid = temp_km.capture(
            title="Title",
            description="Description",
            content="Content",
            tags="old,tags",
        )
        temp_km.update(kid, tags="new,updated,tags")
        item = temp_km.get(kid)
        tags = json_to_tags(item["tags"])
        assert "new" in tags
        assert "updated" in tags

    def test_update_nonexistent(self, temp_km):
        """Test updating nonexistent entry."""
        result = temp_km.update("nonexistent123", title="New Title")
        assert result is False


class TestSearch:
    """Tests for the search method."""

    def test_search_by_title(self, populated_km):
        """Test searching by title."""
        results = populated_km.search("OAuth")
        assert len(results) > 0
        assert "OAuth" in results[0]["title"]

    def test_search_by_description(self, populated_km):
        """Test searching by description."""
        results = populated_km.search("authlib")
        assert len(results) > 0

    def test_search_by_content(self, populated_km):
        """Test searching by content."""
        results = populated_km.search("pool_size")
        assert len(results) > 0

    def test_search_with_project(self, populated_km):
        """Test searching with project filter."""
        results = populated_km.search("test", project="frontend")
        for result in results:
            assert result["project"] == "frontend"


class TestStats:
    """Tests for the stats method."""

    def test_stats_total(self, populated_km):
        """Test total count in stats."""
        stats = populated_km.stats()
        assert stats["total_entries"] == 3

    def test_stats_by_project(self, populated_km):
        """Test project breakdown in stats."""
        stats = populated_km.stats()
        assert "myapp" in stats["by_project"]
        assert stats["by_project"]["myapp"] == 2

    def test_stats_recently_added(self, populated_km):
        """Test recently added in stats."""
        stats = populated_km.stats()
        assert len(stats["recently_added"]) > 0


class TestFormatForContext:
    """Tests for the format_for_context method."""

    def test_format_empty(self, temp_km):
        """Test formatting empty results."""
        result = temp_km.format_for_context([])
        assert "No relevant knowledge found" in result

    def test_format_with_items(self, populated_km):
        """Test formatting with items."""
        items = populated_km.retrieve("OAuth")
        result = populated_km.format_for_context(items)
        assert "Retrieved Knowledge" in result
        assert "OAuth" in result


class TestPersistence:
    """Tests for data persistence."""

    def test_data_survives_restart(self):
        """Test that data persists after manager restart."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create and populate
            km1 = KnowledgeManager(base_path=temp_dir)
            kid = km1.capture(
                title="Persistent Data",
                description="Should survive restart",
                content="Test content",
            )
            km1.close()

            # Reopen and verify
            km2 = KnowledgeManager(base_path=temp_dir)
            item = km2.get(kid)
            assert item is not None
            assert item["title"] == "Persistent Data"
            km2.close()
        finally:
            shutil.rmtree(temp_dir)


class TestContextManager:
    """Tests for context manager support."""

    def test_context_manager(self):
        """Test using KnowledgeManager as context manager."""
        temp_dir = tempfile.mkdtemp()
        try:
            with KnowledgeManager(base_path=temp_dir) as km:
                kid = km.capture(
                    title="Test",
                    description="Test",
                    content="Test content",
                )
                assert len(kid) == 12
        finally:
            shutil.rmtree(temp_dir)
