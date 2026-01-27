"""Tests for the KnowledgeManager class."""

import shutil
import tempfile

import pytest

from claude_knowledge.knowledge_manager import EmbeddingError, KnowledgeManager
from claude_knowledge.utils import (
    create_brief,
    escape_like_pattern,
    estimate_tokens,
    generate_id,
    json_to_tags,
    parse_tags,
    tags_to_json,
)

# temp_km and populated_km fixtures are provided by conftest.py


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

    def test_escape_like_pattern(self):
        """Test escaping LIKE pattern wildcards."""
        assert escape_like_pattern("normal text") == "normal text"
        assert escape_like_pattern("100%") == "100\\%"
        assert escape_like_pattern("test_value") == "test\\_value"
        assert escape_like_pattern("back\\slash") == "back\\\\slash"
        assert escape_like_pattern("%_\\") == "\\%\\_\\\\"


class TestKnowledgeManagerInit:
    """Tests for KnowledgeManager initialization."""

    def test_init_creates_directories(self, temp_km):
        """Test that initialization creates necessary directories."""
        assert temp_km.base_path.exists()
        assert temp_km.chroma_path.exists()

    def test_init_creates_database(self, temp_km):
        """Test that SQLite database is created."""
        assert temp_km.sqlite_path.exists()


class TestEmbeddingError:
    """Tests for embedding error handling."""

    def test_embedding_error_is_raised_for_empty_text(self, temp_km):
        """Test that EmbeddingError is raised for empty text."""
        with pytest.raises(EmbeddingError, match="empty text"):
            temp_km._generate_embedding("")

    def test_embedding_error_is_raised_for_whitespace_only(self, temp_km):
        """Test that EmbeddingError is raised for whitespace-only text."""
        with pytest.raises(EmbeddingError, match="empty text"):
            temp_km._generate_embedding("   \n\t   ")

    def test_embedding_succeeds_for_valid_text(self, temp_km):
        """Test that embedding generation succeeds for valid text."""
        embedding = temp_km._generate_embedding("This is valid text")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)


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

    def test_capture_rejects_title_too_long(self, temp_km):
        """Test that capture rejects title exceeding maximum length."""
        long_title = "A" * (temp_km.MAX_TITLE_LENGTH + 1)
        with pytest.raises(ValueError, match="title exceeds maximum length"):
            temp_km.capture(
                title=long_title,
                description="A test description",
                content="Test content",
            )

    def test_capture_rejects_description_too_long(self, temp_km):
        """Test that capture rejects description exceeding maximum length."""
        long_desc = "A" * (temp_km.MAX_DESCRIPTION_LENGTH + 1)
        with pytest.raises(ValueError, match="description exceeds maximum length"):
            temp_km.capture(
                title="Test Title",
                description=long_desc,
                content="Test content",
            )

    def test_capture_rejects_content_too_long(self, temp_km):
        """Test that capture rejects content exceeding maximum length."""
        long_content = "A" * (temp_km.MAX_CONTENT_LENGTH + 1)
        with pytest.raises(ValueError, match="content exceeds maximum length"):
            temp_km.capture(
                title="Test Title",
                description="A test description",
                content=long_content,
            )

    def test_capture_accepts_content_at_limit(self, temp_km):
        """Test that capture accepts content at exactly the maximum length."""
        max_content = "A" * temp_km.MAX_CONTENT_LENGTH
        kid = temp_km.capture(
            title="Test Title",
            description="A test description",
            content=max_content,
        )
        assert len(kid) == 12


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

    def test_search_with_like_wildcards(self, temp_km):
        """Test that LIKE wildcards in search text are escaped."""
        # Create entries with specific content
        temp_km.capture(
            title="100% Complete",
            description="A fully complete item",
            content="This is 100% done",
        )
        temp_km.capture(
            title="Other Entry",
            description="Something else entirely",
            content="No percentage here",
        )

        # Search for literal "100%" - should only match the first entry
        results = temp_km.search("100%")
        assert len(results) == 1
        assert "100%" in results[0]["title"]

        # Search for literal "%" - should only match entries containing %
        results = temp_km.search("%")
        assert len(results) == 1

    def test_search_with_underscore(self, temp_km):
        """Test that underscore wildcard is escaped in search."""
        temp_km.capture(
            title="test_value",
            description="Has underscore",
            content="Content with test_value",
        )
        temp_km.capture(
            title="testXvalue",
            description="Has X instead",
            content="Content with testXvalue",
        )

        # Search for literal "_" should only match underscore entries
        results = temp_km.search("test_value")
        assert len(results) == 1
        assert "_" in results[0]["title"]


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

    def test_data_survives_restart(self, shared_embedding_service):
        """Test that data persists after manager restart."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create and populate
            km1 = KnowledgeManager(base_path=temp_dir, embedding_service=shared_embedding_service)
            kid = km1.capture(
                title="Persistent Data",
                description="Should survive restart",
                content="Test content",
            )
            km1.close()

            # Reopen and verify
            km2 = KnowledgeManager(base_path=temp_dir, embedding_service=shared_embedding_service)
            item = km2.get(kid)
            assert item is not None
            assert item["title"] == "Persistent Data"
            km2.close()
        finally:
            shutil.rmtree(temp_dir)


class TestContextManager:
    """Tests for context manager support."""

    def test_context_manager(self, shared_embedding_service):
        """Test using KnowledgeManager as context manager."""
        temp_dir = tempfile.mkdtemp()
        try:
            with KnowledgeManager(
                base_path=temp_dir, embedding_service=shared_embedding_service
            ) as km:
                kid = km.capture(
                    title="Test",
                    description="Test",
                    content="Test content",
                )
                assert len(kid) == 12
        finally:
            shutil.rmtree(temp_dir)


class TestDuplicateDetection:
    """Tests for duplicate detection functionality."""

    def test_find_duplicates_empty(self, temp_km):
        """Test finding duplicates with no entries."""
        groups = temp_km.find_duplicates()
        assert groups == []

    def test_find_duplicates_single_entry(self, temp_km):
        """Test finding duplicates with single entry."""
        temp_km.capture(
            title="Test Entry",
            description="A test description",
            content="Test content here",
        )
        groups = temp_km.find_duplicates()
        assert groups == []

    def test_find_duplicates_no_matches(self, temp_km):
        """Test finding duplicates with dissimilar entries."""
        temp_km.capture(
            title="OAuth Implementation",
            description="How to implement OAuth",
            content="Use authlib for OAuth authentication",
        )
        temp_km.capture(
            title="Database Optimization",
            description="How to optimize database queries",
            content="Use indexes and query planning",
        )
        groups = temp_km.find_duplicates()
        assert groups == []

    def test_find_duplicates_similar_entries(self, temp_km):
        """Test finding duplicates with similar entries."""
        temp_km.capture(
            title="OAuth Implementation",
            description="How to implement OAuth with authlib",
            content="Use authlib for OAuth. Configure client ID and secret.",
        )
        temp_km.capture(
            title="OAuth Setup",
            description="Setting up OAuth authentication with authlib",
            content="Authlib OAuth setup requires client ID and secret.",
        )
        groups = temp_km.find_duplicates(threshold=0.7)
        assert len(groups) >= 1
        assert len(groups[0]) == 2

    def test_find_duplicates_threshold(self, temp_km):
        """Test that threshold affects duplicate detection."""
        temp_km.capture(
            title="Python Testing",
            description="How to test Python code",
            content="Use pytest for testing Python applications",
        )
        temp_km.capture(
            title="Python Unit Tests",
            description="Writing unit tests in Python",
            content="Pytest is great for Python unit testing",
        )

        # High threshold should find fewer duplicates
        groups_high = temp_km.find_duplicates(threshold=0.95)
        groups_low = temp_km.find_duplicates(threshold=0.5)

        assert len(groups_low) >= len(groups_high)

    def test_merge_entries(self, temp_km):
        """Test merging two entries."""
        id1 = temp_km.capture(
            title="Entry One",
            description="First entry",
            content="Content one",
            tags="tag1,tag2",
        )
        id2 = temp_km.capture(
            title="Entry Two",
            description="Second entry",
            content="Content two",
            tags="tag2,tag3",
        )

        result = temp_km.merge_entries(id1, id2)
        assert result is True

        # Check target has merged content
        merged = temp_km.get(id1)
        assert "Content one" in merged["content"]
        assert "Content two" in merged["content"]

        # Check tags are merged
        tags = json_to_tags(merged["tags"])
        assert "tag1" in tags
        assert "tag2" in tags
        assert "tag3" in tags

        # Check source is deleted
        assert temp_km.get(id2) is None

    def test_merge_entries_not_found(self, temp_km):
        """Test merging with nonexistent entry."""
        id1 = temp_km.capture(
            title="Entry One",
            description="First entry",
            content="Content one",
        )

        result = temp_km.merge_entries(id1, "nonexistent")
        assert result is False

        result = temp_km.merge_entries("nonexistent", id1)
        assert result is False


class TestStalenessTracking:
    """Tests for staleness tracking functionality."""

    def test_find_stale_empty(self, temp_km):
        """Test finding stale entries with no entries."""
        stale = temp_km.find_stale(days=30)
        assert stale == []

    def test_find_stale_fresh_entries(self, temp_km):
        """Test that recently created entries are not stale."""
        temp_km.capture(
            title="Fresh Entry",
            description="Just created",
            content="New content",
        )
        stale = temp_km.find_stale(days=30)
        assert stale == []

    def test_find_stale_with_old_entries(self, temp_km):
        """Test finding entries that are stale."""
        # Create an entry
        kid = temp_km.capture(
            title="Old Entry",
            description="Will become stale",
            content="Old content",
        )

        # Manually backdate the entry
        from datetime import datetime, timedelta

        old_date = (datetime.now() - timedelta(days=100)).isoformat()
        cursor = temp_km.conn.cursor()
        cursor.execute(
            "UPDATE knowledge SET created = ?, updated_at = ? WHERE id = ?",
            (old_date, old_date, kid),
        )
        temp_km.conn.commit()

        # Should find the stale entry
        stale = temp_km.find_stale(days=30)
        assert len(stale) == 1
        assert stale[0]["id"] == kid
        assert stale[0]["days_stale"] >= 100

    def test_find_stale_respects_threshold(self, temp_km):
        """Test that days parameter is respected."""
        kid = temp_km.capture(
            title="Test Entry",
            description="Test",
            content="Content",
        )

        # Backdate to 50 days ago
        from datetime import datetime, timedelta

        old_date = (datetime.now() - timedelta(days=50)).isoformat()
        cursor = temp_km.conn.cursor()
        cursor.execute(
            "UPDATE knowledge SET created = ?, updated_at = ? WHERE id = ?",
            (old_date, old_date, kid),
        )
        temp_km.conn.commit()

        # Should not be stale with 90 day threshold
        stale_90 = temp_km.find_stale(days=90)
        assert len(stale_90) == 0

        # Should be stale with 30 day threshold
        stale_30 = temp_km.find_stale(days=30)
        assert len(stale_30) == 1

    def test_find_stale_considers_last_used(self, temp_km):
        """Test that last_used date prevents staleness."""
        kid = temp_km.capture(
            title="Used Entry",
            description="Has been used recently",
            content="Content",
        )

        # Backdate creation but set recent last_used
        from datetime import datetime, timedelta

        old_date = (datetime.now() - timedelta(days=100)).isoformat()
        recent_date = datetime.now().isoformat()
        cursor = temp_km.conn.cursor()
        cursor.execute(
            "UPDATE knowledge SET created = ?, updated_at = ?, last_used = ? WHERE id = ?",
            (old_date, old_date, recent_date, kid),
        )
        temp_km.conn.commit()

        # Should not be stale because it was recently used
        stale = temp_km.find_stale(days=30)
        assert len(stale) == 0

    def test_find_stale_project_filter(self, temp_km):
        """Test that project filter works."""
        kid1 = temp_km.capture(
            title="Project A Entry",
            description="In project A",
            content="Content",
            project="project-a",
        )
        kid2 = temp_km.capture(
            title="Project B Entry",
            description="In project B",
            content="Content",
            project="project-b",
        )

        # Backdate both entries
        from datetime import datetime, timedelta

        old_date = (datetime.now() - timedelta(days=100)).isoformat()
        cursor = temp_km.conn.cursor()
        cursor.execute(
            "UPDATE knowledge SET created = ?, updated_at = ? WHERE id IN (?, ?)",
            (old_date, old_date, kid1, kid2),
        )
        temp_km.conn.commit()

        # Filter by project
        stale_a = temp_km.find_stale(days=30, project="project-a")
        assert len(stale_a) == 1
        assert stale_a[0]["id"] == kid1

        stale_b = temp_km.find_stale(days=30, project="project-b")
        assert len(stale_b) == 1
        assert stale_b[0]["id"] == kid2


class TestQualityScoring:
    """Tests for quality scoring functionality."""

    def test_score_quality_empty(self, temp_km):
        """Test scoring with no entries."""
        scored = temp_km.score_quality()
        assert scored == []

    def test_score_quality_full_score(self, temp_km):
        """Test entry with all quality criteria met gets 100."""
        temp_km.capture(
            title="High Quality Entry",
            description="This is a detailed description that explains the entry " * 2,
            content="This content has useful information about quality scoring " * 3,
            tags=["tag1"],
        )

        # Retrieve to increment usage count
        temp_km.retrieve("high quality entry", n_results=1)

        scored = temp_km.score_quality()
        assert len(scored) == 1
        assert scored[0]["quality_score"] == 100

    def test_score_quality_no_tags(self, temp_km):
        """Test entry without tags loses 25 points."""
        temp_km.capture(
            title="Entry Without Tags",
            description="This is a detailed description that explains the entry " * 2,
            content="This content has useful information about quality scoring " * 3,
        )

        # Retrieve to get usage
        temp_km.retrieve("entry without tags", n_results=1)

        scored = temp_km.score_quality()
        assert scored[0]["quality_score"] == 75

    def test_score_quality_short_description(self, temp_km):
        """Test entry with short description loses 25 points."""
        temp_km.capture(
            title="Short Description Entry",
            description="Short",  # Less than 50 chars
            content="This content has useful information about quality scoring " * 3,
            tags=["tag1"],
        )

        # Retrieve to get usage
        temp_km.retrieve("short description entry", n_results=1)

        scored = temp_km.score_quality()
        assert scored[0]["quality_score"] == 75

    def test_score_quality_short_content(self, temp_km):
        """Test entry with short content loses 25 points."""
        temp_km.capture(
            title="Short Content Entry",
            description="This is a detailed description that explains the entry " * 2,
            content="Short",  # Less than 100 chars
            tags=["tag1"],
        )

        # Retrieve to get usage
        temp_km.retrieve("short content entry", n_results=1)

        scored = temp_km.score_quality()
        assert scored[0]["quality_score"] == 75

    def test_score_quality_never_used(self, temp_km):
        """Test entry that's never been used loses 25 points."""
        temp_km.capture(
            title="Never Used",
            description="A" * 50,
            content="B" * 100,
            tags=["tag1"],
        )

        # Don't retrieve it
        scored = temp_km.score_quality()
        assert scored[0]["quality_score"] == 75

    def test_score_quality_sorted_ascending(self, temp_km):
        """Test that results are sorted by score ascending."""
        # Create low quality entry
        temp_km.capture(
            title="Low Quality",
            description="Short",
            content="Short",
        )

        # Create high quality entry
        temp_km.capture(
            title="High Quality",
            description="A" * 50,
            content="B" * 100,
            tags=["tag1"],
        )

        # Use the high quality entry
        temp_km.retrieve("high quality", n_results=1)

        scored = temp_km.score_quality()
        assert len(scored) == 2
        # Low quality should be first
        assert scored[0]["quality_score"] < scored[1]["quality_score"]

    def test_score_quality_min_filter(self, temp_km):
        """Test minimum score filter."""
        # Create low quality entry
        temp_km.capture(
            title="Low Quality",
            description="Short",
            content="Short",
        )

        # Create high quality entry
        kid_high = temp_km.capture(
            title="High Quality",
            description="A" * 50,
            content="B" * 100,
            tags=["tag1"],
        )
        temp_km.retrieve("high quality", n_results=1)

        # Filter for high scores only
        scored = temp_km.score_quality(min_score=50)
        assert len(scored) == 1
        assert scored[0]["id"] == kid_high

    def test_score_quality_max_filter(self, temp_km):
        """Test maximum score filter."""
        # Create low quality entry
        kid_low = temp_km.capture(
            title="Low Quality",
            description="Short",
            content="Short",
        )

        # Create high quality entry
        temp_km.capture(
            title="High Quality",
            description="A" * 50,
            content="B" * 100,
            tags=["tag1"],
        )
        temp_km.retrieve("high quality", n_results=1)

        # Filter for low scores only
        scored = temp_km.score_quality(max_score=50)
        assert len(scored) == 1
        assert scored[0]["id"] == kid_low

    def test_score_quality_project_filter(self, temp_km):
        """Test project filter works with quality scoring."""
        temp_km.capture(
            title="Project A Entry",
            description="In project A",
            content="Content",
            project="project-a",
        )
        temp_km.capture(
            title="Project B Entry",
            description="In project B",
            content="Content",
            project="project-b",
        )

        scored_a = temp_km.score_quality(project="project-a")
        assert len(scored_a) == 1
        assert "Project A" in scored_a[0]["title"]


class TestTagFiltering:
    """Tests for tag-based filtering."""

    def test_list_all_with_tags(self, temp_km):
        """Test list_all filters by tags."""
        temp_km.capture(
            title="Python Entry",
            description="Python code",
            content="Content",
            tags=["python", "api"],
        )
        temp_km.capture(
            title="JavaScript Entry",
            description="JS code",
            content="Content",
            tags=["javascript", "api"],
        )
        temp_km.capture(
            title="Go Entry",
            description="Go code",
            content="Content",
            tags=["go", "cli"],
        )

        # Filter by single tag
        python_entries = temp_km.list_all(tags=["python"])
        assert len(python_entries) == 1
        assert "Python" in python_entries[0]["title"]

        # Filter by multiple tags (AND logic)
        api_entries = temp_km.list_all(tags=["api"])
        assert len(api_entries) == 2

        python_api_entries = temp_km.list_all(tags=["python", "api"])
        assert len(python_api_entries) == 1

    def test_search_with_tags(self, temp_km):
        """Test search filters by tags."""
        temp_km.capture(
            title="Database Query",
            description="SQL query optimization",
            content="Content",
            tags=["sql", "database"],
        )
        temp_km.capture(
            title="Database Schema",
            description="SQL schema design",
            content="Content",
            tags=["sql", "schema"],
        )

        # Search with tag filter
        results = temp_km.search("Database", tags=["schema"])
        assert len(results) == 1
        assert "Schema" in results[0]["title"]


class TestDateFiltering:
    """Tests for date range filtering."""

    def test_list_all_with_since(self, temp_km):
        """Test list_all filters by since date."""
        from datetime import datetime, timedelta

        # Capture an entry
        temp_km.capture(
            title="Recent Entry",
            description="Recent",
            content="Content",
        )

        # Filter by future date (should return nothing)
        future = (datetime.now() + timedelta(days=1)).isoformat()
        entries = temp_km.list_all(since=future)
        assert len(entries) == 0

        # Filter by past date (should return the entry)
        past = (datetime.now() - timedelta(days=1)).isoformat()
        entries = temp_km.list_all(since=past)
        assert len(entries) == 1

    def test_list_all_with_until(self, temp_km):
        """Test list_all filters by until date."""
        from datetime import datetime, timedelta

        temp_km.capture(
            title="Entry",
            description="Description",
            content="Content",
        )

        # Filter by past date (should return nothing)
        past = (datetime.now() - timedelta(days=1)).isoformat()
        entries = temp_km.list_all(until=past)
        assert len(entries) == 0

        # Filter by future date (should return the entry)
        future = (datetime.now() + timedelta(days=1)).isoformat()
        entries = temp_km.list_all(until=future)
        assert len(entries) == 1


class TestFuzzyTagMatching:
    """Tests for fuzzy tag matching."""

    def test_fuzzy_match_exact(self, temp_km):
        """Test exact match still works with fuzzy enabled."""
        temp_km.capture(
            title="Python Entry",
            description="Python code",
            content="Content",
            tags=["python"],
        )

        entries = temp_km.list_all(tags=["python"], fuzzy=True)
        assert len(entries) == 1

    def test_fuzzy_match_typo(self, temp_km):
        """Test fuzzy matching with typos."""
        temp_km.capture(
            title="Python Entry",
            description="Python code",
            content="Content",
            tags=["python"],
        )

        # "pythn" has edit distance 1 from "python"
        entries = temp_km.list_all(tags=["pythn"], fuzzy=True)
        assert len(entries) == 1

        # Without fuzzy, should not match
        entries = temp_km.list_all(tags=["pythn"], fuzzy=False)
        assert len(entries) == 0

    def test_fuzzy_match_multiple_tags(self, temp_km):
        """Test fuzzy matching with multiple tags."""
        temp_km.capture(
            title="Web API",
            description="REST API",
            content="Content",
            tags=["javascript", "api"],
        )

        # Both tags must fuzzy-match
        entries = temp_km.list_all(tags=["javascrpt", "api"], fuzzy=True)
        assert len(entries) == 1

        # One tag too different (edit distance > 2)
        # "jvscrpt" has distance 3 from "javascript"
        entries = temp_km.list_all(tags=["jvscrpt", "api"], fuzzy=True)
        assert len(entries) == 0


class TestRelativeDateParsing:
    """Tests for relative date parsing utility."""

    def test_parse_relative_date_days(self):
        """Test parsing days."""
        from datetime import datetime, timedelta

        from claude_knowledge.utils import parse_relative_date

        result = parse_relative_date("7d")
        assert result is not None
        parsed = datetime.fromisoformat(result)
        expected = datetime.now() - timedelta(days=7)
        # Allow 1 second tolerance
        assert abs((parsed - expected).total_seconds()) < 1

    def test_parse_relative_date_weeks(self):
        """Test parsing weeks."""
        from datetime import datetime, timedelta

        from claude_knowledge.utils import parse_relative_date

        result = parse_relative_date("2w")
        assert result is not None
        parsed = datetime.fromisoformat(result)
        expected = datetime.now() - timedelta(weeks=2)
        assert abs((parsed - expected).total_seconds()) < 1

    def test_parse_relative_date_months(self):
        """Test parsing months."""
        from datetime import datetime, timedelta

        from claude_knowledge.utils import parse_relative_date

        result = parse_relative_date("1m")
        assert result is not None
        parsed = datetime.fromisoformat(result)
        expected = datetime.now() - timedelta(days=30)
        assert abs((parsed - expected).total_seconds()) < 1

    def test_parse_relative_date_iso(self):
        """Test parsing ISO format dates."""
        from claude_knowledge.utils import parse_relative_date

        iso_date = "2026-01-15T10:30:00"
        result = parse_relative_date(iso_date)
        assert result == iso_date

    def test_parse_relative_date_invalid(self):
        """Test parsing invalid dates."""
        from claude_knowledge.utils import parse_relative_date

        assert parse_relative_date("invalid") is None
        assert parse_relative_date("") is None


class TestDateFieldValidation:
    """Tests for date_field SQL injection prevention."""

    def test_valid_date_field_created(self, temp_km):
        """Test that 'created' is accepted as a valid date field."""
        temp_km.capture(
            title="Test Entry",
            description="Test",
            content="Content",
        )
        # Should not raise
        entries = temp_km.list_all(date_field="created")
        assert len(entries) == 1

    def test_valid_date_field_last_used(self, temp_km):
        """Test that 'last_used' is accepted as a valid date field."""
        temp_km.capture(
            title="Test Entry",
            description="Test",
            content="Content",
        )
        # Should not raise
        entries = temp_km.list_all(date_field="last_used")
        assert len(entries) == 1

    def test_invalid_date_field_rejected(self, temp_km):
        """Test that invalid date fields are rejected (SQL injection prevention)."""
        import pytest

        temp_km.capture(
            title="Test Entry",
            description="Test",
            content="Content",
        )

        # Attempt SQL injection via date_field
        with pytest.raises(ValueError, match="Invalid date_field"):
            temp_km.list_all(date_field="created; DROP TABLE knowledge; --")

        with pytest.raises(ValueError, match="Invalid date_field"):
            temp_km.search("test", date_field="id")

        with pytest.raises(ValueError, match="Invalid date_field"):
            temp_km.retrieve("test", date_field="1=1 OR")


class TestLevenshteinDistance:
    """Tests for Levenshtein distance utility."""

    def test_levenshtein_identical(self):
        """Test identical strings have distance 0."""
        from claude_knowledge.utils import levenshtein_distance

        assert levenshtein_distance("python", "python") == 0

    def test_levenshtein_single_edit(self):
        """Test single character edits."""
        from claude_knowledge.utils import levenshtein_distance

        # Deletion
        assert levenshtein_distance("python", "pythn") == 1
        # Insertion
        assert levenshtein_distance("pythn", "python") == 1
        # Substitution
        assert levenshtein_distance("python", "pithon") == 1

    def test_levenshtein_multiple_edits(self):
        """Test multiple character edits."""
        from claude_knowledge.utils import levenshtein_distance

        assert levenshtein_distance("python", "pyton") == 1
        assert levenshtein_distance("python", "pthn") == 2
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_levenshtein_empty_string(self):
        """Test with empty strings."""
        from claude_knowledge.utils import levenshtein_distance

        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("python", "") == 6
        assert levenshtein_distance("", "python") == 6
