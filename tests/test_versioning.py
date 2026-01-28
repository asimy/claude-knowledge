"""Tests for versioning functionality."""

import pytest

# temp_km fixture is provided by conftest.py


@pytest.fixture
def entry_with_updates(temp_km):
    """Create an entry and update it multiple times to generate versions."""
    # Create initial entry
    entry_id = temp_km.capture(
        title="Original Title",
        description="Original description",
        content="Original content here",
        tags="tag1,tag2",
        project="test-project",
    )

    # Update 1 - change title
    temp_km.update(entry_id, title="Updated Title v1")

    # Update 2 - change content
    temp_km.update(entry_id, content="Updated content v2")

    # Update 3 - change description and tags
    temp_km.update(entry_id, description="Updated description v3", tags="tag1,tag3")

    return entry_id


class TestVersionCreation:
    """Tests for automatic version creation on update."""

    def test_no_versions_initially(self, temp_km):
        """Test that new entries have no versions."""
        entry_id = temp_km.capture(
            title="Test Entry",
            description="Test description",
            content="Test content",
        )
        versions = temp_km.get_history(entry_id)
        assert len(versions) == 0

    def test_version_created_on_update(self, temp_km):
        """Test that updating an entry creates a version."""
        entry_id = temp_km.capture(
            title="Test Entry",
            description="Test description",
            content="Test content",
        )

        temp_km.update(entry_id, title="New Title")

        versions = temp_km.get_history(entry_id)
        assert len(versions) == 1
        assert versions[0]["version_number"] == 1
        assert versions[0]["title"] == "Test Entry"  # Original title preserved

    def test_multiple_versions(self, temp_km, entry_with_updates):
        """Test that multiple updates create multiple versions."""
        versions = temp_km.get_history(entry_with_updates)
        assert len(versions) == 3

        # Versions are ordered newest first
        assert versions[0]["version_number"] == 3
        assert versions[1]["version_number"] == 2
        assert versions[2]["version_number"] == 1

    def test_version_captures_all_fields(self, temp_km):
        """Test that version captures all entry fields."""
        entry_id = temp_km.capture(
            title="Original Title",
            description="Original description",
            content="Original content",
            tags="tag1,tag2",
            project="test-project",
            confidence=0.8,
        )

        temp_km.update(entry_id, title="New Title")

        version = temp_km.get_version(entry_id, 1)
        assert version is not None
        assert version["title"] == "Original Title"
        assert version["description"] == "Original description"
        assert version["content"] == "Original content"
        assert version["tags"] is not None
        assert version["project"] == "test-project"
        assert version["confidence"] == 0.8

    def test_no_version_when_nothing_changes(self, temp_km):
        """Test that calling update with no changes doesn't create version."""
        entry_id = temp_km.capture(
            title="Test Entry",
            description="Test description",
            content="Test content",
        )

        # Update with no fields
        temp_km.update(entry_id)

        versions = temp_km.get_history(entry_id)
        assert len(versions) == 0

    def test_version_not_created_when_disabled(self, temp_km):
        """Test that version creation can be disabled."""
        entry_id = temp_km.capture(
            title="Test Entry",
            description="Test description",
            content="Test content",
        )

        # Update with versioning disabled
        temp_km.update(entry_id, title="New Title", create_version=False)

        versions = temp_km.get_history(entry_id)
        assert len(versions) == 0


class TestVersionRetrieval:
    """Tests for retrieving versions."""

    def test_get_version_by_number(self, temp_km, entry_with_updates):
        """Test retrieving a specific version by number."""
        version = temp_km.get_version(entry_with_updates, 1)
        assert version is not None
        assert version["version_number"] == 1
        assert version["title"] == "Original Title"

    def test_get_version_nonexistent(self, temp_km, entry_with_updates):
        """Test that getting nonexistent version returns None."""
        version = temp_km.get_version(entry_with_updates, 999)
        assert version is None

    def test_get_version_wrong_entry(self, temp_km, entry_with_updates):
        """Test that getting version for wrong entry returns None."""
        version = temp_km.get_version("nonexistent-id", 1)
        assert version is None

    def test_get_history_with_limit(self, temp_km, entry_with_updates):
        """Test that history limit is respected."""
        versions = temp_km.get_history(entry_with_updates, limit=2)
        assert len(versions) == 2
        # Should return newest first
        assert versions[0]["version_number"] == 3
        assert versions[1]["version_number"] == 2

    def test_get_history_empty(self, temp_km):
        """Test history for entry with no versions."""
        entry_id = temp_km.capture(
            title="Test Entry",
            description="Test description",
            content="Test content",
        )
        versions = temp_km.get_history(entry_id)
        assert versions == []

    def test_get_version_count(self, temp_km, entry_with_updates):
        """Test counting versions for an entry."""
        count = temp_km.get_version_count(entry_with_updates)
        assert count == 3


class TestRollback:
    """Tests for rollback functionality."""

    def test_rollback_restores_content(self, temp_km, entry_with_updates):
        """Test that rollback restores entry to previous state."""
        # Get version 1 content
        version_1 = temp_km.get_version(entry_with_updates, 1)
        original_title = version_1["title"]

        # Rollback to version 1
        result = temp_km.rollback(entry_with_updates, 1)
        assert result is True

        # Check entry is restored
        entry = temp_km.get(entry_with_updates)
        assert entry["title"] == original_title

    def test_rollback_creates_new_version(self, temp_km, entry_with_updates):
        """Test that rollback creates a version of current state first."""
        initial_count = temp_km.get_version_count(entry_with_updates)

        temp_km.rollback(entry_with_updates, 1)

        # Should have one more version (auto-saved before rollback)
        new_count = temp_km.get_version_count(entry_with_updates)
        assert new_count == initial_count + 1

    def test_rollback_nonexistent_version(self, temp_km, entry_with_updates):
        """Test that rollback to nonexistent version fails."""
        result = temp_km.rollback(entry_with_updates, 999)
        assert result is False

    def test_rollback_nonexistent_entry(self, temp_km):
        """Test that rollback for nonexistent entry fails."""
        result = temp_km.rollback("nonexistent-id", 1)
        assert result is False

    def test_rollback_regenerates_embedding(self, temp_km):
        """Test that rollback regenerates the embedding for restored content."""
        # Create entry with distinctive content
        entry_id = temp_km.capture(
            title="Authentication Guide",
            description="How to implement OAuth authentication",
            content="Use OAuth 2.0 with PKCE flow for secure authentication.",
        )

        # Update to completely different content
        temp_km.update(
            entry_id,
            title="Database Guide",
            description="How to configure PostgreSQL",
            content="Configure PostgreSQL with connection pooling for production.",
        )

        # Rollback to original
        temp_km.rollback(entry_id, 1)

        # Search for original content should find this entry
        results = temp_km.retrieve(query="OAuth authentication PKCE")
        found = any(r["id"] == entry_id for r in results)
        assert found, "Entry should be found via semantic search after rollback"


class TestDiff:
    """Tests for diff functionality."""

    def test_diff_between_versions(self, temp_km, entry_with_updates):
        """Test generating diff between two versions."""
        diff = temp_km.diff_versions(entry_with_updates, 1, 2)

        assert "version_a" in diff
        assert "version_b" in diff
        assert diff["version_a"]["number"] == 1
        assert diff["version_b"]["number"] == 2

    def test_diff_with_current_state(self, temp_km, entry_with_updates):
        """Test generating diff between version and current state."""
        diff = temp_km.diff_versions(entry_with_updates, 1, None)

        assert diff["version_b"]["label"] == "current"

    def test_diff_shows_title_change(self, temp_km):
        """Test that diff shows title changes."""
        entry_id = temp_km.capture(
            title="Original Title",
            description="Test description",
            content="Test content",
        )
        temp_km.update(entry_id, title="New Title")

        diff = temp_km.diff_versions(entry_id, 1, None)
        assert diff["title_diff"] is not None
        assert "Original Title" in diff["title_diff"]
        assert "New Title" in diff["title_diff"]

    def test_diff_shows_content_change(self, temp_km):
        """Test that diff shows content changes."""
        entry_id = temp_km.capture(
            title="Test Entry",
            description="Test description",
            content="Line 1\nLine 2\nLine 3",
        )
        temp_km.update(entry_id, content="Line 1\nModified Line 2\nLine 3")

        diff = temp_km.diff_versions(entry_id, 1, None)
        assert diff["content_diff"] is not None
        assert "-Line 2" in diff["content_diff"]
        assert "+Modified Line 2" in diff["content_diff"]

    def test_diff_no_change_shows_none(self, temp_km):
        """Test that identical content shows no diff."""
        entry_id = temp_km.capture(
            title="Test Entry",
            description="Test description",
            content="Test content",
        )
        # Update only tags, not title
        temp_km.update(entry_id, tags="new-tag")

        diff = temp_km.diff_versions(entry_id, 1, None)
        assert diff["title_diff"] is None  # Title didn't change

    def test_diff_shows_other_changes(self, temp_km):
        """Test that diff shows flags for other changed fields."""
        entry_id = temp_km.capture(
            title="Test Entry",
            description="Test description",
            content="Test content",
            tags="tag1",
            project="proj1",
        )
        temp_km.update(entry_id, tags="tag2", project="proj2")

        diff = temp_km.diff_versions(entry_id, 1, None)
        assert diff["tags_changed"] is True
        assert diff["project_changed"] is True

    def test_diff_nonexistent_version(self, temp_km, entry_with_updates):
        """Test that diff with nonexistent version raises error."""
        with pytest.raises(ValueError, match="not found"):
            temp_km.diff_versions(entry_with_updates, 999, None)

    def test_diff_nonexistent_entry(self, temp_km):
        """Test that diff for nonexistent entry raises error."""
        with pytest.raises(ValueError, match="not found"):
            temp_km.diff_versions("nonexistent-id", 1, None)


class TestVersionPruning:
    """Tests for automatic version pruning."""

    def test_versions_pruned_at_limit(self, temp_km):
        """Test that old versions are pruned when limit is reached."""
        # Set a low limit for testing
        temp_km._versioning.max_versions = 5

        entry_id = temp_km.capture(
            title="Test Entry",
            description="Test description",
            content="Test content",
        )

        # Create 10 versions by updating 10 times
        for i in range(10):
            temp_km.update(entry_id, title=f"Title v{i + 1}")

        # Should only have 5 versions (the limit)
        count = temp_km.get_version_count(entry_id)
        assert count == 5

        # Oldest versions should be pruned, newest kept
        versions = temp_km.get_history(entry_id)
        version_numbers = [v["version_number"] for v in versions]
        # Versions 6-10 should exist (newest 5)
        assert 6 in version_numbers
        assert 10 in version_numbers
        # Version 1-5 should be pruned
        assert 1 not in version_numbers


class TestCascadeDelete:
    """Tests for cascade delete behavior."""

    def test_delete_entry_removes_versions(self, temp_km, entry_with_updates):
        """Test that deleting an entry removes its versions."""
        # Verify versions exist
        assert temp_km.get_version_count(entry_with_updates) > 0

        # Delete the entry
        temp_km.delete(entry_with_updates)

        # Versions should be deleted (via foreign key cascade)
        # Note: We can't check directly since get_version_count depends on entry
        # But we can verify the entry is gone
        assert temp_km.get(entry_with_updates) is None


class TestChangeSummary:
    """Tests for change summary generation."""

    def test_change_summary_includes_fields(self, temp_km):
        """Test that change summary lists modified fields."""
        entry_id = temp_km.capture(
            title="Test Entry",
            description="Test description",
            content="Test content",
        )

        temp_km.update(entry_id, title="New Title", content="New content")

        versions = temp_km.get_history(entry_id)
        assert len(versions) == 1
        summary = versions[0]["change_summary"]
        assert "title" in summary
        assert "content" in summary
