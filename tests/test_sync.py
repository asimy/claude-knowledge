"""Tests for sync functionality."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from claude_knowledge.knowledge_manager import KnowledgeManager
from claude_knowledge.utils import compute_content_hash, get_machine_id

# temp_km and populated_km fixtures are provided by conftest.py
# shared_embedding_service fixture is also provided by conftest.py for manual KM creation


@pytest.fixture
def sync_dir():
    """Create a temporary sync directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestSyncUtils:
    """Tests for sync utility functions."""

    def test_compute_content_hash(self):
        """Test content hash computation."""
        entry = {
            "title": "Test Title",
            "description": "Test Description",
            "content": "Test Content",
        }
        hash1 = compute_content_hash(entry)
        assert len(hash1) == 64  # SHA-256 hex string

        # Same content should produce same hash
        hash2 = compute_content_hash(entry)
        assert hash1 == hash2

        # Different content should produce different hash
        entry["content"] = "Modified Content"
        hash3 = compute_content_hash(entry)
        assert hash1 != hash3

    def test_get_machine_id(self):
        """Test machine ID retrieval."""
        machine_id = get_machine_id()
        assert isinstance(machine_id, str)
        assert len(machine_id) > 0


class TestInitSyncDir:
    """Tests for sync directory initialization."""

    def test_init_creates_structure(self, temp_km, sync_dir):
        """Test that init_sync_dir creates proper directory structure."""
        temp_km.init_sync_dir(sync_dir)

        assert (sync_dir / "manifest.json").exists()
        assert (sync_dir / "entries").is_dir()
        assert (sync_dir / "tombstones").is_dir()
        assert (sync_dir / "tombstones" / "deleted.json").exists()
        assert (sync_dir / "relationships").is_dir()
        assert (sync_dir / "collections").is_dir()

    def test_init_creates_valid_manifest(self, temp_km, sync_dir):
        """Test that manifest is valid JSON with correct structure."""
        temp_km.init_sync_dir(sync_dir)

        with open(sync_dir / "manifest.json") as f:
            manifest = json.load(f)

        assert "version" in manifest
        assert manifest["version"] == 2  # Updated for relationships/collections support
        assert "last_sync" in manifest
        assert "entries" in manifest
        assert "relationships" in manifest
        assert "collections" in manifest

    def test_init_saves_sync_path(self, temp_km, sync_dir):
        """Test that init saves the sync path to config."""
        temp_km.init_sync_dir(sync_dir)
        temp_km.set_sync_path(sync_dir)

        saved_path = temp_km.get_sync_path()
        assert saved_path == sync_dir


class TestSyncPush:
    """Tests for pushing local entries to sync directory."""

    def test_push_creates_entry_files(self, populated_km, sync_dir):
        """Test that sync push creates entry files."""
        result = populated_km.sync(sync_path=sync_dir)

        assert result.pushed == 3
        assert result.pulled == 0

        entries_dir = sync_dir / "entries"
        entry_files = list(entries_dir.glob("*.json"))
        assert len(entry_files) == 3

    def test_push_entry_format(self, populated_km, sync_dir):
        """Test that pushed entries have correct format."""
        populated_km.sync(sync_path=sync_dir)

        entry_files = list((sync_dir / "entries").glob("*.json"))
        with open(entry_files[0]) as f:
            entry = json.load(f)

        assert "id" in entry
        assert "title" in entry
        assert "description" in entry
        assert "content" in entry
        assert "content_hash" in entry
        assert "updated_at" in entry

    def test_push_only_mode(self, populated_km, sync_dir):
        """Test push-only mode."""
        result = populated_km.sync(sync_path=sync_dir, push_only=True)

        assert result.pushed == 3
        assert result.pulled == 0


class TestSyncPull:
    """Tests for pulling remote entries to local database."""

    def test_pull_imports_entries(self, populated_km, sync_dir, shared_embedding_service):
        """Test that sync pull imports remote entries."""
        # Push from populated_km
        populated_km.sync(sync_path=sync_dir)

        # Create a fresh empty km for pulling
        temp_dir2 = tempfile.mkdtemp()
        try:
            km2 = KnowledgeManager(base_path=temp_dir2, embedding_service=shared_embedding_service)
            result = km2.sync(sync_path=sync_dir)

            assert result.pulled == 3
            assert len(km2.list_all()) == 3
            km2.close()
        finally:
            shutil.rmtree(temp_dir2)

    def test_pull_only_mode(self, populated_km, sync_dir, shared_embedding_service):
        """Test pull-only mode."""
        populated_km.sync(sync_path=sync_dir)

        # Create a fresh empty km for pulling
        temp_dir2 = tempfile.mkdtemp()
        try:
            km2 = KnowledgeManager(base_path=temp_dir2, embedding_service=shared_embedding_service)
            result = km2.sync(sync_path=sync_dir, pull_only=True)

            assert result.pulled == 3
            assert result.pushed == 0
            km2.close()
        finally:
            shutil.rmtree(temp_dir2)


class TestSyncBidirectional:
    """Tests for bidirectional sync."""

    def test_bidirectional_no_conflict(self, shared_embedding_service):
        """Test bidirectional sync with no conflicts."""
        temp_dir1 = tempfile.mkdtemp()
        temp_dir2 = tempfile.mkdtemp()
        sync_dir = tempfile.mkdtemp()

        try:
            km1 = KnowledgeManager(base_path=temp_dir1, embedding_service=shared_embedding_service)
            km2 = KnowledgeManager(base_path=temp_dir2, embedding_service=shared_embedding_service)

            # Create different entries on each
            km1.capture(
                title="Entry from KM1",
                description="Created on machine 1",
                content="Content 1",
            )
            km2.capture(
                title="Entry from KM2",
                description="Created on machine 2",
                content="Content 2",
            )

            # Sync km1 first
            km1.sync(sync_path=sync_dir)

            # Sync km2 - should push and pull
            result = km2.sync(sync_path=sync_dir)
            assert result.pushed == 1
            assert result.pulled == 1

            # Sync km1 again - should pull km2's entry
            result = km1.sync(sync_path=sync_dir)
            assert result.pulled == 1

            # Both should now have 2 entries
            assert len(km1.list_all()) == 2
            assert len(km2.list_all()) == 2

            km1.close()
            km2.close()
        finally:
            shutil.rmtree(temp_dir1)
            shutil.rmtree(temp_dir2)
            shutil.rmtree(sync_dir)


class TestSyncConflicts:
    """Tests for conflict detection and resolution."""

    def test_conflict_detection(self, shared_embedding_service):
        """Test that conflicts are detected when both sides change.

        A conflict occurs when both local and remote have changed since the last sync.
        This happens when km2 modifies locally AND km1's sync updates the remote.
        """
        temp_dir1 = tempfile.mkdtemp()
        temp_dir2 = tempfile.mkdtemp()
        sync_dir = tempfile.mkdtemp()

        try:
            km1 = KnowledgeManager(base_path=temp_dir1, embedding_service=shared_embedding_service)
            km2 = KnowledgeManager(base_path=temp_dir2, embedding_service=shared_embedding_service)

            # Create entry on km1 and sync
            kid = km1.capture(
                title="Shared Entry",
                description="Will be modified by both",
                content="Original content",
            )
            km1.sync(sync_path=sync_dir)

            # Pull to km2 - this sets km2's manifest to the original hash
            km2.sync(sync_path=sync_dir)

            # Now modify on both sides WITHOUT syncing in between
            km1.update(kid, content="Modified by km1")
            km2.update(kid, content="Modified by km2")

            # Sync km1 - this updates the remote to km1's version
            # but km2's manifest still has the original hash
            km1.sync(sync_path=sync_dir)

            # When km2 syncs:
            # - local hash = km2's modification (different from manifest)
            # - remote hash = km1's modification (different from manifest)
            # - manifest hash = original (from km2's first sync)
            # Both changed relative to manifest = CONFLICT
            result = km2.sync(sync_path=sync_dir, strategy="manual")
            assert len(result.conflicts) == 1

            km1.close()
            km2.close()
        finally:
            shutil.rmtree(temp_dir1)
            shutil.rmtree(temp_dir2)
            shutil.rmtree(sync_dir)

    def test_conflict_local_wins(self, shared_embedding_service):
        """Test local-wins conflict resolution."""
        temp_dir1 = tempfile.mkdtemp()
        temp_dir2 = tempfile.mkdtemp()
        sync_dir = tempfile.mkdtemp()

        try:
            km1 = KnowledgeManager(base_path=temp_dir1, embedding_service=shared_embedding_service)
            km2 = KnowledgeManager(base_path=temp_dir2, embedding_service=shared_embedding_service)

            kid = km1.capture(
                title="Shared Entry",
                description="Will conflict",
                content="Original",
            )
            km1.sync(sync_path=sync_dir)
            km2.sync(sync_path=sync_dir)

            # Both modify without syncing
            km1.update(kid, content="km1 version")
            km2.update(kid, content="km2 version")

            # km1 syncs first
            km1.sync(sync_path=sync_dir)

            # km2 uses local-wins - should keep its version and push
            result = km2.sync(sync_path=sync_dir, strategy="local-wins")
            assert len(result.conflicts) == 1
            assert result.conflicts[0]["resolution"] == "local"

            # km2 should still have its version
            entry = km2.get(kid)
            assert entry["content"] == "km2 version"

            km1.close()
            km2.close()
        finally:
            shutil.rmtree(temp_dir1)
            shutil.rmtree(temp_dir2)
            shutil.rmtree(sync_dir)

    def test_conflict_remote_wins(self, shared_embedding_service):
        """Test remote-wins conflict resolution."""
        temp_dir1 = tempfile.mkdtemp()
        temp_dir2 = tempfile.mkdtemp()
        sync_dir = tempfile.mkdtemp()

        try:
            km1 = KnowledgeManager(base_path=temp_dir1, embedding_service=shared_embedding_service)
            km2 = KnowledgeManager(base_path=temp_dir2, embedding_service=shared_embedding_service)

            kid = km1.capture(
                title="Shared Entry",
                description="Will conflict",
                content="Original",
            )
            km1.sync(sync_path=sync_dir)
            km2.sync(sync_path=sync_dir)

            # Both modify without syncing
            km1.update(kid, content="km1 version")
            km2.update(kid, content="km2 version")

            # km1 syncs first
            km1.sync(sync_path=sync_dir)

            # km2 uses remote-wins - should take km1's version
            result = km2.sync(sync_path=sync_dir, strategy="remote-wins")
            assert len(result.conflicts) == 1
            assert result.conflicts[0]["resolution"] == "remote"

            # km2 should now have km1's version
            entry = km2.get(kid)
            assert entry["content"] == "km1 version"

            km1.close()
            km2.close()
        finally:
            shutil.rmtree(temp_dir1)
            shutil.rmtree(temp_dir2)
            shutil.rmtree(sync_dir)


class TestSyncDryRun:
    """Tests for dry run mode."""

    def test_dry_run_no_changes(self, populated_km, sync_dir):
        """Test that dry run doesn't make changes."""
        # Initialize first so dry run doesn't fail
        populated_km.init_sync_dir(sync_dir)

        result = populated_km.sync(sync_path=sync_dir, dry_run=True)

        assert result.pushed == 3
        # Files should not exist (dry run)
        entry_files = list((sync_dir / "entries").glob("*.json"))
        assert len(entry_files) == 0

    def test_dry_run_reports_changes(self, populated_km, sync_dir, shared_embedding_service):
        """Test that dry run correctly reports what would happen."""
        populated_km.sync(sync_path=sync_dir)

        # Create a fresh empty km for dry run pull
        temp_dir2 = tempfile.mkdtemp()
        try:
            km2 = KnowledgeManager(base_path=temp_dir2, embedding_service=shared_embedding_service)
            result = km2.sync(sync_path=sync_dir, dry_run=True)

            assert result.pulled == 3
            # But no entries should actually be imported
            assert len(km2.list_all()) == 0
            km2.close()
        finally:
            shutil.rmtree(temp_dir2)


class TestSyncStatus:
    """Tests for sync status command."""

    def test_status_shows_pending(self, populated_km, sync_dir):
        """Test that status shows pending changes."""
        populated_km.init_sync_dir(sync_dir)
        populated_km.set_sync_path(sync_dir)

        status = populated_km.sync_status()

        assert len(status["to_push"]) == 3
        assert len(status["to_pull"]) == 0

    def test_status_after_sync(self, populated_km, sync_dir):
        """Test that status is empty after full sync."""
        populated_km.sync(sync_path=sync_dir)

        status = populated_km.sync_status()

        assert len(status["to_push"]) == 0
        assert len(status["to_pull"]) == 0


class TestSyncConfig:
    """Tests for sync path configuration."""

    def test_save_and_load_sync_path(self, temp_km, sync_dir):
        """Test saving and loading sync path from config."""
        temp_km.set_sync_path(sync_dir)
        loaded = temp_km.get_sync_path()

        assert loaded == sync_dir

    def test_sync_without_path_uses_config(self, populated_km, sync_dir):
        """Test that sync uses saved path when none provided."""
        # First sync with explicit path
        populated_km.sync(sync_path=sync_dir)

        # Add a new entry
        populated_km.capture(
            title="New Entry",
            description="Added after first sync",
            content="New content",
        )

        # Sync without path - should use saved path
        result = populated_km.sync()

        assert result.pushed == 1


class TestSyncProjectFilter:
    """Tests for project-filtered sync."""

    def test_sync_single_project(self, shared_embedding_service):
        """Test syncing only a specific project."""
        temp_dir = tempfile.mkdtemp()
        sync_dir = tempfile.mkdtemp()

        try:
            km = KnowledgeManager(base_path=temp_dir, embedding_service=shared_embedding_service)

            km.capture(
                title="Project A Entry",
                description="For project A",
                content="Content A",
                project="project-a",
            )
            km.capture(
                title="Project B Entry",
                description="For project B",
                content="Content B",
                project="project-b",
            )

            # Sync only project-a
            result = km.sync(sync_path=sync_dir, project="project-a")

            assert result.pushed == 1
            entry_files = list((Path(sync_dir) / "entries").glob("*.json"))
            assert len(entry_files) == 1

            km.close()
        finally:
            shutil.rmtree(temp_dir)
            shutil.rmtree(sync_dir)


class TestUpdatedAtMigration:
    """Tests for updated_at column migration."""

    def test_new_entries_have_updated_at(self, temp_km):
        """Test that new entries have updated_at set."""
        kid = temp_km.capture(
            title="Test",
            description="Test",
            content="Test content",
        )

        entry = temp_km.get(kid)
        assert entry["updated_at"] is not None
        assert entry["updated_at"] == entry["created"]

    def test_update_changes_updated_at(self, temp_km):
        """Test that updates change updated_at timestamp."""
        kid = temp_km.capture(
            title="Test",
            description="Test",
            content="Test content",
        )

        original = temp_km.get(kid)
        original_updated = original["updated_at"]

        import time

        time.sleep(0.01)  # Ensure time difference

        temp_km.update(kid, content="Modified content")

        modified = temp_km.get(kid)
        assert modified["updated_at"] > original_updated
