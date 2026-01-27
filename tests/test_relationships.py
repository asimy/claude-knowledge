"""Tests for relationships and collections functionality."""

import pytest

# temp_km fixture is provided by conftest.py


@pytest.fixture
def entries(temp_km):
    """Create sample entries for relationship testing."""
    id1 = temp_km.capture(
        title="Entry One",
        description="First entry",
        content="Content for entry one",
        tags="tag1",
    )
    id2 = temp_km.capture(
        title="Entry Two",
        description="Second entry",
        content="Content for entry two",
        tags="tag2",
    )
    id3 = temp_km.capture(
        title="Entry Three",
        description="Third entry",
        content="Content for entry three",
        tags="tag3",
    )
    return {"one": id1, "two": id2, "three": id3}


class TestLinking:
    """Tests for link/unlink functionality."""

    def test_link_basic(self, temp_km, entries):
        """Test creating a basic relationship."""
        rel_id = temp_km.link(entries["one"], entries["two"])
        assert len(rel_id) == 12

    def test_link_related_bidirectional(self, temp_km, entries):
        """Test that 'related' type is bidirectional (normalized order)."""
        # Link in one direction
        rel_id1 = temp_km.link(entries["two"], entries["one"], "related")

        # Linking in reverse should return the same relationship
        rel_id2 = temp_km.link(entries["one"], entries["two"], "related")
        assert rel_id1 == rel_id2

    def test_link_depends_on(self, temp_km, entries):
        """Test creating a depends-on relationship."""
        rel_id = temp_km.link(entries["one"], entries["two"], "depends-on")
        assert len(rel_id) == 12

        # Should be able to create reverse direction too (different relationship)
        rel_id2 = temp_km.link(entries["two"], entries["one"], "depends-on")
        assert rel_id2 != rel_id

    def test_link_supersedes(self, temp_km, entries):
        """Test creating a supersedes relationship."""
        rel_id = temp_km.link(entries["two"], entries["one"], "supersedes")
        assert len(rel_id) == 12

    def test_link_invalid_type(self, temp_km, entries):
        """Test that invalid relationship type raises error."""
        with pytest.raises(ValueError, match="Invalid relationship_type"):
            temp_km.link(entries["one"], entries["two"], "invalid-type")

    def test_link_same_entry(self, temp_km, entries):
        """Test that linking entry to itself raises error."""
        with pytest.raises(ValueError, match="Cannot create a relationship.*itself"):
            temp_km.link(entries["one"], entries["one"])

    def test_link_nonexistent_source(self, temp_km, entries):
        """Test that linking with nonexistent source raises error."""
        with pytest.raises(ValueError, match="Source entry not found"):
            temp_km.link("nonexistent123", entries["one"])

    def test_link_nonexistent_target(self, temp_km, entries):
        """Test that linking with nonexistent target raises error."""
        with pytest.raises(ValueError, match="Target entry not found"):
            temp_km.link(entries["one"], "nonexistent123")

    def test_unlink_basic(self, temp_km, entries):
        """Test removing a relationship."""
        temp_km.link(entries["one"], entries["two"])
        result = temp_km.unlink(entries["one"], entries["two"])
        assert result is True

    def test_unlink_nonexistent(self, temp_km, entries):
        """Test removing a nonexistent relationship."""
        result = temp_km.unlink(entries["one"], entries["two"])
        assert result is False

    def test_unlink_with_type(self, temp_km, entries):
        """Test removing a specific relationship type."""
        temp_km.link(entries["one"], entries["two"], "related")
        temp_km.link(entries["one"], entries["two"], "depends-on")

        # Remove only depends-on
        result = temp_km.unlink(entries["one"], entries["two"], "depends-on")
        assert result is True

        # Related should still exist
        relationships = temp_km.get_related(entries["one"])
        assert len(relationships) == 1
        assert relationships[0]["relationship_type"] == "related"

    def test_unlink_all_types(self, temp_km, entries):
        """Test removing all relationships between two entries."""
        temp_km.link(entries["one"], entries["two"], "related")
        temp_km.link(entries["one"], entries["two"], "depends-on")

        # Remove all
        result = temp_km.unlink(entries["one"], entries["two"])
        assert result is True

        # No relationships should remain
        relationships = temp_km.get_related(entries["one"])
        assert len(relationships) == 0


class TestGetRelated:
    """Tests for get_related functionality."""

    def test_get_related_basic(self, temp_km, entries):
        """Test getting related entries."""
        temp_km.link(entries["one"], entries["two"], "related")

        # From entry one's perspective
        related = temp_km.get_related(entries["one"])
        assert len(related) == 1
        assert related[0]["related_id"] == entries["two"]
        assert related[0]["related_title"] == "Entry Two"

        # From entry two's perspective (bidirectional)
        related = temp_km.get_related(entries["two"])
        assert len(related) == 1
        assert related[0]["related_id"] == entries["one"]

    def test_get_related_direction_outgoing(self, temp_km, entries):
        """Test filtering by outgoing direction."""
        temp_km.link(entries["one"], entries["two"], "depends-on")

        # Entry one has outgoing depends-on to entry two
        related = temp_km.get_related(entries["one"], direction="outgoing")
        assert len(related) == 1
        assert related[0]["related_id"] == entries["two"]

        # Entry two has no outgoing relationships
        related = temp_km.get_related(entries["two"], direction="outgoing")
        assert len(related) == 0

    def test_get_related_direction_incoming(self, temp_km, entries):
        """Test filtering by incoming direction."""
        temp_km.link(entries["one"], entries["two"], "depends-on")

        # Entry one has no incoming depends-on
        related = temp_km.get_related(entries["one"], direction="incoming")
        assert len(related) == 0

        # Entry two has incoming depends-on from entry one
        related = temp_km.get_related(entries["two"], direction="incoming")
        assert len(related) == 1
        assert related[0]["related_id"] == entries["one"]

    def test_get_related_type_filter(self, temp_km, entries):
        """Test filtering by relationship type."""
        temp_km.link(entries["one"], entries["two"], "related")
        temp_km.link(entries["one"], entries["three"], "depends-on")

        # Filter by related type only
        related = temp_km.get_related(entries["one"], relationship_type="related")
        assert len(related) == 1
        assert related[0]["related_id"] == entries["two"]

        # Filter by depends-on type only
        related = temp_km.get_related(entries["one"], relationship_type="depends-on")
        assert len(related) == 1
        assert related[0]["related_id"] == entries["three"]

    def test_get_related_invalid_direction(self, temp_km, entries):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="Invalid direction"):
            temp_km.get_related(entries["one"], direction="invalid")


class TestDependencyTree:
    """Tests for dependency tree functionality."""

    def test_dependency_tree_basic(self, temp_km, entries):
        """Test basic dependency tree."""
        temp_km.link(entries["one"], entries["two"], "depends-on")
        temp_km.link(entries["two"], entries["three"], "depends-on")

        tree = temp_km.get_dependency_tree(entries["one"])
        assert tree["id"] == entries["one"]
        assert tree["title"] == "Entry One"
        assert len(tree["dependencies"]) == 1
        assert tree["dependencies"][0]["id"] == entries["two"]
        assert len(tree["dependencies"][0]["dependencies"]) == 1
        assert tree["dependencies"][0]["dependencies"][0]["id"] == entries["three"]

    def test_dependency_tree_depth_limit(self, temp_km, entries):
        """Test that depth limit is respected."""
        temp_km.link(entries["one"], entries["two"], "depends-on")
        temp_km.link(entries["two"], entries["three"], "depends-on")

        # Depth 1 should only show immediate dependencies
        tree = temp_km.get_dependency_tree(entries["one"], depth=1)
        assert len(tree["dependencies"]) == 1
        assert tree["dependencies"][0]["dependencies"] == []

    def test_dependency_tree_nonexistent_entry(self, temp_km):
        """Test dependency tree for nonexistent entry."""
        tree = temp_km.get_dependency_tree("nonexistent123")
        assert tree == {}


class TestCollections:
    """Tests for collection CRUD operations."""

    def test_create_collection(self, temp_km):
        """Test creating a collection."""
        coll_id = temp_km.create_collection("My Collection", "A test collection")
        assert len(coll_id) == 12

    def test_create_collection_duplicate_name(self, temp_km):
        """Test that duplicate collection names raise error."""
        temp_km.create_collection("My Collection")
        with pytest.raises(ValueError, match="already exists"):
            temp_km.create_collection("My Collection")

    def test_create_collection_empty_name(self, temp_km):
        """Test that empty collection name raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            temp_km.create_collection("")

    def test_create_collection_whitespace_name(self, temp_km):
        """Test that whitespace-only name raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            temp_km.create_collection("   ")

    def test_create_collection_name_too_long(self, temp_km):
        """Test that overly long collection name raises error."""
        long_name = "A" * 201
        with pytest.raises(ValueError, match="exceeds maximum length"):
            temp_km.create_collection(long_name)

    def test_get_collection_by_id(self, temp_km):
        """Test getting a collection by ID."""
        coll_id = temp_km.create_collection("My Collection", "Description")
        collection = temp_km.get_collection(coll_id)
        assert collection["name"] == "My Collection"
        assert collection["description"] == "Description"
        assert collection["member_count"] == 0

    def test_get_collection_by_name(self, temp_km):
        """Test getting a collection by name."""
        coll_id = temp_km.create_collection("My Collection")
        collection = temp_km.get_collection("My Collection")
        assert collection["id"] == coll_id

    def test_get_collection_nonexistent(self, temp_km):
        """Test getting a nonexistent collection."""
        collection = temp_km.get_collection("nonexistent")
        assert collection is None

    def test_delete_collection(self, temp_km):
        """Test deleting a collection."""
        coll_id = temp_km.create_collection("My Collection")
        result = temp_km.delete_collection(coll_id)
        assert result is True
        assert temp_km.get_collection(coll_id) is None

    def test_delete_collection_by_name(self, temp_km):
        """Test deleting a collection by name."""
        temp_km.create_collection("My Collection")
        result = temp_km.delete_collection("My Collection")
        assert result is True

    def test_delete_collection_nonexistent(self, temp_km):
        """Test deleting a nonexistent collection."""
        result = temp_km.delete_collection("nonexistent")
        assert result is False

    def test_list_collections(self, temp_km):
        """Test listing collections."""
        temp_km.create_collection("Collection A")
        temp_km.create_collection("Collection B")

        collections = temp_km.list_collections()
        assert len(collections) == 2
        names = {c["name"] for c in collections}
        assert "Collection A" in names
        assert "Collection B" in names


class TestCollectionMembership:
    """Tests for collection membership management."""

    def test_add_to_collection(self, temp_km, entries):
        """Test adding an entry to a collection."""
        coll_id = temp_km.create_collection("My Collection")
        result = temp_km.add_to_collection(coll_id, entries["one"])
        assert result is True

        collection = temp_km.get_collection(coll_id)
        assert collection["member_count"] == 1

    def test_add_to_collection_duplicate(self, temp_km, entries):
        """Test that adding duplicate returns False."""
        coll_id = temp_km.create_collection("My Collection")
        temp_km.add_to_collection(coll_id, entries["one"])

        # Adding again should return False
        result = temp_km.add_to_collection(coll_id, entries["one"])
        assert result is False

    def test_add_to_collection_nonexistent_collection(self, temp_km, entries):
        """Test that adding to nonexistent collection raises error."""
        with pytest.raises(ValueError, match="Collection not found"):
            temp_km.add_to_collection("nonexistent", entries["one"])

    def test_add_to_collection_nonexistent_entry(self, temp_km):
        """Test that adding nonexistent entry raises error."""
        coll_id = temp_km.create_collection("My Collection")
        with pytest.raises(ValueError, match="Entry not found"):
            temp_km.add_to_collection(coll_id, "nonexistent123")

    def test_remove_from_collection(self, temp_km, entries):
        """Test removing an entry from a collection."""
        coll_id = temp_km.create_collection("My Collection")
        temp_km.add_to_collection(coll_id, entries["one"])

        result = temp_km.remove_from_collection(coll_id, entries["one"])
        assert result is True

        collection = temp_km.get_collection(coll_id)
        assert collection["member_count"] == 0

    def test_remove_from_collection_not_member(self, temp_km, entries):
        """Test that removing non-member returns False."""
        coll_id = temp_km.create_collection("My Collection")
        result = temp_km.remove_from_collection(coll_id, entries["one"])
        assert result is False

    def test_remove_from_collection_nonexistent_collection(self, temp_km, entries):
        """Test that removing from nonexistent collection raises error."""
        with pytest.raises(ValueError, match="Collection not found"):
            temp_km.remove_from_collection("nonexistent", entries["one"])

    def test_get_collection_members(self, temp_km, entries):
        """Test getting collection members."""
        coll_id = temp_km.create_collection("My Collection")
        temp_km.add_to_collection(coll_id, entries["one"])
        temp_km.add_to_collection(coll_id, entries["two"])

        members = temp_km.get_collection_members(coll_id)
        assert len(members) == 2
        ids = {m["id"] for m in members}
        assert entries["one"] in ids
        assert entries["two"] in ids

    def test_get_collection_members_nonexistent(self, temp_km):
        """Test that getting members of nonexistent collection raises error."""
        with pytest.raises(ValueError, match="Collection not found"):
            temp_km.get_collection_members("nonexistent")

    def test_get_entry_collections(self, temp_km, entries):
        """Test getting collections that contain an entry."""
        coll_id1 = temp_km.create_collection("Collection A")
        coll_id2 = temp_km.create_collection("Collection B")

        temp_km.add_to_collection(coll_id1, entries["one"])
        temp_km.add_to_collection(coll_id2, entries["one"])

        collections = temp_km.get_entry_collections(entries["one"])
        assert len(collections) == 2
        names = {c["name"] for c in collections}
        assert "Collection A" in names
        assert "Collection B" in names

    def test_cascade_delete_collection_removes_membership(self, temp_km, entries):
        """Test that deleting a collection removes memberships."""
        coll_id = temp_km.create_collection("My Collection")
        temp_km.add_to_collection(coll_id, entries["one"])

        temp_km.delete_collection(coll_id)

        # Entry should no longer be in any collections
        collections = temp_km.get_entry_collections(entries["one"])
        assert len(collections) == 0

    def test_cascade_delete_entry_removes_membership(self, temp_km, entries):
        """Test that deleting an entry removes it from collections."""
        coll_id = temp_km.create_collection("My Collection")
        temp_km.add_to_collection(coll_id, entries["one"])
        temp_km.add_to_collection(coll_id, entries["two"])

        # Delete entry one
        temp_km.delete(entries["one"])

        # Collection should only have entry two
        members = temp_km.get_collection_members(coll_id)
        assert len(members) == 1
        assert members[0]["id"] == entries["two"]


class TestHasRelationshipsOrCollections:
    """Tests for checking relationships/collections before delete."""

    def test_has_relationships(self, temp_km, entries):
        """Test detecting relationships."""
        temp_km.link(entries["one"], entries["two"], "related")
        temp_km.link(entries["one"], entries["three"], "depends-on")

        counts = temp_km.has_relationships_or_collections(entries["one"])
        assert counts["relationships"] == 2
        assert counts["collections"] == 0

    def test_has_collections(self, temp_km, entries):
        """Test detecting collection membership."""
        coll_id1 = temp_km.create_collection("Collection A")
        coll_id2 = temp_km.create_collection("Collection B")
        temp_km.add_to_collection(coll_id1, entries["one"])
        temp_km.add_to_collection(coll_id2, entries["one"])

        counts = temp_km.has_relationships_or_collections(entries["one"])
        assert counts["relationships"] == 0
        assert counts["collections"] == 2

    def test_has_both(self, temp_km, entries):
        """Test detecting both relationships and collections."""
        temp_km.link(entries["one"], entries["two"])
        coll_id = temp_km.create_collection("My Collection")
        temp_km.add_to_collection(coll_id, entries["one"])

        counts = temp_km.has_relationships_or_collections(entries["one"])
        assert counts["relationships"] == 1
        assert counts["collections"] == 1

    def test_has_none(self, temp_km, entries):
        """Test entry with no relationships or collections."""
        counts = temp_km.has_relationships_or_collections(entries["one"])
        assert counts["relationships"] == 0
        assert counts["collections"] == 0


class TestCascadeDeletes:
    """Tests for cascade delete behavior."""

    def test_delete_entry_removes_relationships(self, temp_km, entries):
        """Test that deleting an entry removes its relationships."""
        temp_km.link(entries["one"], entries["two"], "related")
        temp_km.link(entries["one"], entries["three"], "depends-on")

        # Delete entry one
        temp_km.delete(entries["one"])

        # Entry two should have no relationships
        related = temp_km.get_related(entries["two"])
        assert len(related) == 0

        # Entry three should have no relationships
        related = temp_km.get_related(entries["three"])
        assert len(related) == 0
