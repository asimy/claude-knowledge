"""Tests for the code parser module."""

import tempfile
from pathlib import Path

import pytest

from claude_knowledge.code_parser import (
    CodeElement,
    CodeParser,
    ParsedFile,
)


@pytest.fixture
def temp_code_dir():
    """Create a temporary directory with code files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        code_dir = Path(tmpdir)

        # Create Python file
        (code_dir / "main.py").write_text('''"""Main module docstring.

This module contains the main functionality.
"""

import os
from pathlib import Path


class UserService:
    """Service for managing users.

    This class handles all user-related operations including
    creation, retrieval, and deletion.

    Attributes:
        db: Database connection.
    """

    def __init__(self, db):
        """Initialize the service.

        Args:
            db: Database connection.
        """
        self.db = db

    @property
    def users(self):
        """Get all users."""
        return self.db.query("SELECT * FROM users")

    def get_user(self, user_id: int) -> dict:
        """Get a user by ID.

        Args:
            user_id: The user's ID.

        Returns:
            User dict or None if not found.
        """
        return self.db.query_one(f"SELECT * FROM users WHERE id = {user_id}")

    def create_user(self, name: str, email: str) -> int:
        """Create a new user.

        Args:
            name: User's name.
            email: User's email.

        Returns:
            The new user's ID.
        """
        # TODO: Add validation
        return self.db.insert("users", {"name": name, "email": email})


def helper_function():
    """A helper function at module level."""
    pass


# FIXME: This needs to be refactored
GLOBAL_CONFIG = {"debug": True}
''')

        # Create JavaScript file
        (code_dir / "app.js").write_text("""/**
 * Application entry point.
 *
 * @module app
 */

import { Router } from 'express';
const db = require('./database');

/**
 * User controller class.
 * Handles HTTP requests for user operations.
 */
class UserController {
    /**
     * Create a new controller.
     * @param {Database} database - The database instance.
     */
    constructor(database) {
        this.db = database;
    }

    /**
     * Get all users.
     * @returns {Promise<Array>} List of users.
     */
    async getUsers() {
        return await this.db.query('SELECT * FROM users');
    }
}

/**
 * Create a new user.
 * @param {string} name - User name.
 * @param {string} email - User email.
 */
const createUser = async (name, email) => {
    // TODO: Implement validation
    return db.insert('users', { name, email });
};

module.exports = { UserController, createUser };
""")

        # Create Go file
        (code_dir / "main.go").write_text("""package main

import (
    "fmt"
    "log"
)

// UserService handles user operations
type UserService struct {
    db *Database
}

// User represents a user entity
type User struct {
    ID    int
    Name  string
    Email string
}

// GetUser retrieves a user by ID
// Returns nil if not found
func (s *UserService) GetUser(id int) *User {
    return s.db.FindByID(id)
}

// CreateUser creates a new user
func (s *UserService) CreateUser(name, email string) (*User, error) {
    // TODO: Add validation
    return s.db.Insert(&User{Name: name, Email: email})
}

func main() {
    fmt.Println("Hello, World!")
}
""")

        # Create Ruby file
        (code_dir / "service.rb").write_text("""# User service module
# Provides user management functionality
module UserService
  # User repository class
  # Handles database operations for users
  class UserRepository
    def initialize(db)
      @db = db
    end

    # Find a user by ID
    # @param id [Integer] the user ID
    # @return [User, nil] the found user or nil
    def find(id)
      @db.query("SELECT * FROM users WHERE id = #{id}")
    end

    # Create a new user
    def create(name:, email:)
      # FIXME: Add validation here
      @db.insert(:users, name: name, email: email)
    end
  end
end
""")

        # Create subdirectory with files
        (code_dir / "lib").mkdir()
        (code_dir / "lib" / "utils.py").write_text('''"""Utility functions."""

def format_date(date):
    """Format a date object."""
    return date.strftime("%Y-%m-%d")
''')

        # Create test file (should be detected as test)
        (code_dir / "tests").mkdir()
        (code_dir / "tests" / "test_main.py").write_text('''"""Tests for main module."""

import pytest

def test_user_service():
    """Test user service."""
    assert True
''')

        yield code_dir


class TestCodeElement:
    """Tests for CodeElement dataclass."""

    def test_create_code_element(self):
        """Test creating a CodeElement."""
        element = CodeElement(
            element_type="function",
            name="get_user",
            docstring="Get a user by ID.",
            signature="def get_user(user_id: int) -> dict:",
            start_line=10,
            decorators=["property"],
        )

        assert element.element_type == "function"
        assert element.name == "get_user"
        assert element.docstring == "Get a user by ID."
        assert element.start_line == 10
        assert "property" in element.decorators

    def test_method_with_parent(self):
        """Test creating a method element with parent class."""
        element = CodeElement(
            element_type="method",
            name="find",
            parent="UserRepository",
            start_line=20,
        )

        assert element.element_type == "method"
        assert element.parent == "UserRepository"


class TestParsedFile:
    """Tests for ParsedFile dataclass."""

    def test_create_parsed_file(self):
        """Test creating a ParsedFile."""
        parsed = ParsedFile(
            path="/path/to/main.py",
            language="python",
            content_hash="abc123",
            elements=[],
            imports=["os", "pathlib"],
            comments=["[TODO] Add validation"],
            size_bytes=1024,
            line_count=50,
        )

        assert parsed.path == "/path/to/main.py"
        assert parsed.language == "python"
        assert "os" in parsed.imports
        assert parsed.line_count == 50


class TestCodeParser:
    """Tests for CodeParser class."""

    def test_scan_files(self, temp_code_dir):
        """Test scanning for source files."""
        parser = CodeParser(temp_code_dir)
        files = parser.scan_files()

        # Should find Python, JS, Go, and Ruby files
        file_names = [f.name for f in files]
        assert "main.py" in file_names
        assert "app.js" in file_names
        assert "main.go" in file_names
        assert "service.rb" in file_names
        assert "utils.py" in file_names

    def test_scan_files_with_include(self, temp_code_dir):
        """Test scanning with include patterns."""
        parser = CodeParser(temp_code_dir)
        files = parser.scan_files(include_patterns=["*.py"])

        file_names = [f.name for f in files]
        assert "main.py" in file_names
        assert "app.js" not in file_names

    def test_scan_files_with_exclude(self, temp_code_dir):
        """Test scanning with exclude patterns."""
        parser = CodeParser(temp_code_dir)
        files = parser.scan_files(exclude_patterns=["tests/*"])

        file_names = [f.name for f in files]
        assert "main.py" in file_names
        assert "test_main.py" not in file_names

    def test_scan_files_by_language(self, temp_code_dir):
        """Test scanning by language."""
        parser = CodeParser(temp_code_dir)
        files = parser.scan_files(languages=["python"])

        for f in files:
            assert f.suffix == ".py"

    def test_parse_python_file(self, temp_code_dir):
        """Test parsing a Python file."""
        parser = CodeParser(temp_code_dir)
        parsed = parser.parse_file(temp_code_dir / "main.py")

        assert parsed is not None
        assert parsed.language == "python"
        assert parsed.content_hash is not None

        # Check elements
        element_names = [e.name for e in parsed.elements]
        assert "UserService" in element_names
        assert "get_user" in element_names
        assert "create_user" in element_names
        assert "helper_function" in element_names

        # Check class detection
        user_service = next(e for e in parsed.elements if e.name == "UserService")
        assert user_service.element_type == "class"
        assert user_service.docstring is not None
        assert "managing users" in user_service.docstring

        # Check method detection
        get_user = next(e for e in parsed.elements if e.name == "get_user")
        assert get_user.element_type == "method"
        assert get_user.parent == "UserService"
        assert get_user.docstring is not None

        # Check imports
        assert "os" in parsed.imports
        assert "pathlib" in parsed.imports

        # Check significant comments
        assert any("TODO" in c for c in parsed.comments)
        assert any("FIXME" in c for c in parsed.comments)

    def test_parse_javascript_file(self, temp_code_dir):
        """Test parsing a JavaScript file."""
        parser = CodeParser(temp_code_dir)
        parsed = parser.parse_file(temp_code_dir / "app.js")

        assert parsed is not None
        assert parsed.language == "javascript"

        # Check elements
        element_names = [e.name for e in parsed.elements]
        assert "UserController" in element_names
        assert "createUser" in element_names

        # Check imports
        assert any("express" in imp for imp in parsed.imports)

    def test_parse_go_file(self, temp_code_dir):
        """Test parsing a Go file."""
        parser = CodeParser(temp_code_dir)
        parsed = parser.parse_file(temp_code_dir / "main.go")

        assert parsed is not None
        assert parsed.language == "go"

        # Check elements
        element_names = [e.name for e in parsed.elements]
        assert "UserService" in element_names
        assert "User" in element_names
        assert "GetUser" in element_names
        assert "CreateUser" in element_names
        assert "main" in element_names

        # Check Go comment extraction
        get_user = next(e for e in parsed.elements if e.name == "GetUser")
        assert get_user.docstring is not None
        assert "retrieves a user" in get_user.docstring.lower()

    def test_parse_ruby_file(self, temp_code_dir):
        """Test parsing a Ruby file."""
        parser = CodeParser(temp_code_dir)
        parsed = parser.parse_file(temp_code_dir / "service.rb")

        assert parsed is not None
        assert parsed.language == "ruby"

        # Check elements
        element_names = [e.name for e in parsed.elements]
        assert "UserService" in element_names
        assert "UserRepository" in element_names
        assert "find" in element_names
        assert "create" in element_names

    def test_parse_files_batch(self, temp_code_dir):
        """Test parsing multiple files."""
        parser = CodeParser(temp_code_dir)
        parsed_files = parser.parse_files()

        assert len(parsed_files) >= 5  # At least our test files

        languages = {pf.language for pf in parsed_files}
        assert "python" in languages
        assert "javascript" in languages
        assert "go" in languages
        assert "ruby" in languages

    def test_get_file_hash(self, temp_code_dir):
        """Test getting file hash."""
        parser = CodeParser(temp_code_dir)
        hash1 = parser.get_file_hash(temp_code_dir / "main.py")

        assert hash1 is not None
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

        # Same content should give same hash
        hash2 = parser.get_file_hash(temp_code_dir / "main.py")
        assert hash1 == hash2

    def test_detect_language(self, temp_code_dir):
        """Test language detection."""
        parser = CodeParser(temp_code_dir)

        assert parser._detect_language(Path("test.py")) == "python"
        assert parser._detect_language(Path("test.js")) == "javascript"
        assert parser._detect_language(Path("test.ts")) == "typescript"
        assert parser._detect_language(Path("test.go")) == "go"
        assert parser._detect_language(Path("test.rb")) == "ruby"
        assert parser._detect_language(Path("test.txt")) is None

    def test_base_path_validation_nonexistent(self):
        """Test that CodeParser raises ValueError for nonexistent path."""
        with pytest.raises(ValueError, match="does not exist"):
            CodeParser("/nonexistent/path/to/code")

    def test_base_path_validation_file_not_directory(self, temp_code_dir):
        """Test that CodeParser raises ValueError when path is a file."""
        file_path = temp_code_dir / "main.py"
        with pytest.raises(ValueError, match="not a directory"):
            CodeParser(file_path)

    def test_base_path_defaults_to_cwd(self):
        """Test that CodeParser defaults to current working directory."""
        parser = CodeParser()
        assert parser.base_path == Path.cwd()


class TestCodeParserPythonParsing:
    """Tests for Python-specific parsing."""

    def test_extract_docstrings(self, temp_code_dir):
        """Test extracting docstrings from Python code."""
        parser = CodeParser(temp_code_dir)
        parsed = parser.parse_file(temp_code_dir / "main.py")

        # Class docstring
        user_service = next(e for e in parsed.elements if e.name == "UserService")
        assert "Service for managing users" in user_service.docstring

        # Method docstring with Args
        get_user = next(e for e in parsed.elements if e.name == "get_user")
        assert "Args:" in get_user.docstring
        assert "Returns:" in get_user.docstring

    def test_detect_decorators(self, temp_code_dir):
        """Test detecting decorators on functions/methods."""
        parser = CodeParser(temp_code_dir)
        parsed = parser.parse_file(temp_code_dir / "main.py")

        # The 'users' method has @property decorator
        users_method = next((e for e in parsed.elements if e.name == "users"), None)
        if users_method:
            assert "property" in users_method.decorators

    def test_extract_imports(self, temp_code_dir):
        """Test extracting Python imports."""
        parser = CodeParser(temp_code_dir)
        parsed = parser.parse_file(temp_code_dir / "main.py")

        assert "os" in parsed.imports
        assert "pathlib" in parsed.imports


class TestCodeParserComments:
    """Tests for comment extraction."""

    def test_extract_todo_comments(self, temp_code_dir):
        """Test extracting TODO comments."""
        parser = CodeParser(temp_code_dir)
        parsed = parser.parse_file(temp_code_dir / "main.py")

        todos = [c for c in parsed.comments if "TODO" in c]
        assert len(todos) >= 1
        assert any("validation" in c.lower() for c in todos)

    def test_extract_fixme_comments(self, temp_code_dir):
        """Test extracting FIXME comments."""
        parser = CodeParser(temp_code_dir)
        parsed = parser.parse_file(temp_code_dir / "main.py")

        fixmes = [c for c in parsed.comments if "FIXME" in c]
        assert len(fixmes) >= 1


class TestCodeParserExcludePatterns:
    """Tests for file exclusion patterns."""

    def test_default_excludes(self, temp_code_dir):
        """Test default exclude patterns."""
        # Create files that should be excluded
        (temp_code_dir / "node_modules").mkdir()
        (temp_code_dir / "node_modules" / "lib.js").write_text("// library")

        (temp_code_dir / "__pycache__").mkdir()
        (temp_code_dir / "__pycache__" / "main.cpython-39.pyc").write_text("")

        parser = CodeParser(temp_code_dir)
        files = parser.scan_files()

        file_paths = [str(f) for f in files]
        assert not any("node_modules" in p for p in file_paths)
        assert not any("__pycache__" in p for p in file_paths)
