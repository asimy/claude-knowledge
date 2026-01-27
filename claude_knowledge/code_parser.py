"""Parse codebase files for patterns and documentation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path


@dataclass
class CodeElement:
    """A code element (class, function, etc.) extracted from a file."""

    element_type: str  # "class", "function", "method", "constant"
    name: str
    docstring: str | None = None
    signature: str | None = None
    start_line: int = 0
    end_line: int = 0
    decorators: list[str] = field(default_factory=list)
    parent: str | None = None  # For methods, the class name


@dataclass
class ParsedFile:
    """A parsed source file with extracted elements."""

    path: str
    language: str
    content_hash: str
    elements: list[CodeElement] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)  # Significant standalone comments
    size_bytes: int = 0
    line_count: int = 0
    last_modified: datetime | None = None


class CodeParser:
    """Parse source code files to extract structural elements."""

    # File extension to language mapping
    LANGUAGE_MAP = {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rb": "ruby",
        ".rs": "rust",
        ".java": "java",
        ".kt": "kotlin",
        ".swift": "swift",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".php": "php",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
    }

    # Default patterns to exclude
    DEFAULT_EXCLUDES = [
        "*.min.js",
        "*.min.css",
        "*.bundle.js",
        "*.pyc",
        "*.pyo",
        "__pycache__/*",
        ".git/*",
        ".svn/*",
        ".hg/*",
        "node_modules/*",
        "vendor/*",
        "venv/*",
        ".venv/*",
        "env/*",
        ".env/*",
        "dist/*",
        "build/*",
        "*.egg-info/*",
        ".tox/*",
        ".pytest_cache/*",
        ".mypy_cache/*",
        ".coverage",
        "coverage/*",
        "htmlcov/*",
        "*.log",
        "*.lock",
    ]

    def __init__(
        self,
        base_path: str | Path | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ):
        """Initialize the code parser.

        Args:
            base_path: Base directory to scan. Defaults to current directory.
            include_patterns: Glob patterns for files to include.
            exclude_patterns: Glob patterns for files to exclude.

        Raises:
            ValueError: If the path does not exist or is not a directory.
        """
        if base_path is not None:
            resolved = Path(base_path).resolve()
            if not resolved.exists():
                raise ValueError(f"Base path does not exist: {base_path}")
            if not resolved.is_dir():
                raise ValueError(f"Base path is not a directory: {base_path}")
            self.base_path = resolved
        else:
            self.base_path = Path.cwd()
        self.include_patterns = include_patterns or ["*"]
        self.exclude_patterns = exclude_patterns or self.DEFAULT_EXCLUDES

    def scan_files(
        self,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        languages: list[str] | None = None,
    ) -> list[Path]:
        """Scan for source files matching the patterns.

        Args:
            include_patterns: Override include patterns.
            exclude_patterns: Override exclude patterns.
            languages: Only include files of these languages.

        Returns:
            List of file paths.
        """
        include = include_patterns or self.include_patterns
        exclude = exclude_patterns or self.exclude_patterns

        # Collect all matching files
        files = []

        for pattern in include:
            if "**" in pattern:
                # Recursive glob
                matches = self.base_path.rglob(pattern.replace("**/", "").lstrip("/"))
            elif "/" in pattern:
                # Path-like pattern
                matches = self.base_path.glob(pattern)
            else:
                # Simple pattern - search recursively
                matches = self.base_path.rglob(pattern)

            for path in matches:
                if not path.is_file():
                    continue

                # Check excludes
                rel_path = str(path.relative_to(self.base_path))
                if self._is_excluded(rel_path, exclude):
                    continue

                # Check language filter
                if languages:
                    lang = self._detect_language(path)
                    if lang not in languages:
                        continue

                files.append(path)

        return sorted(set(files))

    def _is_excluded(self, rel_path: str, exclude_patterns: list[str]) -> bool:
        """Check if a path matches any exclude pattern.

        Args:
            rel_path: Relative file path.
            exclude_patterns: List of exclude patterns.

        Returns:
            True if the path should be excluded.
        """
        for pattern in exclude_patterns:
            if fnmatch(rel_path, pattern):
                return True
            # Also check against just the filename
            if fnmatch(Path(rel_path).name, pattern):
                return True

        return False

    def _detect_language(self, path: Path) -> str | None:
        """Detect programming language from file extension.

        Args:
            path: File path.

        Returns:
            Language identifier or None.
        """
        return self.LANGUAGE_MAP.get(path.suffix.lower())

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for change detection.

        Args:
            content: File content.

        Returns:
            SHA256 hash (first 16 chars).
        """
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def parse_file(self, path: Path | str) -> ParsedFile | None:
        """Parse a single source file.

        Args:
            path: Path to the file.

        Returns:
            ParsedFile object or None if file cannot be parsed.
        """
        path = Path(path)
        if not path.exists():
            return None

        language = self._detect_language(path)
        if not language:
            return None

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        stat = path.stat()

        parsed = ParsedFile(
            path=str(path),
            language=language,
            content_hash=self._compute_hash(content),
            size_bytes=stat.st_size,
            line_count=content.count("\n") + 1,
            last_modified=datetime.fromtimestamp(stat.st_mtime),
        )

        # Extract elements based on language
        if language == "python":
            parsed.elements = self._parse_python(content)
            parsed.imports = self._extract_python_imports(content)
        elif language in ("javascript", "typescript"):
            parsed.elements = self._parse_javascript(content)
            parsed.imports = self._extract_js_imports(content)
        elif language == "go":
            parsed.elements = self._parse_go(content)
            parsed.imports = self._extract_go_imports(content)
        elif language == "ruby":
            parsed.elements = self._parse_ruby(content)
            parsed.imports = self._extract_ruby_imports(content)
        else:
            # Generic parsing for other languages
            parsed.elements = self._parse_generic(content)

        # Extract significant comments
        parsed.comments = self._extract_significant_comments(content, language)

        return parsed

    def parse_files(self, paths: list[Path] | None = None) -> list[ParsedFile]:
        """Parse multiple source files.

        Args:
            paths: List of file paths. If None, scans using include/exclude patterns.

        Returns:
            List of ParsedFile objects.
        """
        if paths is None:
            paths = self.scan_files()

        parsed_files = []
        for path in paths:
            parsed = self.parse_file(path)
            if parsed:
                parsed_files.append(parsed)

        return parsed_files

    def _parse_python(self, content: str) -> list[CodeElement]:
        """Parse Python source code.

        Args:
            content: Python source code.

        Returns:
            List of CodeElement objects.
        """
        elements = []
        lines = content.split("\n")

        # Track decorators
        pending_decorators: list[str] = []
        current_class: str | None = None

        # Class pattern: class Name(bases):
        class_pattern = re.compile(r"^class\s+(\w+)(?:\s*\([^)]*\))?\s*:")

        # Function pattern: def name(args):
        func_pattern = re.compile(r"^(\s*)def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*[^:]+)?\s*:")

        # Decorator pattern: @decorator or @decorator(...)
        decorator_pattern = re.compile(r"^(\s*)@(\w+(?:\.\w+)*)(?:\([^)]*\))?")

        for i, line in enumerate(lines):
            line_num = i + 1

            # Check for decorator
            dec_match = decorator_pattern.match(line)
            if dec_match:
                pending_decorators.append(dec_match.group(2))
                continue

            # Check for class
            class_match = class_pattern.match(line)
            if class_match:
                class_name = class_match.group(1)
                current_class = class_name

                # Look for docstring
                docstring = self._extract_docstring_after(lines, i + 1)

                elements.append(
                    CodeElement(
                        element_type="class",
                        name=class_name,
                        docstring=docstring,
                        signature=line.strip(),
                        start_line=line_num,
                        decorators=pending_decorators.copy(),
                    )
                )
                pending_decorators.clear()
                continue

            # Check for function/method
            func_match = func_pattern.match(line)
            if func_match:
                indent = func_match.group(1)
                func_name = func_match.group(2)
                args = func_match.group(3)

                # Determine if it's a method (indented) or function
                is_method = len(indent) > 0 and current_class

                # Look for docstring
                docstring = self._extract_docstring_after(lines, i + 1)

                elements.append(
                    CodeElement(
                        element_type="method" if is_method else "function",
                        name=func_name,
                        docstring=docstring,
                        signature=f"def {func_name}({args})",
                        start_line=line_num,
                        decorators=pending_decorators.copy(),
                        parent=current_class if is_method else None,
                    )
                )
                pending_decorators.clear()

            # Reset class context if we hit a non-indented line
            if line and not line[0].isspace() and not line.startswith(("def", "class", "@")):
                current_class = None

        return elements

    def _extract_docstring_after(self, lines: list[str], start_idx: int) -> str | None:
        """Extract docstring from lines following a definition.

        Args:
            lines: All source lines.
            start_idx: Index to start looking from.

        Returns:
            Docstring content or None.
        """
        if start_idx >= len(lines):
            return None

        # Skip blank lines
        while start_idx < len(lines) and not lines[start_idx].strip():
            start_idx += 1

        if start_idx >= len(lines):
            return None

        line = lines[start_idx].strip()

        # Single-line docstring: """content"""
        if line.startswith('"""') and line.endswith('"""') and len(line) > 6:
            return line[3:-3].strip()
        if line.startswith("'''") and line.endswith("'''") and len(line) > 6:
            return line[3:-3].strip()

        # Multi-line docstring
        if line.startswith('"""') or line.startswith("'''"):
            quote = line[:3]
            docstring_lines = [line[3:]]
            start_idx += 1

            while start_idx < len(lines):
                doc_line = lines[start_idx]
                if quote in doc_line:
                    # End of docstring
                    end_idx = doc_line.find(quote)
                    docstring_lines.append(doc_line[:end_idx])
                    break
                docstring_lines.append(doc_line)
                start_idx += 1

            return "\n".join(docstring_lines).strip()

        return None

    def _extract_python_imports(self, content: str) -> list[str]:
        """Extract import statements from Python code.

        Args:
            content: Python source code.

        Returns:
            List of imported module names.
        """
        imports = []

        # import module
        import_pattern = re.compile(r"^import\s+(\S+)", re.MULTILINE)
        for match in import_pattern.finditer(content):
            imports.append(match.group(1).split(",")[0].strip())

        # from module import ...
        from_pattern = re.compile(r"^from\s+(\S+)\s+import", re.MULTILINE)
        for match in from_pattern.finditer(content):
            imports.append(match.group(1))

        return list(set(imports))

    def _parse_javascript(self, content: str) -> list[CodeElement]:
        """Parse JavaScript/TypeScript source code.

        Args:
            content: JavaScript/TypeScript source code.

        Returns:
            List of CodeElement objects.
        """
        elements = []
        lines = content.split("\n")

        # Class pattern: class Name extends Base { or class Name {
        class_pattern = re.compile(
            r"^(?:export\s+)?(?:default\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{"
        )

        # Function patterns
        func_patterns = [
            # function name(args)
            re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)"),
            # const name = (args) =>
            re.compile(r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>"),
            # const name = function(args)
            re.compile(r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?function\s*\(([^)]*)\)"),
        ]

        for i, line in enumerate(lines):
            line_num = i + 1
            line = line.strip()

            # Check for class
            class_match = class_pattern.match(line)
            if class_match:
                class_name = class_match.group(1)
                docstring = self._extract_jsdoc_before(lines, i)

                elements.append(
                    CodeElement(
                        element_type="class",
                        name=class_name,
                        docstring=docstring,
                        signature=line,
                        start_line=line_num,
                    )
                )
                continue

            # Check for functions
            for pattern in func_patterns:
                func_match = pattern.match(line)
                if func_match:
                    func_name = func_match.group(1)
                    args = func_match.group(2)
                    docstring = self._extract_jsdoc_before(lines, i)

                    elements.append(
                        CodeElement(
                            element_type="function",
                            name=func_name,
                            docstring=docstring,
                            signature=f"{func_name}({args})",
                            start_line=line_num,
                        )
                    )
                    break

        return elements

    def _extract_jsdoc_before(self, lines: list[str], line_idx: int) -> str | None:
        """Extract JSDoc comment before a definition.

        Args:
            lines: All source lines.
            line_idx: Index of the definition line.

        Returns:
            JSDoc content or None.
        """
        if line_idx == 0:
            return None

        # Look for /** ... */ before the definition
        doc_lines = []
        i = line_idx - 1

        # Skip blank lines
        while i >= 0 and not lines[i].strip():
            i -= 1

        if i < 0:
            return None

        # Check if previous line ends a JSDoc
        if not lines[i].strip().endswith("*/"):
            return None

        # Collect JSDoc lines
        while i >= 0:
            line = lines[i].strip()
            doc_lines.insert(0, line)
            if line.startswith("/**"):
                break
            i -= 1

        if not doc_lines:
            return None

        # Clean up JSDoc
        content = "\n".join(doc_lines)
        content = re.sub(r"^/\*\*\s*", "", content)
        content = re.sub(r"\s*\*/$", "", content)
        content = re.sub(r"^\s*\*\s?", "", content, flags=re.MULTILINE)

        return content.strip() if content.strip() else None

    def _extract_js_imports(self, content: str) -> list[str]:
        """Extract import statements from JavaScript/TypeScript.

        Args:
            content: Source code.

        Returns:
            List of imported module names.
        """
        imports = []

        # import ... from 'module'
        import_pattern = re.compile(r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]")
        for match in import_pattern.finditer(content):
            imports.append(match.group(1))

        # require('module')
        require_pattern = re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
        for match in require_pattern.finditer(content):
            imports.append(match.group(1))

        return list(set(imports))

    def _parse_go(self, content: str) -> list[CodeElement]:
        """Parse Go source code.

        Args:
            content: Go source code.

        Returns:
            List of CodeElement objects.
        """
        elements = []
        lines = content.split("\n")

        # Struct pattern: type Name struct {
        struct_pattern = re.compile(r"^type\s+(\w+)\s+struct\s*\{")

        # Interface pattern: type Name interface {
        interface_pattern = re.compile(r"^type\s+(\w+)\s+interface\s*\{")

        # Function pattern: func name(args) returns {
        func_pattern = re.compile(r"^func\s+(\w+)\s*\(([^)]*)\)")

        # Method pattern: func (r Receiver) name(args) returns {
        method_pattern = re.compile(r"^func\s+\([^)]+\)\s+(\w+)\s*\(([^)]*)\)")

        for i, line in enumerate(lines):
            line_num = i + 1
            line = line.strip()

            # Check for struct
            struct_match = struct_pattern.match(line)
            if struct_match:
                docstring = self._extract_go_comment_before(lines, i)
                elements.append(
                    CodeElement(
                        element_type="class",
                        name=struct_match.group(1),
                        docstring=docstring,
                        signature=line,
                        start_line=line_num,
                    )
                )
                continue

            # Check for interface
            interface_match = interface_pattern.match(line)
            if interface_match:
                docstring = self._extract_go_comment_before(lines, i)
                elements.append(
                    CodeElement(
                        element_type="class",
                        name=interface_match.group(1),
                        docstring=docstring,
                        signature=line,
                        start_line=line_num,
                    )
                )
                continue

            # Check for method
            method_match = method_pattern.match(line)
            if method_match:
                docstring = self._extract_go_comment_before(lines, i)
                elements.append(
                    CodeElement(
                        element_type="method",
                        name=method_match.group(1),
                        docstring=docstring,
                        signature=line,
                        start_line=line_num,
                    )
                )
                continue

            # Check for function
            func_match = func_pattern.match(line)
            if func_match:
                docstring = self._extract_go_comment_before(lines, i)
                elements.append(
                    CodeElement(
                        element_type="function",
                        name=func_match.group(1),
                        docstring=docstring,
                        signature=line,
                        start_line=line_num,
                    )
                )

        return elements

    def _extract_go_comment_before(self, lines: list[str], line_idx: int) -> str | None:
        """Extract Go comment before a definition.

        Args:
            lines: All source lines.
            line_idx: Index of the definition line.

        Returns:
            Comment content or None.
        """
        if line_idx == 0:
            return None

        doc_lines = []
        i = line_idx - 1

        # Skip blank lines
        while i >= 0 and not lines[i].strip():
            i -= 1

        # Collect // comment lines
        while i >= 0:
            line = lines[i].strip()
            if line.startswith("//"):
                doc_lines.insert(0, line[2:].strip())
                i -= 1
            else:
                break

        if not doc_lines:
            return None

        return "\n".join(doc_lines).strip()

    def _extract_go_imports(self, content: str) -> list[str]:
        """Extract import statements from Go code.

        Args:
            content: Go source code.

        Returns:
            List of imported packages.
        """
        imports = []

        # Single import: import "package"
        single_pattern = re.compile(r'import\s+"([^"]+)"')
        for match in single_pattern.finditer(content):
            imports.append(match.group(1))

        # Group imports: import ( "package" )
        group_pattern = re.compile(r"import\s*\((.*?)\)", re.DOTALL)
        for group in group_pattern.finditer(content):
            for line in group.group(1).split("\n"):
                pkg_match = re.search(r'"([^"]+)"', line)
                if pkg_match:
                    imports.append(pkg_match.group(1))

        return list(set(imports))

    def _parse_ruby(self, content: str) -> list[CodeElement]:
        """Parse Ruby source code.

        Args:
            content: Ruby source code.

        Returns:
            List of CodeElement objects.
        """
        elements = []
        lines = content.split("\n")
        current_class: str | None = None

        # Class pattern: class Name < Base or class Name
        class_pattern = re.compile(r"^\s*class\s+(\w+)(?:\s*<\s*\w+)?")

        # Module pattern: module Name
        module_pattern = re.compile(r"^\s*module\s+(\w+)")

        # Method pattern: def name(args) or def name
        method_pattern = re.compile(r"^(\s*)def\s+(\w+[!?=]?)(?:\s*\(([^)]*)\))?")

        for i, line in enumerate(lines):
            line_num = i + 1

            # Check for class
            class_match = class_pattern.match(line)
            if class_match:
                class_name = class_match.group(1)
                current_class = class_name
                docstring = self._extract_ruby_comment_before(lines, i)

                elements.append(
                    CodeElement(
                        element_type="class",
                        name=class_name,
                        docstring=docstring,
                        signature=line.strip(),
                        start_line=line_num,
                    )
                )
                continue

            # Check for module
            module_match = module_pattern.match(line)
            if module_match:
                module_name = module_match.group(1)
                current_class = module_name
                docstring = self._extract_ruby_comment_before(lines, i)

                elements.append(
                    CodeElement(
                        element_type="class",
                        name=module_name,
                        docstring=docstring,
                        signature=line.strip(),
                        start_line=line_num,
                    )
                )
                continue

            # Check for method
            method_match = method_pattern.match(line)
            if method_match:
                indent = method_match.group(1)
                method_name = method_match.group(2)
                args = method_match.group(3) or ""
                is_method = len(indent) > 0 and current_class
                docstring = self._extract_ruby_comment_before(lines, i)

                elements.append(
                    CodeElement(
                        element_type="method" if is_method else "function",
                        name=method_name,
                        docstring=docstring,
                        signature=f"def {method_name}({args})" if args else f"def {method_name}",
                        start_line=line_num,
                        parent=current_class if is_method else None,
                    )
                )

        return elements

    def _extract_ruby_comment_before(self, lines: list[str], line_idx: int) -> str | None:
        """Extract Ruby comment before a definition.

        Args:
            lines: All source lines.
            line_idx: Index of the definition line.

        Returns:
            Comment content or None.
        """
        if line_idx == 0:
            return None

        doc_lines = []
        i = line_idx - 1

        # Skip blank lines
        while i >= 0 and not lines[i].strip():
            i -= 1

        # Collect # comment lines
        while i >= 0:
            line = lines[i].strip()
            if line.startswith("#"):
                doc_lines.insert(0, line[1:].strip())
                i -= 1
            else:
                break

        if not doc_lines:
            return None

        return "\n".join(doc_lines).strip()

    def _extract_ruby_imports(self, content: str) -> list[str]:
        """Extract require statements from Ruby code.

        Args:
            content: Ruby source code.

        Returns:
            List of required files/gems.
        """
        imports = []

        # require 'gem' or require "file"
        require_pattern = re.compile(r"require\s+['\"]([^'\"]+)['\"]")
        for match in require_pattern.finditer(content):
            imports.append(match.group(1))

        # require_relative 'file'
        require_rel_pattern = re.compile(r"require_relative\s+['\"]([^'\"]+)['\"]")
        for match in require_rel_pattern.finditer(content):
            imports.append(match.group(1))

        return list(set(imports))

    def _parse_generic(self, content: str) -> list[CodeElement]:
        """Generic parsing for languages without specific support.

        Args:
            content: Source code.

        Returns:
            List of CodeElement objects.
        """
        # Try to find basic patterns
        elements = []
        lines = content.split("\n")

        # Generic function patterns
        func_patterns = [
            # func name(
            re.compile(r"^\s*(?:public|private|protected)?\s*func\s+(\w+)\s*\("),
            # def name(
            re.compile(r"^\s*def\s+(\w+)\s*\("),
            # function name(
            re.compile(r"^\s*function\s+(\w+)\s*\("),
            # void/int/etc name(
            re.compile(r"^\s*(?:void|int|string|bool|float)\s+(\w+)\s*\("),
        ]

        for i, line in enumerate(lines):
            line_num = i + 1
            for pattern in func_patterns:
                match = pattern.match(line)
                if match:
                    elements.append(
                        CodeElement(
                            element_type="function",
                            name=match.group(1),
                            signature=line.strip(),
                            start_line=line_num,
                        )
                    )
                    break

        return elements

    def _extract_significant_comments(self, content: str, language: str) -> list[str]:
        """Extract significant standalone comments (TODO, FIXME, NOTE, etc.).

        Args:
            content: Source code.
            language: Programming language.

        Returns:
            List of significant comments.
        """
        comments = []

        # Common comment markers
        markers = ["TODO", "FIXME", "NOTE", "HACK", "XXX", "BUG", "OPTIMIZE"]
        markers_pattern = "|".join(markers)
        marker_pattern = re.compile(
            rf"(?://|#|/\*)\s*({markers_pattern})\s*:?\s*(.+?)(?:\*/)?$",
            re.MULTILINE | re.IGNORECASE,
        )

        for match in marker_pattern.finditer(content):
            marker = match.group(1).upper()
            text = match.group(2).strip()
            if text:
                comments.append(f"[{marker}] {text}")

        return comments

    def get_file_hash(self, path: Path | str) -> str | None:
        """Get content hash for a file without fully parsing it.

        Args:
            path: File path.

        Returns:
            Content hash or None if file cannot be read.
        """
        path = Path(path)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            return self._compute_hash(content)
        except OSError:
            return None
