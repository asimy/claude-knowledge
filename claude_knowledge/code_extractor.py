"""Extract knowledge from codebase patterns and documentation."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from claude_knowledge.code_parser import CodeElement, CodeParser, ParsedFile


@dataclass
class ExtractedKnowledge:
    """A piece of knowledge extracted from code patterns."""

    title: str
    description: str
    content: str
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    extraction_type: str = "pattern"  # "pattern", "docstring", "architecture"
    source_files: list[str] = field(default_factory=list)


@dataclass
class PatternMatch:
    """A detected code pattern."""

    pattern_type: str  # "repository", "service", "factory", "singleton", etc.
    file_path: str
    element_name: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5


class CodeExtractor:
    """Extract knowledge from parsed codebase."""

    # Architectural pattern indicators
    PATTERN_INDICATORS = {
        "repository": {
            "name_patterns": [r"repo(?:sitory)?$", r"_repo$", r"Repo(?:sitory)?$"],
            "method_patterns": ["find", "get", "save", "delete", "create", "update", "all", "by"],
            "file_patterns": [r"repositor", r"_repo\."],
        },
        "service": {
            "name_patterns": [r"service$", r"_service$", r"Service$", r"Svc$"],
            "method_patterns": ["execute", "process", "handle", "perform", "run"],
            "file_patterns": [r"service", r"_svc\."],
        },
        "factory": {
            "name_patterns": [r"factory$", r"Factory$", r"_factory$"],
            "method_patterns": ["create", "make", "build", "new", "get_instance"],
            "file_patterns": [r"factor"],
        },
        "singleton": {
            "name_patterns": [r"singleton$", r"Singleton$"],
            "method_patterns": ["instance", "get_instance", "getInstance"],
            "file_patterns": [r"singleton"],
        },
        "controller": {
            "name_patterns": [r"controller$", r"Controller$", r"_controller$"],
            "method_patterns": ["index", "show", "create", "update", "destroy", "get", "post"],
            "file_patterns": [r"controller"],
        },
        "model": {
            "name_patterns": [r"Model$", r"Entity$", r"_model$"],
            "file_patterns": [r"models?/", r"entities/"],
        },
        "middleware": {
            "name_patterns": [r"middleware$", r"Middleware$"],
            "method_patterns": ["before", "after", "around", "process", "handle"],
            "file_patterns": [r"middleware"],
        },
        "handler": {
            "name_patterns": [r"handler$", r"Handler$", r"_handler$"],
            "method_patterns": ["handle", "process", "on_", "dispatch"],
            "file_patterns": [r"handler"],
        },
    }

    # Configuration patterns
    CONFIG_PATTERNS = {
        "environment": [r"env", r"environment", r"config"],
        "database": [r"database", r"db_", r"connection"],
        "api": [r"api_", r"endpoint", r"url"],
        "auth": [r"auth", r"secret", r"key", r"token"],
        "logging": [r"log", r"logger"],
    }

    # Error handling patterns
    ERROR_PATTERNS = {
        "custom_exception": r"(?:class|def)\s+\w*(?:Error|Exception)",
        "try_catch": r"(?:try|begin|rescue|except|catch)",
        "error_handler": r"(?:handle_error|on_error|error_handler)",
    }

    def __init__(self, parser: CodeParser | None = None):
        """Initialize the code extractor.

        Args:
            parser: CodeParser instance. Creates default if None.
        """
        self.parser = parser or CodeParser()

        # Compile patterns
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency.

        Only name_patterns and file_patterns are compiled as regex.
        method_patterns are kept as strings for simple substring matching.
        """
        self._pattern_indicators: dict[str, dict[str, list]] = {}

        for pattern_type, indicators in self.PATTERN_INDICATORS.items():
            self._pattern_indicators[pattern_type] = {}
            for key, patterns in indicators.items():
                if key in ("name_patterns", "file_patterns"):
                    # Compile regex patterns
                    self._pattern_indicators[pattern_type][key] = [
                        re.compile(p, re.IGNORECASE) for p in patterns
                    ]
                else:
                    # Keep method_patterns as strings
                    self._pattern_indicators[pattern_type][key] = patterns

    def extract(
        self,
        parsed_files: list[ParsedFile],
        project_name: str | None = None,
    ) -> list[ExtractedKnowledge]:
        """Extract knowledge from parsed files.

        Args:
            parsed_files: List of ParsedFile objects.
            project_name: Optional project name for tagging.

        Returns:
            List of ExtractedKnowledge entries.
        """
        extractions = []

        # Extract from docstrings
        extractions.extend(self._extract_from_docstrings(parsed_files, project_name))

        # Detect architectural patterns
        extractions.extend(self._extract_architectural_patterns(parsed_files, project_name))

        # Extract from significant comments
        extractions.extend(self._extract_from_comments(parsed_files, project_name))

        return extractions

    def extract_from_codebase(
        self,
        base_path: str | Path | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        project_name: str | None = None,
    ) -> list[ExtractedKnowledge]:
        """Scan and extract knowledge from a codebase.

        Args:
            base_path: Base directory to scan.
            include_patterns: File patterns to include.
            exclude_patterns: File patterns to exclude.
            project_name: Optional project name.

        Returns:
            List of ExtractedKnowledge entries.
        """
        if base_path:
            self.parser = CodeParser(
                base_path=base_path,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )

        parsed_files = self.parser.parse_files()

        if not project_name and base_path:
            project_name = Path(base_path).name

        return self.extract(parsed_files, project_name)

    def _extract_from_docstrings(
        self,
        parsed_files: list[ParsedFile],
        project_name: str | None,
    ) -> list[ExtractedKnowledge]:
        """Extract knowledge from docstrings.

        Args:
            parsed_files: List of parsed files.
            project_name: Optional project name.

        Returns:
            List of docstring-based extractions.
        """
        extractions = []

        for parsed_file in parsed_files:
            for element in parsed_file.elements:
                if not element.docstring:
                    continue

                # Skip short docstrings (likely just one-liners)
                if len(element.docstring) < 50:
                    continue

                # Calculate confidence
                confidence = self._calculate_docstring_confidence(element, parsed_file)

                if confidence < 0.3:
                    continue

                # Generate title
                title = self._generate_docstring_title(element, parsed_file)

                # Generate description
                description = self._generate_docstring_description(element, parsed_file)

                # Generate content
                content = self._generate_docstring_content(element, parsed_file)

                # Derive tags
                tags = self._derive_tags(
                    [element],
                    [parsed_file],
                    project_name,
                    "docstring",
                )

                extractions.append(
                    ExtractedKnowledge(
                        title=title,
                        description=description,
                        content=content,
                        tags=tags,
                        confidence=confidence,
                        extraction_type="docstring",
                        source_files=[parsed_file.path],
                    )
                )

        return extractions

    def _calculate_docstring_confidence(
        self,
        element: CodeElement,
        parsed_file: ParsedFile,
    ) -> float:
        """Calculate confidence for a docstring extraction.

        Confidence scoring:
        - Base: 0.3
        - +0.15 documented with docstrings
        - +0.1 has examples or code blocks
        - +0.1 has parameter documentation
        - +0.1 has return value documentation
        - -0.1 very short docstring

        Args:
            element: Code element with docstring.
            parsed_file: Parsed file containing the element.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        score = 0.3  # Base score

        docstring = element.docstring or ""

        # Has substantial content: +0.15
        if len(docstring) > 100:
            score += 0.15

        # Has examples or code blocks: +0.1
        if "```" in docstring or ">>>" in docstring or "Example" in docstring:
            score += 0.1

        # Has parameter documentation: +0.1
        param_patterns = [r":param", r"@param", r"Args:", r"Parameters:"]
        if any(re.search(p, docstring) for p in param_patterns):
            score += 0.1

        # Has return documentation: +0.1
        return_patterns = [r":return", r"@return", r"Returns:", r"@returns"]
        if any(re.search(p, docstring) for p in return_patterns):
            score += 0.1

        # Very short: -0.1
        if len(docstring) < 100:
            score -= 0.1

        # Class or module level documentation is more valuable: +0.1
        if element.element_type == "class":
            score += 0.1

        return max(0.0, min(1.0, score))

    def _generate_docstring_title(
        self,
        element: CodeElement,
        parsed_file: ParsedFile,
    ) -> str:
        """Generate title for docstring knowledge.

        Args:
            element: Code element.
            parsed_file: Parsed file.

        Returns:
            Generated title.
        """
        type_prefix = {
            "class": "Class:",
            "function": "Function:",
            "method": "Method:",
        }.get(element.element_type, "")

        name = element.name
        if element.parent:
            name = f"{element.parent}.{name}"

        return f"{type_prefix} {name}".strip()

    def _generate_docstring_description(
        self,
        element: CodeElement,
        parsed_file: ParsedFile,
    ) -> str:
        """Generate description from docstring.

        Args:
            element: Code element.
            parsed_file: Parsed file.

        Returns:
            Generated description.
        """
        docstring = element.docstring or ""

        # Take first paragraph or first few sentences
        paragraphs = docstring.split("\n\n")
        first_para = paragraphs[0].strip()

        # Clean up
        first_para = re.sub(r"\s+", " ", first_para)

        # Truncate if needed
        if len(first_para) > 200:
            first_para = first_para[:197] + "..."

        return first_para

    def _generate_docstring_content(
        self,
        element: CodeElement,
        parsed_file: ParsedFile,
    ) -> str:
        """Generate content from element and docstring.

        Args:
            element: Code element.
            parsed_file: Parsed file.

        Returns:
            Generated content.
        """
        parts = []

        # Signature
        if element.signature:
            lang = parsed_file.language
            parts.append(f"```{lang}\n{element.signature}\n```")

        # Full docstring
        if element.docstring:
            parts.append(f"\n## Documentation\n\n{element.docstring}")

        # File location
        rel_path = Path(parsed_file.path).name
        parts.append(f"\n**File:** `{rel_path}` (line {element.start_line})")

        # Decorators
        if element.decorators:
            decs = ", ".join(f"`@{d}`" for d in element.decorators)
            parts.append(f"\n**Decorators:** {decs}")

        return "\n".join(parts)

    def _extract_architectural_patterns(
        self,
        parsed_files: list[ParsedFile],
        project_name: str | None,
    ) -> list[ExtractedKnowledge]:
        """Detect and extract architectural patterns.

        Args:
            parsed_files: List of parsed files.
            project_name: Optional project name.

        Returns:
            List of pattern-based extractions.
        """
        extractions = []

        # Detect patterns across files
        patterns = self._detect_patterns(parsed_files)

        # Group by pattern type
        patterns_by_type: dict[str, list[PatternMatch]] = defaultdict(list)
        for pattern in patterns:
            patterns_by_type[pattern.pattern_type].append(pattern)

        # Generate extraction for each pattern type with multiple instances
        for pattern_type, matches in patterns_by_type.items():
            if len(matches) < 2:  # Require at least 2 instances
                continue

            confidence = self._calculate_pattern_confidence(matches)
            if confidence < 0.3:
                continue

            title = self._generate_pattern_title(pattern_type, matches)
            description = self._generate_pattern_description(pattern_type, matches)
            content = self._generate_pattern_content(pattern_type, matches, parsed_files)
            tags = self._derive_tags(
                [],
                parsed_files,
                project_name,
                pattern_type,
            )

            extractions.append(
                ExtractedKnowledge(
                    title=title,
                    description=description,
                    content=content,
                    tags=tags,
                    confidence=confidence,
                    extraction_type="pattern",
                    source_files=[m.file_path for m in matches],
                )
            )

        return extractions

    def _detect_patterns(self, parsed_files: list[ParsedFile]) -> list[PatternMatch]:
        """Detect architectural patterns across files.

        Args:
            parsed_files: List of parsed files.

        Returns:
            List of pattern matches.
        """
        matches = []

        for parsed_file in parsed_files:
            for element in parsed_file.elements:
                if element.element_type != "class":
                    continue

                # Check against each pattern type
                for pattern_type, indicators in self._pattern_indicators.items():
                    evidence = []
                    match_score = 0

                    # Check name patterns
                    for pattern in indicators.get("name_patterns", []):
                        if pattern.search(element.name):
                            evidence.append(f"Name matches {pattern_type} pattern")
                            match_score += 1
                            break

                    # Check file path patterns
                    for pattern in indicators.get("file_patterns", []):
                        if pattern.search(parsed_file.path):
                            evidence.append(f"File path matches {pattern_type} pattern")
                            match_score += 1
                            break

                    # Check method patterns (look at methods in same file)
                    methods = [
                        e
                        for e in parsed_file.elements
                        if e.element_type == "method" and e.parent == element.name
                    ]
                    method_names = [m.name.lower() for m in methods]

                    for method_pattern in indicators.get("method_patterns", []):
                        if any(method_pattern in name for name in method_names):
                            evidence.append(f"Has {method_pattern} method")
                            match_score += 0.5

                    # Only consider it a match if we have enough evidence
                    if match_score >= 1.5:
                        confidence = min(1.0, match_score / 3)
                        matches.append(
                            PatternMatch(
                                pattern_type=pattern_type,
                                file_path=parsed_file.path,
                                element_name=element.name,
                                evidence=evidence,
                                confidence=confidence,
                            )
                        )

        return matches

    def _calculate_pattern_confidence(self, matches: list[PatternMatch]) -> float:
        """Calculate confidence for pattern extraction.

        Confidence scoring:
        - Base: 0.3
        - +0.2 multiple examples (3+ instances)
        - +0.1 consistent naming convention
        - +0.1 average match confidence > 0.5
        - -0.1 single instance only

        Args:
            matches: List of pattern matches.

        Returns:
            Confidence score.
        """
        score = 0.3  # Base

        # Multiple examples: +0.2
        if len(matches) >= 3:
            score += 0.2
        elif len(matches) == 1:
            score -= 0.1

        # Average match confidence: +0.1
        avg_confidence = sum(m.confidence for m in matches) / len(matches)
        if avg_confidence > 0.5:
            score += 0.1

        # Check naming consistency: +0.1
        names = [m.element_name for m in matches]
        if self._has_consistent_naming(names):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _has_consistent_naming(self, names: list[str]) -> bool:
        """Check if names follow a consistent pattern.

        Args:
            names: List of element names.

        Returns:
            True if naming is consistent.
        """
        if len(names) < 2:
            return False

        # Check for common suffix
        suffixes = set()
        for name in names:
            # Extract suffix (last word in CamelCase or snake_case)
            if "_" in name:
                suffix = name.split("_")[-1]
            else:
                # CamelCase: find last uppercase start
                parts = re.findall(r"[A-Z][a-z]*", name)
                suffix = parts[-1] if parts else name

            suffixes.add(suffix.lower())

        # Consistent if most have the same suffix
        return len(suffixes) == 1

    def _generate_pattern_title(
        self,
        pattern_type: str,
        matches: list[PatternMatch],
    ) -> str:
        """Generate title for pattern extraction.

        Args:
            pattern_type: Type of pattern.
            matches: Pattern matches.

        Returns:
            Generated title.
        """
        type_names = {
            "repository": "Repository Pattern",
            "service": "Service Layer Pattern",
            "factory": "Factory Pattern",
            "singleton": "Singleton Pattern",
            "controller": "Controller Pattern",
            "model": "Model/Entity Pattern",
            "middleware": "Middleware Pattern",
            "handler": "Handler Pattern",
        }

        return type_names.get(pattern_type, f"{pattern_type.title()} Pattern")

    def _generate_pattern_description(
        self,
        pattern_type: str,
        matches: list[PatternMatch],
    ) -> str:
        """Generate description for pattern extraction.

        Args:
            pattern_type: Type of pattern.
            matches: Pattern matches.

        Returns:
            Generated description.
        """
        num_instances = len(matches)
        example_names = ", ".join(m.element_name for m in matches[:3])
        if num_instances > 3:
            example_names += f", and {num_instances - 3} more"

        type_descriptions = {
            "repository": "Data access layer abstraction for database operations",
            "service": "Business logic layer for complex operations",
            "factory": "Object creation pattern for flexible instantiation",
            "singleton": "Single instance pattern for shared resources",
            "controller": "Request handling layer for HTTP endpoints",
            "model": "Data structure definitions and domain objects",
            "middleware": "Request/response pipeline processing",
            "handler": "Event or message processing pattern",
        }

        base_desc = type_descriptions.get(pattern_type, f"{pattern_type.title()} design pattern")
        return f"{base_desc}. Found {num_instances} instances: {example_names}."

    def _generate_pattern_content(
        self,
        pattern_type: str,
        matches: list[PatternMatch],
        parsed_files: list[ParsedFile],
    ) -> str:
        """Generate content for pattern extraction.

        Args:
            pattern_type: Type of pattern.
            matches: Pattern matches.
            parsed_files: All parsed files.

        Returns:
            Generated content.
        """
        parts = []

        parts.append(f"## {pattern_type.title()} Pattern Instances\n")

        # Create a lookup for parsed files
        file_lookup = {pf.path: pf for pf in parsed_files}

        for match in matches[:5]:  # Limit to 5 examples
            parts.append(f"\n### {match.element_name}\n")
            parts.append(f"**File:** `{Path(match.file_path).name}`\n")

            if match.evidence:
                parts.append("**Evidence:**\n")
                for ev in match.evidence:
                    parts.append(f"- {ev}")

            # Include signature if available
            parsed_file = file_lookup.get(match.file_path)
            if parsed_file:
                for element in parsed_file.elements:
                    if element.name == match.element_name:
                        if element.signature:
                            parts.append(f"\n```{parsed_file.language}")
                            parts.append(element.signature)
                            parts.append("```")
                        if element.docstring:
                            first_line = element.docstring.split("\n")[0].strip()
                            parts.append(f"\n_{first_line}_")
                        break

        if len(matches) > 5:
            parts.append(f"\n_... and {len(matches) - 5} more instances_")

        return "\n".join(parts)

    def _extract_from_comments(
        self,
        parsed_files: list[ParsedFile],
        project_name: str | None,
    ) -> list[ExtractedKnowledge]:
        """Extract knowledge from significant comments.

        Args:
            parsed_files: List of parsed files.
            project_name: Optional project name.

        Returns:
            List of comment-based extractions.
        """
        extractions = []

        # Group comments by type (TODO, FIXME, etc.)
        comments_by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for parsed_file in parsed_files:
            for comment in parsed_file.comments:
                # Extract type from [TYPE] prefix
                match = re.match(r"\[(\w+)\]\s*(.+)", comment)
                if match:
                    comment_type = match.group(1)
                    comment_text = match.group(2)
                    comments_by_type[comment_type].append((parsed_file.path, comment_text))

        # Create extraction for each type with multiple comments
        for comment_type, comments in comments_by_type.items():
            if len(comments) < 3:  # Require at least 3 comments
                continue

            # TODO comments are generally lower value
            base_confidence = 0.35 if comment_type == "TODO" else 0.45

            title = f"Code {comment_type}s"
            description = f"Found {len(comments)} {comment_type} comments across the codebase."

            content_parts = [f"## {comment_type} Comments\n"]
            for file_path, text in comments[:20]:  # Limit
                rel_path = Path(file_path).name
                content_parts.append(f"- `{rel_path}`: {text}")

            if len(comments) > 20:
                content_parts.append(f"\n_... and {len(comments) - 20} more_")

            tags = [comment_type.lower()]
            if project_name:
                tags.append(project_name.lower())

            extractions.append(
                ExtractedKnowledge(
                    title=title,
                    description=description,
                    content="\n".join(content_parts),
                    tags=tags,
                    confidence=base_confidence,
                    extraction_type="comment",
                    source_files=list({c[0] for c in comments}),
                )
            )

        return extractions

    def _derive_tags(
        self,
        elements: list[CodeElement],
        parsed_files: list[ParsedFile],
        project_name: str | None,
        extraction_type: str,
    ) -> list[str]:
        """Derive tags from extraction context.

        Args:
            elements: Code elements involved.
            parsed_files: Parsed files involved.
            project_name: Optional project name.
            extraction_type: Type of extraction.

        Returns:
            List of derived tags.
        """
        tags = set()

        # Add project name
        if project_name:
            tags.add(project_name.lower())

        # Add languages
        for pf in parsed_files:
            tags.add(pf.language)

        # Add extraction type
        if extraction_type not in ("docstring", "general"):
            tags.add(extraction_type.replace("_", "-"))

        # Add element type for docstrings
        for element in elements:
            if element.element_type:
                tags.add(element.element_type)

        return sorted(tags)[:5]

    def detect_all_patterns(self, parsed_files: list[ParsedFile]) -> list[PatternMatch]:
        """Detect all patterns across files.

        Public method for testing and inspection.

        Args:
            parsed_files: List of parsed files.

        Returns:
            List of all pattern matches.
        """
        return self._detect_patterns(parsed_files)
