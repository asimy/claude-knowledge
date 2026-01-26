"""Extract knowledge from Claude Code session transcripts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from claude_knowledge.session_parser import SessionMessage, SessionTranscript


@dataclass
class ExtractedKnowledge:
    """A piece of knowledge extracted from a session."""

    title: str
    description: str
    content: str
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    extraction_type: str = "general"  # "problem_solution", "decision", "pattern", "general"
    source_messages: list[str] = field(default_factory=list)  # Message UUIDs


class SessionExtractor:
    """Extract knowledge entries from parsed session transcripts."""

    # Question indicators for identifying problem statements
    QUESTION_PATTERNS = [
        r"\?",  # Direct questions
        r"\bhow\s+(do|can|should|would|to)\b",
        r"\bwhy\s+(does|is|are|do|can|would)\b",
        r"\bwhat\s+(is|are|does|should|would)\b",
        r"\bwhere\s+(is|are|do|does|should)\b",
        r"\bcan\s+you\b",
        r"\bcould\s+you\b",
        r"\bhelp\s+(me|with)\b",
        r"\bfix\b",
        r"\berror\b",
        r"\bfailing\b",
        r"\bnot\s+working\b",
        r"\bdoesn't\s+work\b",
        r"\bproblem\b",
        r"\bissue\b",
        r"\bimplement\b",
        r"\badd\b.*\b(feature|function|method)\b",
        r"\bcreate\b.*\b(function|class|file)\b",
    ]

    # Tool names that indicate code modifications
    CODE_TOOLS = {"Edit", "Write", "NotebookEdit"}

    # Tool names that indicate successful operations
    SUCCESS_TOOLS = {"Bash", "Read", "Glob", "Grep"}

    def __init__(self) -> None:
        """Initialize the session extractor."""
        self._question_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.QUESTION_PATTERNS
        ]

    def extract(self, transcript: SessionTranscript) -> list[ExtractedKnowledge]:
        """Extract knowledge entries from a session transcript.

        Args:
            transcript: Parsed session transcript.

        Returns:
            List of extracted knowledge entries.
        """
        extractions = []

        # Get conversation pairs
        pairs = transcript.get_conversation_pairs()
        if not pairs:
            return extractions

        # Process each user-assistant pair
        for i, (user_msg, assistant_msg) in enumerate(pairs):
            # Check if user message contains a question/problem
            if not self._is_question_or_problem(user_msg.text_content):
                continue

            # Check if assistant response contains a solution
            tool_uses = assistant_msg.tool_uses
            has_code = any(t.get("name") in self.CODE_TOOLS for t in tool_uses)

            # Look for follow-up messages to assess success
            success_indicators = self._check_success_indicators(pairs[i:] if i < len(pairs) else [])

            # Calculate confidence
            confidence = self._calculate_confidence(
                user_msg=user_msg,
                assistant_msg=assistant_msg,
                has_code=has_code,
                success_indicators=success_indicators,
            )

            # Only extract if confidence meets threshold
            if confidence < 0.3:
                continue

            # Generate extraction
            extraction = self._create_extraction(
                user_msg=user_msg,
                assistant_msg=assistant_msg,
                transcript=transcript,
                confidence=confidence,
                has_code=has_code,
            )
            if extraction:
                extractions.append(extraction)

        return extractions

    def _is_question_or_problem(self, text: str) -> bool:
        """Check if text contains a question or problem statement.

        Args:
            text: User message text.

        Returns:
            True if text appears to be asking a question or describing a problem.
        """
        if not text:
            return False

        text_lower = text.lower()

        # Check against patterns
        for pattern in self._question_patterns:
            if pattern.search(text_lower):
                return True

        return False

    def _check_success_indicators(
        self,
        remaining_pairs: list[tuple[SessionMessage, SessionMessage]],
    ) -> dict[str, bool]:
        """Check for indicators of successful resolution in follow-up messages.

        Args:
            remaining_pairs: Conversation pairs after the current one.

        Returns:
            Dict with success indicator flags.
        """
        indicators = {
            "user_acknowledged": False,
            "has_follow_up_corrections": False,
            "tool_success": False,
            "tool_error": False,
        }

        # Only look at the next few pairs
        for user_msg, _assistant_msg in remaining_pairs[:3]:
            user_text = user_msg.text_content.lower()

            # Check for acknowledgment
            ack_patterns = [
                r"\b(thanks|thank\s+you|perfect|great|works?|working)\b",
                r"\b(that('s)?|it('s)?)\s+(great|perfect|exactly|what\s+i)\b",
                r"^\s*(ok|okay|cool|nice|good|yes)\s*[.!]?\s*$",
            ]
            for pattern in ack_patterns:
                if re.search(pattern, user_text, re.IGNORECASE):
                    indicators["user_acknowledged"] = True
                    break

            # Check for corrections/follow-ups suggesting it didn't work
            correction_patterns = [
                r"\b(but|however|actually|still|doesn't|doesn't work)\b",
                r"\b(not quite|almost|close but)\b",
                r"\b(try again|one more|another)\b",
                r"\berror\b",
            ]
            for pattern in correction_patterns:
                if re.search(pattern, user_text, re.IGNORECASE):
                    indicators["has_follow_up_corrections"] = True
                    break

        return indicators

    def _calculate_confidence(
        self,
        user_msg: SessionMessage,
        assistant_msg: SessionMessage,
        has_code: bool,
        success_indicators: dict[str, bool],
    ) -> float:
        """Calculate confidence score for an extraction.

        Args:
            user_msg: User message.
            assistant_msg: Assistant response message.
            has_code: Whether the response includes code modifications.
            success_indicators: Dict of success indicator flags.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        score = 0.4  # Base score

        # Has code implementation: +0.15
        if has_code:
            score += 0.15

        # Tool results in conversation: +0.1
        if assistant_msg.tool_uses:
            score += 0.1

        # User acknowledged solution: +0.2
        if success_indicators.get("user_acknowledged"):
            score += 0.2

        # Has follow-up corrections: -0.15
        if success_indicators.get("has_follow_up_corrections"):
            score -= 0.15

        # Long, detailed response: +0.05
        if len(assistant_msg.text_content) > 500:
            score += 0.05

        # Clamp to valid range
        return max(0.0, min(1.0, score))

    def _create_extraction(
        self,
        user_msg: SessionMessage,
        assistant_msg: SessionMessage,
        transcript: SessionTranscript,
        confidence: float,
        has_code: bool,
    ) -> ExtractedKnowledge | None:
        """Create an ExtractedKnowledge entry from a message pair.

        Args:
            user_msg: User message with question/problem.
            assistant_msg: Assistant response message.
            transcript: Parent transcript for context.
            confidence: Calculated confidence score.
            has_code: Whether the response includes code.

        Returns:
            ExtractedKnowledge instance, or None if extraction fails.
        """
        # Generate title from user message
        title = self._generate_title(user_msg.text_content)
        if not title:
            return None

        # Generate description
        description = self._generate_description(user_msg.text_content, assistant_msg)

        # Extract content (code blocks and key explanation)
        content = self._extract_content(assistant_msg)
        if not content or len(content.strip()) < 20:
            return None

        # Derive tags
        tags = self._derive_tags(transcript, assistant_msg, content)

        # Determine extraction type
        extraction_type = "problem_solution" if has_code else "general"

        return ExtractedKnowledge(
            title=title,
            description=description,
            content=content,
            tags=tags,
            confidence=confidence,
            extraction_type=extraction_type,
            source_messages=[user_msg.uuid, assistant_msg.uuid],
        )

    def _generate_title(self, user_text: str) -> str:
        """Generate a title from the user's question/problem.

        Args:
            user_text: User message text.

        Returns:
            Generated title string.
        """
        if not user_text:
            return ""

        # Clean and truncate
        text = user_text.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            r"^(hi|hello|hey)[,!.]?\s*",
            r"^(can|could)\s+you\s+(please\s+)?",
            r"^(please\s+)?",
            r"^(i\s+(need|want)\s+(to|help)\s+)",
            r"^(help\s+me\s+)",
        ]
        for pattern in prefixes_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Take first sentence or line
        first_sentence = re.split(r"[.!?\n]", text)[0].strip()

        # Truncate if too long
        if len(first_sentence) > 80:
            # Try to break at word boundary
            truncated = first_sentence[:77]
            last_space = truncated.rfind(" ")
            if last_space > 40:
                truncated = truncated[:last_space]
            first_sentence = truncated + "..."

        # Capitalize first letter
        if first_sentence:
            first_sentence = first_sentence[0].upper() + first_sentence[1:]

        return first_sentence

    def _generate_description(
        self,
        user_text: str,
        assistant_msg: SessionMessage,
    ) -> str:
        """Generate a description for the knowledge entry.

        Args:
            user_text: User message text.
            assistant_msg: Assistant response message.

        Returns:
            Description string.
        """
        # Use the first part of the assistant's text response
        assistant_text = assistant_msg.text_content.strip()

        if assistant_text:
            # Take first paragraph or first few sentences
            paragraphs = assistant_text.split("\n\n")
            first_para = paragraphs[0].strip()

            # Truncate if too long
            if len(first_para) > 200:
                first_para = first_para[:197] + "..."

            return first_para

        # Fallback: describe what was done based on tools
        tools_used = [t.get("name", "") for t in assistant_msg.tool_uses]
        if tools_used:
            unique_tools = list(dict.fromkeys(tools_used))  # Preserve order, remove dupes
            return f"Solution using {', '.join(unique_tools[:3])}."

        return "Solution to the described problem."

    def _extract_content(self, assistant_msg: SessionMessage) -> str:
        """Extract the main content/solution from assistant message.

        Args:
            assistant_msg: Assistant response message.

        Returns:
            Extracted content string.
        """
        content_parts = []

        # Get text content
        text_content = assistant_msg.text_content.strip()
        if text_content:
            content_parts.append(text_content)

        # Extract code from tool uses (Edit/Write operations)
        for tool_use in assistant_msg.tool_uses:
            tool_name = tool_use.get("name", "")
            tool_input = tool_use.get("input", {})

            if tool_name == "Edit":
                old_str = tool_input.get("old_string", "")
                new_str = tool_input.get("new_string", "")
                file_path = tool_input.get("file_path", "")
                if new_str and new_str != old_str:
                    content_parts.append(f"\n\nEdit to {file_path}:\n```\n{new_str}\n```")

            elif tool_name == "Write":
                file_content = tool_input.get("content", "")
                file_path = tool_input.get("file_path", "")
                if file_content:
                    # Truncate very long file contents
                    if len(file_content) > 2000:
                        file_content = file_content[:2000] + "\n... (truncated)"
                    content_parts.append(f"\n\nFile {file_path}:\n```\n{file_content}\n```")

            elif tool_name == "Bash":
                command = tool_input.get("command", "")
                if command:
                    content_parts.append(f"\n\nCommand:\n```bash\n{command}\n```")

        return "\n".join(content_parts)

    def _derive_tags(
        self,
        transcript: SessionTranscript,
        assistant_msg: SessionMessage,
        content: str,
    ) -> list[str]:
        """Derive tags from the transcript and content.

        Args:
            transcript: Parent transcript.
            assistant_msg: Assistant response message.
            content: Extracted content.

        Returns:
            List of derived tags.
        """
        tags = set()

        # Add project name as tag (extract last component)
        if transcript.project_path:
            project_name = transcript.project_path.rstrip("/").split("/")[-1]
            if project_name:
                tags.add(project_name.lower())

        # Detect language from content
        language = self._detect_language(content)
        if language:
            tags.add(language)

        # Add tags based on tools used
        tool_names = {t.get("name", "").lower() for t in assistant_msg.tool_uses}
        if "bash" in tool_names:
            tags.add("cli")
        if "edit" in tool_names or "write" in tool_names:
            tags.add("code")

        # Detect common patterns/topics
        content_lower = content.lower()
        topic_patterns = {
            "api": r"\b(api|endpoint|rest|graphql)\b",
            "database": r"\b(database|sql|query|table|migration)\b",
            "testing": r"\b(test|spec|assert|expect|mock)\b",
            "auth": r"\b(auth|login|password|token|jwt|oauth)\b",
            "config": r"\b(config|setting|environment|env)\b",
            "error-handling": r"\b(error|exception|try|catch|raise)\b",
            "performance": r"\b(performance|optimize|cache|speed)\b",
            "refactor": r"\b(refactor|clean|improve|simplify)\b",
        }
        for tag, pattern in topic_patterns.items():
            if re.search(pattern, content_lower, re.IGNORECASE):
                tags.add(tag)

        return sorted(tags)[:5]  # Limit to 5 tags

    def _detect_language(self, content: str) -> str | None:
        """Detect programming language from content.

        Args:
            content: Code content.

        Returns:
            Language identifier or None.
        """
        # Language detection patterns
        patterns = {
            "python": [
                r"\bdef\s+\w+\(",
                r"\bimport\s+\w+",
                r"\bfrom\s+\w+\s+import\b",
                r"\bclass\s+\w+[:\(]",
            ],
            "javascript": [
                r"\bconst\s+\w+\s*=",
                r"\blet\s+\w+\s*=",
                r"\bfunction\s+\w+\(",
                r"=>\s*\{",
            ],
            "typescript": [
                r":\s*(string|number|boolean|any)\b",
                r"\binterface\s+\w+",
                r"<\w+>",
            ],
            "go": [
                r"\bfunc\s+\w+\(",
                r"\bpackage\s+\w+",
                r"\bgo\s+(func|routine)\b",
            ],
            "ruby": [
                r"\bdef\s+\w+\b(?!\()",
                r"\bend\b",
                r"\bdo\s*\|",
                r"\.rb\b",
            ],
            "rust": [
                r"\bfn\s+\w+\(",
                r"\blet\s+mut\b",
                r"::\w+::",
            ],
            "bash": [
                r"^\s*#!.*\b(ba)?sh\b",
                r"\b(export|echo|cd|ls|grep)\b",
                r"\$\{?\w+\}?",
            ],
            "sql": [
                r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER)\b",
            ],
        }

        for language, lang_patterns in patterns.items():
            for pattern in lang_patterns:
                if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                    return language

        return None

    def identify_problem_solutions(
        self,
        transcript: SessionTranscript,
    ) -> list[ExtractedKnowledge]:
        """Identify problem-solution pairs in a transcript.

        This is a convenience method that filters extractions to only
        include problem-solution type entries.

        Args:
            transcript: Parsed session transcript.

        Returns:
            List of problem-solution extractions.
        """
        all_extractions = self.extract(transcript)
        return [e for e in all_extractions if e.extraction_type == "problem_solution"]
