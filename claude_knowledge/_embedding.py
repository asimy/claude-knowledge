"""Embedding service for generating vector representations of text."""

from sentence_transformers import SentenceTransformer

from claude_knowledge.utils import sanitize_for_embedding


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

    pass


class EmbeddingService:
    """Handles embedding generation using sentence-transformers.

    This service manages the lazy loading of the embedding model and
    provides methods for generating embeddings from text.
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model: SentenceTransformer | None = None) -> None:
        """Initialize the embedding service.

        Args:
            model: Optional pre-loaded SentenceTransformer model.
                   If None, model will be lazy-loaded on first use.
        """
        self._model: SentenceTransformer | None = model

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model on first use.

        Returns:
            The initialized SentenceTransformer model.

        Raises:
            EmbeddingError: If the model fails to load.
        """
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.EMBEDDING_MODEL)
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load embedding model '{self.EMBEDDING_MODEL}': {e}"
                ) from e
        return self._model

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If text is empty after sanitization or encoding fails.
        """
        clean_text = sanitize_for_embedding(text)
        if not clean_text:
            raise EmbeddingError("Cannot generate embedding for empty text")

        try:
            embedding = self.model.encode(clean_text, convert_to_numpy=True)
            return embedding.tolist()
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to encode text: {e}") from e

    def create_embedding_text(self, title: str, description: str, content: str) -> str:
        """Create combined text for embedding generation.

        Args:
            title: Knowledge entry title.
            description: Knowledge entry description.
            content: Knowledge entry content.

        Returns:
            Combined text for embedding.
        """
        return f"{title}. {description}. {content}"
