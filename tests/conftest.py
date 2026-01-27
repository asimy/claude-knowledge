"""Pytest configuration and shared fixtures."""

import shutil
import tempfile

import pytest
from sentence_transformers import SentenceTransformer

from claude_knowledge._embedding import EmbeddingService
from claude_knowledge.knowledge_manager import KnowledgeManager


@pytest.fixture(scope="session")
def embedding_model():
    """Load the embedding model once per test session.

    This avoids the ~1.5-2s model loading time for each test.
    """
    return SentenceTransformer(EmbeddingService.EMBEDDING_MODEL)


@pytest.fixture(scope="session")
def shared_embedding_service(embedding_model):
    """Create a shared EmbeddingService with the pre-loaded model.

    This service is shared across all tests in the session.
    """
    return EmbeddingService(model=embedding_model)


@pytest.fixture
def temp_km(shared_embedding_service):
    """Create a temporary knowledge manager with shared embedding service.

    This fixture provides a fresh KnowledgeManager for each test while
    reusing the same embedding model to avoid repeated model loading.
    """
    temp_dir = tempfile.mkdtemp()
    km = KnowledgeManager(base_path=temp_dir, embedding_service=shared_embedding_service)
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
