import importlib
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


@pytest.fixture(scope="module")
def helper_module():
    """Import the repository helper under patched LLM/embedding constructors."""
    os.environ.setdefault("GOOGLE_API_KEY", "test-key")

    # Patch the classes that are instantiated when Langchain_helper is imported.
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_llm_cls, \
         patch("langchain_huggingface.HuggingFaceEmbeddings") as mock_embedding_cls:

        mock_llm_cls.return_value = Mock(name="MockLLM")
        mock_embedding_cls.return_value = Mock(name="MockEmbedding")

        # Ensure a clean import for deterministic test behavior.
        sys.modules.pop("Langchain_helper", None)
        module = importlib.import_module("Langchain_helper")
        return module


def test_load_csv_documents_missing_file(helper_module):
    """Missing CSV inputs should fail fast with a clear file error."""
    with pytest.raises(FileNotFoundError, match="not found"):
        helper_module.load_csv_documents("/tmp/does-not-exist.csv", "tiktok")


def test_load_csv_documents_populates_source_metadata(tmp_path, helper_module):
    """The loader should keep track of where each document came from."""
    csv_path = tmp_path / "restaurant.csv"
    csv_path.write_text(
        "restaurant,description\n"
        "The Green Room,A friendly vegetarian cafe in Croydon.\n",
        encoding="utf-8",
    )

    documents = helper_module.load_csv_documents(str(csv_path), "tiktok")

    assert len(documents) == 1
    assert documents[0].metadata["knowledge_source"] == "tiktok"
    assert "The Green Room" in documents[0].page_content


def test_get_retrieved_documents_requires_existing_faiss_index(helper_module):
    """Retrieval should stop cleanly when the FAISS knowledge base has not been created."""
    with patch.object(helper_module.os.path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="FAISS index does not exist"):
            helper_module.get_retrieved_documents("Indian restaurant", k=5)


def test_get_retrieved_documents_returns_similarity_docs(helper_module):
    """Retrieval should pass the user's search through to the FAISS vector store."""
    fake_docs = [Mock(page_content="restaurant", metadata={"knowledge_source": "tiktok"})]
    fake_vector_db = Mock()
    fake_vector_db.similarity_search.return_value = fake_docs

    with patch.object(helper_module.os.path, "exists", return_value=True):
        with patch.object(helper_module.FAISS, "load_local", return_value=fake_vector_db):
            docs = helper_module.get_retrieved_documents("Indian restaurant", k=5)

    assert docs == fake_docs
    fake_vector_db.similarity_search.assert_called_once_with("Indian restaurant", k=5)


def test_create_combined_vector_db_builds_and_saves_mock_faiss(helper_module):
    """The combined DB builder should load CSV documents, build a FAISS database, and save it locally."""
    fake_docs = [
        Mock(page_content="Restaurant data", metadata={"knowledge_source": "tiktok"}),
        Mock(page_content="Restaurant data", metadata={"knowledge_source": "reddit"}),
    ]

    with patch.object(helper_module, "load_csv_documents", side_effect=[fake_docs[:1], fake_docs[1:]]):
        helper_module.instruction_embedding.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        fake_faiss = Mock()
        fake_faiss.save_local.return_value = None

        with patch.object(helper_module.FAISS, "from_documents", return_value=fake_faiss):
            count = helper_module.Create_combined_vector_DB()

    assert count == 2
    fake_faiss.save_local.assert_called_once_with(helper_module.VECTOR_DB_PATH)


def test_get_qachain_requires_existing_faiss_index(helper_module):
    """The QA graph builder should guard against constructing a chain without a vector store."""
    with patch.object(helper_module.os.path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="FAISS index does not exist"):
            helper_module.get_QA_Chain()
