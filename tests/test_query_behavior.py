import importlib
import os
import sys
from unittest.mock import Mock, patch

import pytest


@pytest.fixture(scope="module")
def helper_module():
    """Import the Langchain_helper module, but only include mock modules for simple query testing."""
    os.environ.setdefault("GOOGLE_API_KEY", "test-key")

    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_llm_cls, \
         patch("langchain_huggingface.HuggingFaceEmbeddings") as mock_embedding_cls:

        mock_llm_cls.return_value = Mock(name="MockLLM")
        mock_embedding_cls.return_value = Mock(name="MockEmbedding")

        # Import Langchain module (but remove existing instances first to avoid conflicts)
        sys.modules.pop("Langchain_helper", None)
        module = importlib.import_module("Langchain_helper")
        return module


def test_query_returns_empty_list_when_no_documents_match(helper_module):
    """A query with no matching documents should return an empty doc list cleanly."""
    fake_vector_db = Mock()
    fake_vector_db.similarity_search.return_value = []

    with patch.object(helper_module.os.path, "exists", return_value=True):
        with patch.object(helper_module.FAISS, "load_local", return_value=fake_vector_db):
            docs = helper_module.get_retrieved_documents(
                "Find a vegan restaurant in Croydon",
                k=5,
            )

    assert docs == []
    fake_vector_db.similarity_search.assert_called_once_with(
        "Find a vegan restaurant in Croydon",
        k=5,
    )


def test_query_returns_top_5_documents_from_faiss(helper_module):
    """A query should return the top 5 documents when the FAISS mock is populated."""
    fake_docs = [
        Mock(page_content=f"Restaurant {index}", metadata={"knowledge_source": "tiktok"})
        for index in range(5)
    ]
    fake_vector_db = Mock()
    fake_vector_db.similarity_search.return_value = fake_docs

    with patch.object(helper_module.os.path, "exists", return_value=True):
        with patch.object(helper_module.FAISS, "load_local", return_value=fake_vector_db):
            docs = helper_module.get_retrieved_documents(
                "Recommend a Croydon restaurant",
                k=5,
            )

    assert len(docs) == 5
    assert [doc.page_content for doc in docs] == [
        "Restaurant 0",
        "Restaurant 1",
        "Restaurant 2",
        "Restaurant 3",
        "Restaurant 4",
    ]
    fake_vector_db.similarity_search.assert_called_once_with(
        "Recommend a Croydon restaurant",
        k=5,
    )
