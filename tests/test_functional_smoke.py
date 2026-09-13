import importlib
import os
import sys

import pytest


@pytest.mark.functional
def test_real_query_smoke():
    """Do a basic query with Google Gemini and check that a response is received."""

    # Only run the test if the GOOGLE_API_KEY is set
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        pytest.skip("GOOGLE_API_KEY is required for integration smoke tests")

    # Import the helper with the real environment
    sys.modules.pop("Langchain_helper", None)
    helper = importlib.import_module("Langchain_helper")

    # Build the vector store if the index is missing
    if not os.path.exists(helper.VECTOR_DB_PATH):
        helper.Create_combined_vector_DB()

    # Send a basic query
    chain = helper.get_QA_Chain()
    response = chain.invoke("Recommend a restaurant in Croydon")

    # Check for a response
    assert isinstance(response, str)
    assert len(response.strip()) > 0


@pytest.mark.functional
def test_real_retrieval_smoke():
    """Do a basic query with Google Gemini and check that some documents are retrieved"""

    # Only run the test if the GOOGLE_API_KEY is set
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        pytest.skip("GOOGLE_API_KEY is required for integration smoke tests")

    # Import the helper with the real environment
    sys.modules.pop("Langchain_helper", None)
    helper = importlib.import_module("Langchain_helper")

    # Build the vector store if the index is missing
    if not os.path.exists(helper.VECTOR_DB_PATH):
        helper.Create_combined_vector_DB()

    # Send a basic query and get the top 5 documents that were returned
    docs = helper.get_retrieved_documents("Recommend a restaurant in Croydon", k=5)

    # Check that we get a valid list of docs
    assert isinstance(docs, list)
    assert len(docs) >= 1
    # Make sure that the contents are sensible
    assert all(hasattr(doc, "page_content") and hasattr(doc, "metadata") for doc in docs)
    assert all(doc.page_content.strip() for doc in docs)
