import importlib
import os
import sys

import pytest


@pytest.mark.integration
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
