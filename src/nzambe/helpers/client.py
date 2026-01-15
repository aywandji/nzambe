import json
import os
from typing import Optional

import requests

from nzambe.constants import NZAMBE_SERVER_DEFAULT_BASE_URL
from nzambe.server.server import HealthResponse


def _get_base_url(explicit: Optional[str] = None) -> str:
    """Resolve the server base URL.

    Priority: explicit arg > NZAMBE_API_URL env var > NZAMBE_SERVER_DEFAULT_BASE_URL
    """
    if explicit:
        return explicit.rstrip("/")
    env_val = os.getenv("NZAMBE_API_URL")
    return (env_val or NZAMBE_SERVER_DEFAULT_BASE_URL).rstrip("/")


def health_check(base_url: Optional[str] = None) -> HealthResponse:
    """Call the server health endpoint and return the parsed status."""
    url = f"{_get_base_url(base_url)}/health"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return HealthResponse(**resp.json())


def query_server(
    question: str, base_url: Optional[str] = None, docs_only: bool = False
) -> tuple[str, list[str]]:
    """Send a question to the server and return the answer text.

    Expects FastAPI server to respond with JSON matching `QueryResponse`:
      {"answer": "..."}
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    if docs_only:
        url = f"{_get_base_url(base_url)}/retrieve_docs"
    else:
        url = f"{_get_base_url(base_url)}/query"

    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"question": question}),
        timeout=120,
    )
    # Raise for non-2xx; FastAPI provides useful error payloads which will be shown as HTTPError
    resp.raise_for_status()
    data = resp.json()
    answer = data.get("answer")
    docs = data.get("nodes")

    if not docs_only:
        if not isinstance(answer, str):
            raise ValueError(f"Unexpected response format: {data}")
    if not isinstance(docs, list):
        raise ValueError(f"Unexpected response format: {data}")

    return answer, docs
