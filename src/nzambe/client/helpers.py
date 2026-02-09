import json
import logging
import os
from typing import Optional

import boto3
import requests
from botocore.exceptions import ClientError

from nzambe.constants import NZAMBE_SERVER_DEFAULT_BASE_URL
from nzambe.utils import HealthResponse

logger = logging.getLogger(__name__)


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


def upload_txt_files_to_s3(
    folder_path: str, bucket_name: str, s3_prefix: str = ""
) -> dict:
    """
    Uploads all .txt files from a local folder to an S3 bucket,
    skipping files that already exist in the bucket.

    Args:
        folder_path: Path to the local folder containing .txt files.
        bucket_name: Name of the S3 bucket.
        s3_prefix: Optional prefix (folder path) in S3. E.g. "documents/books/"

    Returns:
        A summary dict with lists of uploaded and skipped files.
    """
    s3_client = boto3.client("s3")
    uploaded = []
    skipped = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        s3_key = f"{s3_prefix}{filename}" if s3_prefix else filename
        local_path = os.path.join(folder_path, filename)

        if _s3_object_exists(s3_client, bucket_name, s3_key):
            logger.debug(f"⏭️  Skipped (already exists): {s3_key}")
            skipped.append(filename)
        else:
            s3_client.upload_file(local_path, bucket_name, s3_key)
            logger.debug(f"✅ Uploaded: {s3_key}")
            uploaded.append(filename)

    logger.debug(f"\nDone — Uploaded: {len(uploaded)}, Skipped: {len(skipped)}")
    return {"uploaded": uploaded, "skipped": skipped}


def _s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    """Check if an object already exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise
