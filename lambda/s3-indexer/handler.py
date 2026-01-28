"""
Lambda function handler for processing S3 document uploads and building vector indices.

This function is triggered by S3 ObjectCreated events, downloads the .txt file,
chunks it, generates embeddings via OpenAI, and persists the index to S3 vector store.
"""

import asyncio
import json
import logging
import os
import tempfile
from urllib.parse import unquote_plus

import aioboto3
import boto3

# import nest_asyncio
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
import tiktoken
from llama_index.vector_stores.s3 import S3VectorStore

# # Apply nest_asyncio to allow nested event loops
# nest_asyncio.apply()

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
secrets_client = boto3.client("secretsmanager")

# Environment variables
S3_VECTORS_BUCKET_ARN = os.environ["S3_VECTORS_BUCKET_ARN"]
S3_VECTORS_INDEX_ARN = os.environ["S3_VECTORS_INDEX_ARN"]
OPENAI_SECRET_ARN = os.environ["OPENAI_SECRET_ARN"]
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "120"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_INDEX_DISTANCE_METRIC = os.environ.get("VECTOR_INDEX_DISTANCE_METRIC", "cosine")
VECTOR_INDEX_DATA_TYPE = os.environ.get("VECTOR_INDEX_DATA_TYPE", "float32")


def setup_llama_index() -> VectorStoreIndex:
    """Configure LlamaIndex with the OpenAI embedding model."""
    Settings.embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL,
        api_key=get_openai_api_key(),
        embed_batch_size=10,
    )
    Settings.tokenizer = tiktoken.get_encoding("cl100k_base").encode

    logger.info(f"Configured embedding model: {EMBEDDING_MODEL}")

    # Connect to the remote index
    vector_store = S3VectorStore(
        index_name_or_arn=S3_VECTORS_INDEX_ARN,
        bucket_name_or_arn=S3_VECTORS_BUCKET_ARN,
        insert_batch_size=500,
        # the below values must be the same as the ones used in the vector store creation in terraform
        data_type=VECTOR_INDEX_DATA_TYPE,
        distance_metric=VECTOR_INDEX_DISTANCE_METRIC,
    )
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


s3_index = setup_llama_index()


def get_openai_api_key():
    """Retrieve OpenAI API key from AWS Secrets Manager."""
    try:
        response = secrets_client.get_secret_value(SecretId=OPENAI_SECRET_ARN)
        return response["SecretString"]
    except Exception as e:
        logger.error(f"Failed to retrieve OpenAI API key: {str(e)}")
        raise


# def download_file_from_s3(bucket: str, key: str, local_path: str) -> None:
#     """Download a file from S3 to local filesystem (sync version for backwards compatibility)."""
#     logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
#     s3_client.download_file(bucket, key, local_path)


async def download_file_from_s3_async(bucket: str, key: str, local_path: str) -> None:
    """Download a file from S3 to local filesystem asynchronously using aioboto3."""
    logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
    session = aioboto3.Session()
    async with session.client("s3") as s3:
        await s3.download_file(bucket, key, local_path)


async def process_document(file_path: str, index: VectorStoreIndex):
    """
    Process a document: load, chunk, embed, and index.

    Args:
        file_path: Path to the .txt file
        index: VectorStoreIndex to update

    Returns:
        Updated VectorStoreIndex
    """
    logger.info(f"Processing document: {file_path}")

    # Load document
    documents = await SimpleDirectoryReader(
        input_files=[file_path], exclude_hidden=False
    ).aload_data()

    logger.info(f"Loaded {len(documents)} document(s)")

    # Create an ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                paragraph_separator="\n\n",
            ),
            Settings.embed_model,
        ]
    )

    # Process documents into nodes
    logger.info("Creating text chunks and generating embeddings...")
    nodes = await pipeline.arun(documents=documents)
    logger.info(f"Created {len(nodes)} nodes with embeddings")

    # Insert nodes into index
    await index.ainsert_nodes(nodes)
    logger.info(f"Inserted {len(nodes)} nodes into index")


async def process_single_record(
    bucket_key: tuple[str, str], index: VectorStoreIndex
) -> tuple[bool, str, str]:
    """
    Process a single S3 record asynchronously.

    Args:
        bucket_key: Tuple of (bucket: str, key: str)
        index: VectorStoreIndex to update

    Returns:
        Tuple of (success: bool, key: str, error: Optional[str])
    """
    bucket, key = bucket_key
    try:
        logger.info(f"Processing file: s3://{bucket}/{key}")

        # Create a temporary directory for file download
        with tempfile.TemporaryDirectory() as temp_dir:
            local_file_path = os.path.join(temp_dir, os.path.basename(key))

            # Download file from S3 asynchronously using aioboto3
            await download_file_from_s3_async(bucket, key, local_file_path)

            # Process the document in thread pool (llama-index is sync: API calls, S3 writes)
            # await asyncio.to_thread(process_document, local_file_path, s3_index)
            await process_document(local_file_path, index)
            logger.info(f"Successfully processed {key}")
            return True, key, ""

    except Exception as e:
        logger.error(f"Error processing {key}: {str(e)}", exc_info=True)
        return False, key, str(e)


async def async_handler(event):
    """
    Async AWS Lambda handler implementation.

    Processes all S3 records concurrently for better performance.
    """
    logger.info(f"Received event: {json.dumps(event)}")
    # Create tasks for all records
    records = []
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        # Validate file extension
        if not key.lower().endswith(".txt"):
            logger.warning(f"Skipping non-.txt file: {key}")
            continue
        records.append((bucket, key))

    tasks = [process_single_record(bucket_key, s3_index) for bucket_key in records]

    # Process all records concurrently
    logger.info(f"Processing {len(tasks)} records concurrently...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Analyze results
    successes = []
    failures = []

    for result in results:
        if isinstance(result, BaseException):
            failures.append(("unknown", str(result)))
        else:
            success, key, error = result  # ty: ignore[not-iterable]
            if success:
                successes.append(key)
            else:
                failures.append((key, error))

    # Log summary
    logger.info(
        f"Processing complete: {len(successes)} succeeded, {len(failures)} failed"
    )
    if failures:
        for key, error in failures:
            logger.warning(f"Failed to process {key}: {error}")

    return {
        "statusCode": 200 if len(successes) > 0 else 500,
        "body": json.dumps(
            {
                "message": "Processing complete",
                "processed_count": len(successes),
                "failed_count": len(failures),
                "total_records": len(event["Records"]),
            }
        ),
    }


def lambda_handler(event, context):
    """
    AWS Lambda handler function.

    Triggered by S3 ObjectCreated events for .txt files.
    Uses the existing event loop to run async operations concurrently.
    """
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(async_handler(event))
    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
