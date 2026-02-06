"""
Unit tests for the Lambda S3 indexer handler.

Tests cover the main features and behaviors of the handler using pytest and unittest.mock.
"""

import os

os.environ["S3_VECTORS_BUCKET_NAME"] = "test"
os.environ["S3_VECTORS_INDEX_NAME"] = "test"
os.environ["OPENAI_SECRET_ARN"] = "test"
os.environ["CHUNK_SIZE"] = "10"
os.environ["CHUNK_OVERLAP"] = "2"
os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"
os.environ["VECTOR_INDEX_DISTANCE_METRIC"] = "test"
os.environ["VECTOR_INDEX_DATA_TYPE"] = "test"

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest

# Module under test
import handler  # ty: ignore[unresolved-import]


class TestGetOpenAIApiKey:
    """Tests for get_openai_api_key function."""

    @patch("handler.secrets_client")
    def test_successfully_retrieves_api_key(self, mock_secrets_client):
        """Test successful retrieval of API key from Secrets Manager."""
        expected_key = "sk-test-api-key-12345"
        mock_secrets_client.get_secret_value.return_value = {
            "SecretString": expected_key
        }

        result = handler.get_openai_api_key("test_secret_arn")

        assert result == expected_key
        mock_secrets_client.get_secret_value.assert_called_once_with(
            SecretId="test_secret_arn"
        )

    @patch("handler.secrets_client")
    @patch("handler.logger")
    def test_handles_retrieval_failure(self, mock_logger, mock_secrets_client):
        """Test that exceptions are logged and re-raised when retrieval fails."""
        error_msg = "Access denied"
        mock_secrets_client.get_secret_value.side_effect = Exception(error_msg)

        with pytest.raises(Exception, match=error_msg):
            handler.get_openai_api_key("test_secret_arn")

        mock_logger.error.assert_called_once()


class TestDownloadFileFromS3Async:
    """Tests for download_file_from_s3_async function."""

    @pytest.mark.asyncio
    @patch("handler.aioboto3.Session")
    @patch("handler.logger")
    async def test_successfully_downloads_file(self, mock_logger, mock_session_class):
        """Test successful file download from S3."""
        bucket = "test-bucket"
        key = "documents/test.txt"
        local_path = "/tmp/test.txt"

        # Mock the async context manager for s3 client
        mock_s3_client = AsyncMock()
        mock_session = MagicMock()
        mock_session.client.return_value.__aenter__.return_value = mock_s3_client
        mock_session_class.return_value = mock_session

        await handler.download_file_from_s3_async(bucket, key, local_path)

        mock_s3_client.download_file.assert_called_once_with(bucket, key, local_path)
        mock_logger.info.assert_called_once()


@pytest.fixture()
def config():
    return handler.get_config()


class TestProcessDocument:
    """Tests for process_document function."""

    @pytest.mark.asyncio
    @patch("handler.IngestionPipeline")
    @patch("handler.SimpleDirectoryReader")
    async def test_processes_document_successfully(
        self, mock_reader_class, mock_pipeline_class, config
    ):
        """Test successful document processing with loading, chunking, and indexing."""
        file_path = "/tmp/test.txt"
        mock_index = AsyncMock()

        # Mock document loading
        mock_documents = [Mock(text="Test document content")]
        mock_reader = Mock()
        mock_reader.aload_data = AsyncMock(return_value=mock_documents)
        mock_reader_class.return_value = mock_reader

        # Mock ingestion pipeline
        mock_nodes = [Mock(id="node1"), Mock(id="node2")]
        mock_pipeline = Mock()
        mock_pipeline.arun = AsyncMock(return_value=mock_nodes)
        mock_pipeline_class.return_value = mock_pipeline

        await handler.process_document(file_path, mock_index, config=config)

        # Verify document loading
        mock_reader_class.assert_called_once_with(
            input_files=[file_path], exclude_hidden=False
        )
        mock_reader.aload_data.assert_called_once()

        # Verify pipeline processing
        mock_pipeline.arun.assert_called_once_with(documents=mock_documents)

        # Verify index insertion
        mock_index.ainsert_nodes.assert_called_once_with(mock_nodes)


class TestProcessSingleRecord:
    """Tests for the process_single_record function."""

    @pytest.mark.asyncio
    @patch("handler.process_document")
    @patch("handler.download_file_from_s3_async")
    async def test_successfully_processes_record(
        self, mock_download, mock_process_doc, config
    ):
        """Test successful processing of a single S3 record."""
        bucket = "test-bucket"
        key = "documents/test.txt"
        mock_index = AsyncMock()

        mock_download.return_value = None
        mock_process_doc.return_value = None

        success, result_key, error = await handler.process_single_record(
            (bucket, key), mock_index, config
        )

        assert success is True
        assert result_key == key
        assert error == ""
        mock_download.assert_called_once()
        mock_process_doc.assert_called_once()

    @pytest.mark.asyncio
    @patch("handler.download_file_from_s3_async")
    @patch("handler.logger")
    async def test_handles_processing_failure(self, mock_logger, mock_download, config):
        """Test that exceptions are caught and returned as failure tuple."""
        bucket = "test-bucket"
        key = "documents/test.txt"
        mock_index = AsyncMock()
        error_msg = "Download failed"

        mock_download.side_effect = Exception(error_msg)

        success, result_key, error = await handler.process_single_record(
            (bucket, key), mock_index, config
        )

        assert success is False
        assert result_key == key
        assert error_msg in error
        mock_logger.error.assert_called_once()


class TestAsyncHandler:
    """Tests for async_handler function."""

    @pytest.mark.asyncio
    @patch("handler.get_openai_api_key")
    @patch("handler.get_s3_index")
    @patch("handler.process_single_record")
    async def test_processes_multiple_txt_files(
        self, mock_process, mock_get_index, mock_get_key, config
    ):
        """Test processing multiple .txt files concurrently."""
        event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "doc1.txt"},
                    }
                },
                {
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "doc2.txt"},
                    }
                },
            ]
        }

        # Mock successful processing
        mock_process.return_value = (True, "doc1.txt", "")
        mock_get_key.return_value = "test-key"

        response = await handler.async_handler(event, config)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["processed_count"] == 2
        assert body["failed_count"] == 0
        assert body["total_records"] == 2
        assert mock_process.call_count == 2

    @pytest.mark.asyncio
    @patch("handler.get_openai_api_key")
    @patch("handler.get_s3_index")
    @patch("handler.process_single_record")
    async def test_filters_non_txt_files(
        self, mock_process, mock_get_index, mock_get_key, config
    ):
        """Test that non-.txt files are skipped."""
        event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "doc.txt"},
                    }
                },
                {
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "image.jpg"},
                    }
                },
                {
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "data.pdf"},
                    }
                },
            ]
        }

        mock_process.return_value = (True, "doc.txt", "")
        mock_get_key.return_value = "test-key"

        response = await handler.async_handler(event, config)

        # Only .txt file should be processed
        assert mock_process.call_count == 1
        body = json.loads(response["body"])
        assert body["processed_count"] == 1
        assert body["total_records"] == 3

    @pytest.mark.asyncio
    @patch("handler.get_openai_api_key")
    @patch("handler.get_s3_index")
    @patch("handler.process_single_record")
    async def test_handles_mixed_success_and_failure(
        self, mock_process, mock_get_index, mock_get_key, config
    ):
        """Test handling of mixed success and failure results."""
        event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "success.txt"},
                    }
                },
                {
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "failure.txt"},
                    }
                },
            ]
        }

        # Mock one success and one failure
        mock_process.side_effect = [
            (True, "success.txt", ""),
            (False, "failure.txt", "Processing error"),
        ]
        mock_get_key.return_value = "test-key"

        response = await handler.async_handler(event, config)

        assert response["statusCode"] == 200  # At least one success
        body = json.loads(response["body"])
        assert body["processed_count"] == 1
        assert body["failed_count"] == 1

    @pytest.mark.asyncio
    @patch("handler.get_openai_api_key")
    @patch("handler.get_s3_index")
    @patch("handler.process_single_record")
    async def test_returns_500_when_all_fail(
        self, mock_process, mock_get_index, mock_get_key, config
    ):
        """Test that 500 status is returned when all records fail."""
        event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "doc.txt"},
                    }
                }
            ]
        }

        mock_process.return_value = (False, "doc.txt", "Error processing")
        mock_get_key.return_value = "test-key"

        response = await handler.async_handler(event, config)

        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert body["processed_count"] == 0
        assert body["failed_count"] == 1

    @pytest.mark.asyncio
    @patch("handler.get_openai_api_key")
    @patch("handler.get_s3_index")
    @patch("handler.process_single_record")
    async def test_handles_url_encoded_keys(
        self, mock_process, mock_get_index, mock_get_key, config
    ):
        """Test that URL-encoded S3 keys are properly decoded."""
        event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "folder%2Ffile+name.txt"},
                    }
                }
            ]
        }

        mock_process.return_value = (True, "folder/file name.txt", "")
        mock_get_key.return_value = "test-key"

        await handler.async_handler(event, config)

        # Verify the key was decoded
        call_args = mock_process.call_args[0]
        bucket, key = call_args[0]
        assert key == "folder/file name.txt"


class TestLambdaHandler:
    """Tests for lambda_handler function."""

    @patch("handler.async_handler")
    @patch("handler.asyncio.get_event_loop")
    def test_successfully_runs_async_handler(self, mock_get_loop, mock_async_handler):
        """Test that lambda_handler successfully runs async_handler in event loop."""
        event = {"Records": []}
        context = Mock()
        expected_response = {
            "statusCode": 200,
            "body": json.dumps({"message": "Success"}),
        }

        mock_loop = Mock()
        mock_loop.run_until_complete.return_value = expected_response
        mock_get_loop.return_value = mock_loop

        response = handler.lambda_handler(event, context)

        assert response == expected_response
        mock_loop.run_until_complete.assert_called_once()

    @patch("handler.async_handler")
    @patch("handler.asyncio.get_event_loop")
    @patch("handler.logger")
    def test_handles_exception_and_returns_500(
        self, mock_logger, mock_get_loop, mock_async_handler
    ):
        """Test that exceptions are caught and 500 error is returned."""
        event = {"Records": []}
        context = Mock()
        error_msg = "Fatal error"

        mock_loop = Mock()
        mock_loop.run_until_complete.side_effect = Exception(error_msg)
        mock_get_loop.return_value = mock_loop

        response = handler.lambda_handler(event, context)

        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert error_msg in body["error"]
        mock_logger.error.assert_called_once()
