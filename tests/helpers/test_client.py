import json
import os
from unittest.mock import Mock, patch

import pytest
import requests

from nzambe.constants import NZAMBE_SERVER_DEFAULT_BASE_URL
from nzambe.helpers.client import health_check, query_server, HealthResponse


class TestHealthCheck:
    """Unit tests for the health_check method."""

    @patch("nzambe.helpers.client.requests.get")
    def test_health_check_success(self, mock_get):
        """Test successful health check with default base URL."""
        # Arrange
        mock_response = Mock()
        server_response = {"status": "healthy", "query_engine_loaded": True}
        mock_response.json.return_value = server_response
        mock_get.return_value = mock_response

        # Act
        result = health_check()

        # Assert
        mock_get.assert_called_once_with(
            f"{NZAMBE_SERVER_DEFAULT_BASE_URL}/health", timeout=30
        )
        mock_get.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
        assert isinstance(result, HealthResponse)
        assert result.status == server_response["status"]
        assert result.query_engine_loaded == server_response["query_engine_loaded"]

    @patch("nzambe.helpers.client.requests.get")
    def test_health_check_with_custom_base_url(self, mock_get):
        """Test health check with custom base URL."""
        # Arrange
        custom_url = "http://example.com:9000"
        mock_response = Mock()
        server_response = {"status": "healthy", "query_engine_loaded": False}
        mock_response.json.return_value = server_response
        mock_get.return_value = mock_response

        # Act
        result = health_check(base_url=custom_url)

        # Assert
        mock_get.assert_called_once_with(f"{custom_url}/health", timeout=30)
        assert result.status == server_response["status"]
        assert result.query_engine_loaded == server_response["query_engine_loaded"]

    @patch.dict(os.environ, {"NZAMBE_API_URL": "http://env-server:7000"})
    @patch("nzambe.helpers.client.requests.get")
    def test_health_check_with_env_variable(self, mock_get):
        """Test health check uses NZAMBE_API_URL environment variable."""
        # Arrange
        mock_response = Mock()
        server_response = {"status": "healthy", "query_engine_loaded": True}
        mock_response.json.return_value = server_response
        mock_get.return_value = mock_response

        # Act
        result = health_check()

        # Assert
        mock_get.assert_called_once_with("http://env-server:7000/health", timeout=30)
        assert result.status == server_response["status"]

    @patch("nzambe.helpers.client.requests.get")
    def test_health_check_strips_trailing_slash(self, mock_get):
        """Test that trailing slashes are stripped from base URL."""
        # Arrange
        mock_response = Mock()
        server_response = {"status": "healthy", "query_engine_loaded": True}
        mock_response.json.return_value = server_response
        mock_get.return_value = mock_response

        # Act
        health_check(base_url="http://example.com:9000/")

        # Assert
        mock_get.assert_called_once_with("http://example.com:9000/health", timeout=30)

    @patch("nzambe.helpers.client.requests.get")
    def test_health_check_http_error(self, mock_get):
        """Test health check handles HTTP errors properly."""
        # Arrange
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error"
        )
        mock_get.return_value = mock_response

        # Act & Assert
        with pytest.raises(requests.HTTPError):
            health_check()


class TestQueryServer:
    """Unit tests for the query_server method."""

    @patch("nzambe.helpers.client.requests.post")
    def test_query_server_success(self, mock_post):
        """Test a successful query with normal mode."""
        # Arrange
        question = "What is the meaning of life?"
        mock_response = Mock()
        server_response = {"answer": "42", "nodes": ["doc1", "doc2"]}
        mock_response.json.return_value = server_response
        mock_post.return_value = mock_response

        # Act
        answer, docs = query_server(question)

        # Assert
        mock_post.assert_called_once_with(
            f"{NZAMBE_SERVER_DEFAULT_BASE_URL}/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"question": question}),
            timeout=120,
        )
        mock_response.raise_for_status.assert_called_once()
        assert answer == server_response["answer"]
        assert docs == server_response["nodes"]

    @patch("nzambe.helpers.client.requests.post")
    def test_query_server_docs_only_mode(self, mock_post):
        """Test a successful query with docs_only mode."""
        # Arrange
        question = "What is the meaning of life?"
        mock_response = Mock()
        server_response = {"answer": None, "nodes": ["doc1", "doc2", "doc3"]}
        mock_response.json.return_value = server_response
        mock_post.return_value = mock_response

        # Act
        answer, docs = query_server(question, docs_only=True)

        # Assert
        mock_post.assert_called_once_with(
            f"{NZAMBE_SERVER_DEFAULT_BASE_URL}/retrieve_docs",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"question": question}),
            timeout=120,
        )
        assert answer == server_response["answer"]
        assert docs == server_response["nodes"]

    @patch("nzambe.helpers.client.requests.post")
    def test_query_server_with_custom_base_url(self, mock_post):
        """Test query with custom base URL."""
        # Arrange
        question = "test question"
        custom_url = "http://custom-server:5000"
        mock_response = Mock()
        server_response = {"answer": "test answer", "nodes": []}
        mock_response.json.return_value = server_response
        mock_post.return_value = mock_response

        # Act
        query_server(question, base_url=custom_url)

        # Assert
        mock_post.assert_called_once_with(
            f"{custom_url}/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"question": question}),
            timeout=120,
        )

    def test_query_server_empty_whitespace_question(self):
        """Test that empty/whitespace-only question raises ValueError."""
        with pytest.raises(ValueError, match="question must be a non-empty string"):
            query_server("")

        with pytest.raises(ValueError, match="question must be a non-empty string"):
            query_server("   ")

    @patch("nzambe.helpers.client.requests.post")
    def test_query_server_http_error(self, mock_post):
        """Test query handles HTTP errors properly."""
        # Arrange
        question = "test question"
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_post.return_value = mock_response

        # Act & Assert
        with pytest.raises(requests.HTTPError):
            query_server(question)

    @patch("nzambe.helpers.client.requests.post")
    def test_query_server_invalid_answer_format(self, mock_post):
        """Test query handles an invalid answer format in normal mode."""
        # Arrange
        question = "test question"
        mock_response = Mock()
        mock_response.json.return_value = {
            "answer": 123,  # Not a string
            "nodes": [],
        }
        mock_post.return_value = mock_response

        # Act & Assert
        with pytest.raises(ValueError, match="Unexpected response format"):
            query_server(question)

    @patch("nzambe.helpers.client.requests.post")
    def test_query_server_invalid_nodes_format(self, mock_post):
        """Test query handles invalid nodes format."""
        # Arrange
        question = "test question"
        mock_response = Mock()
        mock_response.json.return_value = {
            "answer": "valid answer",
            "nodes": "not a list",  # Not a list
        }
        mock_post.return_value = mock_response

        # Act & Assert
        with pytest.raises(ValueError, match="Unexpected response format"):
            query_server(question)

    @patch.dict(os.environ, {"NZAMBE_API_URL": "http://env-server:3000"})
    @patch("nzambe.helpers.client.requests.post")
    def test_query_server_with_env_variable(self, mock_post):
        """Test query uses NZAMBE_API_URL environment variable."""
        # Arrange
        question = "test question"
        mock_response = Mock()
        mock_response.json.return_value = {
            "answer": "test answer",
            "nodes": [],
        }
        mock_post.return_value = mock_response

        # Act
        query_server(question)

        # Assert
        mock_post.assert_called_once_with(
            "http://env-server:3000/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"question": question}),
            timeout=120,
        )

    @patch("nzambe.helpers.client.requests.post")
    def test_query_server_strips_trailing_slash(self, mock_post):
        """Test that trailing slashes are stripped from base URL."""
        # Arrange
        question = "test question"
        mock_response = Mock()
        server_response = {"answer": "test answer", "nodes": []}
        mock_response.json.return_value = server_response
        mock_post.return_value = mock_response

        # Act
        customers_base_url = "http://example.com:9000/"
        query_server(question, base_url=customers_base_url)

        # Assert
        mock_post.assert_called_once_with(
            f"{customers_base_url.rstrip('/')}/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"question": question}),
            timeout=120,
        )
