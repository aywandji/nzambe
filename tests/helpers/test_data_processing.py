import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from nzambe.helpers.data_processing import (
    extract_testament_books_names,
    extract_book_text,
    extract_book_verses,
    split_bible_text_by_books,
    from_documents_to_nodes,
    build_documents_index,
)


class TestExtractTestamentBooksNames:
    """Unit tests for extract_testament_books_names function."""

    def test_extract_single_book(self):
        """Test extracting a single book between first and last."""
        bible_text = "Genesis\n\nExodus\n\nLeviticus"
        result = extract_testament_books_names(bible_text, "Genesis", "Leviticus")
        assert result == ["Genesis", "Exodus", "Leviticus"]

    def test_extract_multiple_books(self):
        """Test extracting multiple books."""
        bible_text = "Genesis\nExodus\nLeviticus\nNumbers\nDeuteronomy"
        result = extract_testament_books_names(bible_text, "Genesis", "Deuteronomy")
        assert "Genesis" in result
        assert "Deuteronomy" in result
        assert len(result) >= 2

    def test_extract_with_empty_lines(self):
        """Test that empty lines are filtered out."""
        bible_text = "Genesis\n\n\nExodus\n\n\nLeviticus"
        result = extract_testament_books_names(bible_text, "Genesis", "Leviticus")
        # Should not contain empty strings
        assert all(len(book) > 0 for book in result)

    def test_extract_adjacent_books(self):
        """Test extracting when the first and last books are adjacent."""
        bible_text = "Genesis\nExodus\nLeviticus"
        result = extract_testament_books_names(bible_text, "Genesis", "Exodus")
        assert result == ["Genesis", "Exodus"]


class TestExtractBookText:
    """Unit tests for the extract_book_text function."""

    def test_extract_book_text_basic(self):
        """Test basic book text extraction."""
        bible_text = "Genesis\n\n\nChapter 1 content\n\nExodus"
        result = extract_book_text(bible_text, "Genesis", "Exodus")
        assert "Chapter 1 content" == result

    def test_extract_book_text_removes_asterisks(self):
        """Test that *** markers, whitespaces are removed."""
        bible_text = "Genesis\n\n\n  ***Chapter 1***  \n\nExodus"
        result = extract_book_text(bible_text, "Genesis", "Exodus")
        assert "***" not in result
        assert "Chapter 1" == result

    def test_extract_book_text_removes_testament_marker(self):
        """Test that New Testament marker is removed."""
        bible_text = (
            "Matthew\n\n\nThe New Testament of the King James Bible\nChapter 1\n\nMark"
        )
        result = extract_book_text(bible_text, "Matthew", "Mark")
        assert "The New Testament of the King James Bible" not in result
        assert "Chapter 1" in result


class TestExtractBookVerses:
    """Unit tests for extract_book_verses function."""

    def test_extract_single_verse(self):
        """Test extracting a single verse."""
        book_text = "1:1 In the beginning God created the heaven and the earth."
        result = extract_book_verses(book_text)
        assert "1:1" in result
        assert result["1:1"] == "In the beginning God created the heaven and the earth."

    def test_extract_multiple_verses(self):
        """Test extracting multiple verses."""
        book_text = "1:1 First verse. 1:2 Second verse. 1:3 Third verse."
        result = extract_book_verses(book_text)
        assert len(result) == 3
        assert "1:1" in result
        assert "1:2" in result
        assert "1:3" in result

    def test_extract_verses_removes_newlines(self):
        """Test that newlines within verses are replaced with spaces."""
        book_text = "1:1 First line\nsecond line\nthird line. 1:2 Next verse."
        result = extract_book_verses(book_text)
        assert "\n" not in result["1:1"]
        assert "First line second line third line." in result["1:1"]

    def test_extract_verses_strips_whitespace(self):
        """Test that verse text is stripped of extra whitespace."""
        book_text = "1:1   Verse with spaces   1:2   Another verse   "
        result = extract_book_verses(book_text)
        assert result["1:1"] == "Verse with spaces"
        assert result["1:2"] == "Another verse"

    def test_extract_verses_with_chapter_verse_format(self):
        """Test various chapter:verse number formats."""
        book_text = "1:1 First. 10:25 Second. 100:999 Third."
        result = extract_book_verses(book_text)
        assert "1:1" in result
        assert "10:25" in result
        assert "100:999" in result


@pytest.fixture()
def bible_text_path() -> Path:
    """Path to a sample Bible text file for testing."""
    return Path(__file__).parents[1] / "data" / "bible_text_sample.txt"


class TestSplitBibleTextByBooks:
    """Integration tests for the split_bible_text_by_books function."""

    def test_split_bible_text_basic(self, bible_text_path: Path):
        """Test basic bible text splitting without saving."""
        # Act
        result = split_bible_text_by_books(str(bible_text_path))

        # Assert - Check Old Testament books
        assert "The First Book of Moses: Called Genesis" in result
        assert "Exodus" in result
        assert "Malachi" in result

        # Check New Testament books
        assert "The Gospel According to Saint Matthew" in result
        assert "The Gospel According to Saint Mark" in result
        assert "The Revelation of Saint John the Divine" in result

        # Verify Genesis verses
        genesis = result["The First Book of Moses: Called Genesis"]
        assert "1:1" in genesis
        assert (
            "In the beginning God created the heaven and the earth." in genesis["1:1"]
        )
        assert "2:1" in genesis

        # Verify Matthew verses
        matthew = result["The Gospel According to Saint Matthew"]
        assert "1:1" in matthew
        assert "generation of Jesus Christ" in matthew["1:1"]

    def test_split_bible_text_with_destination(
        self, bible_text_path: Path, tmp_path: Path
    ):
        """Test bible text splitting with destination directory."""
        # Arrange
        output_dir = tmp_path / "output"

        # Act
        result = split_bible_text_by_books(
            str(bible_text_path), destination_directory=str(output_dir)
        )

        # Assert - Check that result contains books
        assert "The First Book of Moses: Called Genesis" in result
        assert "The Gospel According to Saint Matthew" in result

        # Check that files were created
        assert output_dir.exists()
        genesis_file = output_dir / "The First Book of Moses: Called Genesis.txt"
        matthew_file = output_dir / "The Gospel According to Saint Matthew.txt"

        assert genesis_file.exists()
        assert matthew_file.exists()

        # Verify file contents
        genesis_content = genesis_file.read_text()
        assert "1:1 In the beginning God created" in genesis_content
        assert "2:1 Thus the heavens and the earth" in genesis_content

    def test_split_bible_text_skip_existing_files(
        self, bible_text_path: Path, tmp_path: Path
    ):
        """Test that existing book files are not overwritten."""
        # Arrange
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create existing file with custom content
        genesis_file = output_dir / "The First Book of Moses: Called Genesis.txt"
        original_content = "ORIGINAL CONTENT - DO NOT OVERWRITE"
        genesis_file.write_text(original_content)

        # Act
        result = split_bible_text_by_books(
            str(bible_text_path), destination_directory=str(output_dir)
        )

        # Assert - Result should still contain Genesis data
        assert "The First Book of Moses: Called Genesis" in result

        # But the file should not be overwritten
        assert genesis_file.read_text() == original_content

        # Other files should still be created
        matthew_file = output_dir / "The Gospel According to Saint Matthew.txt"
        assert matthew_file.exists()

    def test_split_bible_uses_alternative_book_names(self, bible_text_path: Path):
        """Test that alternative book names are used when extracting."""
        # Act
        result = split_bible_text_by_books(str(bible_text_path))

        # Assert - The book should be in result with its display name (from table of contents)
        assert "The First Book of the Kings" in result

        # Verify the content was extracted correctly using the alternative name
        # (The actual content is under "The Third Book of the Kings" but mapped to "The First Book of the Kings")
        book_content = result["The First Book of the Kings"]
        assert "1:1" in book_content
        assert "Now king David was old" in book_content["1:1"]


class TestFromDocumentsToNodes:
    """Unit tests for from_documents_to_nodes async function."""

    @pytest.mark.asyncio
    @patch("nzambe.helpers.data_processing.IngestionPipeline")
    @patch("nzambe.helpers.data_processing.Settings")
    async def test_from_documents_to_nodes_async_mode(
        self, mock_settings, mock_pipeline_class
    ):
        """Test document-to-node conversion in async mode (num_workers=None)."""
        # Setup
        mock_documents = [Mock(name="doc1"), Mock(name="doc2")]
        mock_tokenizer = Mock()
        mock_embed_model = Mock()
        mock_settings.embed_model = mock_embed_model

        mock_pipeline_instance = Mock()
        ingestion_pipeline_outputs = ["node1", "node2"]
        mock_pipeline_instance.arun = AsyncMock(return_value=ingestion_pipeline_outputs)
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Act
        result = await from_documents_to_nodes(
            documents=mock_documents,
            model_tokenizer=mock_tokenizer,
            chunk_size=512,
            chunk_overlap=50,
        )

        # Assert
        assert result == ingestion_pipeline_outputs
        mock_pipeline_instance.arun.assert_called_once_with(documents=mock_documents)
        # Verify embed_model was added to transformations
        assert mock_embed_model in mock_pipeline_class.call_args[1]["transformations"]

    @pytest.mark.asyncio
    @patch("nzambe.helpers.data_processing.IngestionPipeline")
    @patch("nzambe.helpers.data_processing.Settings")
    async def test_from_documents_to_nodes_multiprocessing_mode(
        self, mock_settings, mock_pipeline_class
    ):
        """Test document to node conversion in multiprocessing mode."""
        # Setup
        mock_documents = [Mock(name="doc1"), Mock(name="doc2")]
        mock_tokenizer = Mock()
        mock_embed_model = Mock()
        mock_settings.embed_model = mock_embed_model

        mock_pipeline_instance = Mock()
        ingestion_pipeline_outputs = ["node1", "node2", "node3"]
        mock_pipeline_instance.run = Mock(return_value=ingestion_pipeline_outputs)
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Act
        result = await from_documents_to_nodes(
            documents=mock_documents,
            model_tokenizer=mock_tokenizer,
            chunk_size=512,
            chunk_overlap=50,
            num_workers=4,
        )

        # Assert
        assert result == ingestion_pipeline_outputs
        mock_pipeline_instance.run.assert_called_once_with(
            documents=mock_documents, num_workers=4
        )
        # Verify embed_model was NOT added when using multiprocessing
        assert (
            mock_embed_model not in mock_pipeline_class.call_args[1]["transformations"]
        )

    @pytest.mark.asyncio
    @patch("nzambe.helpers.data_processing.IngestionPipeline")
    @patch("nzambe.helpers.data_processing.SentenceSplitter")
    @patch("nzambe.helpers.data_processing.Settings")
    async def test_from_documents_to_nodes_custom_parameters(
        self, mock_settings, mock_splitter_class, mock_pipeline_class
    ):
        """Test that custom chunk parameters are passed to SentenceSplitter."""
        # Setup
        mock_documents = [Mock()]
        mock_settings.embed_model = Mock()
        mock_tokenizer = Mock()
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.arun = AsyncMock(return_value=["node1"])
        mock_pipeline_class.return_value = mock_pipeline_instance

        chunk_size = 1024
        chunk_overlap = 100
        paragraph_separator = "\n\n\n"

        # Act
        await from_documents_to_nodes(
            documents=mock_documents,
            model_tokenizer=mock_tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator=paragraph_separator,
        )

        # Assert
        mock_splitter_class.assert_called_once_with(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator=paragraph_separator,
            tokenizer=mock_tokenizer,
        )


class TestBuildDocumentsIndex:
    """Unit tests for build_documents_index async function."""

    @pytest.mark.asyncio
    @patch("nzambe.helpers.data_processing.StorageContext")
    @patch("nzambe.helpers.data_processing.load_index_from_storage")
    async def test_build_documents_index_loads_from_disk(
        self, mock_load_index, mock_storage_context, tmp_path
    ):
        """Test that the existing index is loaded from the disk."""
        # Setup
        mock_index = Mock()
        mock_load_index.return_value = mock_index

        with tempfile.NamedTemporaryFile(mode="w", delete=True, dir=tmp_path):
            # create a temporary index file to ensure the loading process will be called
            result = await build_documents_index(
                index_storage_dir=tmp_path,
                input_data_directory=None,
                input_data_files=None,
                document_split_chunk_size=512,
                document_split_chunk_overlap=50,
            )

        # Assert
        assert result == mock_index
        mock_load_index.assert_called_once()
        mock_storage_context.from_defaults.assert_called_once_with(
            persist_dir=str(tmp_path)
        )

    @pytest.mark.asyncio
    async def test_build_documents_index_requires_tokenizer(self, tmp_path):
        """Test that model_tokenizer is required when building a new index."""
        # Act & Assert
        with pytest.raises(
            Exception, match="model_tokenizer is required to build index"
        ):
            await build_documents_index(
                index_storage_dir=tmp_path,
                input_data_directory="/data",
                input_data_files=None,
                document_split_chunk_size=512,
                document_split_chunk_overlap=50,
                model_tokenizer=None,
            )

    @pytest.mark.asyncio
    # @patch("os.path.exists", return_value=False)
    @patch("nzambe.helpers.data_processing.Settings")
    async def test_build_documents_index_unsupported_embed_model(
        self, mock_settings, tmp_path
    ):
        """Test that a non-HuggingFace embed model raises an exception."""
        # Setup
        mock_settings.embed_model = Mock(spec=object)  # Not HuggingFaceEmbedding

        # Act & Assert
        with pytest.raises(Exception, match="Unsupported embed model"):
            await build_documents_index(
                index_storage_dir=tmp_path,
                input_data_directory="/data",
                input_data_files=None,
                document_split_chunk_size=512,
                document_split_chunk_overlap=50,
                model_tokenizer=Mock(),
            )

    @pytest.mark.asyncio
    @patch("nzambe.helpers.data_processing.Settings")
    @patch("nzambe.helpers.data_processing.SimpleDirectoryReader")
    @patch("nzambe.helpers.data_processing.from_documents_to_nodes")
    async def test_build_documents_index_creates_new_index(
        self, mock_from_docs, mock_reader_class, mock_settings, tmp_path
    ):
        """Test creating a new index from scratch."""
        # Setup
        embed_model = Mock(spec=HuggingFaceEmbedding, model_name="test-model")
        embed_model.get_text_embedding_batch.return_value = [
            np.random.rand(5),
            np.random.rand(5),
        ]
        mock_settings.embed_model = embed_model

        mock_reader = Mock()
        mock_reader.load_data.return_value = [Mock(name="doc1"), Mock(name="doc2")]
        mock_reader_class.return_value = mock_reader

        mock_from_docs.return_value = [TextNode(text="node1"), TextNode(text="node2")]

        index_dir = tmp_path
        # Act
        result = await build_documents_index(
            index_storage_dir=index_dir,
            input_data_directory="/data",
            input_data_files=None,
            document_split_chunk_size=512,
            document_split_chunk_overlap=50,
            model_tokenizer=Mock(),
        )

        # Assert
        assert isinstance(result, VectorStoreIndex)

        mock_reader_class.assert_called_once()
        mock_from_docs.assert_called_once()

        # Check metadata was written
        with open(index_dir / "nzambe_metadata.json", "r") as f:
            metadata = json.load(f)

        assert metadata["embed_model_id"] == mock_settings.embed_model.model_name
        assert metadata["platform"] == "huggingface"

        # check index was written
        assert len(list(index_dir.glob("*store.json"))) > 0

    @pytest.mark.asyncio
    @patch("nzambe.helpers.data_processing.Settings")
    @patch("nzambe.helpers.data_processing.SimpleDirectoryReader")
    @patch("nzambe.helpers.data_processing.from_documents_to_nodes")
    async def test_build_documents_index_with_input_files(
        self,
        mock_from_docs,
        mock_reader_class,
        mock_settings,
        tmp_path,
    ):
        """Test creating an index with specific input files."""
        # Setup
        embed_model = Mock(spec=HuggingFaceEmbedding, model_name="test-model")
        embed_model.get_text_embedding_batch.return_value = [np.random.rand(5)]
        mock_settings.embed_model = embed_model

        mock_reader = Mock()
        mock_reader.load_data.return_value = [Mock(name="doc1")]
        mock_reader_class.return_value = mock_reader

        mock_from_docs.return_value = [TextNode(text="node1")]
        input_files = [tmp_path / "file1.txt", tmp_path / "file2.txt"]

        # Act
        _ = await build_documents_index(
            index_storage_dir=tmp_path,
            input_data_directory=None,
            input_data_files=input_files,
            document_split_chunk_size=512,
            document_split_chunk_overlap=50,
            model_tokenizer=Mock(),
        )

        # Assert
        mock_reader_class.assert_called_once_with(
            input_dir=None,
            input_files=input_files,
            exclude_hidden=False,
        )

    @pytest.mark.asyncio
    @patch("nzambe.helpers.data_processing.Settings")
    @patch("nzambe.helpers.data_processing.SimpleDirectoryReader")
    @patch("nzambe.helpers.data_processing.from_documents_to_nodes")
    async def test_build_documents_index_with_multiprocessing(
        self, mock_from_docs, mock_reader_class, mock_settings, tmp_path
    ):
        """Test creating index with multiprocessing."""
        # Setup
        mock_embed_model = Mock(spec=HuggingFaceEmbedding, model_name="test-model")
        mock_embed_model.get_text_embedding_batch.return_value = [np.random.rand(5)]
        mock_settings.embed_model = mock_embed_model

        mock_reader = Mock()
        mock_reader.load_data.return_value = [Mock(name="doc1")]
        mock_reader_class.return_value = mock_reader

        mock_from_docs.return_value = [TextNode(text="node1")]

        # Act
        num_workers = 4
        _ = await build_documents_index(
            index_storage_dir=tmp_path,
            input_data_directory=str(tmp_path / "data"),
            input_data_files=None,
            document_split_chunk_size=512,
            document_split_chunk_overlap=50,
            model_tokenizer=Mock(),
            num_workers=num_workers,
        )

        # Assert
        # Verify from_documents_to_nodes was called with num_workers
        call_kwargs = mock_from_docs.call_args[1]
        assert call_kwargs.get("num_workers") == num_workers
