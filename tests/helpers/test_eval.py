import datetime
import json
import os
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import pytest
from langfuse.api import ObservationLevel
from llama_index.core.llama_dataset.legacy.embedding import EmbeddingQAFinetuneDataset

from nzambe.helpers.eval import (
    run_nightly_benchmark,
    generate_new_questions_from_index,
)


class TestRunNightlyBenchmark:
    """Unit tests for run_nightly_benchmark function."""

    @patch("nzambe.helpers.eval.evaluate")
    @patch("nzambe.helpers.eval.setup_langfuse_client")
    def test_run_nightly_benchmark_success(self, mock_setup_langfuse, mock_evaluate):
        """Test successful nightly benchmark run."""
        # Setup Langfuse client
        mock_langfuse = Mock()
        mock_setup_langfuse.return_value = mock_langfuse

        # Setup trace data
        mock_trace = Mock(
            id="trace-123", input="What is the meaning of life?", output="42"
        )

        # Setup observations
        mock_observation = {
            "metadata": {
                "attributes": {
                    "retrieval.documents.0.document.content": "Context 1",
                    "retrieval.documents.1.document.content": "Context 2",
                }
            }
        }

        mock_langfuse.api.trace.list.return_value.data = [mock_trace]
        mock_langfuse.api.observations.get_many.return_value.dict.return_value = {
            "data": [mock_observation]
        }

        # Setup evaluation results
        mock_result = Mock()
        mock_df = Mock()
        mock_df.iterrows.return_value = [
            (0, {"faithfulness": 0.9, "answer_relevancy": 0.85})
        ]
        mock_result.to_pandas.return_value = mock_df
        mock_evaluate.return_value = mock_result

        # Act
        run_nightly_benchmark(last_n_hours=24, num_traces_limit=1)

        # Assert
        mock_langfuse.api.trace.list.assert_called_once()
        assert mock_langfuse.api.observations.get_many.call_count >= 1
        mock_evaluate.assert_called_once()
        mock_langfuse.create_score.assert_called()

    @patch("nzambe.helpers.eval.setup_langfuse_client")
    def test_run_nightly_benchmark_no_traces(self, mock_setup_langfuse):
        """Test when no traces are found."""
        # Setup
        mock_langfuse = Mock()
        mock_setup_langfuse.return_value = mock_langfuse
        mock_langfuse.api.trace.list.return_value.data = []

        # Act
        run_nightly_benchmark(last_n_hours=24, num_traces_limit=50)

        # Assert
        mock_langfuse.api.trace.list.assert_called_once()
        # Should return early without evaluating
        assert (
            not hasattr(mock_langfuse, "create_score")
            or mock_langfuse.create_score.call_count == 0
        )

    @patch("nzambe.helpers.eval.setup_langfuse_client")
    @patch("nzambe.helpers.eval.evaluate")
    def test_run_nightly_benchmark_skips_traces_without_observations(
        self, mock_evaluate, mock_setup_langfuse
    ):
        """Tests that traces without observations are skipped."""
        # Setup
        mock_langfuse = Mock()
        mock_setup_langfuse.return_value = mock_langfuse

        mock_trace1 = Mock(id="trace-1", input="Question 1", output="Answer 1")
        mock_trace2 = Mock(id="trace-2", input="Question 2", output="Answer 2")

        mock_langfuse.api.trace.list.return_value.data = [mock_trace1, mock_trace2]

        # The first trace has no observations, the second has observations
        mock_langfuse.api.observations.get_many.return_value.dict.side_effect = [
            {"data": []},  # No RETRIEVER observations for first call
            {"data": []},  # No retrieve observations for first call
            dict(
                data=[
                    dict(
                        id="id",
                        type="type",
                        metadata={
                            "attributes": {
                                "retrieval.documents.1.document.content": "test"
                            }
                        },
                        startTime=datetime.datetime.now(),
                        level=ObservationLevel.DEBUG,
                    )
                ],
            ),  # Has observations for the second trace
            {"data": []},
        ]

        # Act
        with (
            patch("nzambe.helpers.eval.embedding_factory"),
        ):
            run_nightly_benchmark(last_n_hours=1, num_traces_limit=2)

        # Assert evaluate is called with a dataset having only one trace
        assert mock_evaluate.call_args[0][0].num_rows == 1
        # Assert - should have tried to get observations for both traces
        assert mock_langfuse.api.observations.get_many.call_count == 4

    @patch("nzambe.helpers.eval.evaluate")
    @patch("nzambe.helpers.eval.setup_langfuse_client")
    def test_run_nightly_benchmark_extracts_contexts(
        self, mock_setup_langfuse, mock_evaluate
    ):
        """Test that contexts are correctly extracted from observations."""
        # Setup
        mock_langfuse = Mock()
        mock_setup_langfuse.return_value = mock_langfuse

        mock_trace = Mock(id="trace-123", input="Question", output="Answer")

        # Create observation with multiple retrieval documents (out of order)
        mock_observation = {
            "metadata": {
                "attributes": {
                    "retrieval.documents.0.document.content": "First context",
                    "retrieval.documents.2.document.content": "Third context",
                    "retrieval.documents.1.document.content": "Second context",
                    "other.attribute": "ignored",
                }
            }
        }

        mock_langfuse.api.trace.list.return_value.data = [mock_trace]
        mock_langfuse.api.observations.get_many.return_value.dict.return_value = {
            "data": [mock_observation]
        }

        # Mock evaluation to capture what dataset was passed
        mock_result = Mock()
        mock_result.to_pandas.return_value = Mock(iterrows=lambda: [])
        mock_evaluate.return_value = mock_result

        # Act
        run_nightly_benchmark(last_n_hours=1, num_traces_limit=1)

        # Assert - contexts should be in sorted order
        call_args = mock_evaluate.call_args[0]
        dataset = call_args[0]
        contexts = dataset["contexts"][0]
        assert contexts == ["First context", "Second context", "Third context"]

    @patch("nzambe.helpers.eval.evaluate")
    @patch("nzambe.helpers.eval.setup_langfuse_client")
    def test_run_nightly_benchmark_pushes_scores_back(
        self, mock_setup_langfuse, mock_evaluate
    ):
        """Test that evaluation scores are pushed back to Langfuse."""
        # Setup
        mock_langfuse = Mock()
        mock_setup_langfuse.return_value = mock_langfuse

        mock_trace = Mock(id="trace-456", input="Question", output="Answer")

        mock_observation = {
            "metadata": {
                "attributes": {
                    "retrieval.documents.0.document.content": "Context",
                }
            }
        }

        mock_langfuse.api.trace.list.return_value.data = [mock_trace]
        mock_langfuse.api.observations.get_many.return_value.dict.return_value = {
            "data": [mock_observation]
        }

        # Setup evaluation results with specific scores
        mock_result = Mock()
        mock_df_row = {"faithfulness": 0.95, "answer_relevancy": 0.88}
        mock_df = Mock()
        mock_df.iterrows.return_value = [(0, mock_df_row)]
        mock_result.to_pandas.return_value = mock_df
        mock_evaluate.return_value = mock_result

        # Act
        run_nightly_benchmark(last_n_hours=1, num_traces_limit=1)

        # Assert
        assert mock_langfuse.create_score.call_count == 2  # One for each metric

        # Check that scores were created with correct values
        score_calls = mock_langfuse.create_score.call_args_list
        trace_ids = [call[1]["trace_id"] for call in score_calls]
        assert all(tid == "trace-456" for tid in trace_ids)

        metric_names = [call[1]["name"] for call in score_calls]
        assert "faithfulness" in metric_names
        assert "answer_relevancy" in metric_names


class TestGenerateNewQuestionsFromIndex:
    """Unit tests for generate_new_questions_from_index function."""

    @patch("nzambe.helpers.eval.generate_question_context_pairs")
    @patch("nzambe.helpers.eval.load_index_from_storage")
    def test_generate_new_questions_success(
        self, mock_load_index, mock_generate_pairs, tmp_path
    ):
        """Test successful question generation from index."""
        # Setup real metadata file
        index_storage_dir = str(tmp_path)
        with open(os.path.join(index_storage_dir, "nzambe_metadata.json"), "w") as f:
            json.dump({"platform": "huggingface", "embed_model_id": "test-model"}, f)

        # Mock index
        mock_index = Mock()
        mock_index.storage_context.vector_store.data.embedding_dict.keys.return_value = [
            "node1",
            "node2",
            "node3",
        ]
        mock_node1 = Mock(node_id="node1")
        mock_node2 = Mock(node_id="node2")
        mock_index.docstore.get_nodes.return_value = [mock_node1, mock_node2]
        mock_load_index.return_value = mock_index

        # Mock generated dataset
        mock_qa_dataset = Mock(spec=EmbeddingQAFinetuneDataset)
        mock_qa_dataset.queries = {}
        mock_qa_dataset.corpus = {}
        mock_qa_dataset.relevant_docs = {}
        mock_generate_pairs.return_value = mock_qa_dataset

        # Act
        with (
            patch("nzambe.helpers.eval.HuggingFaceEmbedding"),
            patch("nzambe.helpers.eval.StorageContext"),
            patch("nzambe.helpers.eval.ollama_llm"),
        ):
            result = generate_new_questions_from_index(
                ollama_model_name="llama2",
                index_storage_dir=index_storage_dir,
                num_questions_per_node=2,
                num_nodes_to_sample=2,
                random_seed=42,
            )

        # Assert
        assert result == mock_qa_dataset
        mock_generate_pairs.assert_called_once()
        assert not mock_qa_dataset.save_json.called  # No path provided

    @patch("nzambe.helpers.eval.load_index_from_storage")
    @patch("nzambe.helpers.eval.generate_question_context_pairs")
    def test_generate_new_questions_with_existing_dataset(
        self, mock_generate_pairs, mock_load_index, tmp_path
    ):
        """Test question generation updates existing dataset."""
        # Mock index with 3 nodes, but the first node already has questions
        mock_index = Mock()
        mock_index.storage_context.vector_store.data.embedding_dict.keys.return_value = [
            "23e8f67c-bb2e-4d0e-bd72-e5f48737e650",
            "node2",
            "node3",
        ]
        mock_node2 = Mock(node_id="node2")
        # mock_node3 = Mock(node_id="node3")
        mock_index.docstore.get_nodes.return_value = [mock_node2]
        mock_load_index.return_value = mock_index

        # Mock new questions
        mock_new_dataset = Mock(spec=EmbeddingQAFinetuneDataset)
        mock_new_dataset.queries = {"q1": "Question 1"}
        mock_new_dataset.corpus = {"node2": "Corpus 2"}
        mock_new_dataset.relevant_docs = {"q1": ["node2"]}
        mock_generate_pairs.return_value = mock_new_dataset

        # Act
        index_storage_dir = str(tmp_path)
        questions_dataset_path = str(
            Path(__file__).parents[1] / "data" / "qa_dataset.json"
        )
        with open(os.path.join(index_storage_dir, "nzambe_metadata.json"), "w") as f:
            json.dump({"platform": "huggingface", "embed_model_id": "test-model"}, f)

        with (
            patch("nzambe.helpers.eval.ollama_llm"),
            patch("nzambe.helpers.eval.HuggingFaceEmbedding"),
            patch("nzambe.helpers.eval.StorageContext"),
        ):
            updated_dataset = generate_new_questions_from_index(
                ollama_model_name="llama2",
                index_storage_dir=index_storage_dir,
                num_questions_per_node=1,
                num_nodes_to_sample=1,
                random_seed=42,
                questions_dataset_path=questions_dataset_path,
            )

        # Should merge with existing dataset
        existing_dataset = EmbeddingQAFinetuneDataset.from_json(questions_dataset_path)
        assert (
            updated_dataset.queries
            == existing_dataset.queries | mock_new_dataset.queries
        )
        assert (
            updated_dataset.corpus == existing_dataset.corpus | mock_new_dataset.corpus
        )
        assert (
            updated_dataset.relevant_docs
            == existing_dataset.relevant_docs | mock_new_dataset.relevant_docs
        )
        # Should save merged dataset
        mock_new_dataset.save_json.assert_called_once_with(questions_dataset_path)

    @patch("builtins.open", new_callable=mock_open)
    def test_generate_new_questions_missing_metadata(self, mock_file):
        """Test that missing metadata file raises exception."""
        # Setup
        mock_file.side_effect = FileNotFoundError()

        # Act & Assert
        with pytest.raises(Exception, match="Could not find nzambe_metadata.json"):
            generate_new_questions_from_index(
                ollama_model_name="llama2",
                index_storage_dir="/fake/index",
                num_questions_per_node=2,
                num_nodes_to_sample=5,
                random_seed=42,
            )

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"platform": "openai", "embed_model_id": "test"}',
    )
    def test_generate_new_questions_unsupported_platform(self, mock_file):
        """Test that unsupported platform raises exception."""
        # Act & Assert
        with pytest.raises(Exception, match="Unsupported platform"):
            generate_new_questions_from_index(
                ollama_model_name="llama2",
                index_storage_dir="/fake/index",
                num_questions_per_node=2,
                num_nodes_to_sample=5,
                random_seed=42,
            )

    @patch("nzambe.helpers.eval.load_index_from_storage")
    def test_generate_new_questions_no_candidate_nodes(self, mock_load_index, tmp_path):
        """Test when all nodes already have questions."""
        questions_dataset_path = str(
            Path(__file__).parents[1] / "data" / "qa_dataset.json"
        )
        existing_dataset = EmbeddingQAFinetuneDataset.from_json(questions_dataset_path)
        # Mock index with same nodes
        index_storage_dir = str(tmp_path)
        with open(os.path.join(index_storage_dir, "nzambe_metadata.json"), "w") as f:
            json.dump({"platform": "huggingface", "embed_model_id": "test-model"}, f)
        mock_index = Mock()
        mock_index.storage_context.vector_store.data.embedding_dict.keys.return_value = list(
            existing_dataset.corpus.keys()
        )
        mock_load_index.return_value = mock_index

        # Act
        with (
            patch("nzambe.helpers.eval.HuggingFaceEmbedding"),
            patch("nzambe.helpers.eval.StorageContext"),
        ):
            result = generate_new_questions_from_index(
                ollama_model_name="llama2",
                index_storage_dir=index_storage_dir,
                num_questions_per_node=2,
                num_nodes_to_sample=5,
                random_seed=42,
                questions_dataset_path=questions_dataset_path,
            )

        # Assert
        assert result == []

    @patch("nzambe.helpers.eval.generate_question_context_pairs")
    @patch("nzambe.helpers.eval.load_index_from_storage")
    @patch("nzambe.helpers.eval.nzambe_settings")
    @patch(
        "nzambe.helpers.eval.DEFAULT_QA_GENERATE_PROMPT_TMPL",
        "Default prompt with questions",
    )
    def test_generate_new_questions_prompt_template(
        self, mock_settings, mock_load_index, mock_generate_pairs, tmp_path
    ):
        """Test that QA prompt template is correctly formatted."""
        # Setup
        mock_settings.eval.qa_generate_prompt = "Custom: {{default_qa_generate_prompt}}"

        index_storage_dir = str(tmp_path)
        with open(os.path.join(index_storage_dir, "nzambe_metadata.json"), "w") as f:
            json.dump({"platform": "huggingface", "embed_model_id": "test-model"}, f)

        mock_index = Mock()
        mock_index.storage_context.vector_store.data.embedding_dict.keys.return_value = [
            "node1"
        ]
        mock_node1 = Mock(node_id="node1")
        mock_index.docstore.get_nodes.return_value = [mock_node1]
        mock_load_index.return_value = mock_index

        mock_qa_dataset = Mock(spec=EmbeddingQAFinetuneDataset)
        mock_generate_pairs.return_value = mock_qa_dataset

        # Act
        with (
            patch("nzambe.helpers.eval.HuggingFaceEmbedding"),
            patch("nzambe.helpers.eval.StorageContext"),
            patch("nzambe.helpers.eval.ollama_llm"),
        ):
            generate_new_questions_from_index(
                ollama_model_name="llama2",
                index_storage_dir=index_storage_dir,
                num_questions_per_node=3,
                num_nodes_to_sample=1,
                random_seed=42,
            )

        # Assert
        # Check that generate_question_context_pairs was called with formatted prompt
        call_kwargs = mock_generate_pairs.call_args[1]
        assert "qa_generate_prompt_tmpl" in call_kwargs
        # The prompt should have the default replaced and " questions" -> " question(s)"
        expected_prompt = "Custom: Default prompt with question(s)"
        assert call_kwargs["qa_generate_prompt_tmpl"] == expected_prompt
        assert call_kwargs["num_questions_per_chunk"] == 3

    @patch("nzambe.helpers.eval.generate_question_context_pairs")
    @patch("nzambe.helpers.eval.load_index_from_storage")
    def test_generate_new_questions_sample_size_limit(
        self, mock_load_index, mock_generate_pairs, tmp_path
    ):
        """Test that requesting more samples than available nodes raises an error."""
        # Setup
        index_storage_dir = str(tmp_path)
        with open(os.path.join(index_storage_dir, "nzambe_metadata.json"), "w") as f:
            json.dump({"platform": "huggingface", "embed_model_id": "test-model"}, f)

        mock_index = Mock()
        # Only 2 candidate nodes available
        mock_index.storage_context.vector_store.data.embedding_dict.keys.return_value = [
            "node1",
            "node2",
        ]
        mock_node1 = Mock(node_id="node1")
        mock_node2 = Mock(node_id="node2")
        mock_index.docstore.get_nodes.return_value = [mock_node1, mock_node2]
        mock_load_index.return_value = mock_index

        # Act - request 10 samples but only 2 available should raise an error
        with (
            patch("nzambe.helpers.eval.HuggingFaceEmbedding"),
            patch("nzambe.helpers.eval.StorageContext"),
            patch("nzambe.helpers.eval.ollama_llm"),
            pytest.raises(ValueError, match="Sample larger than population"),
        ):
            generate_new_questions_from_index(
                ollama_model_name="llama2",
                index_storage_dir=index_storage_dir,
                num_questions_per_node=2,
                num_nodes_to_sample=10,  # More than available
                random_seed=42,
            )
