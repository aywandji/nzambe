import datetime
import json
import logging
import os
import random

from datasets import Dataset
from langchain_ollama import ChatOllama
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core.llama_dataset.legacy.embedding import (
    EmbeddingQAFinetuneDataset,
    DEFAULT_QA_GENERATE_PROMPT_TMPL,
)
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

from ragas import evaluate
from ragas.embeddings.base import embedding_factory
from ragas.metrics import answer_relevancy, faithfulness
from tqdm import tqdm

from nzambe.config import nzambe_settings
from nzambe.helpers.llm import ollama_llm
from nzambe.helpers.observability import setup_langfuse_client

logger = logging.getLogger(__name__)


def run_nightly_benchmark(last_n_hours: int, num_traces_limit: int = 1):
    if nzambe_settings.eval is None:
        raise Exception("Evaluation settings not found in the config.")

    langfuse = setup_langfuse_client()

    # 1. Fetch today's traces (random sample of 50)
    logger.info("Fetching traces from Langfuse...")
    # fetch last n hours traces
    traces = langfuse.api.trace.list(
        limit=num_traces_limit,
        from_timestamp=datetime.datetime.now() - datetime.timedelta(hours=last_n_hours),
    )
    if len(traces.data) == 0:
        logger.warning(
            f"No traces found for the last {last_n_hours} hours. Please check the time range and try again."
        )
        return

    # 2. Format for Ragas
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "trace_id": [],  # Keep track so we can push scores back!
    }

    for t in tqdm(traces.data):
        # Extract data from the trace object (simplified logic)
        # You'd typically parse the 'generation' and 'retrieval' spans here
        if t.output and t.input:
            # get retriever observations (old traces used .retrieve)
            trace_observations = (
                langfuse.api.observations.get_many(
                    trace_id=t.id,
                    type="RETRIEVER",
                    name="VectorIndexRetriever.aretrieve",
                ).dict()["data"]
                + langfuse.api.observations.get_many(
                    trace_id=t.id,
                    type="RETRIEVER",
                    name="VectorIndexRetriever.retrieve",
                ).dict()["data"]
            )
            if len(trace_observations) == 0:
                logger.warning(
                    f"No observations found for trace {t.id}. Please check the trace ID and try again. Skipping..."
                )
                continue

            # extract used contexts from the trace "retriever" observations
            contexts_positions = set()
            retrieval_documents_attributes = {}
            for observation in trace_observations:
                for k, v in observation["metadata"]["attributes"].items():
                    if k.startswith("retrieval"):
                        contexts_positions.add(k.split(".")[2])
                        retrieval_documents_attributes[k] = v

            contexts = [
                retrieval_documents_attributes[
                    f"retrieval.documents.{p}.document.content"
                ]
                for p in sorted(list(contexts_positions))
            ]

            data["question"].append(t.input)
            data["answer"].append(t.output)
            data["contexts"].append(contexts)
            data["trace_id"].append(t.id)

    # 3. Run Evaluation

    dataset = Dataset.from_dict(data)
    ollama_eval_model = ChatOllama(
        model=nzambe_settings.eval.ollama_model,
        base_url=nzambe_settings.eval.ollama_base_url,
    )
    embeddings_model = embedding_factory(
        nzambe_settings.eval.embedding_model.platform,
        nzambe_settings.eval.embedding_model.name,
    )

    ragas_metrics = [faithfulness, answer_relevancy]
    results = evaluate(
        dataset,
        metrics=ragas_metrics,
        llm=ollama_eval_model,
        embeddings=embeddings_model,
    )

    # 4. Push Scores BACK to the Dashboard
    logger.info("Uploading scores to dashboard...")
    df = results.to_pandas()
    for trace_position, row in df.iterrows():
        trace_id = data["trace_id"][trace_position]
        for metric in ragas_metrics:
            langfuse.create_score(
                trace_id=trace_id,
                name=metric.name,
                value=row[metric.name],
                score_id=f"{trace_id}-{metric.name}-{ollama_eval_model.model}",  # should add the app-release version
                # which represents all the eval models
            )


def generate_new_questions_from_index(
    ollama_model_name: str,
    index_storage_dir: str,
    num_questions_per_node: int,
    num_nodes_to_sample: int,
    random_seed: int,
    questions_dataset_path: str | None = None,
):
    if nzambe_settings.eval is None:
        raise Exception("Evaluation settings not found in the config.")

    try:
        with open(os.path.join(index_storage_dir, "nzambe_metadata.json"), "r") as f:
            metadata = json.load(f)
        platform = metadata["platform"]
        if nzambe_settings.env in ("local", "test") and platform == "ollama":
            Settings.embed_model = OllamaEmbedding(
                model_name=metadata["embed_model_name"]
            )
            index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=index_storage_dir),
            )
        else:
            raise Exception(f"Unsupported platform: {platform}")

    except FileNotFoundError:
        raise Exception(
            "Error: Could not find nzambe_metadata.json. Cannot determine original embedding model."
        )

    # 1. Retrieve all nodes from the index
    # Note: For a very large index, this might be slow. A better approach for
    # massive datasets might be to sample the underlying vector store directly,
    # but this is the standard LlamaIndex approach.
    current_dataset = (
        EmbeddingQAFinetuneDataset.from_json(questions_dataset_path)
        if (questions_dataset_path and os.path.exists(questions_dataset_path))
        else None
    )
    existing_node_ids = []
    if current_dataset is not None:
        existing_node_ids = list(current_dataset.corpus.keys())

    # 2. Filter out nodes that already have existing questions
    # The doc_id is the unique identifier for the node in the index's docstore/graph store.
    vector_store: SimpleVectorStore = index.storage_context.vector_store  # type: ignore
    candidate_node_ids = [
        node_id
        for node_id in vector_store.data.embedding_dict.keys()
        if node_id not in existing_node_ids
    ]

    logger.info(f"Nodes to skip (existing questions): {len(existing_node_ids)}")
    logger.info(f"Candidate nodes for generation: {len(candidate_node_ids)}")

    # 3. Randomly select nodes for question generation
    random.seed(random_seed)

    if len(candidate_node_ids) == 0:
        logger.info("No new nodes available for question generation.")
        return []

    selected_nodes: list[TextNode] = random.sample(  # type: ignore
        index.docstore.get_nodes(candidate_node_ids), num_nodes_to_sample
    )
    logger.info(
        f"Randomly selected {len(selected_nodes)} new nodes for question generation."
    )

    qa_prompt_template = nzambe_settings.eval.qa_generate_prompt.replace(
        "{{default_qa_generate_prompt}}",
        DEFAULT_QA_GENERATE_PROMPT_TMPL.replace(" questions", " question(s)"),
    )
    qa_dataset = generate_question_context_pairs(
        selected_nodes,
        llm=ollama_llm(ollama_model_name),
        num_questions_per_chunk=num_questions_per_node,
        qa_generate_prompt_tmpl=qa_prompt_template,
    )

    if questions_dataset_path is not None:
        if current_dataset is not None:
            logger.info("updating the existing questions dataset")
            qa_dataset.queries.update(current_dataset.queries)
            qa_dataset.corpus.update(current_dataset.corpus)
            qa_dataset.relevant_docs.update(current_dataset.relevant_docs)

        qa_dataset.save_json(questions_dataset_path)

    return qa_dataset
