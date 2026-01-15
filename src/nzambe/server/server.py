import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request
from llama_index.core import Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import BaseModel

from nzambe import __version__
from nzambe.config import nzambe_settings
from nzambe.constants import APP_DESCRIPTION, APP_TITLE
from nzambe.helpers.data_processing import build_documents_index
from nzambe.helpers.llm import setup_llama_index_llms
from nzambe.helpers.observability import setup_observability

# Turn on debug logging to check what the llamaindex app is doing
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    nodes: list[str]


class HealthResponse(BaseModel):
    status: str
    query_engine_loaded: bool


class RetrieverResponse(BaseModel):
    nodes: list[NodeWithScore]


def get_query_engine(request: Request) -> RetrieverQueryEngine:
    engine = getattr(request.app.state, "query_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Query engine not initialized.")
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup event to load index and models once at application startup.
    """
    logger.info("Starting server initialization...")

    # Setup observability
    setup_observability()

    # Load configuration from settings
    data_folder_path = Path(nzambe_settings.data.folder_path)
    custom_qa_prompt = RichPromptTemplate(nzambe_settings.query_engine.qa_prompt)
    custom_refine_prompt = RichPromptTemplate(
        nzambe_settings.query_engine.refine_prompt
    )

    # Setup LLMs
    embedding_model_name = nzambe_settings.llm.embedding_model_name
    llm_name = nzambe_settings.llm.model_name
    model_tokenizer = setup_llama_index_llms(
        embedding_model_name=embedding_model_name,
        embed_batch_size=nzambe_settings.llm.embed_batch_size,
        llm_model_name=llm_name,
        context_window=nzambe_settings.llm.context_window,
        request_timeout=nzambe_settings.llm.request_timeout,
    )

    logger.info(f"Using embedding model: {embedding_model_name}")
    logger.info(f"Using LLM: {llm_name}")

    # Index storage directory
    index_storage_dir = (
        data_folder_path / f"documents_index_{embedding_model_name.replace('/', '--')}"
    )
    input_data_directory = None
    input_data_files = list(
        (data_folder_path / nzambe_settings.data.books_subfolder).glob("*.txt")
    )

    # Build or load index
    index = await build_documents_index(
        index_storage_dir,
        input_data_directory,
        input_data_files,
        document_split_chunk_size=Settings.embed_model.max_length
        - 10,  # chunk size based on embedding model max length
        document_split_chunk_overlap=nzambe_settings.index.chunk_overlap,
        model_tokenizer=model_tokenizer,
        paragraph_separator=nzambe_settings.index.paragraph_separator,
        insert_batch_size=nzambe_settings.index.insert_batch_size,
    )

    # Create a query engine
    node_postprocessors = []
    if nzambe_settings.query_engine.similarity_cutoff is not None:
        node_postprocessors.append(
            SimilarityPostprocessor(
                similarity_cutoff=nzambe_settings.query_engine.similarity_cutoff
            )
        )

    app.state.query_engine = index.as_query_engine(
        similarity_top_k=nzambe_settings.query_engine.similarity_top_k,
        response_mode=nzambe_settings.query_engine.response_mode,
        node_postprocessors=node_postprocessors,
        text_qa_template=custom_qa_prompt or DEFAULT_TEXT_QA_PROMPT,
        refine_template=custom_refine_prompt or DEFAULT_REFINE_PROMPT,
    )
    logger.info("Server initialization complete. Ready to accept requests.")

    yield

    # Cleanup (if needed)
    logger.info("Server shutting down...")


# Create FastAPI app with lifespan
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check(query_engine: RetrieverQueryEngine = Depends(get_query_engine)):
    """
    Health check endpoint to verify the server is running.
    """
    return HealthResponse(
        status="healthy", query_engine_loaded=query_engine is not None
    )


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    query_engine: RetrieverQueryEngine = Depends(get_query_engine),
):
    """
    Query endpoint to ask questions to the RAG system.
    """
    if query_engine is None:
        raise HTTPException(
            status_code=503, detail="Query engine not initialized. Server starting up."
        )

    try:
        logger.info(f"Received query: {request.question}")
        response = await query_engine.aquery(request.question)
        answer = str(response)
        source_nodes = [str(nodewscore) for nodewscore in response.source_nodes]
        logger.debug(f"Generated answer: {answer[:100]}...")
        return QueryResponse(answer=answer, nodes=source_nodes)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/retrieve_docs", response_model=RetrieverResponse)
async def retrieve_docs(
    request: QueryRequest,
    query_engine: RetrieverQueryEngine = Depends(get_query_engine),
):
    """
    Retrieve docs from the index matching the query
    """
    if query_engine is None:
        raise HTTPException(
            status_code=503,
            detail="retriever_query engine not initialized. Server starting up.",
        )

    try:
        logger.info(f"Received query: {request.question}")
        nodes_with_score: list[NodeWithScore] = await query_engine.aretrieve(
            QueryBundle(request.question)
        )
        return RetrieverResponse(nodes=nodes_with_score)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
