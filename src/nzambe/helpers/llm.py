import tiktoken
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from transformers import AutoTokenizer

from nzambe.config import nzambe_settings


def setup_llama_index_llms(
    embed_batch_size: int,
    llm_model_name: str,
    context_window: int,
    request_timeout: float,
):
    """
    Setup LlamaIndex global LLM settings using provided configuration.

    Args:
        embed_batch_size: Batch size for embedding generation
        llm_model_name: Query LLM model name (deprecated, use config instead)
        context_window: LLM context window size (deprecated, use config instead)
        request_timeout: Request timeout in seconds (deprecated, use config instead)

    Returns:
        Tuple of (embedding_model_name, llm_name, chunk_size, chunk_overlap, tokenizer)
    """

    if nzambe_settings.llm.embedding_model.platform == "ollama":
        Settings.embed_model = OllamaEmbedding(
            model_name=nzambe_settings.llm.embedding_model.name,
            embed_batch_size=embed_batch_size,
            **(nzambe_settings.llm.embedding_model.other_kwargs or {}),
        )
        Settings.tokenizer = AutoTokenizer.from_pretrained(
            nzambe_settings.llm.embedding_model.tokenizer
        )
    elif nzambe_settings.llm.embedding_model.platform == "openai":
        Settings.embed_model = OpenAIEmbedding(
            model=nzambe_settings.llm.embedding_model.name,
            api_key=nzambe_settings.llm.embedding_model.api_key,
            embed_batch_size=embed_batch_size,
            **(nzambe_settings.llm.embedding_model.other_kwargs or {}),
        )
        Settings.tokenizer = tiktoken.get_encoding(
            nzambe_settings.llm.embedding_model.tokenizer
        ).encode
    else:
        raise Exception(
            f"Unsupported embedding model platform: {nzambe_settings.llm.embedding_model.platform}"
        )

    if nzambe_settings.llm.query_model.platform == "ollama":
        Settings.llm = Ollama(
            model=nzambe_settings.llm.query_model.name,
            request_timeout=nzambe_settings.llm.query_model.request_timeout,
            context_window=nzambe_settings.llm.query_model.context_window,
            **(nzambe_settings.llm.query_model.other_kwargs or {}),
        )
    elif nzambe_settings.llm.query_model.platform == "openai":
        Settings.llm = OpenAI(
            model=nzambe_settings.llm.query_model.name,
            api_key=nzambe_settings.llm.query_model.api_key,
            timeout=nzambe_settings.llm.query_model.request_timeout,
            **(nzambe_settings.llm.query_model.other_kwargs or {}),
        )
    else:
        raise Exception(
            f"Unsupported LLM platform: {nzambe_settings.llm.query_model.platform}"
        )


def ollama_llm(model_name: str = "llama3.1"):
    return Ollama(
        model=model_name,
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=4096,
    )
