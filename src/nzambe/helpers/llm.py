from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
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
        llm_model_name: Ollama LLM model name
        context_window: LLM context window size
        request_timeout: Request timeout in seconds

    Returns:
        Tuple of (embedding_model_name, llm_name, chunk_size, chunk_overlap, tokenizer)
    """

    if nzambe_settings.platform == "ollama":
        Settings.embed_model = OllamaEmbedding(
            model_name=nzambe_settings.llm.embedding_model.name,
            embed_batch_size=embed_batch_size,
        )
        Settings.tokenizer = AutoTokenizer.from_pretrained(
            nzambe_settings.llm.embedding_model.tokenizer
        )
    else:
        raise Exception(f"Unsupported embedding model: {nzambe_settings.embed_model}")

    Settings.llm = Ollama(
        model=llm_model_name,
        request_timeout=request_timeout,
        context_window=context_window,
    )


def ollama_llm(model_name: str = "llama3.1"):
    return Ollama(
        model=model_name,
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=4096,
    )
