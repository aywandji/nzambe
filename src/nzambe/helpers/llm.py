from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from transformers import AutoTokenizer


def setup_llama_index_llms(
    embedding_model_name: str,
    embed_batch_size: int,
    llm_model_name: str,
    context_window: int,
    request_timeout: float,
):
    """
    Setup LlamaIndex global LLM settings using provided configuration.

    Args:
        embedding_model_name: HuggingFace embedding model name
        embed_batch_size: Batch size for embedding generation
        llm_model_name: Ollama LLM model name
        context_window: LLM context window size
        request_timeout: Request timeout in seconds

    Returns:
        Tuple of (embedding_model_name, llm_name, chunk_size, chunk_overlap, tokenizer)
    """
    model_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embedding_model_name, embed_batch_size=embed_batch_size
    )

    Settings.llm = Ollama(
        model=llm_model_name,
        request_timeout=request_timeout,
        context_window=context_window,
    )

    return model_tokenizer


def ollama_llm(model_name: str = "llama3.1"):
    return Ollama(
        model=model_name,
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=4096,
    )
