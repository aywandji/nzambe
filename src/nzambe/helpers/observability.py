import logging
import os

from langfuse._client.get_client import get_client
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor


logger = logging.getLogger(__name__)


def setup_langfuse_client():
    # set up langfuse client. The keys are supposed to be set as environment variables.
    langfuse = get_client()

    # Verify connection
    if langfuse.auth_check():
        logger.info("Langfuse client is authenticated and ready!")
    else:
        raise Exception(
            "Langfuse authentication failed. Please check your credentials and host."
        )

    return langfuse


def setup_observability():
    # Skip Langfuse setup if keys are not configured
    if (
        os.getenv("LANGFUSE_PUBLIC_KEY") is None
        or os.getenv("LANGFUSE_SECRET_KEY") is None
    ):
        logger.warning(
            "Langfuse env vars not configured. Skipping observability setup."
        )
        return

    try:
        setup_langfuse_client()
    except Exception as e:
        logger.error(
            f"Failed to setup Langfuse client: {str(e)}. Skipping observability setup."
        )
        return
    # This hooks into LlamaIndex and auto-logs every step
    # This third-party instrumentation automatically captures LlamaIndex operations
    # and exports OpenTelemetry (OTel) spans to Langfuse.
    LlamaIndexInstrumentor().instrument()
