import logging

from langfuse._client.get_client import get_client
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

from nzambe.config import nzambe_settings

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
        nzambe_settings.langfuse.public_key is None
        or nzambe_settings.langfuse.secret_key is None
    ):
        logger.warning("Langfuse keys not configured. Skipping observability setup.")
        return

    setup_langfuse_client()
    # This hooks into LlamaIndex and auto-logs every step
    # This third-party instrumentation automatically captures LlamaIndex operations
    # and exports OpenTelemetry (OTel) spans to Langfuse.
    LlamaIndexInstrumentor().instrument()
