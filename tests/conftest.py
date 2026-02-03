import os
from unittest.mock import MagicMock

from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding

# Set the environment variable to 'test' before any other imports happen.
# This ensures that when your config.py is imported, it sees NZAMBE_ENV=test
os.environ["NZAMBE_ENV"] = "test"
os.environ["AWS_DEFAULT_REGION"] = "test"
os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"


def set_mock_embed_model():
    from nzambe.config import nzambe_settings

    Settings.embed_model = MagicMock(
        spec=OllamaEmbedding, model_name=nzambe_settings.llm.embedding_model.name
    )


set_mock_embed_model()
