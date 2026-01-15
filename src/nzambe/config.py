"""
Configuration management for Nzambe RAG server.
Uses Pydantic Settings with YAML configuration files and environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv

# Load .env file if it exists (primarily for local development)
load_dotenv()


class LangfuseConfig(BaseModel):
    """Langfuse observability configuration."""

    # Using validation_alias allows these to be found even if NZAMBE_ prefix is set globally
    public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    host: str = Field(
        default="http://localhost:3000", validation_alias="LANGFUSE_BASE_URL"
    )


class LLMConfig(BaseModel):
    """LLM and embedding model configuration."""

    model_name: str = "gemma3:1b-it-qat"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    context_window: int = 4096
    request_timeout: float = 360.0
    embed_batch_size: int = 10


class IndexConfig(BaseModel):
    """Vector index configuration."""

    chunk_overlap: int = 120
    paragraph_separator: str = "\n\n"
    insert_batch_size: int = 2048

    # chunk_size is computed dynamically from embedding model max_length
    # so it's not included here as a configurable parameter


class QueryEngineConfig(BaseModel):
    """Query engine configuration."""

    similarity_top_k: int = 2
    response_mode: str = "compact"
    similarity_cutoff: float | None = None
    qa_prompt: str
    refine_prompt: str


class DataConfig(BaseModel):
    """Data paths configuration."""

    folder_path: str = ".debug/data/bible"
    books_subfolder: str = "books"

    @field_validator("folder_path")
    @classmethod
    def resolve_path(cls, v: str) -> str:
        """Resolve relative paths to absolute paths."""
        path = Path(v)
        if not path.is_absolute():
            # Resolve relative to project root (3 levels up from this file)
            project_root = Path(__file__).parents[2]
            path = project_root / v
        return str(path)


class EvalConfig(BaseModel):
    """Evaluation model configuration."""

    ollama_model: str = "gemma3:1b-it-qat"
    ollama_base_url: str = "http://localhost:11434"
    embeddings_provider: str = "huggingface"
    embeddings_model: str = "BAAI/bge-small-en-v1.5"
    qa_generate_prompt: str


class Settings(BaseSettings):
    """
    Root settings class for Nzambe RAG server.
    Loads configuration from YAML files based on the environment.
    """

    model_config = SettingsConfigDict(
        env_prefix="NZAMBE_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # Allows overrides like NZAMBE__LLM__MODEL_NAME
        case_sensitive=False,
        extra="ignore",
    )

    env: str = Field(default="local", alias="NZAMBE_ENV")

    # Nested configuration groups
    llm: LLMConfig
    index: IndexConfig
    query_engine: QueryEngineConfig
    langfuse: LangfuseConfig = LangfuseConfig()
    data: DataConfig
    eval: EvalConfig

    def __init__(self, **kwargs):
        """
        Load configuration from YAML files and merge with environment variables.
        """
        # Determine the config directory (relative to the project root)
        project_root = Path(__file__).parents[2]
        config_dir = project_root / "config"

        # Load base configuration
        base_config_path = config_dir / "base.yaml"
        if not base_config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {base_config_path}. It must exist at the root of the project "
                f"in config/base.yaml."
            )
        config_data = self._load_yaml(base_config_path)

        # Load environment-specific configuration if it exists
        env = os.getenv("NZAMBE_ENV", "local")
        env_config_path = config_dir / f"{env}.yaml"
        if env_config_path.exists():
            env_config = self._load_yaml(env_config_path)
            config_data = self._deep_merge(config_data, env_config)

        # Merge with any provided kwargs
        config_data = self._deep_merge(config_data, kwargs)

        # Initialize with merged configuration
        super().__init__(**config_data)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        """Load YAML configuration file."""
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """
        Deep merge two dictionaries.
        Override values take precedence over base values.
        """
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = Settings._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


# Global settings instance
# Import this instance throughout the application
nzambe_settings = Settings()
