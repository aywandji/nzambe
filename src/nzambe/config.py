"""
Configuration management for Nzambe RAG server.
Uses Pydantic Settings with YAML configuration files and environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any, Literal

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


class EmbeddingModel(BaseModel):
    """Embedding model configuration."""

    name: str
    platform: str
    tokenizer: str
    embed_batch_size: int = 10
    api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    other_kwargs: dict | None = None

    def __str__(self):
        return f"{self.platform}--{self.name}--{self.tokenizer}"


class QueryModel(BaseModel):
    """Query model configuration for RAG final response generation."""

    name: str
    platform: str
    context_window: int = 4096
    request_timeout: float = 360.0
    api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    other_kwargs: dict | None = None

    def __str__(self):
        return f"{self.platform}--{self.name}"


class LLMConfig(BaseModel):
    """LLM and embedding model configuration."""

    query_model: QueryModel
    embedding_model: EmbeddingModel


class IndexConfig(BaseModel):
    """Vector index configuration."""

    type: Literal["memory", "s3vectors_index"] = "memory"
    chunk_overlap: int = 120
    paragraph_separator: str = "\n\n"
    insert_batch_size: int = 2048
    # remote index configuration
    s3vectors_bucket_arn: str | None = Field(
        default=None, validation_alias="S3_VECTORS_BUCKET_ARN"
    )
    s3vectors_index_arn: str | None = Field(
        default=None, validation_alias="S3_VECTORS_INDEX_ARN"
    )
    s3vectors_index_data_type: str | None = Field(
        default=None, validation_alias="S3_VECTORS_INDEX_DATA_TYPE"
    )
    s3vectors_index_distance_metric: str | None = Field(
        default=None, validation_alias="S3_VECTORS_INDEX_DISTANCE_METRIC"
    )


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
            # Resolve relative to the current working directory (where the app is run from)
            # This works for Docker (WORKDIR /app) and local dev (running from root)
            path = Path.cwd() / v
        return str(path)


class EvalConfig(BaseModel):
    """Evaluation model configuration."""

    ollama_model: str = "gemma3:1b-it-qat"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: EmbeddingModel
    qa_generate_prompt: str


class NzambeSettings(BaseSettings):
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
        populate_by_name=True,
    )

    env: str = Field(validation_alias="NZAMBE_ENV")

    # Nested configuration groups
    llm: LLMConfig
    index: IndexConfig
    query_engine: QueryEngineConfig
    langfuse: LangfuseConfig = LangfuseConfig()
    data: DataConfig
    eval: EvalConfig | None = None

    def __init__(self, **kwargs):
        """
        Load configuration from YAML files and merge with environment variables.
        """
        # Look for config in CWD or explicit Env Var
        # 1. Check for explicit env var
        config_dir_env = os.getenv("NZAMBE_CONFIG_DIR")
        if config_dir_env:
            config_dir = Path(config_dir_env)
        else:
            # 2. Fallback to the 'config' folder in the current working directory
            # In Docker, WORKDIR is /app, so this resolves to /app/config
            config_dir = Path.cwd() / "config"

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
        config_data["env"] = env
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
                result[key] = NzambeSettings._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


# Global settings instance
# Import this instance throughout the application
nzambe_settings = NzambeSettings()
