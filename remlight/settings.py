"""REMLight settings with environment variable support."""

import os
from pathlib import Path

# Load .env file from project root
from dotenv import load_dotenv

# Find .env file - check current dir and parent dirs
def _find_env_file() -> Path | None:
    """Find .env file in current or parent directories."""
    current = Path.cwd()
    for _ in range(5):  # Check up to 5 levels
        env_file = current / ".env"
        if env_file.exists():
            return env_file
        current = current.parent
    # Also check package directory
    pkg_dir = Path(__file__).parent.parent
    env_file = pkg_dir / ".env"
    if env_file.exists():
        return env_file
    return None

env_file = _find_env_file()
if env_file:
    load_dotenv(env_file, override=True)  # Override bash profile vars with .env

from pydantic import BaseModel


class PostgresSettings(BaseModel):
    """PostgreSQL connection settings."""

    connection_string: str = os.getenv(
        "POSTGRES__CONNECTION_STRING",
        "postgresql://remlight:remlight@localhost:5432/remlight"
    )

    @property
    def enabled(self) -> bool:
        """Check if postgres is enabled via POSTGRES_ENABLED env var (default: true)."""
        return os.getenv("POSTGRES_ENABLED", "true").lower() == "true"


class LLMSettings(BaseModel):
    """LLM provider settings."""

    default_model: str = os.getenv("LLM__DEFAULT_MODEL", "openai:gpt-4o-mini")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    temperature: float = float(os.getenv("LLM__TEMPERATURE", "0.5"))
    max_iterations: int = int(os.getenv("LLM__MAX_ITERATIONS", "20"))


class OTELSettings(BaseModel):
    """OpenTelemetry settings for Phoenix integration."""

    enabled: bool = os.getenv("OTEL__ENABLED", "false").lower() == "true"
    service_name: str = os.getenv("OTEL__SERVICE_NAME", "remlight-api")
    collector_endpoint: str = os.getenv("OTEL__COLLECTOR_ENDPOINT", "http://localhost:6006")
    protocol: str = os.getenv("OTEL__PROTOCOL", "http")  # http or grpc
    export_timeout: int = int(os.getenv("OTEL__EXPORT_TIMEOUT", "30000"))


class Settings(BaseModel):
    """Application settings."""

    postgres: PostgresSettings = PostgresSettings()
    llm: LLMSettings = LLMSettings()
    otel: OTELSettings = OTELSettings()
    environment: str = os.getenv("ENVIRONMENT", "development")


settings = Settings()
