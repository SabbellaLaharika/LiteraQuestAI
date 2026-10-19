import os
from pathlib import Path
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "agent_run.log"
LITERATURE_REVIEW_FILE = OUTPUT_DIR / "literature_review.md"
REFERENCES_BIB_FILE = OUTPUT_DIR / "references.bib"


class Settings(BaseModel):
    # Redis configuration
    redis_host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = Field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = Field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))

    # Ollama configuration
    ollama_host: str = Field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))

    # Optional external API keys
    anthropic_api_key: str | None = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    semantic_scholar_api_key: str | None = Field(default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
    crossref_mailto: str = Field(default_factory=lambda: os.getenv("CROSSREF_MAILTO", "research_assistant@literaquest.local"))

    # Web Dashboard configuration
    web_host: str = Field(default_factory=lambda: os.getenv("WEB_HOST", "0.0.0.0"))
    web_port: int = Field(default_factory=lambda: int(os.getenv("WEB_PORT", "8000")))


settings = Settings()
