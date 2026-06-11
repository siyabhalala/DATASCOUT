"""
datascout.infrastructure.config.settings
──────────────────────────────────────────
Complete Settings class for DataScout.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings, Field  # type: ignore


class Settings(BaseSettings):
    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = Field(default="DATASCOUT", alias="APP_NAME")
    app_version: str = Field(default="3.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_key: Optional[str] = Field(default=None, alias="API_KEY")
    api_cors_origins: list[str] = Field(
        default=["*"], alias="API_CORS_ORIGINS"
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_structured: bool = Field(default=True, alias="LOG_STRUCTURED")

    # ── Dataset Providers ─────────────────────────────────────────────────────
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
    openml_api_key: Optional[str] = Field(default=None, alias="OPENML_API_KEY")
    kaggle_username: Optional[str] = Field(default=None, alias="KAGGLE_USERNAME")
    kaggle_key: Optional[str] = Field(default=None, alias="KAGGLE_KEY")

    # ── LLM Providers — Gemini only ───────────────────────────────────────────
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")

    llm_provider: Literal["gemini", "mock"] = Field(
        default="gemini",
        description="LLM provider for explanations only — never used for ranking.",
    )
    llm_model: str = Field(
        default="gemini-2.5-flash",
          description="Gemini model to use"
   )  
    llm_timeout: float = Field(default=25.0)
    llm_max_explanation_tokens: int = Field(default=1024)
    llm_explanation_detail: str = Field(default="standard")

    # ── Storage ───────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///./datascout.db",
        alias="DATABASE_URL",
    )

    # ── Elasticsearch — partner track ─────────────────────────────────────────
    elasticsearch_url: Optional[str] = Field(default=None, alias="ELASTICSEARCH_URL")
    elasticsearch_api_key: Optional[str] = Field(default=None, alias="ELASTICSEARCH_API_KEY")
    elasticsearch_index: str = Field(default="datascout-datasets", alias="ELASTICSEARCH_INDEX")
    elastic_enabled: bool = Field(default=False, alias="ELASTIC_ENABLED")

    # ── Search ────────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    search_timeout_seconds: float = Field(default=28.0)
    adapter_timeout_seconds: float = Field(default=8.0)
    embedding_cache_ttl_seconds: int = Field(default=3600)

    # ── Demo / Hackathon ──────────────────────────────────────────────────────
    demo_mode: bool = Field(default=False, alias="DEMO_MODE")
    demo_max_results: int = Field(default=8)
    demo_warmup_queries: list[str] = Field(
        default_factory=lambda: [
            "image classification dataset",
            "sentiment analysis NLP",
            "tabular classification healthcare",
        ]
    )

    # ── Cloud Run ─────────────────────────────────────────────────────────────
    cloud_run_service: Optional[str] = Field(default=None, alias="K_SERVICE")
    cloud_run_revision: Optional[str] = Field(default=None, alias="K_REVISION")
    gcp_project_id: Optional[str] = Field(default=None, alias="GCP_PROJECT_ID")

    @property
    def is_cloud_run(self) -> bool:
        return self.cloud_run_service is not None

    @property
    def is_demo_environment(self) -> bool:
        return self.demo_mode

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
    }


def _find_env_file() -> str:
    """
    Walk up from this file's directory looking for .env.
    Handles running from any working directory — datascout_project/,
    datascout_project/datascout/, or anywhere else.
    """
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent, here.parent.parent.parent]:
        env = candidate / ".env"
        if env.exists():
            return str(env)
    return ".env"  # last-resort fallback


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    import os
    env_path = _find_env_file()
    # pydantic-settings v2 supports _env_file constructor override.
    # Also manually load via dotenv as belt-and-suspenders for edge cases.
    try:
        from dotenv import load_dotenv  # type: ignore[import]
        load_dotenv(env_path, override=False)  # override=False: existing os.environ wins
    except ImportError:
        pass
    settings = Settings(_env_file=env_path)
    # Push critical credentials into os.environ so third-party SDKs
    # (kaggle, google-generativeai) that read os.environ directly can find them.
    import os
    _env_exports = {
        "GOOGLE_API_KEY":      settings.google_api_key,
        "KAGGLE_USERNAME":     settings.kaggle_username,
        "KAGGLE_KEY":          settings.kaggle_key,
        "HF_TOKEN":            getattr(settings, "hf_token", None),
        "ELASTICSEARCH_URL":   getattr(settings, "elasticsearch_url", None),
        "ELASTICSEARCH_API_KEY": getattr(settings, "elasticsearch_api_key", None),
    }
    for key, val in _env_exports.items():
        if val and not os.environ.get(key):
            os.environ[key] = str(val)
    return settings