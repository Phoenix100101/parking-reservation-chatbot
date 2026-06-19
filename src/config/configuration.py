from functools import lru_cache

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    """OpenAI / LLM configuration (env vars prefixed OPENAI_)."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_")

    api_key: SecretStr
    # Primary model for the agents (router, retrieval, dynamic, reservation).
    model: str = "gpt-4o"
    # Lighter model for low-stakes nodes (e.g. out-of-scope replies).
    mini_model: str = "gpt-4o-mini"
    temperature: float = 0.0


class PostgreSQLSettings(BaseSettings):
    """Postgres connection settings (env vars prefixed POSTGRES_)."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: SecretStr = SecretStr("mysecretpassword")
    database: str = "postgres"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> str:
        """psycopg/SQLAlchemy-style connection URL."""
        password = self.password.get_secret_value()
        return (
            f"postgresql://{self.user}:{password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class WeaviateSettings(BaseSettings):
    """Weaviate connection settings (env vars prefixed WEAVIATE_)."""

    model_config = SettingsConfigDict(env_prefix="WEAVIATE_")

    host: str = "localhost"
    http_port: int = 8080
    grpc_port: int = 50051


class GuardrailSettings(BaseSettings):
    """PII / safety guardrail configuration (env vars prefixed GUARDRAIL_).

    Set-valued fields accept a comma-separated string from the environment, e.g.
    ``GUARDRAIL_MASK_LABELS=PHONE,SSN,CREDIT_CARD``.
    """

    # enable_decoding=False: keep set-valued env vars as raw strings so the
    # validator below can split them on commas (instead of JSON-parsing).
    model_config = SettingsConfigDict(env_prefix="GUARDRAIL_", enable_decoding=False)

    enabled: bool = True
    injection_check: bool = True
    spacy_model: str = "en_core_web_sm"

    # Labels the output guardrail redacts. Structured PII only (regex-matched,
    # language-agnostic). Excludes EMAIL/DATE/TIME so reservation fields survive,
    # and excludes NER labels like PERSON which the English-only spaCy model
    # mislabels on other languages. Add PERSON via the env var if appropriate.
    mask_labels: frozenset[str] = frozenset(
        {"PHONE", "CREDIT_CARD", "SSN", "IP_ADDRESS"}
    )
    # Labels that cause the input guardrail to reject the message outright.
    block_input_labels: frozenset[str] = frozenset({"CREDIT_CARD", "SSN"})

    @field_validator("mask_labels", "block_input_labels", mode="before")
    @classmethod
    def _split_labels(cls, value: object) -> object:
        if isinstance(value, str):
            return frozenset(s.strip().upper() for s in value.split(",") if s.strip())
        return value


class Settings(BaseSettings):
    """Top-level application settings, loaded from the environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Root log level (env var LOG_LEVEL). One of DEBUG/INFO/WARNING/ERROR.
    log_level: str = "INFO"

    openai: OpenAISettings = Field(default_factory=OpenAISettings)  # type: ignore[arg-type]
    postgres: PostgreSQLSettings = Field(default_factory=PostgreSQLSettings)
    weaviate: WeaviateSettings = Field(default_factory=WeaviateSettings)
    guardrail: GuardrailSettings = Field(default_factory=GuardrailSettings)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()


def build_chat_model(model: str | None = None, temperature: float | None = None):
    """Build a ``ChatOpenAI`` from settings, with optional per-call overrides.

    Centralises the API key, default model, and temperature so nodes never
    hardcode them. Pass ``model`` / ``temperature`` to override a single field
    (e.g. a node that needs the lighter model or a higher temperature).
    """
    # Imported lazily so importing settings doesn't pull in langchain.
    from langchain_openai import ChatOpenAI

    openai = get_settings().openai
    return ChatOpenAI(
        model=model or openai.model,
        temperature=openai.temperature if temperature is None else temperature,
        api_key=openai.api_key,
    )