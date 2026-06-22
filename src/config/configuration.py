from functools import lru_cache

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_")
    api_key: SecretStr
    # Primary model for the agents (router, retrieval, dynamic, reservation).
    model: str = "gpt-4o"
    # Lighter model for low-stakes nodes (e.g. out-of-scope replies).
    mini_model: str = "gpt-4o-mini"
    temperature: float = 0.0


class PostgreSQLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_")
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: SecretStr = SecretStr("mysecretpassword")
    database: str = "postgres"
    # Schema pinned on the connection search_path (env var POSTGRES_SCHEMA).
    schema_name: str = Field(default="parking", validation_alias="POSTGRES_SCHEMA")

    # Connection-pool sizing/timeout (env vars POSTGRES_POOL_*).
    pool_min_size: int = 2
    pool_max_size: int = 5
    # Seconds a caller waits for a free connection before timing out.
    pool_timeout: float = 5.0

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
    return Settings()


def build_chat_model(model: str | None = None, temperature: float | None = None):
    from langchain_openai import ChatOpenAI
    openai = get_settings().openai
    return ChatOpenAI(
        model=model or openai.model,
        temperature=openai.temperature if temperature is None else temperature,
        api_key=openai.api_key,
    )