from enum import Enum
import os
from pathlib import Path
import re
from typing import List

from pydantic import ConfigDict, field_validator, model_validator
from pydantic_settings import BaseSettings


class AppEnv(str, Enum):
    """Supported application operating modes."""

    TEST = "test"
    DEVELOPMENT = "development"
    TESTNET = "testnet"
    PRODUCTION = "production"


_EVM_ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")
_ZERO_EVM_ADDRESS = "0x0000000000000000000000000000000000000000"

class Settings(BaseSettings):
    """Typed configuration that rejects incomplete live escrow environments."""

    APP_ENV: AppEnv = AppEnv.DEVELOPMENT

    # Database Settings
    DATABASE_URL: str = "sqlite+aiosqlite:///./mediverse.db"

    # Security & Auth Settings
    JWT_SECRET_KEY: str = "supersecretjwtkeychangeitinproduction"
    JWT_ALGORITHM: str = "HS256"
    GUEST_SESSION_EXPIRE_MINUTES: int = 1440

    # AI API Keys
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Redis Settings
    REDIS_URL: str = "redis://localhost:6379/0"

    # Qdrant Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # LiveKit Settings
    LIVEKIT_API_KEY: str = ""
    LIVEKIT_API_SECRET: str = ""
    LIVEKIT_URL: str = ""

    # Web3 Escrow Settings
    POLYGON_AMOY_RPC_URL: str = ""
    POLYGON_MAINNET_RPC_URL: str = ""
    POLYGON_FALLBACK_RPC_URLS: List[str] = []
    WEB3_PROVIDER_URL: str = ""
    PRIVATE_KEY: str = ""
    ESCROW_CONTRACT_ADDRESS: str = ""
    ESCROW_ORACLE_ADDRESS: str = ""
    ESCROW_START_BLOCK: int = 0

    # Object storage settings. Credentials belong only in the deployment secret store.
    OBJECT_STORAGE_PROVIDER: str = "local"
    S3_BUCKET: str = ""
    S3_ENDPOINT: str = ""
    S3_REGION: str = ""

    # Monitoring & Alerting
    SENTRY_DSN: str = ""
    SENTRY_ENVIRONMENT: str = ""
    SENTRY_TRACES_SAMPLE_RATE: float = 0.1
    SLACK_ALERT_WEBHOOK_URL: str = ""

    # Gnosis Safe Multi-Sig
    GNOSIS_SAFE_ADDRESS: str = ""

    @field_validator("OBJECT_STORAGE_PROVIDER")
    @classmethod
    def validate_object_storage_provider(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in {"local", "s3"}:
            raise ValueError("OBJECT_STORAGE_PROVIDER must be either 'local' or 's3'.")
        return normalized

    @model_validator(mode="after")
    def validate_live_environment_configuration(self) -> "Settings":
        if self.APP_ENV not in {AppEnv.TESTNET, AppEnv.PRODUCTION}:
            return self

        if (
            not self.ESCROW_CONTRACT_ADDRESS
            or self.ESCROW_CONTRACT_ADDRESS == _ZERO_EVM_ADDRESS
            or not _EVM_ADDRESS_PATTERN.fullmatch(self.ESCROW_CONTRACT_ADDRESS)
        ):
            raise ValueError(
                "ESCROW_CONTRACT_ADDRESS must be a non-zero deployed EVM address "
                f"when APP_ENV is '{self.APP_ENV.value}'."
            )
        if not (self.POLYGON_AMOY_RPC_URL or self.WEB3_PROVIDER_URL or self.POLYGON_MAINNET_RPC_URL):
            raise ValueError(
                "POLYGON_AMOY_RPC_URL, POLYGON_MAINNET_RPC_URL, or WEB3_PROVIDER_URL must be configured "
                f"when APP_ENV is '{self.APP_ENV.value}'."
            )
        if self.OBJECT_STORAGE_PROVIDER == "s3" and not self.S3_BUCKET:
            raise ValueError("S3_BUCKET is required when OBJECT_STORAGE_PROVIDER is 's3'.")
        return self

    @property
    def requires_live_escrow(self) -> bool:
        """Whether simulated payment responses are forbidden."""
        return self.APP_ENV in {AppEnv.TESTNET, AppEnv.PRODUCTION}

    @property
    def active_rpc_urls(self) -> List[str]:
        """Return prioritized list of RPC URLs for fallback provider."""
        urls = []
        if self.POLYGON_MAINNET_RPC_URL:
            urls.append(self.POLYGON_MAINNET_RPC_URL)
        if self.POLYGON_AMOY_RPC_URL:
            urls.append(self.POLYGON_AMOY_RPC_URL)
        if self.WEB3_PROVIDER_URL:
            urls.append(self.WEB3_PROVIDER_URL)
        # Append fallback URLs
        for url in self.POLYGON_FALLBACK_RPC_URLS:
            if url not in urls:
                urls.append(url)
        return urls

    # Resolve .env from project root regardless of cwd
    _DOTENV = Path(__file__).resolve().parents[3] / ".env"
    model_config = ConfigDict(
        env_file=str(_DOTENV) if _DOTENV.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
