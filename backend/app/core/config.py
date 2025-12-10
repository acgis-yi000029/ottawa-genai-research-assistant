"""
Configuration Management

Centralized configuration using Pydantic Settings for type safety.
"""

from functools import lru_cache

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # Project Information
    PROJECT_NAME: str = Field(default="Ottawa GenAI Research Assistant")
    API_V1_STR: str = Field(default="/api/v1")

    # Environment
    DEBUG: bool = Field(default=True)
    ENVIRONMENT: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")

    # API Keys
    OPENAI_API_KEY: str = Field(default="")
    GROQ_API_KEY: str = Field(default="")
    GEMINI_API_KEY: str = Field(default="")
    GOOGLE_CLIENT_ID: str = Field(default="")

    # Database
    DATABASE_URL: str = Field(default="sqlite:///./ottawa_genai.db")

    # Vector Database
    # Note: Actual path is managed by VectorConfig using monk/vector_db
    CHROMA_PERSIST_DIR: str = Field(default="monk/vector_db")

    # Security
    SECRET_KEY: str = Field(default="change-this-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)

    # CORS
    ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"]
    )

    # File Upload
    MAX_FILE_SIZE_MB: int = Field(default=50)
    UPLOAD_DIR: str = Field(default="uploads")
    ALLOWED_FILE_TYPES: list[str] = Field(default=["pdf"])
    
    # Data Storage - All data must be stored in monk/ directory
    MONK_DATA_DIR: str = Field(default="monk")
    ENFORCE_MONK_CONSTRAINT: bool = Field(default=True)

    # AI Model Settings
    DEFAULT_AI_MODEL: str = Field(default="gpt-3.5-turbo")
    MAX_TOKENS: int = Field(default=4000)
    TEMPERATURE: float = Field(default=0.7)

    # Logging
    LOG_FILE: str = Field(default="./logs/app.log")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
