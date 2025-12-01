"""Chat data models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ConversationMetadata(BaseModel):
    """Conversation metadata model."""

    topic: str | None = None
    message_count: int = 0


class Conversation(BaseModel):
    """Conversation model."""

    id: str
    user_id: str
    title: str
    language: str = Field(default="en", pattern="^(en|fr)$")
    created_at: datetime
    updated_at: datetime
    status: str = Field(default="active", pattern="^(active|completed|archived)$")
    metadata: ConversationMetadata = Field(default_factory=ConversationMetadata)


class MessageMetadata(BaseModel):
    """Message metadata model."""

    language: str | None = None
    intent: str | None = None
    sources: list[str] | None = None
    confidence: float | None = None


class Message(BaseModel):
    """Message model."""

    id: str
    conversation_id: str
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str
    conversation_id: str | None = None
    language: str = Field(default="en", pattern="^(en|fr)$")
    user_id: str
    context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    """Chat response model."""

    id: str
    response: str
    language: str
    timestamp: datetime
    sources: list[str] | None = None
    charts: dict[str, Any] | None = None


class ConversationSummary(BaseModel):
    """Conversation summary model."""

    id: str
    title: str
    last_message: str
    last_activity: datetime
    message_count: int
    language: str
