"""Document data models."""

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Document metadata model."""

    pages: int | None = None
    author: str | None = None
    version: str | None = None
    section: str | None = None
    subsection: str | None = None
    word_count: int | None = None
    start_char: int | None = None
    end_char: int | None = None


class Document(BaseModel):
    """Document model."""

    id: str
    user_id: str  
    filename: str
    title: str
    description: str
    file_path: str
    file_size: int
    mime_type: str
    upload_date: datetime
    last_modified: datetime
    status: str = Field(
        default="pending", pattern="^(pending|processing|processed|error)$"
    )
    language: str = Field(default="en", pattern="^(en|fr)$")
    tags: list[str] = Field(default_factory=list)
    content: str = Field(default="")  
    chunks: list[str] = Field(default_factory=list)  
    vector_id: str | None = Field(default=None)  
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class DocumentChunk(BaseModel):
    """Document chunk model."""

    id: str
    doc_id: str
    chunk_index: int
    page_number: int
    content: str
    embedding: list[float] | None = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    created_at: datetime


class DocumentUpload(BaseModel):
    """Document upload request model."""

    filename: str
    title: str | None = None
    description: str | None = None
    language: str = Field(default="en", pattern="^(en|fr)$")
    tags: list[str] = Field(default_factory=list)


class DocumentSearch(BaseModel):
    """Document search request model."""

    query: str
    language: str | None = Field(default=None, pattern="^(en|fr)$")
    tags: list[str] | None = None
    max_results: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class DocumentSearchResult(BaseModel):
    """Document search result model."""

    doc_id: str
    filename: str
    chunk_text: str
    page_number: int
    similarity_score: float
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
