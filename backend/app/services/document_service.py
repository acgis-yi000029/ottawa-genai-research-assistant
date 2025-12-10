"""
📄 Document Service

Handles PDF document processing, text extraction, and vector storage.
Now properly uses the Repository layer for data persistence.
"""

import hashlib
import logging
import math
import os
import random
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import pdfplumber
import pypdf
from app.core.config import Settings
from app.models.document import Document, DocumentMetadata, DocumentChunk
from app.repositories.document_repository import DocumentRepository, DocumentChunkRepository
from app.services.vector_store import VectorStore
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class DocumentService:
    """Service for handling document processing and management."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.upload_dir = settings.UPLOAD_DIR
        self.document_repo = DocumentRepository()
        self.chunk_repo = DocumentChunkRepository()
        self.vector_store = VectorStore()
        
        # Load sentence-transformer model once
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Ensure upload directory exists
        os.makedirs(self.upload_dir, exist_ok=True)

    async def process_document(
        self,
        file_path: str,
        doc_id: str,
        filename: str,
        user_id: str,
        language: str | None = "en",
    ) -> dict[str, Any]:
        """
        Process an uploaded PDF document and save to repository.

        Args:
            file_path: Path to the uploaded file
            doc_id: Unique document identifier
            filename: Original filename
            language: Document language

        Returns:
            Dictionary containing processing results
        """
        try:
            # Extract text from PDF
            extracted_text = await self._extract_text_from_pdf(file_path)

            # Get document metadata
            pdf_metadata = await self._extract_metadata(file_path)

            # Get file size
            file_size = os.path.getsize(file_path)

            # Create document metadata
            metadata = DocumentMetadata(
                pages=pdf_metadata.get("page_count", 0),
                author=pdf_metadata.get("author"),
                word_count=len(extracted_text.split()) if extracted_text else 0,
            )

            # NEW: Text Blocking
            chunks = await self._chunk_text(extracted_text)

            # NEW: Generate embeddings for each block
            embeddings = await self._generate_embeddings(chunks)

            # NEW: Metadata used by the vector library (dict, not Pydantic objects)
            vector_metadata: dict[str, Any] = {
                "pages": metadata.pages,
                "author": metadata.author,
                "word_count": metadata.word_count,
                "language": language,
                "filename": filename,
            }

            # NEW: Write chunks and vectors to the vector library and chunk repository
            await self._store_in_vector_db(
                doc_id=doc_id,
                chunks=chunks,
                embeddings=embeddings,
                metadata=vector_metadata,
            )

            # Create document model
            document = Document(
                id=doc_id,
                user_id=user_id,
                filename=filename,
                title=pdf_metadata.get("title", filename),
                description=f"Processed PDF document: {filename}",
                file_path=file_path,
                file_size=file_size,
                mime_type="application/pdf",
                upload_date=datetime.now(timezone.utc),
                last_modified=datetime.now(timezone.utc),
                status="processed",
                language=language,
                content=extracted_text,
                chunks=chunks,  # Originally it was [], now it saves a list of text blocks
                metadata=metadata,
                vector_id=self.vector_store.collection.name,  # Originally it was None, now record the collection name used 
            )

            # Save through repository
            self.document_repo.create(document)

            processing_result = {
                "status": "processed",
                "doc_id": doc_id,
                "filename": filename,
                "language": language,
                "page_count": metadata.pages,
                "word_count": metadata.word_count,
                "text_preview": (
                    extracted_text[:500] + "..."
                    if len(extracted_text) > 500
                    else extracted_text
                ),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "file_size": file_size,
            }

            return processing_result

        except Exception as e:
            # Create failed document record
            failed_document = Document(
                id=doc_id,
                user_id=user_id, 
                filename=filename,
                title=filename,
                description=f"Failed to process: {str(e)}",
                file_path=file_path,
                file_size=(
                    os.path.getsize(file_path) if os.path.exists(file_path) else 0
                ),
                mime_type="application/pdf",
                upload_date=datetime.now(timezone.utc),
                last_modified=datetime.now(timezone.utc),
                status="error",
                language=language or "unknown",
                content="",
                chunks=[],
                metadata=DocumentMetadata(),
                vector_id=None,
            )

            # Save failed document
            self.document_repo.create(failed_document)

            return {
                "status": "error",
                "doc_id": doc_id,
                "filename": filename,
                "error": str(e),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }

    async def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF file."""

        extracted_text = ""

        try:
            # Try with pdfplumber first (better for complex layouts)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"

            # If pdfplumber didn't extract much text, try pypdf
            if len(extracted_text.strip()) < 100:
                with open(file_path, "rb") as file:
                    pdf_reader = pypdf.PdfReader(file)
                    extracted_text = ""

                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"

            return extracted_text.strip()

        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    async def _extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from PDF file."""

        metadata = {}

        try:
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)

                # Basic metadata
                metadata["page_count"] = len(pdf_reader.pages)
                metadata["file_size"] = os.path.getsize(file_path)

                # PDF metadata if available
                if pdf_reader.metadata:
                    pdf_meta = pdf_reader.metadata
                    metadata["title"] = pdf_meta.get("/Title", "")
                    metadata["author"] = pdf_meta.get("/Author", "")
                    metadata["subject"] = pdf_meta.get("/Subject", "")
                    metadata["creator"] = pdf_meta.get("/Creator", "")

                    # Convert creation date if available
                    if "/CreationDate" in pdf_meta:
                        metadata["creation_date"] = str(pdf_meta["/CreationDate"])

            return metadata

        except Exception as e:
            return {"error": f"Error extracting metadata: {str(e)}"}

    async def _chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> list[str]:
        """Split text into overlapping chunks for vector storage."""

        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size

            # Try to break at sentence boundaries
            if end < text_length:
                # Look for sentence endings near the chunk boundary
                sentence_endings = [". ", "! ", "? ", "\n\n"]
                best_break = end

                for ending in sentence_endings:
                    pos = text.rfind(ending, start + chunk_size - 200, end)
                    if pos > start:
                        best_break = pos + len(ending)
                        break

                end = best_break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    async def _generate_embeddings(self, chunks: list[str]) -> list[list[float]]:
        """Generate embeddings for text chunks using a simple hash-based approach."""


        if not chunks:
            return []

        try:
            # Encode all chunks in one batch
            embeddings = self.embedding_model.encode(
                chunks,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # Ensure numpy float32 array with shape (n_chunks, embedding_dim)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Hard fail: if embeddings cannot be generated, let the caller see the error
            raise

    async def _store_in_vector_db(
        self,
        doc_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: dict[str, Any],
    ) -> bool:
        """Store document chunks and embeddings in vector database."""

        try:
            if not chunks or not embeddings:
                logger.warning("No chunks or embeddings to store for document %s", doc_id)
                return False

            if len(chunks) != len(embeddings):
                raise ValueError(
                    f"Chunks and embeddings length mismatch: "
                    f"{len(chunks)} vs {len(embeddings)}"
                )

            # Preparing per-chunk metadata for the vector library
            pages = metadata.get("pages")
            author = metadata.get("author")
            word_count = metadata.get("word_count")
            language = metadata.get("language")
            filename = metadata.get("filename")

            metadatas: list[dict[str, Any]] = []
            for i, chunk in enumerate(chunks):
                meta: dict[str, Any] = {
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "page_number": 1,  # placeholder
                    }
                if pages is not None:
                    meta["pages"] = int(pages)
                if author is not None:
                    meta["author"] = str(author)
                if word_count is not None:
                    meta["word_count"] = int(word_count)
                if language is not None:
                    meta["language"] = str(language)
                if filename is not None:
                    meta["filename"] = str(filename)

                metadatas.append(meta)

            # Write to the Chroma vector library
            chunk_ids = self.vector_store.add_documents(
                doc_id=doc_id,
                chunks=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )

             # Write to local chunks persistent repository（Monk JSON）
            now = datetime.now(timezone.utc)
            for i, (chunk_id, chunk_text, emb) in enumerate(
                zip(chunk_ids, chunks, embeddings)
            ):
                chunk_model = DocumentChunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=i,
                    page_number=1,
                    content=chunk_text,
                    embedding=emb,
                    metadata=DocumentMetadata(
                        pages=pages,
                        author=author,
                        word_count=word_count,
                    ),
                    created_at=now,
                )
                self.chunk_repo.create(chunk_model)

            return True

        except Exception as e:
            raise Exception(f"Error storing in vector database: {str(e)}")

    async def search_documents(
        self, query: str, limit: int = 5, similarity_threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        """Search for relevant document chunks based on query using cosine similarity."""

        try:
            # 1. Generate embedding for the query
            query_embedding = await self._generate_embeddings([query])
            if not query_embedding:
                return []

            query_vector = query_embedding[0]

            # 2. Use VectorStore to perform similarity search on stored chunks
            raw_results = self.vector_store.search(
                query_embedding=query_vector,
                top_k=limit,
                doc_id=None,
                where=None,
            )
            logger.info(f"VectorStore returned {len(raw_results)} items")

            if not raw_results:
                return []

            results: list[dict[str, Any]] = []

            # 3. Format results and enrich with document info
            for item in raw_results:
                try:
                    # Similarity score from vector store
                    similarity = float(item.get("similarity_score", 0.0))
                    if similarity < similarity_threshold:
                        continue

                    # Chunk text
                    chunk_text = item.get("text") or ""
                    metadata = item.get("metadata") or {}

                    doc_id = metadata.get("doc_id")
                    page_number = metadata.get("page_number") or 1

                    # Look up document record from repository
                    document = self.document_repo.find_by_id(doc_id) if doc_id else None

                    results.append(
                        {
                            "doc_id": doc_id,
                            "filename": (
                                document.filename if document else metadata.get("filename")
                            ),
                            "title": document.title if document else None,
                            "chunk_text": chunk_text,
                            "page_number": page_number,
                            "similarity_score": round(similarity, 3),
                            "metadata": metadata,
                        }
                    )

                except Exception as inner_e:
                    logger.warning(f"Error building search result from vector item: {inner_e}")
                    continue

            # 4. Sort by similarity score and return top N
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Error in search_documents: {e}")
            raise Exception(f"Error searching documents: {str(e)}")

    def _calculate_cosine_similarity(
        self, vec1: list[float], vec2: list[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        try:

            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))

            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))

            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = dot_product / (magnitude1 * magnitude2)

            # Ensure the result is between 0 and 1
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    async def get_document_content(
        self, doc_id: str, page: int | None = None
    ) -> dict[str, Any]:
        """Get content from a specific document using repository."""

        try:
            document = self.document_repo.find_by_id(doc_id)
            if not document:
                raise ValueError(f"Document with ID {doc_id} not found")

            return {
                "doc_id": doc_id,
                "content": document.content,
                "page": page,
                "total_pages": document.metadata.pages if document.metadata else 0,
                "filename": document.filename,
                "title": document.title,
                "status": document.status,
            }

        except Exception as e:
            raise Exception(f"Error retrieving document content: {str(e)}")

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all associated data."""

        try:
            # Get document info first
            document = self.document_repo.find_by_id(doc_id)
            if not document:
                return False

            # 1. Remove file from storage
            if os.path.exists(document.file_path):
                os.remove(document.file_path)

            # 2. Delete record from repository
            deleted = self.document_repo.delete(doc_id)

            # 3. Remove chunks from chunk repository
            try:
                deleted_chunks = self.chunk_repo.delete_by_doc_id(doc_id)
                logger.info(
                    "Deleted %s chunks for document %s from chunk repository",
                    deleted_chunks,
                    doc_id,
                )
            except Exception as e:
                logger.error(
                    "Error deleting chunks for document %s: %s",
                    doc_id,
                    e,
                )

            # 4. Remove vectors from vector database
            try:
                self.vector_store.delete_document(doc_id)
            except Exception as e:
                logger.error(
                    "Error deleting vectors for document %s: %s",
                    doc_id,
                    e,
                )

            return deleted

        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")

    async def get_documents_list(self) -> list[dict[str, Any]]:
        """Get list of all uploaded documents from repository."""

        try:
            documents = self.document_repo.get_all()

            return [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "title": doc.title,
                    "size": doc.file_size,
                    "upload_date": doc.upload_date.isoformat(),
                    "processed": doc.status == "processed",
                    "page_count": doc.metadata.pages if doc.metadata else None,
                    "language": doc.language,
                    "status": doc.status,
                }
                for doc in documents
            ]

        except Exception as e:
            raise Exception(f"Error retrieving documents list: {str(e)}")

    async def get_document_by_filename(self, filename: str) -> dict[str, Any] | None:
        """Get document by filename."""

        try:
            document = self.document_repo.find_by_filename(filename)
            if not document:
                return None

            return {
                "id": document.id,
                "filename": document.filename,
                "title": document.title,
                "size": document.file_size,
                "upload_date": document.upload_date.isoformat(),
                "status": document.status,
                "language": document.language,
                "content_preview": (
                    document.content[:500] + "..."
                    if len(document.content) > 500
                    else document.content
                ),
            }

        except Exception as e:
            raise Exception(f"Error retrieving document by filename: {str(e)}")

    async def get_documents_by_status(self, status: str) -> list[dict[str, Any]]:
        """Get documents by processing status."""

        try:
            documents = self.document_repo.find_by_status(status)

            return [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "title": doc.title,
                    "status": doc.status,
                    "upload_date": doc.upload_date.isoformat(),
                    "language": doc.language,
                }
                for doc in documents
            ]

        except Exception as e:
            raise Exception(f"Error retrieving documents by status: {str(e)}")

    async def update_document_status(self, doc_id: str, status: str) -> bool:
        """Update document processing status."""

        try:
            document = self.document_repo.find_by_id(doc_id)
            if not document:
                return False

            document.status = status
            document.last_modified = datetime.now(timezone.utc)

            updated_doc = self.document_repo.update(doc_id, document)
            return updated_doc is not None

        except Exception as e:
            raise Exception(f"Error updating document status: {str(e)}")

    async def reprocess_document(self, doc_id: str, user_id: str) -> dict[str, Any]:
        """Reprocess a document that has already been uploaded."""

        try:
            # Get document info
            document = self.document_repo.find_by_id(doc_id)
            if not document:
                raise ValueError(f"Document with ID {doc_id} not found")

            # Verify ownership
            if document.user_id != user_id:
                raise ValueError("Document does not belong to the current user")

            # Check if file still exists
            if not os.path.exists(document.file_path):
                raise ValueError("Original document file not found")

            # Update status to processing
            self.document_repo.update_status(doc_id, "processing")

            # Reprocess the document using the existing process_document method
            processing_result = await self.process_document(
                file_path=document.file_path,
                doc_id=doc_id,
                filename=document.filename,
                user_id=user_id,
                language=document.language
            )

            return {
                "status": processing_result["status"],
                "processing_id": doc_id,
                "message": "Document reprocessing started successfully"
            }

        except Exception as e:
            # Update status to error if reprocessing fails
            self.document_repo.update_status(doc_id, "error")
            raise Exception(f"Error reprocessing document: {str(e)}")

    def get_user_documents(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> list[Document]:
        """
        Get documents for a specific user.

        Args:
            user_id: User identifier
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            List of user documents
        """
        try:
            # Get all documents for the user
            all_user_docs = []
            all_docs = self.document_repo.find_all()

            # Filter by user_id
            for doc in all_docs:
                if doc.user_id == user_id:
                    all_user_docs.append(doc)

            # Apply pagination
            return all_user_docs[offset: offset + limit]

        except Exception as e:
            raise Exception(f"Error getting user documents: {str(e)}")

    def get_document_by_id(self, doc_id: str, user_id: str) -> Document | None:
        """
        Get a specific document by ID, ensuring it belongs to the user.
        
        Args:
            doc_id: Document identifier
            user_id: User identifier
            
        Returns:
            Document if found and belongs to user, None otherwise
        """
        try:
            # Get document from repository
            document = self.document_repo.find_by_id(doc_id)
            
            # Check if document exists and belongs to the user
            if document and document.user_id == user_id:
                return document
            
            return None
            
        except Exception as e:
            raise Exception(f"Error getting document by ID: {str(e)}")
