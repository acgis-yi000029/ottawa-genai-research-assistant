"""
📄 Document Management API Endpoints

Handles PDF document upload, processing, and management.
"""

import logging
import os
import uuid
from datetime import datetime, timezone

from app.api.auth import get_current_user
from app.core.config import get_settings
from app.models.user import User
from app.services.document_service import DocumentService
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


# Response Models
class DocumentInfo(BaseModel):
    id: str
    filename: str
    size: int
    upload_date: datetime
    processed: bool
    page_count: int | None = None
    language: str | None = None


class DocumentList(BaseModel):
    documents: list[DocumentInfo]
    total: int


class UploadResponse(BaseModel):
    id: str
    filename: str
    size: int
    message: str
    processing_status: str
    status: str = "uploaded"  

class DocumentSearchRequest(BaseModel):
    query: str
    limit: int = 5
    similarity_threshold: float = 0.7

class DocumentSearchResult(BaseModel):
    doc_id: str | None = None
    filename: str | None = None
    title: str | None = None
    chunk_text: str
    page_number: int
    similarity_score: float
    metadata: dict | None = None


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    language: str | None = "en",
    current_user: User = Depends(get_current_user),
    settings=Depends(get_settings),
):
    """
    Upload a PDF document for processing.

    - **file**: PDF file to upload (max 50MB)
    - **language**: Document language (en/fr)
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)

        max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        if file_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB",
            )
        
        # Generate unique ID and save file
        doc_id = str(uuid.uuid4())
        file_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}_{file.filename}")

        with open(file_path, "wb") as f:
            f.write(content)
        
        # Initialize document service and process file
        doc_service = DocumentService(settings)
        
        processing_result = await doc_service.process_document(
            file_path=file_path,
            doc_id=doc_id,
            filename=file.filename,
            user_id=current_user.id,
            language=language,
        )

        return UploadResponse(
            id=doc_id,
            filename=file.filename,
            size=file_size,
            message="Document uploaded successfully",
            processing_status=processing_result["status"],
            status="uploaded",
        )

    except HTTPException:
        # Re-raise HTTP exceptions (these are expected validation errors)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error uploading document: {str(e)}"
        )


@router.get("/list", response_model=DocumentList)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    settings=Depends(get_settings),
):
    """
    Get list of uploaded documents for the current user.

    - **limit**: Maximum number of documents to return
    - **offset**: Number of documents to skip (for pagination)
    """
    try:
        doc_service = DocumentService(settings)
        user_documents = doc_service.get_user_documents(
            user_id=current_user.id, limit=limit, offset=offset
        )

        # Convert to DocumentInfo format
        document_infos = []
        for doc in user_documents:
            document_infos.append(
                DocumentInfo(
                    id=doc.id,
                    filename=doc.filename,
                    size=doc.file_size,
                    upload_date=doc.upload_date,
                    processed=(doc.status == "processed"),
                    page_count=doc.metadata.pages,
                    language=doc.language,
                )
            )

        # Get total count of user documents (not just the paginated results)
        all_user_docs = doc_service.get_user_documents(user_id=current_user.id, limit=1000, offset=0)
        total_count = len(all_user_docs)
        
        return DocumentList(documents=document_infos, total=total_count)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing documents: {str(e)}"
        )


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    settings=Depends(get_settings),
):
    """
    Get information about a specific document.

    - **doc_id**: Unique document identifier
    """
    try:
        doc_service = DocumentService(settings)
        document = doc_service.get_document_by_id(doc_id, current_user.id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentInfo(
            id=document.id,
            filename=document.filename,
            size=document.file_size,
            upload_date=document.upload_date,
            processed=(document.status == "processed"),
            page_count=document.metadata.pages if document.metadata else None,
            language=document.language,
        )

    except Exception:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    settings=Depends(get_settings),
):
    """
    Delete a document and its associated data.

    - **doc_id**: Unique document identifier
    """
    try:
        doc_service = DocumentService(settings)
        
        # Check if document exists and belongs to user
        document = doc_service.get_document_by_id(doc_id, current_user.id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete the document using the service
        success = await doc_service.delete_document(doc_id)
        
        if success:
            return {"message": f"Document {doc_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting document: {str(e)}"
        )


@router.post("/{doc_id}/reprocess")
async def reprocess_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    settings=Depends(get_settings),
):
    """
    Reprocess a document (useful if processing failed or needs updating).

    - **doc_id**: Unique document identifier
    """
    try:
        doc_service = DocumentService(settings)
        
        # Check if document exists and belongs to user
        document = doc_service.get_document_by_id(doc_id, current_user.id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Reprocess the document
        result = await doc_service.reprocess_document(doc_id, current_user.id)
        
        return {
            "message": f"Document {doc_id} queued for reprocessing",
            "status": result.get("status", "processing"),
            "processing_id": result.get("processing_id"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reprocessing document: {str(e)}"
        )


@router.get("/{doc_id}/content")
async def get_document_content(
    doc_id: str,
    page: int | None = None,
    current_user: User = Depends(get_current_user),
    settings=Depends(get_settings),
):
    """
    Get extracted text content from a document.

    - **doc_id**: Unique document identifier
    - **page**: Specific page number (optional, returns all if not specified)
    """
    try:
        doc_service = DocumentService(settings)
        
        # Check if document exists and belongs to user
        document = doc_service.get_document_by_id(doc_id, current_user.id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document content
        content = await doc_service.get_document_content(doc_id, page)
        
        return content

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving document content: {str(e)}"
        )

@router.post("/search", response_model=list[DocumentSearchResult])
async def search_documents_endpoint(
    request: DocumentSearchRequest,
    current_user: User = Depends(get_current_user),
    settings=Depends(get_settings),
):
    """
    Search processed documents using semantic search on stored chunks.

    - **query**: Search text
    - **limit**: Maximum number of results
    - **similarity_threshold**: Minimum similarity score (0–1) to include in results
    """
    try:
        doc_service = DocumentService(settings)

        results = await doc_service.search_documents(
            query=request.query,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
        )

        return results

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}",
        )