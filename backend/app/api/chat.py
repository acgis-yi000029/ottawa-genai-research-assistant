"""
💬 Chat API Endpoints

Handles natural language conversation with AI assistant.
"""

import uuid
from datetime import datetime, timezone

from app.api.auth import get_current_user
from app.core.config import get_settings
from app.models.chat import Conversation, Message, ChatResponse
from app.models.user import User
from app.repositories.chat_repository import (ConversationRepository,
                                              MessageRepository)
from app.repositories.document_repository import DocumentRepository
from app.services.chat_service import ChatService
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter()


# Request/Response Models
class ChatMessage(BaseModel):
    message: str
    language: str | None = "en"
    context: str | None = None



class ChatHistory(BaseModel):
    messages: list[dict]
    total: int


# Public demo endpoint (no authentication required)
@router.post("/demo/message", response_model=ChatResponse)
async def send_demo_message(
    request: ChatMessage,
    settings=Depends(get_settings),
):
    """
    Send a message to the AI assistant (demo mode - no authentication required).
    
    - **message**: The user's question or message
    - **language**: Language preference (en/fr)
    - **context**: Optional context for the conversation
    """
    try:
        chat_service = ChatService(settings)
        
        # Process the message without saving to database
        response = await chat_service.process_message(
            message=request.message,
            language=request.language,
            conversation_id=request.context,
        )
        
        return ChatResponse(
            id=str(uuid.uuid4()),
            response=response["text"],
            language=request.language or "en",
            timestamp=datetime.now(timezone.utc),
            sources=response.get("sources", []),
            charts=response.get("charts"),
            confidence=response.get("confidence"),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing message: {str(e)}"
        )


@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatMessage,
    current_user: User = Depends(get_current_user),
    settings=Depends(get_settings),
):
    """
    Send a message to the AI assistant and get a response.

    - **message**: The user's question or message
    - **language**: Language preference (en/fr)
    - **context**: Optional context for the conversation
    """
    try:
        chat_service = ChatService(settings)
        conversation_repo = ConversationRepository()
        message_repo = MessageRepository()

        # Process the message
        response = await chat_service.process_message(
            message=request.message,
            language=request.language,
            conversation_id=request.context,
        )

        # Create or get conversation
        user_conversations = conversation_repo.find_by_user(current_user.id)
        if not user_conversations:
            # Create new conversation
            conversation_id = str(uuid.uuid4())
            conversation = Conversation(
                id=conversation_id,
                user_id=current_user.id,
                title=(
                    request.message[:50] + "..." 
                    if len(request.message) > 50 
                    else request.message
                ),
                language=request.language or "en",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            conversation_repo.create(conversation)
        else:
            # Use the most recent active conversation
            conversation_id = user_conversations[0].id
            # Update the conversation's updated_at timestamp
            conversation_repo.update(
                conversation_id, {"updated_at": datetime.now(timezone.utc).isoformat()}
            )

        # Save user message
        user_message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="user",
            content=request.message,
            timestamp=datetime.now(timezone.utc),
        )
        message_repo.create(user_message)

        # Save assistant response
        assistant_message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="assistant",
            content=response["text"],
            timestamp=datetime.now(timezone.utc),
        )
        assistant_message.metadata.sources = response.get("sources", [])
        message_repo.create(assistant_message)

        return ChatResponse(
            id=assistant_message.id,
            response=response["text"],
            language=request.language or "en",
            timestamp=datetime.now(timezone.utc),
            sources=response.get("sources", []),
            charts=response.get("charts"),
            confidence=response.get("confidence"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing message: {str(e)}"
        )


@router.get("/history", response_model=ChatHistory)
async def get_chat_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
):
    """
    Get chat history for the current session.

    - **limit**: Maximum number of messages to return
    - **offset**: Number of messages to skip (for pagination)
    """
    try:
        conversation_repo = ConversationRepository()
        message_repo = MessageRepository()

        # Get user's conversations
        conversations = conversation_repo.find_by_user(current_user.id)
        
        all_messages = []
        for conversation in conversations:
            # Get messages for each conversation
            messages = message_repo.find_by_conversation(conversation.id)
            for message in messages:
                all_messages.append({
                    "id": message.id,
                    "conversation_id": message.conversation_id,
                    "role": message.role,
                    "message": (
                        message.content if message.role == "user" else ""
                    ),
                    "response": (
                        message.content if message.role == "assistant" else ""
                    ),
                    "timestamp": message.timestamp.isoformat(),
                    "language": conversation.language,
                    "sources": message.metadata.sources if message.metadata else [],
                })

        # Sort by timestamp (newest first)
        all_messages.sort(key=lambda x: x["timestamp"], reverse=True)

        # Apply pagination
        paginated_messages = all_messages[offset: offset + limit]

        return ChatHistory(
            messages=paginated_messages,
            total=len(all_messages),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving chat history: {str(e)}"
        )


@router.delete("/history")
async def clear_chat_history(current_user: User = Depends(get_current_user)):
    """
    Clear all chat history for the current session.
    """
    try:
        conversation_repo = ConversationRepository()
        message_repo = MessageRepository()

        # Get user's conversations
        conversations = conversation_repo.find_by_user(current_user.id)
        
        # Clear all messages for each conversation
        for conversation in conversations:
            message_repo.clear_conversation_messages(conversation.id)
        
        # Clear all conversations for the user
        conversation_repo.clear_user_conversations(current_user.id)

        return {"message": "Chat history cleared successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clearing chat history: {str(e)}"
        )


@router.get("/suggestions")
async def get_suggestions(current_user: User = Depends(get_current_user)):
    """
    Get suggested questions based on available documents and common queries.
    """
    try:
        # Generate dynamic suggestions based on available documents
        
        doc_repo = DocumentRepository()
        documents = doc_repo.find_all()
        
        # Base suggestions
        suggestions = [
            "What are the main economic trends in Ottawa this quarter?",
            "Show me small business support program statistics",
            "How has employment changed in the tech sector?",
            "What are the key infrastructure projects planned?",
            "Summarize the latest economic development initiatives",
        ]
        
        # Add document-specific suggestions if documents are available
        if documents:
            recent_docs = sorted(
                documents, key=lambda x: x.upload_date, reverse=True
            )[:3]
            for doc in recent_docs:
                suggestions.append(f"What are the key insights from {doc.title}?")
                suggestions.append(f"Summarize the main points in {doc.filename}")

        return {"suggestions": suggestions[:8]}  # Limit to 8 suggestions

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting suggestions: {str(e)}"
        )
