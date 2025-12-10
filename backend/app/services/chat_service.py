"""
Chat Service with Free AI Integration

This service provides AI-powered chat functionality using completely free AI providers:
- Groq AI (ultra-fast, free)
- Google Gemini (high-quality, free)
- Local fallback when no API keys are available

All services are completely free and provide enterprise-grade AI responses.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.config import Settings
from app.models.chat import Conversation, Message
from app.repositories.chat_repository import (ConversationRepository,
                                              MessageRepository)
from app.services.ai_providers import GeminiService, GroqService
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)


class ChatService:
    """
    Chat service with free AI integration and intelligent fallback
    """
    # Minimum similarity required to trust ED PDF matches
    RAG_MIN_SIMILARITY: float = 0.4
    
    def __init__(self, settings: Settings):
        """Initialize chat service with free AI providers"""
        self.settings = settings
        self.conversation_repository = ConversationRepository()
        self.message_repository = MessageRepository()
        self.document_service = DocumentService(settings)
        
        # Initialize free AI services
        self.groq_service = GroqService(settings)
        self.gemini_service = GeminiService(settings)
        
        # Service priority: Groq first (fastest), then Gemini (high quality)
        self.ai_services = [
            ("Groq", self.groq_service),
            ("Gemini", self.gemini_service)
        ]
        
        # Check available services
        available_services = [name for name, service in self.ai_services if service.is_available]
        
        if available_services:
            logger.info(f"Free AI services available: {', '.join(available_services)}")
        else:
            logger.warning("No AI services available - will use mock responses")
    
    async def process_message(
        self, 
        message: str, 
        language: str = "en",
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user message and generate AI response
        
        Args:
            message: User input message
            language: Response language (en, fr, zh)
            conversation_id: Optional conversation ID
            user_id: Optional user ID
            
        Returns:
            Dict containing AI response and metadata
        """
        # Store document excerpts for citation in the response
        display_snippets: list[str] = []

         # Unified fixed text when ED PDFs do not provide a trustworthy match
        NO_ED_MATCH_TEXT = (
            "No relevant information was found in the current ED update reports."
        )

        confidence_label: Optional[str] = None

        try:
            # Try document search
            rag_results: list[dict[str, Any]] = []
            try:
                rag_results = await self.document_service.search_documents(
                    query=message,
                    limit=20,
                    similarity_threshold=0.0,
                )
            except Exception as rag_error:
                logger.warning(f"RAG search failed, using LLM only: {rag_error}")
                rag_results = []

            # If search results are found, construct the context and sources.
            sources: list[str] = []
            model_input: str

            # 1) No RAG results at all -> return fixed message
            if not rag_results:
                logger.info(
                    "No RAG results found; returning fixed ED 'not found' message "
                    "without calling any external LLM."
                )
                response: Dict[str, Any] = {
                    "text": NO_ED_MATCH_TEXT,
                    "provider": "RAG-filter",
                    "model": "ed-pdf-only",
                    "language": language,
                    "tokens_used": 0,
                    "cost": 0.0,
                    "sources": [],
                    "charts": {},
                }
                if conversation_id and user_id:
                    await self._save_conversation(
                        conversation_id, user_id, message, response["text"], language
                    )
                return response
            
            # 2) Have results but best similarity is below threshold
            try:
                best_score = max(
                    float(item.get("similarity_score") or 0.0)
                    for item in rag_results
                )
            except Exception as score_error:
                logger.warning(
                    f"Failed to read similarity_score from RAG results: {score_error}"
                )
                best_score = 0.0

            if best_score < self.RAG_MIN_SIMILARITY:
                logger.info(
                    "RAG best similarity %.3f below threshold %.3f; "
                    "returning fixed ED 'not found' message without calling any external LLM",
                    best_score,
                    self.RAG_MIN_SIMILARITY,
                )
                response = {
                    "text": NO_ED_MATCH_TEXT,
                    "provider": "RAG-filter",
                    "model": "ed-pdf-only",
                    "language": language,
                    "tokens_used": 0,
                    "cost": 0.0,
                    "sources": [],
                    "charts": {},
                    "confidence": confidence_label,
                }
                if conversation_id and user_id:
                    await self._save_conversation(
                        conversation_id, user_id, message, response["text"], language
                    )
                return response
            
            mid = self.RAG_MIN_SIMILARITY + 0.05
            high = self.RAG_MIN_SIMILARITY + 0.10

            if best_score >= high:
                confidence_label = "High"
            elif best_score >= mid:
                confidence_label = "Medium"
            else:
                confidence_label = "Low"

            
            # 3）Have results and best similarity is upper threshold
            if rag_results:
                context_parts: list[str] = []

                for item in rag_results:
                    filename = (
                        item.get("title")
                        or item.get("filename")
                        or item.get("doc_id")
                        or "Document"
                    )
                    chunk_text = (item.get("chunk_text") or "").strip()
                    if not chunk_text:
                        continue
                    
                    snippet = f"[{filename}]\n{chunk_text}"
                    context_parts.append(snippet)

                # Select a single best citation from the top-ranked chunk
                top_item = rag_results[0]
                top_filename = (
                    top_item.get("title")
                    or top_item.get("filename")
                    or top_item.get("doc_id")
                    or "Document"
                )
                top_chunk_text = (top_item.get("chunk_text") or "").strip()
                if top_chunk_text:
                    top_label = f"{top_filename}"
                    # Only keep this one label as the citation source
                    sources = [top_label]
                    # Only this snippet will be exposed in charts["citations"]
                    display_snippets = [f"[{top_label}]\n{top_chunk_text}"]

                if context_parts:
                    context_text = "\n\n".join(context_parts)

                    # Prevent prompts from being too long
                    max_chars = 20000
                    if len(context_text) > max_chars:
                        context_text = context_text[:max_chars]

                    # Combine the document content and the original question into a message and send
                    model_input = (
                            "Use the following internal document excerpts as context. "
                            "If they are relevant, base your answer on them. "
                            "If they do not contain the answer, answer normally.\n\n"
                            f"{context_text}\n\n"
                            f"User question: {message}"
                    )
                else: 
                    # empty text
                    model_input = message
                    sources = []
            else:
                # No document hits
                model_input = message

            # call free AI providers with the prepared model_input
            for service_name, service in self.ai_services:
                if service.is_available:
                    try:
                        logger.info(f"Trying {service_name} AI service...")
                        response = await service.generate_response(model_input, language)
                        
                        # Use the final answer to reselect the best-matching chunk for citation.
                        if rag_results and isinstance(response.get("text"), str):
                            aligned_sources, aligned_snippets = self._align_citations_with_answer(
                                answer_text=response["text"],
                                rag_results=rag_results,
                            )
                            # Only override the original top1 setting when a better alignment result is found.
                            if aligned_sources:
                                sources = aligned_sources
                            if aligned_snippets:
                                display_snippets = aligned_snippets

                        # If documentation was used, include the source in the response for API/frontend use.
                        if sources:
                            existing_sources = response.get("sources") or []
                            # Remove duplicates and maintain order
                            merged_sources = list(
                                dict.fromkeys(existing_sources + sources)
                            )
                            response["sources"] = merged_sources
                        
                        # Attach original document excerpts for citations
                        if display_snippets:
                            existing_charts = response.get("charts") or {}
                            if not isinstance(existing_charts, dict):
                                existing_charts = {}
                            existing_charts["citations"] = display_snippets
                            response["charts"] = existing_charts
                            citations_text = "\n\n---\nOriginal excerpts used:\n\n" + "\n\n".join(display_snippets)
                            response["text"] = (response.get("text") or "").rstrip() + citations_text

                        # Attach confidence level if available
                        if confidence_label is not None:
                            response["confidence"] = confidence_label

                        # Save conversation if IDs provided
                        if conversation_id and user_id:
                            await self._save_conversation(
                                conversation_id, user_id, message, response["text"], language
                            )
    
                        logger.info(f"{service_name} response generated successfully")
                        return response
                        
                    except Exception as e:
                        logger.warning(f"{service_name} failed: {e}")
                        continue
            
            # If all AI services fail, use fallback
            logger.warning("All AI services failed, using mock response")
            fallback = await self._get_fallback_response(message, language)
            if sources:
                fallback["sources"] = sources
            if display_snippets:
                existing_charts = fallback.get("charts") or {}
                if not isinstance(existing_charts, dict):
                    existing_charts = {}
                existing_charts["citations"] = display_snippets
                fallback["charts"] = existing_charts
            return fallback
            
        except Exception as e:
            logger.error(f"Chat service error: {e}")
            fallback = await self._get_fallback_response(message, language)
            return fallback
    
    async def _get_fallback_response(self, message: str, language: str) -> Dict[str, Any]:
        """
        Generate fallback response when AI services are unavailable
        """
        fallback_responses = {
            "en": "Thank you for your question about Ottawa's economic development. Based on our local data, I can provide detailed analysis. What specific information would you like to know about business opportunities, investment climate, or economic programs?",
            "fr": "Merci pour votre question sur le développement économique d'Ottawa. Basé sur nos données locales, je peux fournir une analyse détaillée. Quelles informations spécifiques aimeriez-vous connaître sur les opportunités d'affaires, le climat d'investissement ou les programmes économiques?",
        }
        
        return {
            "text": fallback_responses.get(language, fallback_responses["en"]),
            "provider": "Mock",
            "model": "local-data",
            "language": language,
            "tokens_used": 50,
            "cost": 0.0,
            "note": "This is a fallback response. Please set up free API keys for Groq and Gemini for best experience."
        }
    
    def _align_citations_with_answer(
        self,
        answer_text: str,
        rag_results: list[dict[str, Any]],
    ) -> tuple[list[str], list[str]]:
        """
        Based on the model's final answer, select the best-matching chunk from `rag_results` 
        and return (sources, display_snippets). If no match is found, return ([], []) 
        to allow the caller to backtrack.
        """
        try:
            if not answer_text or not rag_results:
                return [], []

            # Use only the main body of the answer, avoiding automatic matching with already included citations.
            lower_answer = answer_text.lower()
            if "---" in lower_answer:
                lower_answer = lower_answer.split("---", 1)[0]

            # 1) Extract numbers from the answer and keep only “real” values
            number_pattern = re.compile(r"\d[\d,.]*")
            all_numbers = number_pattern.findall(lower_answer)

            # Keep numbers that look like years or larger values (>= 4 digits),
            # to avoid noise from small numbers.
            raw_numbers: list[str] = []
            for n in all_numbers:
                digits_only = re.sub(r"[^\d]", "", n)
                if len(digits_only) >= 4:
                    raw_numbers.append(n)

            # Normalized numbers: remove commas and trailing dots.
            normalized_numbers = {
                n.replace(",", "").strip(".")
                for n in raw_numbers
                if n
            }

            # 2）Extracting keywords
            word_pattern = re.compile(r"[a-z]{4,}")
            answer_tokens = word_pattern.findall(lower_answer)
            answer_token_set = set(answer_tokens)

            best_item = None
            best_score = 0.0

            for item in rag_results:
                chunk_text = (item.get("chunk_text") or "")
                if not chunk_text:
                    continue

                chunk_lower = chunk_text.lower()
                chunk_no_commas = chunk_lower.replace(",", "")

                score = 0.0

                # Number matching: Each matched number is given a higher weight.
                for num in raw_numbers:
                    if num and num in chunk_lower:
                        score += 3.0
                for num in normalized_numbers:
                    if num and num in chunk_no_commas:
                        score += 3.0

                # Keyword overlap: low weight, used to break ties when numbers are the same or missing.
                chunk_tokens = set(word_pattern.findall(chunk_lower))
                common_tokens = answer_token_set & chunk_tokens
                if common_tokens:
                    score += 0.1 * len(common_tokens)

                if score > best_score:
                    best_score = score
                    best_item = item

            # If a chunk with a score > 0 is found, it will be used for citation.
            if best_item and best_score > 0.0:
                filename = (
                    best_item.get("title")
                    or best_item.get("filename")
                    or best_item.get("doc_id")
                    or "Document"
                )
                chunk_text = (best_item.get("chunk_text") or "").strip()
                if chunk_text:
                    label = f"{filename}"
                    sources = [label]
                    display_snippets = [f"[{label}]\n{chunk_text}"]
                    return sources, display_snippets

            # If no match is found, the caller should revert to the original top 1 logic.
            return [], []
        except Exception:
            # Any anomalies will not affect the main process; a rollback will be performed directly.
            return [], []
        
    async def _save_conversation(
        self, 
        conversation_id: str, 
        user_id: str, 
        user_message: str, 
        ai_response: str,
        language: str
    ) -> None:
        """Save conversation to repository"""
        try:
            # Create user message
            user_msg = Message(
                id=f"msg_{datetime.now(timezone.utc).timestamp()}",
                conversation_id=conversation_id,
                role="user",
                content=user_message,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Create AI response message
            ai_msg = Message(
                id=f"msg_{datetime.now(timezone.utc).timestamp()}_ai",
                conversation_id=conversation_id,
                role="assistant",
                content=ai_response,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Save messages
            self.message_repository.create(user_msg)
            self.message_repository.create(ai_msg)
            
        except Exception as e:
            logger.warning(f"Failed to save conversation: {e}")
    
    async def get_conversation_history(
        self, 
        conversation_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get conversation history"""
        try:
            messages = self.message_repository.find_by_conversation(
                conversation_id, limit
            )
            return [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def create_conversation(
        self, 
        user_id: str, 
        title: Optional[str] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """Create new conversation"""
        try:
            now = datetime.now(timezone.utc)
            conversation = Conversation(
                id=f"conv_{now.timestamp()}",
                user_id=user_id,
                title=title or f"New Chat - {now.strftime('%Y-%m-%d %H:%M')}",
                created_at=now,
                updated_at=now,
                language=language
            )
            
            self.conversation_repository.create(conversation)
            
            return {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat(),
                "language": conversation.language
            }
            
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise Exception(f"Could not create conversation: {str(e)}")
    
    def get_ai_service_status(self) -> Dict[str, Any]:
        """Get status of all AI services"""
        services_status = []
        
        for service_name, service in self.ai_services:
            services_status.append(service.get_status())
        
        return {
            "total_services": len(self.ai_services),
            "available_services": len([s for s in services_status if s["available"]]),
            "services": services_status,
            "fallback_available": True,
            "recommendation": "Set up free API keys for Groq and Gemini for best experience"
        }
