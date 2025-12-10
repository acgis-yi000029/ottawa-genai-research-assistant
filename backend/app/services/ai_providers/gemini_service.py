"""
Google Gemini AI Service Provider
Provides completely free AI chat capabilities using Google's Gemini 1.5 Flash model.

Features:
- Completely free with API key
- High-quality responses
- Multi-language support (English, French, Chinese)
- Rate limit: 60 requests per minute
"""

import logging
from typing import Any, Dict, Optional

import google.generativeai as genai
from app.core.config import Settings

logger = logging.getLogger(__name__)


class GeminiService:
    """Google Gemini AI service provider for free AI chat"""
    
    def __init__(self, settings: Settings):
        """Initialize Gemini service with API key"""
        self.settings = settings
        self.api_key = settings.GEMINI_API_KEY
        self.model = None
        self.is_available = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.is_available = True
                logger.info("Google Gemini AI service initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                self.is_available = False
        else:
            logger.warning("GEMINI_API_KEY not found in environment variables")
    
    async def generate_response(self, message: str, language: str = "en") -> Dict[str, Any]:
        """
        Generate AI response using Google Gemini
        
        Args:
            message: User input message
            language: Response language (en, fr, zh)
            
        Returns:
            Dict with response text and metadata
        """
        if not self.is_available:
            raise Exception("Gemini service not available - check GEMINI_API_KEY")
        
        try:
            # Prepare language-specific prompt
            language_prompts = {
                "en": "You are a helpful AI assistant specialized in Ottawa economic development and business research. Please provide detailed, accurate information.",
                "fr": "Vous êtes un assistant IA spécialisé dans le développement économique et la recherche commerciale d'Ottawa. Veuillez fournir des informations détaillées et précises."
            }
            
            system_prompt = language_prompts.get(language, language_prompts["en"])
            full_prompt = f"{system_prompt}\n\nUser question: {message}"
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            return {
                "text": response.text,
                "provider": "Google Gemini",
                "model": "gemini-1.5-flash",
                "language": language,
                "tokens_used": len(response.text.split()),  # Approximate token count
                "cost": 0.0  # Completely free
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise Exception(f"Gemini service error: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status and capabilities"""
        return {
            "name": "Google Gemini",
            "model": "gemini-1.5-flash",
            "available": self.is_available,
            "cost": "Free",
            "rate_limit": "60 requests/minute",
            "features": [
                "Multi-language support",
                "High-quality responses", 
                "Context understanding",
                "No usage costs"
            ]
        } 