# app/services/agent_wrapper.py (UPDATED WITH RATE LIMITER)

import time
from typing import Optional, Type, TypeVar
from pydantic import BaseModel
from google import genai
from google.genai import types
from app.core.config import settings
from app.services.circuit_breaker import CircuitBreaker
from app.services.rate_limiter import RateLimiter  # NEW

T = TypeVar('T', bound=BaseModel)

class AgentWrapper:
    """
    A low-level wrapper for the Gemini API call with rate limiting protection.
    """
    def __init__(
        self, 
        circuit_breaker: CircuitBreaker,
        rate_limiter: Optional[RateLimiter] = None
    ):
        self.cb = circuit_breaker 
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        # Initialize rate limiter (default: 8 requests/minute for free tier)
        self.rate_limiter = rate_limiter or RateLimiter(
            max_requests=8,  # Conservative: 8/min vs. 10/min limit
            window_seconds=60
        )
        
    def execute(
        self, 
        prompt: str, 
        response_schema: Optional[Type[BaseModel]] = None
    ) -> str:
        """
        Executes the prompt against the Gemini API with rate limiting.
        
        Args:
            prompt: The user prompt
            response_schema: Pydantic model class for response validation (optional)
        
        Returns the raw JSON text response.
        """
        def api_request():
            """Inner function containing the actual blocking API call."""
            # Apply rate limiting BEFORE making the request
            self.rate_limiter.wait_if_needed()
            
            config_params = {}
            
            if response_schema:
                config_params = {
                    "response_mime_type": "application/json",
                    "response_schema": response_schema.model_json_schema()
                }
            
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(**config_params)
            )
            return response.text

        # Use the injected Circuit Breaker to make the API call
        return self.cb.call(api_request)

    def health_check(self) -> dict:
        """Simple connectivity check via Circuit Breaker."""
        def _ping():
            self.rate_limiter.wait_if_needed()
            return self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents="Ping",
                config=types.GenerateContentConfig(max_output_tokens=5)
            ).text

        try:
            response = self.cb.call(_ping)
            return {
                "status": "healthy", 
                "model": settings.GEMINI_MODEL, 
                "response": response, 
                "circuit_state": self.cb.state.value,
                "rate_limit": self.rate_limiter.get_current_usage()
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e), 
                "circuit_state": self.cb.state.value,
                "rate_limit": self.rate_limiter.get_current_usage()
            }