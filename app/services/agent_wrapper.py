import time
from google import genai
from google.genai import types
from app.core.config import settings
from app.schemas.decision import TradeDecision
from app.services.circuit_breaker import CircuitBreaker

class AgentWrapper:
    """
    A low-level wrapper for the Gemini API call, responsible for 
    enforcing JSON structure and integrating the Circuit Breaker.
    """
    def __init__(self, circuit_breaker: CircuitBreaker):
        # The Circuit Breaker is injected
        self.cb = circuit_breaker 
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
    def execute(self, prompt: str) -> str:
        """
        Executes the prompt against the Gemini API, protected by the Circuit Breaker.
        Returns the raw JSON text response.
        """
        # We define the simplified schema here, as it's common to all calls
        simplified_schema = TradeDecision.model_json_schema()
        
        def api_request():
            """Inner function containing the actual blocking API call."""
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=simplified_schema,
                )
            )
            return response.text

        # Use the injected Circuit Breaker to make the API call
        return self.cb.call(api_request)

    def health_check(self) -> dict:
        """Simple connectivity check via Circuit Breaker."""
        def _ping():
            return self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents="Ping",
                config=types.GenerateContentConfig(max_output_tokens=5)
            ).text

        try:
            response = self.cb.call(_ping)
            return {"status": "healthy", "model": settings.GEMINI_MODEL, "response": response, "circuit_state": self.cb.state.value}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "circuit_state": self.cb.state.value}