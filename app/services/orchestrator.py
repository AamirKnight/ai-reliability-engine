# app/services/orchestrator.py (FIXED VERSION)

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.config import settings
from app.schemas.decision import TradeDecision
from app.services.confidence_engine import ConfidenceEngine
import json
import time

class ReliabilityOrchestrator:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.confidence_engine = ConfidenceEngine()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_gemini_3(self, prompt: str, schema_dict: dict):
        """
        Calls Gemini with strict JSON enforcement.
        """
        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_dict,
                )
            )
            return response.text
        except Exception as e:
            # Print the ACTUAL error for debugging
            print(f"🔥 Gemini API Error: {type(e).__name__}: {str(e)}")
            raise

    def run_reliable_workflow(
        self, 
        user_prompt: str, 
        confidence_threshold: float = 0.7
    ) -> dict:
        schema = TradeDecision.model_json_schema()
        
        # SIMPLIFIED SCHEMA - Gemini might be rejecting complex schemas
        simplified_schema = {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string"}
            },
            "required": ["symbol", "action", "confidence", "reasoning"]
        }
        
        messages = f"""
You are a trading analyst. Analyze this request and respond in JSON format:

Request: {user_prompt}

Provide:
- symbol: stock ticker (e.g., AAPL, TSLA)
- action: BUY, SELL, or HOLD
- confidence: number between 0 and 1
- reasoning: brief explanation (50-100 words)
        """

        for attempt in range(3):
            try:
                print(f"🧠 Gemini Thinking... (Attempt {attempt+1})")
                
                start_time = time.time()
                raw_json = self._call_gemini_3(messages, simplified_schema)
                response_time_ms = (time.time() - start_time) * 1000
                
                print(f"📦 Raw Response: {raw_json[:200]}...")  # Debug output
                
                # Parse & Validate
                decision = TradeDecision.model_validate_json(raw_json)
                
                # Calculate Confidence
                confidence_metrics = self.confidence_engine.calculate_confidence(
                    decision=decision.model_dump(),
                    validation_passed=True,
                    response_time_ms=response_time_ms
                )
                
                # Confidence Gate
                should_proceed, reason = self.confidence_engine.should_proceed(
                    confidence_metrics,
                    threshold=confidence_threshold
                )
                
                if not should_proceed:
                    print(f"⛔ Confidence Gate Blocked: {reason}")
                    messages += f"\n\nPREVIOUS OUTPUT REJECTED: {reason}. Provide more detailed reasoning."
                    continue
                
                print(f"✅ Confidence: {confidence_metrics.overall_confidence:.2f}")
                
                return {
                    "status": "success",
                    "model": settings.GEMINI_MODEL,
                    "data": decision.model_dump(),
                    "confidence_metrics": confidence_metrics.model_dump(),
                    "response_time_ms": response_time_ms
                }

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"⚠️ Error: {error_msg}")
                
                # Add detailed error to next attempt
                messages += f"\n\nPREVIOUS ATTEMPT FAILED: {error_msg}. Fix the JSON output."
        
        return {
            "status": "failure", 
            "error": "Max retries exceeded. Check logs for details.",
            "confidence_metrics": None
        }
    
    def health_check(self) -> dict:
        """
        Simple connectivity check with better error reporting.
        """
        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents="Say 'OK'",
                config=types.GenerateContentConfig(
                    max_output_tokens=10
                )
            )
            return {
                "status": "healthy", 
                "model": settings.GEMINI_MODEL,
                "response": response.text
            }
        except Exception as e:
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "model_attempted": settings.GEMINI_MODEL
            }
            print(f"🔥 Health Check Failed: {error_details}")
            return {
                "status": "unhealthy", 
                **error_details
            }