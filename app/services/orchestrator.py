from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.config import settings
from app.schemas.decision import TradeDecision
import json

class ReliabilityOrchestrator:
    def __init__(self):
        # 1. Native Client (Automatically picks up GEMINI_API_KEY from env)
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    # --- RELIABILITY LAYER ---
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_gemini_3(self, prompt: str, schema_dict: dict):
        """
        Calls Gemini 3 with 'Thinking' enabled and strict JSON enforcement.
        """
        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                # Force JSON output
                response_mime_type="application/json",
                # Pass the Pydantic Schema to guide the model
                response_schema=schema_dict, 
                # OPTIONAL: Control "Thinking" depth (Low/High) for Gemini 3
                # thinking_level="HIGH" 
            )
        )
        return response.text

    def run_reliable_workflow(self, user_prompt: str) -> dict:
        # Get the schema structure from Pydantic
        # Note: We don't send the schema in the prompt text anymore; 
        # we send it in the 'config' above (Native Structured Output).
        schema = TradeDecision.model_json_schema()

        messages = f"""
        You are a high-precision trading agent.
        Analyze this request: {user_prompt}
        """

        # --- SELF-HEALING LOOP ---
        for attempt in range(3):
            try:
                print(f"🧠 Gemini 3 Thinking... (Attempt {attempt+1})")
                
                # 1. Call the Model
                raw_json = self._call_gemini_3(messages, schema)
                
                # 2. Parse & Validate
                decision = TradeDecision.model_validate_json(raw_json)
                
                return {
                    "status": "success", 
                    "model": settings.GEMINI_MODEL,
                    "data": decision.model_dump()
                }

            except Exception as e:
                print(f"⚠️ Error: {str(e)}")
                # Recursion/Retry Logic:
                # If it failed, we add the error to the prompt for the next try
                messages += f"\n\nPREVIOUS ATTEMPT FAILED: {str(e)}. FIX THE JSON."
        
        return {"status": "failure", "error": "Max retries exceeded"}
    
# ... inside ReliabilityOrchestrator class ...

    def health_check(self) -> dict:
        """
        Simple connectivity check to ensure API Key and Model are valid.
        """
        try:
            # Send a minimal prompt to test connectivity
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents="Ping",
                config=types.GenerateContentConfig(
                    max_output_tokens=5
                )
            )
            return {"status": "healthy", "model": settings.GEMINI_MODEL}
        except Exception as e:
            # Return the exact error so we can debug the ClientError
            print(f"🔥 Health Check Failed: {e}")
            return {"status": "unhealthy", "error": str(e)}