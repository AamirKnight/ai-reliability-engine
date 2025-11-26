# app/services/orchestrator.py (FIXED VERSION)

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.config import settings
from app.schemas.decision import TradeDecision
from app.services.confidence_engine import ConfidenceEngine
import json
import time

# app/services/orchestrator.py (Refactored)

# ... (Imports remain the same) ...
# 🟢 NEW IMPORT
from app.services.parallel_executor import ParallelExecutor 
import asyncio # Need asyncio to run the new executor

from app.services.circuit_breaker import CircuitBreaker # 🟢 NEW IMPORT

class ReliabilityOrchestrator:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.confidence_engine = ConfidenceEngine()
        # 🟢 Initialize Circuit Breaker
        self.cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60) 

    # Modify the _call_gemini_3 method (or the method used by ParallelExecutor)
    def _call_gemini_3(self, prompt: str, schema_dict: dict):
        """
        Wrapped in Circuit Breaker to prevent cascading failures.
        """
        # Define the actual API call as an inner function or lambda
        def api_request():
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_dict,
                )
            )
            return response.text

        # 🟢 Execute via Circuit Breaker
        # If the circuit is OPEN, this raises an exception immediately without calling Google
        return self.cb.call(api_request)
    
    # ... (run_reliable_workflow now becomes async) ...
    async def run_reliable_workflow(
        self, 
        user_prompt: str, 
        confidence_threshold: float = 0.7
    ) -> dict:
        # Schema definition is only needed for the underlying executor
        
        messages = f"""
You are a trading analyst. Analyze this request and respond in JSON format:

Request: {user_prompt}
... (rest of the prompt remains the same for context in error messages)
        """
        
        # --- TOP-LEVEL RELIABILITY LOOP ---
        for attempt in range(3):
            try:
                print(f"🔄 Running Parallel Execution (Attempt {attempt+1})")
                
                start_time = time.time()
                
                # 1. 🟢 RUN PARALLEL EXECUTION AND VOTING
                final_decision, all_decisions = await self.parallel_executor.run_parallel_execution(messages)
                
                response_time_ms = (time.time() - start_time) * 1000
                
                # 2. Calculate Confidence on the VOTED decision
                confidence_metrics = self.confidence_engine.calculate_confidence(
                    decision=final_decision.model_dump(),
                    validation_passed=True, # It must pass schema validation to be a TradeDecision object
                    response_time_ms=response_time_ms
                )
                
                # 3. Confidence Gate
                should_proceed, reason = self.confidence_engine.should_proceed(
                    confidence_metrics,
                    threshold=confidence_threshold
                )
                
                if not should_proceed:
                    # Confidence too low, trigger smart retry with new prompt context
                    print(f"⛔ Confidence Gate Blocked: {reason}")
                    messages += f"\n\nPREVIOUS PARALLEL OUTPUT REJECTED: {reason}. Rerun the agents with a focus on resolving this issue."
                    continue
                
                # 4. Success
                print(f"✅ Consensus Confidence: {confidence_metrics.overall_confidence:.2f}")
                
                # Record success for historical tracking
                self.confidence_engine.record_outcome(success=True)
                
                return {
                    "status": "success",
                    "model": settings.GEMINI_MODEL,
                    "data": final_decision.model_dump(),
                    "confidence_metrics": confidence_metrics.model_dump(),
                    "response_time_ms": response_time_ms
                }

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"⚠️ Top-Level Error: {error_msg}")
                self.confidence_engine.record_outcome(success=False)
                
                # Add detailed error to next attempt
                messages += f"\n\nPREVIOUS TOP-LEVEL ATTEMPT FAILED: {error_msg}. Resolve the execution error."
                
        # ... (Max retries failure remains the same) ...
        
        return {
            "status": "failure", 
            "error": "Max retries exceeded. Check logs for details.",
            "confidence_metrics": None
        }

    # ... (health_check method remains the same) ...
    
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
