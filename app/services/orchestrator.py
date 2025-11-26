from google import genai
from google.genai import types
import asyncio
import time
from app.core.config import settings
from app.schemas.decision import TradeDecision
from app.services.confidence_engine import ConfidenceEngine
from app.services.parallel_executor import ParallelExecutor
from app.services.circuit_breaker import CircuitBreaker
# 🟢 NEW IMPORT
from app.services.agent_wrapper import AgentWrapper 
# 🔴 REMOVED: tenacity imports (as we use our CB now)

class ReliabilityOrchestrator:
    def __init__(self):
        # 1. Initialize Core Components (Lowest level first)
        # We still need the client for the health check/initial config check
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY) 
        self.confidence_engine = ConfidenceEngine()
        self.cb = CircuitBreaker(failure_threshold=3, recovery_timeout=45)
        
        # 2. Inject CB into Agent Wrapper (the client layer)
        self.agent_wrapper = AgentWrapper(circuit_breaker=self.cb)
        
        # 3. Inject Agent Wrapper into Parallel Executor (the execution layer)
        self.parallel_executor = ParallelExecutor(
            num_instances=3, 
            agent_wrapper=self.agent_wrapper
        )

    async def run_reliable_workflow(self, user_prompt: str, confidence_threshold: float = 0.7) -> dict:
        """
        Executes the full reliability pipeline:
        Parallel Agents -> Voting -> Confidence Gate -> Retry Loop
        """
        
        base_prompt = f"""
        You are a senior trading analyst. 
        Analyze the following request and return a JSON decision.
        User Request: {user_prompt}
        """

        current_prompt = base_prompt
        
        # --- RELIABILITY LOOP (Max 3 Retries) ---
        for attempt in range(3):
            try:
                print(f"\n🔄 WORKFLOW START: Attempt {attempt+1}")
                start_time = time.time()
                
                # STEP 1: Parallel Execution (Implicitly uses AgentWrapper/CircuitBreaker)
                final_decision, all_decisions = await self.parallel_executor.run_parallel_execution(current_prompt)
                
                response_time_ms = (time.time() - start_time) * 1000
                
                # STEP 2: Confidence Calculation
                confidence_metrics = self.confidence_engine.calculate_confidence(
                    decision=final_decision.model_dump(),
                    validation_passed=True,
                    response_time_ms=response_time_ms
                )
                
                # STEP 3: The Confidence Gate
                should_proceed, reason = self.confidence_engine.should_proceed(
                    confidence_metrics, 
                    threshold=confidence_threshold
                )
                
                if should_proceed:
                    print(f"✅ SUCCESS: Workflow passed with confidence {confidence_metrics.overall_confidence:.2f}")
                    self.confidence_engine.record_outcome(success=True)
                    
                    return {
                        "status": "success",
                        "model": settings.GEMINI_MODEL,
                        "data": final_decision.model_dump(),
                        "confidence_metrics": confidence_metrics.model_dump(),
                        "meta": {
                            "attempts": attempt + 1,
                            "parallel_agents": len(all_decisions),
                            "response_time_ms": response_time_ms
                        }
                    }
                else:
                    print(f"⛔ GATE BLOCKED: {reason}")
                    # Adaptive Prompting for next retry
                    current_prompt += f"\n\nCRITICAL FEEDBACK: Your previous analysis was rejected because: {reason}. You must be more specific and confident."
                    continue

            except Exception as e:
                print(f"⚠️ Attempt {attempt+1} Failed: {str(e)}")
                self.confidence_engine.record_outcome(success=False)
                # Ensure we add the error to the prompt for the next attempt
                current_prompt += f"\n\nSYSTEM ERROR: The previous attempt crashed. Please ensure you output valid JSON or resolve the execution error."
                
        self.confidence_engine.record_outcome(success=False)
        return {
            "status": "failure",
            "error": "Workflow failed after maximum retries or circuit breaker open."
        }

    # The Orchestrator's health check is now delegated to the AgentWrapper
    def health_check(self) -> dict:
        return self.agent_wrapper.health_check()