# app/services/orchestrator.py (FREE TIER OPTIMIZED)

from google import genai
from google.genai import types
import asyncio
import time
from typing import Optional, Type
from pydantic import BaseModel
from app.core.config import settings
from app.schemas.decision import TradeDecision
from app.services.confidence_engine import ConfidenceEngine
from app.services.circuit_breaker import CircuitBreaker
from app.services.agent_wrapper import AgentWrapper
from app.services.rate_limiter import RateLimiter
from app.services.sequential_executor import SequentialExecutor  # NEW


class ReliabilityOrchestrator:
    def __init__(self, use_parallel: bool = False):
        """
        Args:
            use_parallel: If True, use parallel execution (needs paid tier).
                         If False, use sequential execution (free tier friendly).
        """
        # 1. Initialize Core Components
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY) 
        self.confidence_engine = ConfidenceEngine()
        self.cb = CircuitBreaker(failure_threshold=3, recovery_timeout=45)
        
        # 2. Initialize rate limiter for free tier
        self.rate_limiter = RateLimiter(
            max_requests=8,  # Conservative limit for free tier
            window_seconds=60
        )
        
        # 3. Inject dependencies into Agent Wrapper
        self.agent_wrapper = AgentWrapper(
            circuit_breaker=self.cb,
            rate_limiter=self.rate_limiter
        )
        
        # 4. Choose execution strategy based on tier
        self.use_parallel = use_parallel
        
        if use_parallel:
            # Parallel execution (requires paid tier)
            from app.services.parallel_executor import ParallelExecutor
            self.executor = ParallelExecutor(
                num_instances=3, 
                agent_wrapper=self.agent_wrapper
            )
        else:
            # Sequential execution (free tier friendly)
            self.executor = SequentialExecutor(
                agent_wrapper=self.agent_wrapper
            )

    async def run_reliable_workflow(
        self, 
        user_prompt: str, 
        confidence_threshold: float = 0.7,
        response_schema: Optional[Type[BaseModel]] = None
    ) -> dict:
        """
        Executes the reliability pipeline with adaptive retry logic.
        """
        schema = response_schema or TradeDecision
        
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
                
                # STEP 1: Execute (parallel or sequential based on tier)
                if self.use_parallel:
                    final_decision, all_decisions = await self.executor.run_parallel_execution(
                        current_prompt,
                        response_schema=schema
                    )
                    num_agents = len(all_decisions)
                else:
                    final_decision = await self.executor.run_single_execution(
                        current_prompt,
                        response_schema=schema
                    )
                    num_agents = 1
                
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
                            "parallel_agents": num_agents,
                            "response_time_ms": response_time_ms,
                            "execution_mode": "parallel" if self.use_parallel else "sequential"
                        }
                    }
                else:
                    print(f"⛔ GATE BLOCKED: {reason}")
                    current_prompt += f"\n\nCRITICAL FEEDBACK: Your previous analysis was rejected because: {reason}. You must be more specific and confident."
                    continue

            except Exception as e:
                print(f"⚠️ Attempt {attempt+1} Failed: {str(e)}")
                self.confidence_engine.record_outcome(success=False)
                current_prompt += f"\n\nSYSTEM ERROR: The previous attempt crashed. Please ensure you output valid JSON or resolve the execution error."
                
        self.confidence_engine.record_outcome(success=False)
        return {
            "status": "failure",
            "error": "Workflow failed after maximum retries or circuit breaker open."
        }

    def health_check(self) -> dict:
        """Health check with rate limit info"""
        return self.agent_wrapper.health_check()