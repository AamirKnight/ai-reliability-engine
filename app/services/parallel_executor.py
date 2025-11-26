import asyncio
from typing import List, Tuple, Optional, Any
from collections import Counter
from app.schemas.decision import TradeDecision
# 🟢 NEW IMPORT: Now imports the AgentWrapper
from app.services.agent_wrapper import AgentWrapper 
# 🔴 FIX: Removed circular import: from app.services.orchestrator import ReliabilityOrchestrator # To call the model

class ParallelExecutor:
    """
    Runs multiple agent instances concurrently and uses a voting mechanism.
    
    It requires an AgentWrapper instance for making protected API calls.
    """
    def __init__(self, num_instances: int = 3, agent_wrapper: Optional[AgentWrapper] = None):
        self.num_instances = num_instances
        # 🟢 FIX: Dependency Injection is required here.
        if not agent_wrapper:
            raise ValueError("AgentWrapper must be injected into ParallelExecutor.")
        self.agent_wrapper = agent_wrapper


    def _generate_prompt_variations(self, prompt: str) -> List[str]:
        """
        Creates slightly varied prompts to encourage diverse thinking.
        """
        variations = [
            f"{prompt}\n\n[Perspective: Conservative Risk Manager]",
            f"{prompt}\n\n[Perspective: Aggressive Growth Analyst]",
            f"{prompt}\n\n[Perspective: Technical Chart Analyst]",
        ]
        # Pad if we need more instances than variations
        while len(variations) < self.num_instances:
            variations.append(prompt)
        
        return variations[:self.num_instances]

    def _sync_call_model(self, prompt: str) -> Optional[TradeDecision]:
        """
        The actual blocking I/O call to Gemini via the AgentWrapper.
        """
        try:
            # 🟢 FIX: Use the injected AgentWrapper to execute the API call
            # The AgentWrapper already handles the Circuit Breaker and JSON schema
            raw_text = self.agent_wrapper.execute(prompt)
            
            # Schema validate immediately after receiving response
            decision = TradeDecision.model_validate_json(raw_text)
            return decision
        except Exception as e:
            # Note the failure, but return None so the other parallel runs can continue
            print(f"⚠️ Single Agent Instance Failed: {type(e).__name__} - {str(e)}")
            return None

    async def _execute_single_run(self, prompt: str) -> Optional[TradeDecision]:
        """
        Async wrapper for the sync model call, running it in a thread.
        """
        return await asyncio.to_thread(self._sync_call_model, prompt)

    def _majority_vote(self, decisions: List[TradeDecision]) -> Tuple[TradeDecision, List[TradeDecision]]:
        """
        Voting Logic: Majority Action -> Highest Confidence Breakpoint.
        """
        if not decisions:
            raise ValueError("No successful decisions to vote on.")
            
        # Count votes for BUY, SELL, HOLD
        action_votes = Counter(d.action for d in decisions)
        
        # Find the most common action(s)
        most_common_count = max(action_votes.values())
        
        # Filter for the winners
        winning_decisions = [d for d in decisions if action_votes[d.action] == most_common_count]
        
        # Tie-breaker: Pick the one with highest confidence among the winners
        final_decision = max(winning_decisions, key=lambda d: d.confidence)
        
        return final_decision, decisions

    async def run_parallel_execution(self, prompt: str) -> Tuple[TradeDecision, List[TradeDecision]]:
        """
        Main entry point for parallel execution.
        """
        prompts = self._generate_prompt_variations(prompt)
        
        # Launch all tasks
        tasks = [self._execute_single_run(p) for p in prompts]
        results = await asyncio.gather(*tasks)
        
        # Filter failures
        successful = [r for r in results if r is not None]
        
        if len(successful) < (self.num_instances // 2) + 1:
            raise RuntimeError(f"Consensus Failed: Only {len(successful)}/{self.num_instances} agents survived.")

        return self._majority_vote(successful)