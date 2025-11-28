# app/services/parallel_executor.py

import asyncio
from typing import List, Tuple, Optional, Any, Type
from collections import Counter
from pydantic import BaseModel
from app.schemas.decision import TradeDecision
from app.services.agent_wrapper import AgentWrapper


class ParallelExecutor:
    """
    Runs multiple agent instances concurrently and uses a voting mechanism.
    
    It requires an AgentWrapper instance for making protected API calls.
    """
    def __init__(self, num_instances: int = 3, agent_wrapper: Optional[AgentWrapper] = None):
        self.num_instances = num_instances
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

    def _sync_call_model(
        self, 
        prompt: str, 
        response_schema: Optional[Type[BaseModel]] = None
    ) -> Optional[BaseModel]:
        """
        The actual blocking I/O call to Gemini via the AgentWrapper.
        
        Args:
            prompt: The prompt to send
            response_schema: Pydantic model class for validation (defaults to TradeDecision)
        
        Returns:
            Validated Pydantic model instance or None on failure
        """
        try:
            # Default to TradeDecision if no schema provided (for backward compatibility)
            schema = response_schema or TradeDecision
            
            # Pass the schema to the agent wrapper
            raw_text = self.agent_wrapper.execute(prompt, response_schema=schema)
            
            # Schema validate immediately after receiving response
            validated_model = schema.model_validate_json(raw_text)
            return validated_model
            
        except Exception as e:
            # Note the failure, but return None so the other parallel runs can continue
            print(f"⚠️ Single Agent Instance Failed: {type(e).__name__} - {str(e)}")
            return None

    async def _execute_single_run(
        self, 
        prompt: str,
        response_schema: Optional[Type[BaseModel]] = None
    ) -> Optional[BaseModel]:
        """
        Async wrapper for the sync model call, running it in a thread.
        """
        return await asyncio.to_thread(self._sync_call_model, prompt, response_schema)

    def _majority_vote(
        self, 
        decisions: List[BaseModel]
    ) -> Tuple[BaseModel, List[BaseModel]]:
        """
        Voting Logic: Majority Action -> Highest Confidence Breakpoint.
        
        Works with any Pydantic model that has 'action' and 'confidence' fields.
        """
        if not decisions:
            raise ValueError("No successful decisions to vote on.")
        
        # Check if decisions have the required fields
        if not hasattr(decisions[0], 'action') or not hasattr(decisions[0], 'confidence'):
            # If no voting fields, just return the first one
            return decisions[0], decisions
            
        # Count votes for actions (BUY, SELL, HOLD, etc.)
        action_votes = Counter(d.action for d in decisions)
        
        # Find the most common action(s)
        most_common_count = max(action_votes.values())
        
        # Filter for the winners
        winning_decisions = [d for d in decisions if action_votes[d.action] == most_common_count]
        
        # Tie-breaker: Pick the one with highest confidence among the winners
        final_decision = max(winning_decisions, key=lambda d: d.confidence)
        
        return final_decision, decisions

    async def run_parallel_execution(
        self, 
        prompt: str,
        response_schema: Optional[Type[BaseModel]] = None
    ) -> Tuple[BaseModel, List[BaseModel]]:
        """
        Main entry point for parallel execution.
        
        Args:
            prompt: The prompt to execute
            response_schema: Optional Pydantic model for response validation
        
        Returns:
            Tuple of (final_decision, all_decisions)
        """
        prompts = self._generate_prompt_variations(prompt)
        
        # Launch all tasks with the schema
        tasks = [self._execute_single_run(p, response_schema) for p in prompts]
        results = await asyncio.gather(*tasks)
        
        # Filter failures
        successful = [r for r in results if r is not None]
        
        if len(successful) < (self.num_instances // 2) + 1:
            raise RuntimeError(
                f"Consensus Failed: Only {len(successful)}/{self.num_instances} agents survived."
            )

        return self._majority_vote(successful)