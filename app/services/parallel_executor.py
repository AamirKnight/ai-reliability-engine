
import asyncio
from typing import List, Dict, Any, Tuple
from collections import Counter
from app.schemas.decision import TradeDecision
from app.services.orchestrator import ReliabilityOrchestrator # To call the model

class ParallelExecutor:
    """
    Runs multiple agent instances concurrently and uses a voting mechanism 
    to achieve consensus on the final decision.
    """
    def __init__(self, num_instances: int = 3):
        self.num_instances = num_instances
        # We need a separate orchestrator instance to prevent shared state/retries
        self.orchestrator = ReliabilityOrchestrator() 

    def _generate_prompt_variations(self, prompt: str) -> List[str]:
        """
        Creates slightly varied prompts to avoid identical latent space
        results from the LLM, encouraging diverse outputs.
        """
        variations = [
            f"{prompt}. Focus your analysis on short-term market indicators.",
            f"{prompt}. Provide a justification based on fundamental analysis.",
            f"{prompt}. Offer a decision with the highest possible confidence score.",
        ]
        # Pad with the original prompt if num_instances > len(variations)
        while len(variations) < self.num_instances:
            variations.append(prompt)
        
        return variations[:self.num_instances]

    async def _execute_single_run(self, prompt: str) -> Optional[TradeDecision]:
        """
        Wrapper to run a single, isolated agent call.
        """
        try:
            # Call the model wrapper directly, bypassing the orchestrator's
            # external run_reliable_workflow, which has its own retry loop.
            raw_json = await asyncio.to_thread(
                self.orchestrator._call_gemini_3, 
                prompt, 
                TradeDecision.model_json_schema()
            )
            # Schema validate immediately after receiving response
            decision = TradeDecision.model_validate_json(raw_json)
            return decision
        except Exception as e:
            print(f"⚠️ Parallel run failed: {type(e).__name__}")
            return None

    def _majority_vote(self, decisions: List[TradeDecision]) -> Tuple[TradeDecision, List[TradeDecision]]:
        """
        Identifies the action (BUY/SELL/HOLD) with the most votes.
        If a tie, the decision with the highest average confidence wins.
        """
        if not decisions:
            raise ValueError("No successful decisions to vote on.")
            
        action_votes = Counter(d.action for d in decisions)
        
        # Find the most common action(s)
        most_common_count = max(action_votes.values())
        top_actions = [
            action for action, count in action_votes.items() 
            if count == most_common_count
        ]
        
        # If there's a tie, break it using confidence
        if len(top_actions) > 1:
            best_confidence = -1
            winning_action = top_actions[0] # Fallback
            
            for action in top_actions:
                action_confidences = [
                    d.confidence for d in decisions 
                    if d.action == action
                ]
                avg_confidence = sum(action_confidences) / len(action_confidences)
                
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    winning_action = action
        else:
            winning_action = top_actions[0]
            
        # Select the winning decision object (e.g., the one with the highest confidence among the winners)
        winning_decision = max(
            (d for d in decisions if d.action == winning_action), 
            key=lambda d: d.confidence
        )
        
        return winning_decision, decisions

    async def run_parallel_execution(self, prompt: str) -> Tuple[TradeDecision, List[TradeDecision]]:
        """
        Executes the agent in parallel, filters results, and returns the consensus decision.
        """
        prompts = self._generate_prompt_variations(prompt)
        
        tasks = [self._execute_single_run(p) for p in prompts]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Filter out failed runs (where result is None)
        successful_decisions = [r for r in results if r is not None]
        
        if len(successful_decisions) < (self.num_instances // 2) + 1:
            raise RuntimeError(f"Consensus failed: Only {len(successful_decisions)} out of {self.num_instances} succeeded. Requires majority.")

        # Apply the voting logic
        final_decision, all_decisions = self._majority_vote(successful_decisions)
        
        return final_decision, all_decisions