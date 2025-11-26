import random
from typing import Callable, Any
from app.services.agent_wrapper import AgentWrapper

class FailureInjector:
    """
    Wrapper that injects controlled failures for testing.
    """
    def __init__(self, agent_wrapper: AgentWrapper, failure_rate: float = 0.0):
        self.agent_wrapper = agent_wrapper
        self.failure_rate = failure_rate  # 0.0 = no failures, 0.3 = 30% fail
        self.total_calls = 0
        self.injected_failures = 0
    
    def execute(self, prompt: str) -> str:
        """
        Execute with potential failure injection.
        """
        self.total_calls += 1
        
        # Randomly inject failure
        if random.random() < self.failure_rate:
            self.injected_failures += 1
            raise Exception(f"💉 INJECTED FAILURE (Rate: {self.failure_rate*100}%)")
        
        # Normal execution
        return self.agent_wrapper.execute(prompt)
    
    def health_check(self):
        return self.agent_wrapper.health_check()
    
    def get_stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "injected_failures": self.injected_failures,
            "failure_rate": f"{self.failure_rate*100:.1f}%",
            "actual_failure_rate": f"{(self.injected_failures/self.total_calls*100):.1f}%" if self.total_calls > 0 else "0%"
        }