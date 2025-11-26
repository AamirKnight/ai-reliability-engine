import time
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "CLOSED"     # Normal operation (Current flows)
    OPEN = "OPEN"         # Circuit broken (Fails fast)
    HALF_OPEN = "HALF_OPEN" # Testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Executes the function if circuit is CLOSED or HALF_OPEN.
        Raises Exception if OPEN.
        """
        if self.state == CircuitState.OPEN:
            # Check if we should try to recover (HALF_OPEN)
            time_since_failure = time.time() - self.last_failure_time
            if time_since_failure > self.recovery_timeout:
                print("⚠️ Circuit Breaker: Entering HALF_OPEN state (Testing recovery)...")
                self.state = CircuitState.HALF_OPEN
            else:
                remaining = int(self.recovery_timeout - time_since_failure)
                raise Exception(f"⛔ Circuit Breaker is OPEN. Blocking call to protect system. Retry in {remaining}s")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            # We re-raise the exception so the caller knows it failed
            raise e

    def _on_success(self):
        """Reset failure count on success"""
        if self.state != CircuitState.CLOSED:
            print("✅ Circuit Breaker: Call successful. Closing circuit (System Healthy).")
            self.state = CircuitState.CLOSED
        self.failure_count = 0

    def _on_failure(self):
        """Track failures and open circuit if threshold reached"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state != CircuitState.OPEN and self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"🔥 Circuit Breaker: Threshold reached ({self.failure_count} failures). Opening circuit!")