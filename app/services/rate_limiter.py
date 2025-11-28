# app/services/rate_limiter.py

import time
from collections import deque
from typing import Callable, Any

class RateLimiter:
    """
    Token bucket rate limiter to prevent API quota exhaustion.
    """
    def __init__(self, max_requests: int = 8, window_seconds: int = 60):
        """
        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times = deque()
    
    def _clean_old_requests(self):
        """Remove requests outside the current time window"""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
    
    def wait_if_needed(self):
        """
        Block if rate limit would be exceeded.
        Returns immediately if under limit.
        """
        self._clean_old_requests()
        
        if len(self.request_times) >= self.max_requests:
            # Calculate wait time
            oldest_request = self.request_times[0]
            wait_time = (oldest_request + self.window_seconds) - time.time()
            
            if wait_time > 0:
                print(f"⏳ Rate limit: Waiting {wait_time:.1f}s before next request...")
                time.sleep(wait_time + 0.1)  # Add small buffer
                self._clean_old_requests()
        
        # Record this request
        self.request_times.append(time.time())
    
    def get_current_usage(self) -> dict:
        """Get current rate limit statistics"""
        self._clean_old_requests()
        return {
            "requests_in_window": len(self.request_times),
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "utilization": f"{(len(self.request_times) / self.max_requests) * 100:.1f}%"
        }