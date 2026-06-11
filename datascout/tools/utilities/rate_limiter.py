"""
datascout.tools.utilities.rate_limiter
--------------------------------------
Token bucket rate limiter for API calls.

Ensures adapters don't exceed platform rate limits.

Algorithm:
    Token bucket with async-safe implementation.
    Tokens refill at constant rate.
    Operations block until tokens available.

Usage:
    limiter = RateLimiter(rate=10, per=60.0)  # 10 calls per 60 seconds
    
    async with limiter:
        response = await api_call()
"""

import asyncio
import time
from typing import AsyncContextManager


class RateLimiter:
    """
    Async-safe token bucket rate limiter.
    
    Limits operations to a maximum rate over a time window.
    Thread-safe and async-safe via asyncio.Lock.
    """
    
    def __init__(self, rate: int, per: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            rate: Maximum number of operations allowed
            per: Time window in seconds (default: 60.0 = 1 minute)
            
        Example:
            # Allow 10 API calls per minute
            limiter = RateLimiter(rate=10, per=60.0)
        """
        self.rate = rate
        self.per = per
        self.allowance = float(rate)  # Current tokens available
        self.last_check = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        """
        Async context manager entry.
        
        Blocks until a token is available.
        """
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return False
    
    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens from the bucket.
        
        Blocks (sleeps) if insufficient tokens available.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        async with self._lock:
            while True:
                current = time.monotonic()
                time_passed = current - self.last_check
                self.last_check = current
                
                # Refill tokens based on time passed
                self.allowance += time_passed * (self.rate / self.per)
                
                # Cap at maximum rate
                if self.allowance > self.rate:
                    self.allowance = float(self.rate)
                
                # Check if we have enough tokens
                if self.allowance >= tokens:
                    self.allowance -= tokens
                    return
                
                # Calculate wait time needed
                tokens_needed = tokens - self.allowance
                wait_time = tokens_needed * (self.per / self.rate)
                
                # Release lock while waiting
                self._lock.release()
                await asyncio.sleep(wait_time)
                await self._lock.acquire()
    
    def reset(self) -> None:
        """
        Reset the rate limiter to full capacity.
        
        Useful for testing or when switching contexts.
        """
        self.allowance = float(self.rate)
        self.last_check = time.monotonic()


class AdapterRateLimiters:
    """
    Centralized rate limiter registry for all adapters.
    
    Maintains one rate limiter per adapter to enforce
    platform-specific rate limits.
    """
    
    def __init__(self):
        self._limiters: dict[str, RateLimiter] = {}
    
    def get_limiter(self, adapter_name: str, rate: int, per: float = 60.0) -> RateLimiter:
        """
        Get or create a rate limiter for an adapter.
        
        Args:
            adapter_name: Adapter identifier (e.g., "kaggle")
            rate: Calls per time window
            per: Time window in seconds
            
        Returns:
            RateLimiter instance for this adapter
        """
        if adapter_name not in self._limiters:
            self._limiters[adapter_name] = RateLimiter(rate=rate, per=per)
        return self._limiters[adapter_name]
    
    def reset_all(self) -> None:
        """Reset all rate limiters."""
        for limiter in self._limiters.values():
            limiter.reset()


# Global rate limiter registry
_rate_limiters = AdapterRateLimiters()


def get_rate_limiter(adapter_name: str, rate: int, per: float = 60.0) -> RateLimiter:
    """
    Get rate limiter for an adapter.
    
    Args:
        adapter_name: Adapter identifier
        rate: Max calls per time window
        per: Time window in seconds
        
    Returns:
        RateLimiter instance
        
    Example:
        limiter = get_rate_limiter("kaggle", rate=10, per=60.0)
        async with limiter:
            results = await kaggle_api.search(query)
    """
    return _rate_limiters.get_limiter(adapter_name, rate, per)