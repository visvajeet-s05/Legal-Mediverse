"""
Rate Limiting Middleware
========================
Redis-backed rate limiting for FastAPI endpoints.
Falls back to in-memory rate limiting if Redis is not available.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("rate_limiter")

# ─── Configuration ──────────────────────────────────────────────────────────

DEFAULT_RATE_LIMIT = 60  # requests
DEFAULT_RATE_WINDOW = 60  # seconds
STRICT_RATE_LIMIT = 10    # requests for strict endpoints
STRICT_RATE_WINDOW = 60   # seconds

# Endpoints that need stricter rate limiting
STRICT_ENDPOINTS = [
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/health/triage",
    "/api/v1/community/campaigns",
]

# Try to import Redis
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis.asyncio not available. Using in-memory rate limiting (not recommended for production).")


class InMemoryRateLimiter:
    """Simple in-memory rate limiter (fallback when Redis is unavailable)."""

    def __init__(self):
        self._buckets: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def check(self, key: str, max_requests: int, window_seconds: int) -> Tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining)."""
        now = time.time()
        async with self._lock:
            # Clean old entries
            self._buckets[key] = [
                t for t in self._buckets[key] if now - t < window_seconds
            ]

            if len(self._buckets[key]) >= max_requests:
                # Rate limited
                reset_time = int(self._buckets[key][0] + window_seconds)
                retry_after = max(0, reset_time - int(now))
                return False, retry_after

            self._buckets[key].append(now)
            remaining = max_requests - len(self._buckets[key])
            return True, remaining


class RedisRateLimiter:
    """Redis-based rate limiter using sliding window counter."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            try:
                self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
                await self._redis.ping()
                logger.info("Connected to Redis for rate limiting")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._redis

    async def check(self, key: str, max_requests: int, window_seconds: int) -> Tuple[bool, int]:
        """Check rate limit using Redis sorted sets."""
        try:
            redis = await self._get_redis()
            now = time.time()
            window_start = now - window_seconds

            # Remove old entries
            await redis.zremrangebyscore(key, 0, window_start)

            # Count requests in current window
            request_count = await redis.zcard(key)

            if request_count >= max_requests:
                # Get oldest timestamp to calculate retry-after
                oldest = await redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = max(0, int(oldest[0][1] + window_seconds - now))
                else:
                    retry_after = window_seconds
                return False, retry_after

            # Add current request
            await redis.zadd(key, {str(now): now})
            # Set expiry on the key
            await redis.expire(key, window_seconds * 2)

            remaining = max_requests - request_count - 1
            return True, remaining

        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fail open - allow request if Redis is down
            return True, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app: FastAPI, redis_url: Optional[str] = None):
        super().__init__(app)
        if REDIS_AVAILABLE and redis_url:
            self.limiter = RedisRateLimiter(redis_url)
            logger.info("Using Redis rate limiter")
        else:
            self.limiter = InMemoryRateLimiter()
            logger.info("Using in-memory rate limiter")

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for non-API routes and static files
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)

        # Determine rate limit config based on endpoint
        is_strict = any(path.startswith(ep) for ep in STRICT_ENDPOINTS)
        max_requests = STRICT_RATE_LIMIT if is_strict else DEFAULT_RATE_LIMIT
        window = STRICT_RATE_WINDOW if is_strict else DEFAULT_RATE_WINDOW

        # Create a unique key based on IP + endpoint
        client_ip = request.client.host if request.client else "unknown"
        key = f"ratelimit:{client_ip}:{path}"

        try:
            allowed, retry_after = await self.limiter.check(key, max_requests, window)

            if not allowed:
                logger.warning(f"Rate limit exceeded for {client_ip} on {path}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "Too many requests. Please slow down.",
                        "retry_after_seconds": retry_after,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(max_requests),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            response = await call_next(request)

            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(max_requests)
            response.headers["X-RateLimit-Remaining"] = str(retry_after)

            return response

        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open
            return await call_next(request)


def add_rate_limiting(app: FastAPI, redis_url: Optional[str] = None):
    """Helper to add rate limiting middleware to a FastAPI app."""
    app.add_middleware(RateLimitMiddleware, redis_url=redis_url)
    logger.info("Rate limiting middleware added")

