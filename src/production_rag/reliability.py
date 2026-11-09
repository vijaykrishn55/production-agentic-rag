from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_seconds: float = 20.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failures = 0
        self.state = CircuitState.CLOSED
        self.opened_at = 0.0
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            if self.state == CircuitState.OPEN and time.monotonic() - self.opened_at >= self.recovery_seconds:
                self.state = CircuitState.HALF_OPEN
            return self.state != CircuitState.OPEN

    async def success(self) -> None:
        async with self._lock:
            self.failures = 0
            self.state = CircuitState.CLOSED

    async def failure(self) -> None:
        async with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()


async def retry_async(operation: Callable[[], Awaitable[T]], attempts: int = 3,
                      base_delay: float = 0.15, retry_exceptions: tuple[type[BaseException], ...] = (Exception,)) -> T:
    for attempt in range(attempts):
        try:
            return await operation()
        except retry_exceptions:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** attempt) + random.uniform(0, base_delay / 2))
    raise RuntimeError("retry loop exhausted")
