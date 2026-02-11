"""Inference concurrency layer.

Architecture:
    FastAPI (async) -> asyncio.Semaphore(N) -> ThreadPoolExecutor(N) -> ONNX inference

Requests beyond the semaphore limit queue with a 5s timeout, then get 503.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from recognizex.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

SEMAPHORE_TIMEOUT_SECONDS: float = 5.0


class InferencePool:
    """Manages the semaphore and thread pool for ML inference."""

    def __init__(self, settings: Settings) -> None:
        self._semaphore = asyncio.Semaphore(settings.max_concurrent)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent,
            thread_name_prefix="onnx-inference",
        )
        self._active_count: int = 0
        self._queue_depth: int = 0
        self._counter_lock = threading.Lock()

    async def run(self, func: Callable[..., T], *args: object) -> T:
        """Submit a synchronous function to the inference thread pool.

        Acquires the semaphore (with timeout), runs the function in the
        executor, then releases.

        Raises:
            TimeoutError: If the semaphore cannot be acquired within the timeout.
        """
        with self._counter_lock:
            self._queue_depth += 1
        try:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=SEMAPHORE_TIMEOUT_SECONDS,
                )
            finally:
                with self._counter_lock:
                    self._queue_depth -= 1

            with self._counter_lock:
                self._active_count += 1
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(self._executor, func, *args)
            finally:
                self._semaphore.release()
                with self._counter_lock:
                    self._active_count -= 1
        except TimeoutError:
            raise

    @property
    def active_count(self) -> int:
        """Number of currently running inference tasks."""
        with self._counter_lock:
            return self._active_count

    @property
    def queue_depth(self) -> int:
        """Number of requests waiting for a semaphore slot."""
        with self._counter_lock:
            return self._queue_depth

    def shutdown(self) -> None:
        """Shut down the thread pool executor."""
        self._executor.shutdown(wait=True)
