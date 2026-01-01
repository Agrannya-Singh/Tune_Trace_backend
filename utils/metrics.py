import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def track_latency(operation: str):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        logger.info(f"METRIC: [{operation}] took {latency_ms:.2f}ms")
