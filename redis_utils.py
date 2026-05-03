# redis_utils.py
import redis
import logging
from typing import Optional
from config import REDIS_URL

logger = logging.getLogger(__name__)

redis_client: Optional[redis.Redis] = None

if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Connection to Redis established successfully.")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        redis_client = None

def get_redis_client() -> Optional[redis.Redis]:
    return redis_client
