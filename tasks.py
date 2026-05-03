# tasks.py
import json
import logging
from typing import Set
from redis_utils import redis_client
from config import REDIS_TTL_SECONDS
from utils.metrics import track_latency

logger = logging.getLogger(__name__)

def update_redis_user_likes(user_id: str, all_liked_ids: Set[int]):
    """
    Background task to cache the user's FULL set of liked song IDs in Redis.
    """
    if not redis_client:
        logger.warning(
            "Redis client not available. Skipping cache update for user %s.", user_id)
        return

    try:
        redis_key = f"user_likes:{user_id}"
        value = json.dumps(list(all_liked_ids))

        with track_latency("Redis:Write"):
            redis_client.set(redis_key, value, ex=REDIS_TTL_SECONDS)
        logger.info("Successfully cached %d liked songs for user %s in Redis.",
                    len(all_liked_ids), user_id)
    except Exception as e:
        logger.error("Failed to update Redis cache for user %s: %s", user_id, e)
