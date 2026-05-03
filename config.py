# config.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

try:
    REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "3600"))
except (ValueError, TypeError):
    logger.warning("Invalid REDIS_TTL_SECONDS environment variable, using default 3600")
    REDIS_TTL_SECONDS = 3600

if not YOUTUBE_API_KEY:
    logger.critical("FATAL: YOUTUBE_API_KEY environment variable not set.")
