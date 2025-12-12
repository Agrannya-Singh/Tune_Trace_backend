# main.py

# --- Standard Library Imports ---
import logging
import os
import json
from typing import List, Set, Optional

# --- Third-Party Imports ---
import asyncio
import redis
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

# --- Local Application Imports ---
from db import SessionLocal
from ml_engine import MLEngine
from services import SuggestionService
from repository import MusicRepository
from api_models import SuggestionResponse, LikedSongsRequest, SongSuggestion, LikedSongResponse
from dependencies import get_repo, get_suggestion_service

# ==============================================================================
# --- Initial Application Setup ---
# ==============================================================================

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# --- Environment & Configuration ---
# ==============================================================================

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
try:
    REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "3600"))
except ValueError:
    logger.warning("Invalid REDIS_TTL_SECONDS environment variable, using default 3600")
    REDIS_TTL_SECONDS = 3600

if not YOUTUBE_API_KEY:
    logger.critical("FATAL: YOUTUBE_API_KEY environment variable not set.")

# ==============================================================================
# --- ML Engine Initialization ---
# ==============================================================================

ml_engine = MLEngine()

# ==============================================================================
# --- FastAPI App Initialization ---
# ==============================================================================

app = FastAPI(
    title="Hybrid Music Suggestion API",
    description="Generates music suggestions using a hybrid model with a genre-based fallback.",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ==============================================================================
# --- Service Connections ---
# ==============================================================================

redis_client: Optional[redis.Redis] = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Connection to Redis established successfully.")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        redis_client = None

# ==============================================================================
# --- FastAPI Application Events ---
# ==============================================================================


@app.on_event("startup")
def on_startup() -> None:
    """Initialize services and verify connections on application startup."""
    # Initialize and store suggestion service
    app.state.suggestion_service = SuggestionService(api_key=YOUTUBE_API_KEY)
    logger.info("Application starting up...")
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        logger.info("Connection to the database established successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Could not connect to the database: {e}")
        raise RuntimeError(f"Database connection failed: {e}") from e
    logger.info("Application startup complete.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Close the httpx client gracefully on application shutdown."""
    logger.info("Application shutting down...")
    service = getattr(app.state, "suggestion_service", None)
    if service:
        await service.close()
        logger.info("SuggestionService client closed.")

# ==============================================================================
# --- Background Tasks ---
# ==============================================================================


def update_redis_user_likes(user_id: str, song_ids: Set[int]):
    """
    Background task to update a user's liked songs in the Redis cache.
    This runs after the HTTP response is sent.
    """
    if not redis_client:
        logger.warning(
            "Redis client not available. Skipping cache update for user %s.", user_id)
        return

    try:
        redis_key = f"user_likes:{user_id}"
        # Convert the set of integer IDs to a JSON string for storage
        value = json.dumps(list(song_ids))

        redis_client.set(redis_key, value, ex=REDIS_TTL_SECONDS)
        logger.info("Successfully cached liked songs for user %s in Redis.", user_id)
    except Exception as e:
        logger.error("Failed to update Redis cache for user %s: %s", user_id, e)

# ==============================================================================
# --- API Endpoints ---
# ==============================================================================


@app.post("/suggestions", response_model=SuggestionResponse, tags=["Suggestions"])
async def post_suggestions(
    request: LikedSongsRequest,
    background_tasks: BackgroundTasks,
    repo: MusicRepository = Depends(get_repo),
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
):
    if not YOUTUBE_API_KEY:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Service is not configured.")

    try:
        # 1. Create a list of concurrent tasks
        tasks = [
            suggestion_service._search_youtube_for_song_async(song_name)
            for song_name in request.songs
        ]

        # 2. Execute all searches in parallel
        results = await asyncio.gather(*tasks)

        # 3. Process the results
        song_metadata_ids_to_like = set()
        for video_info in results:
            if not video_info:
                continue

            song_meta = repo.get_song_metadata_by_video_id(
                video_info["video_id"])
            if not song_meta:
                song_meta = repo.create_song_metadata(video_info)
            song_metadata_ids_to_like.add(song_meta.id)

        user = repo.get_or_create_user(request.user_id)

        # 1. Perform the primary (PostgreSQL) write. The user waits for this.
        repo.persist_user_likes(user, song_metadata_ids_to_like)

        # 2. Schedule the secondary (Redis) write. The user does NOT wait for this.
        background_tasks.add_task(
            update_redis_user_likes, user.user_id, song_metadata_ids_to_like
        )

        # 3. Fetch data for ML-driven recommendations
        user_likes = repo.get_user_liked_songs_objects(user.user_id)
        candidate_songs = repo.get_candidate_songs(limit=1000)

        # 4. Run the ML engine to get content-based suggestions
        ai_suggestions = ml_engine.recommend(
            user_history=[s.to_dict() for s in user_likes],
            all_songs=[s.to_dict() for s in candidate_songs],
            top_n=10
        )

        # 5. Fallback to collaborative/genre-based suggestions if ML fails
        if not ai_suggestions:
            logger.warning(
                f"ML engine returned no suggestions for user {user.user_id}. Using fallback.")
            ai_suggestions = suggestion_service.get_suggestions(
                user, repo, genre=request.genre)

        # We explicitly map the data to our strict DTO.
        response_suggestions = [
            SongSuggestion(
                title=s['title'],
                artist=s['artist'],
                youtube_video_id=s.get(
                    'video_id') or s.get('youtube_video_id')
            )
            for s in ai_suggestions
        ]
        return {"suggestions": response_suggestions}

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "An internal server error occurred.")


@app.get("/liked-songs", response_model=List[LikedSongResponse], tags=["User Data"])
async def get_liked_songs(
    user_id: str = Query(..., max_length=255, min_length=1),
    repo: MusicRepository = Depends(get_repo),
):
    """Returns the list of liked songs for a given user.

    Args:
        user_id: User email or unique identifier from OAuth

    Returns:
        List of liked songs with video_id, title, artist, and created_at timestamp
    """
    try:
        liked_songs = repo.get_user_liked_songs(user_id)

        return [
            LikedSongResponse(
                video_id=video_id,
                title=title,
                artist=artist,
                created_at=created_at.isoformat()
            )
            for video_id, title, artist, created_at in liked_songs
        ]
    except Exception as e:
        logger.exception(f"Error fetching liked songs for user {user_id}: {e}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve liked songs."
        )


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    """A simple endpoint to confirm the service is running."""
    return {"status": "healthy"}
