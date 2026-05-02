# main.py

# --- Standard Library Imports ---
import logging
import os
import json
from contextlib import asynccontextmanager
import threading
from typing import List, Set, Optional

# --- Third-Party Imports ---
import asyncio
import redis
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

# --- Local Application Imports ---
from db import SessionLocal, get_session
from ml_engine import MLEngine
from services import SuggestionService
from repository import MusicRepository
from api_models import SuggestionResponse, LikedSongsRequest, SongSuggestion, LikedSongResponse
from dependencies import get_repo, get_suggestion_service
from utils.metrics import track_latency
from utils.enrichment import run_enrichment

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
# --- Application Lifespan ---
# ==============================================================================


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manages startup and shutdown for the application."""
    # --- Startup ---
    application.state.suggestion_service = SuggestionService(api_key=YOUTUBE_API_KEY)
    logger.info("Application starting up...")
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        logger.info("Connection to the database established successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Could not connect to the database: {e}")
        raise RuntimeError(f"Database connection failed: {e}") from e

    # --- Pre-load SentenceTransformer model into RAM (~90 MB) ---
    ml_engine.load_model()
    logger.info("Application startup complete.")

    # --- Launch enrichment in a background thread (non-blocking) ---
    enrichment_thread = threading.Thread(
        target=run_enrichment,
        kwargs={"api_key": YOUTUBE_API_KEY},
        daemon=True,
        name="song-enrichment",
    )
    enrichment_thread.start()
    logger.info("Enrichment background thread started.")

    yield  # --- Application runs here ---

    # --- Shutdown ---
    logger.info("Application shutting down...")
    service = getattr(application.state, "suggestion_service", None)
    if service:
        await service.close()
        logger.info("SuggestionService client closed.")


# ==============================================================================
# --- FastAPI App Initialization ---
# ==============================================================================

app = FastAPI(
    title="TuneTrace Semantic Music API",
    description="Generates music suggestions using semantic vector search (pgvector) with genre-based fallback.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
# --- Background Tasks ---
# ==============================================================================


def update_redis_user_likes(user_id: str, all_liked_ids: Set[int]):
    """
    Background task to cache the user's FULL set of liked song IDs in Redis.
    Receives the complete set (not just current request) to avoid partial overwrites.
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
        with track_latency("YouTube:Search_Parallel"):
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
        with track_latency("PostgreSQL:Write_Likes"):
            repo.persist_user_likes(user, song_metadata_ids_to_like)

        # 2. Schedule the secondary (Redis) write with the FULL liked set.
        #    We compute user_liked_ids first, then pass it to the background task.

        # 3. Fetch data for ML-driven recommendations
        with track_latency("PostgreSQL:Fetch_History"):
            user_likes = repo.get_user_liked_songs_objects(user.user_id)

        user_liked_ids = {s.id for s in user_likes}

        # Now schedule Redis cache update with the complete set
        background_tasks.add_task(
            update_redis_user_likes, user.user_id, user_liked_ids
        )

        # 4. Run the semantic ML engine (pgvector cosine search)
        #    — pass previously recommended video IDs so they are never repeated

        # Fetch previously recommended video IDs from Redis (last 24h)
        previously_recommended: set = set()
        if redis_client:
            try:
                rec_key = f"prev_recs:{user.user_id}"
                cached_recs = redis_client.get(rec_key)
                if cached_recs:
                    previously_recommended = set(json.loads(cached_recs))
                    logger.info(
                        "Excluding %d previously recommended songs for user %s.",
                        len(previously_recommended), user.user_id,
                    )
            except Exception as e:
                logger.warning("Failed to read prev_recs from Redis: %s", e)

        # Get a fresh DB session for the pgvector query
        db_session = next(get_session())
        try:
            with track_latency("MLEngine:SemanticSearch"):
                ai_suggestions = ml_engine.recommend(
                    user_history=[s.to_dict() for s in user_likes],
                    db_session=db_session,
                    top_n=10,
                    excluded_video_ids=previously_recommended,
                )
        finally:
            db_session.close()

        # 5. Fallback to genre/trending YouTube search if semantic engine yields no results.
        #    Collaborative filtering is disabled (low user count); see services.py for the flag.
        if not ai_suggestions:
            logger.warning(
                "Semantic engine returned no suggestions for user %s. Using genre/trending fallback.",
                user.user_id,
            )
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

        # Store the recommended video IDs in Redis so future calls exclude them
        if redis_client and ai_suggestions:
            try:
                new_rec_ids = [
                    s.get('video_id') or s.get('youtube_video_id')
                    for s in ai_suggestions
                    if s.get('video_id') or s.get('youtube_video_id')
                ]
                rec_key = f"prev_recs:{user.user_id}"
                # Merge with existing set so exclusions accumulate
                merged = list(previously_recommended | set(new_rec_ids))
                redis_client.set(rec_key, json.dumps(merged), ex=60 * 60 * 24)  # 24h TTL
                logger.info(
                    "Stored %d prev_recs for user %s (total exclusions: %d).",
                    len(new_rec_ids), user.user_id, len(merged),
                )
            except Exception as e:
                logger.warning("Failed to write prev_recs to Redis: %s", e)

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
    Attempts to read from Redis cache first, falls back to PostgreSQL.
    """
    try:
        # 1. Try to read from Redis Cache
        if redis_client:
            with track_latency("Redis:Read"):
                cached_data = redis_client.get(f"user_likes:{user_id}")
            
            if cached_data:
                logger.info(f"Cache HIT for user {user_id}")
                song_ids = json.loads(cached_data)
                
                # Fetch minimal details from DB for the cached IDs
                # (Optimization: We still need Title/Artist, so we hit DB for the subset)
                with track_latency("PostgreSQL:Read_Cached"):
                    liked_songs = repo.get_songs_by_ids(song_ids)
                    
                return [
                    LikedSongResponse(
                        video_id=s.video_id,
                        title=s.title,
                        artist=s.artist,
                        created_at="from_cache" # For simplicity in this version
                    )
                    for s in liked_songs
                ]

        # 2. Fallback to PostgreSQL
        logger.info(f"Cache MISS for user {user_id}. Fetching from DB.")
        with track_latency("PostgreSQL:Read_Full"):
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
