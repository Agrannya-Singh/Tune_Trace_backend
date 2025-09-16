# main.py
"""
FastAPI application for the Hybrid Music Suggestion microservice.

This application provides a RESTful API for generating personalized song
suggestions. It accepts a user's liked songs, persists this data, and then
returns a list of new suggestions based on a hybrid model of collaborative
and content-based filtering.

The application is built with a three-tier architecture:
- API Layer (main.py): Handles HTTP requests, validation, and responses.
- Service Layer (SuggestionService): Contains the core business logic.
- Repository Layer (MusicRepository): Manages all database interactions.
"""

# --- Standard Library Imports ---
import logging
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Set

# --- Third-Party Imports ---
import redis
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

# --- Local Application Imports ---
from db import (
    User,
    UserLikedSong,
    SongMetadata,
    get_read_session,
    get_write_sessions,
)

# ==============================================================================
# --- Initial Application Setup ---
# ==============================================================================

# Load environment variables from a .env file for local development.
# In production, environment variables should be set directly.
load_dotenv()

# Configure logging to provide visibility into application behavior.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# --- Environment & Configuration ---
# ==============================================================================

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "3600"))

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hybrid Music Suggestion API",
    description="Generates music suggestions using collaborative and content-based filtering.",
    version="2.0.0",
    # Add OpenAPI documentation URL.
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS Middleware Configuration ---
# NOTE FOR AUDITORS: The `allow_origins=["*"]` is permissive for development.
# For production, this should be restricted to the specific frontend domain
# e.g., `allow_origins=["https://www.your-frontend-app.com"]`.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ==============================================================================
# --- Service Connections (Redis, etc.) ---
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

# Critical check for the YouTube API key on startup.
if not YOUTUBE_API_KEY:
    logger.critical(
        "FATAL: YOUTUBE_API_KEY environment variable not set. "
        "The suggestion service will be non-functional."
    )


# ==============================================================================
# --- FastAPI Application Events ---
# ==============================================================================

@app.on_event("startup")
def on_startup() -> None:
    """
    FastAPI startup event handler.

    This function is executed when the application starts up.
    NOTE: Database initialization (`init_db`) has been removed from this function.
    Database schema management is now handled exclusively by Alembic migrations,
    which should be run as a separate step in the deployment process.
    """
    logger.info("Application startup complete.")
    pass


# ==============================================================================
# --- Pydantic Data Models (API Contracts) ---
# ==============================================================================

class SongSuggestion(BaseModel):
    """Represents a single song suggestion returned to the client."""
    title: str
    artist: str
    youtube_video_id: str

class SuggestionResponse(BaseModel):
    """The response model for the /suggestions endpoint."""
    suggestions: List[SongSuggestion]

class LikedSongsRequest(BaseModel):
    """The request model for submitting liked songs."""
    user_id: str = Field(..., description="Unique client-generated identifier for the user.")
    songs: List[str] = Field(..., min_length=1, description="A list of song titles the user has liked.")


# ==============================================================================
# --- Repository Layer (Data Access) ---
# ==============================================================================

class MusicRepository:
    """
    Handles all database interactions for users, songs, and their relationships.
    This class abstracts the database operations away from the business logic.
    """

    def __init__(self, db: Session):
        """
        Initializes the repository with a database session.
        :param db: An active SQLAlchemy Session object.
        """
        self.db = db

    def get_or_create_user(self, user_id: str) -> User:
        """
        Retrieves a user by their ID, creating them if they don't exist.
        :param user_id: The external unique identifier for the user.
        :return: The User ORM object.
        """
        user = self.db.query(User).filter_by(user_id=user_id).one_or_none()
        if not user:
            user = User(user_id=user_id)
            self.db.add(user)
            self.db.flush()  # Use flush to get the user's ID without committing.
        return user

    def get_song_metadata_by_video_id(self, video_id: str) -> Optional[SongMetadata]:
        """
        Finds song metadata by its YouTube video ID.
        :param video_id: The YouTube video ID.
        :return: A SongMetadata object or None if not found.
        """
        return self.db.query(SongMetadata).filter_by(video_id=video_id).one_or_none()

    def create_song_metadata(self, video_data: dict) -> SongMetadata:
        """
        Creates and stores a new SongMetadata record from video data.
        :param video_data: A dictionary containing video metadata.
        :return: The newly created SongMetadata object.
        """
        song = SongMetadata(
            video_id=video_data["video_id"],
            title=video_data["title"],
            artist=video_data["artist"],
            tags=",".join(video_data.get("tags", [])),
        )
        self.db.add(song)
        self.db.flush()
        return song

    def persist_user_likes(self, user: User, song_metadata_ids: Set[int]):
        """
        Synchronizes the user's liked songs with the provided set of song IDs.
        :param user: The User object.
        :param song_metadata_ids: A set of song metadata primary keys the user likes.
        """
        existing_liked_ids = {like.song_id for like in user.likes}

        ids_to_add = song_metadata_ids - existing_liked_ids
        ids_to_remove = existing_liked_ids - song_metadata_ids

        if ids_to_remove:
            self.db.query(UserLikedSong).filter(
                UserLikedSong.user_id == user.id,
                UserLikedSong.song_id.in_(ids_to_remove),
            ).delete(synchronize_session='fetch')

        for song_id in ids_to_add:
            self.db.add(UserLikedSong(user_id=user.id, song_id=song_id))

        self.db.commit()

    def get_collaborative_suggestions(self, user: User, limit: int) -> List[SongMetadata]:
        """
        Implements collaborative filtering by finding songs liked by "taste neighbors."
        :param user: The target user for whom to generate suggestions.
        :param limit: The maximum number of suggestions to return.
        :return: A list of suggested SongMetadata objects.
        """
        liked_song_ids = {like.song_id for like in user.likes}
        if not liked_song_ids:
            return []

        # Subquery to find users who have liked at least one same song.
        similar_users_subquery = (
            self.db.query(UserLikedSong.user_id)
            .filter(UserLikedSong.song_id.in_(liked_song_ids))
            .filter(UserLikedSong.user_id != user.id)
            .distinct()
        )

        # Main query to find songs liked by those similar users.
        suggestions_query = (
            self.db.query(SongMetadata)
            .join(UserLikedSong)
            .filter(UserLikedSong.user_id.in_(similar_users_subquery))
            .filter(SongMetadata.id.notin_(liked_song_ids)) # Exclude songs user already likes.
            .group_by(SongMetadata.id)
            .order_by(func.count(SongMetadata.id).desc()) # Order by popularity.
            .limit(limit)
        )
        return suggestions_query.all()


# ==============================================================================
# --- Service Layer (Business Logic) ---
# ==============================================================================

class SuggestionService:
    """Orchestrates the business logic for finding and ranking song suggestions."""

    def __init__(self, api_key: str):
        """
        Initializes the service with the necessary API key.
        :param api_key: The YouTube Data API key.
        """
        self.api_key = api_key

    @lru_cache(maxsize=512)
    def _search_youtube_for_song(self, song_name: str) -> Optional[Dict]:
        """
        Searches YouTube for a song and returns its metadata. Uses LRU cache.
        :param song_name: The name of the song to search for.
        :return: A dictionary of video metadata or None.
        """
        if not self.api_key:
            return None
        try:
            # Sanitize query for better search results.
            query = re.sub(r'[^\w\s]', '', song_name).lower().strip()
            search_url = (
                f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video"
                f"&videoCategoryId=10&maxResults=1&key={self.api_key}"
            )
            resp = requests.get(search_url, timeout=5)
            resp.raise_for_status()
            items = resp.json().get('items', [])
            if not items:
                return None

            snippet = items[0]['snippet']
            return {
                "video_id": items[0]['id']['videoId'],
                "title": snippet['title'],
                "artist": snippet['channelTitle'],
            }
        except requests.RequestException as e:
            logger.error(f"YouTube search API error for '{song_name}': {e}")
            return None

    @lru_cache(maxsize=256)
    def _get_content_based_suggestions(self, video_id: str) -> List[Dict]:
        """
        Gets 'related' videos from YouTube to use as content-based candidates.
        :param video_id: The YouTube video ID to find related videos for.
        :return: A list of related video metadata dictionaries.
        """
        if not self.api_key:
            return []
        try:
            related_url = (
                f"https://www.googleapis.com/youtube/v3/search?part=snippet&relatedToVideoId={video_id}"
                f"&type=video&videoCategoryId=10&maxResults=15&key={self.api_key}"
            )
            resp = requests.get(related_url, timeout=5)
            resp.raise_for_status()
            items = resp.json().get('items', [])

            return [
                {
                    "video_id": item['id']['videoId'],
                    "title": item['snippet']['title'],
                    "artist": item['snippet']['channelTitle'],
                }
                for item in items if 'videoId' in item.get('id', {})
            ]
        except requests.RequestException as e:
            logger.error(f"YouTube related videos API error for '{video_id}': {e}")
            return []

    def get_suggestions(self, user: User, repo: MusicRepository, num_suggestions: int = 10) -> List[Dict]:
        """
        Generates suggestions using a hybrid of collaborative and content-based methods.
        :param user: The User object.
        :param repo: The MusicRepository for database access.
        :param num_suggestions: The desired number of final suggestions.
        :return: A ranked list of song suggestion dictionaries.
        """
        collaborative_raw = repo.get_collaborative_suggestions(user, limit=20)

        content_based_raw = []
        if user.likes:
            most_recent_like = sorted(user.likes, key=lambda x: x.created_at, reverse=True)[0]
            video_id_for_content = most_recent_like.song.video_id
            content_based_raw = self._get_content_based_suggestions(video_id_for_content)

        # Combine and rank suggestions, boosting songs found by both methods.
        suggestion_pool: Dict[str, Dict] = {}
        for song in collaborative_raw:
            suggestion_pool[song.video_id] = {
                "title": song.title, "artist": song.artist,
                "youtube_video_id": song.video_id, "score": 1.0,
            }

        for song_data in content_based_raw:
            vid = song_data["video_id"]
            if vid in suggestion_pool:
                suggestion_pool[vid]["score"] += 0.5  # Hybrid boost
            else:
                suggestion_pool[vid] = {
                    "title": song_data["title"], "artist": song_data["artist"],
                    "youtube_video_id": vid, "score": 0.8,
                }

        if not suggestion_pool:
            logger.warning(f"No suggestions found for user {user.user_id}. Consider a fallback.")
            return []

        final_suggestions = sorted(suggestion_pool.values(), key=lambda x: x['score'], reverse=True)
        return final_suggestions[:num_suggestions]


# ==============================================================================
# --- Dependency Injection ---
# ==============================================================================

def get_repo(db_session: Session = Depends(get_read_session)) -> MusicRepository:
    """Dependency to get a MusicRepository with a read-only session."""
    return MusicRepository(db=db_session)

def get_write_repos(db_sessions: List[Session] = Depends(get_write_sessions)) -> List[MusicRepository]:
    """Dependency to get a list of MusicRepositories with write-enabled sessions."""
    return [MusicRepository(db=s) for s in db_sessions]

def get_suggestion_service() -> SuggestionService:
    """Dependency to get an instance of the SuggestionService."""
    return SuggestionService(api_key=YOUTUBE_API_KEY)


# ==============================================================================
# --- API Endpoints ---
# ==============================================================================

@app.post("/suggestions", response_model=SuggestionResponse, tags=["Suggestions"])
async def post_suggestions(
    request: LikedSongsRequest,
    user_repos_write: List[MusicRepository] = Depends(get_write_repos),
    user_repo_read: MusicRepository = Depends(get_repo),
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
):
    """
    Accepts a user's liked songs, persists them, and returns personalized suggestions.
    This is the primary endpoint of the service.
    """
    if not YOUTUBE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: API key not configured.",
        )

    # Persist Likes
    song_metadata_ids_to_like = set()
    write_repo = user_repos_write[0]

    for song_name in request.songs:
        video_info = suggestion_service._search_youtube_for_song(song_name)
        if not video_info:
            continue

        song_meta = write_repo.get_song_metadata_by_video_id(video_info["video_id"])
        if not song_meta:
            song_meta = write_repo.create_song_metadata(video_info)

        song_metadata_ids_to_like.add(song_meta.id)

    # Sync likes across all write-enabled databases.
    for repo in user_repos_write:
        user = repo.get_or_create_user(request.user_id)
        repo.persist_user_likes(user, song_metadata_ids_to_like)

    # Generate Suggestions using the read replica for fresh data.
    user = user_repo_read.get_or_create_user(request.user_id)
    suggestions = suggestion_service.get_suggestions(user, user_repo_read)

    if not suggestions:
        raise HTTPException(
            status_code=404,
            detail="Could not find any personalized suggestions for the provided songs.",
        )

    return {"suggestions": suggestions}


@app.get("/health", status_code=200, tags=["Health"])
async def health_check():
    """
    Provides a simple health check endpoint to verify the service is running.
    """
    return {"status": "healthy"}
