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
import urllib.parse

# --- Third-Party Imports ---
import redis
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, text
from sqlalchemy.orm import Session, joinedload

# --- Local Application Imports ---
from db import SessionLocal, SongMetadata, User, UserLikedSong, get_session

# ==============================================================================
# --- Initial Application Setup ---
# ==============================================================================

# Load environment variables from a .env file for local development.
# In production, these should be set directly in the environment.
load_dotenv()

# Configure a logger for consistent application-wide logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# --- Environment & Configuration ---
# ==============================================================================

# Centralize configuration for better management and validation.
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

# Initialize the FastAPI application.
app = FastAPI(
    title="Hybrid Music Suggestion API",
    description="Generates music suggestions using collaborative and content-based filtering.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS) middleware.
# Using a wildcard ("*") is convenient for development but should be
# restricted to the actual frontend domain in a production environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ==============================================================================
# --- Service Connections (Redis, etc.) ---
# ==============================================================================

# Establish a connection to Redis for caching if a URL is provided.
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
    logger.critical("FATAL: YOUTUBE_API_KEY environment variable not set.")


# ==============================================================================
# --- FastAPI Application Events ---
# ==============================================================================


@app.on_event("startup")
def on_startup() -> None:
    """
    FastAPI startup event handler. Verifies the database connection.
    This ensures the application fails fast if the database is unavailable.
    """
    logger.info("Application starting up...")
    try:
        # Use a temporary session to check the database connection.
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        logger.info("Connection to the database established successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Could not connect to the database: {e}")
        # Raising an error here will prevent the application from starting.
        raise RuntimeError(f"Database connection failed: {e}") from e

    logger.info("Application startup complete.")


# ==============================================================================
# --- Pydantic Data Models (API Contracts) ---
# ==============================================================================


class SongSuggestion(BaseModel):
    """Defines the structure for a single song suggestion returned to the client."""

    title: str
    artist: str
    youtube_video_id: str


class SuggestionResponse(BaseModel):
    """Defines the structure of the successful response from the /suggestions endpoint."""

    suggestions: List[SongSuggestion]


class LikedSongsRequest(BaseModel):
    """Defines the structure of the request body for the /suggestions endpoint."""

    user_id: str = Field(
        ..., description="Unique client-generated identifier for the user."
    )
    songs: List[str] = Field(
        ..., min_length=1, description="A list of song titles the user has liked."
    )


# ==============================================================================
# --- Repository Layer (Data Access) ---
# ==============================================================================


class MusicRepository:
    """Handles all database interactions for users, songs, and their relationships."""

    def __init__(self, db: Session):
        self.db = db

    def get_or_create_user(self, user_id: str) -> User:
        """
        Retrieves a user by their unique ID or creates a new one if not found.
        This method is atomic and ensures a user object is always returned with a valid ID.
        """
        user = (
            self.db.query(User)
            .options(joinedload(User.likes))
            .filter_by(user_id=user_id)
            .one_or_none()
        )
        if not user:
            user = User(user_id=user_id)
            self.db.add(user)
            # Flush the session to persist the new user and get back the auto-generated ID.
            self.db.flush()
        return user

    def get_song_metadata_by_video_id(
        self, video_id: str
    ) -> Optional[SongMetadata]:
        """Retrieves song metadata from the database by its YouTube video ID."""
        return self.db.query(SongMetadata).filter_by(video_id=video_id).one_or_none()

    def create_song_metadata(self, video_data: dict) -> SongMetadata:
        """Creates and persists a new song metadata record."""
        song = SongMetadata(
            video_id=video_data["video_id"],
            title=video_data["title"],
            artist=video_data["artist"],
        )
        self.db.add(song)
        # Flush to get the ID for immediate use, maintaining transactional integrity.
        self.db.flush()
        return song

    def persist_user_likes(self, user: User, song_metadata_ids: Set[int]):
        """
        Synchronizes the user's liked songs in the database with the provided set of song IDs.
        It efficiently adds new likes and removes old ones.
        """
        existing_liked_ids = {like.song_id for like in user.likes}
        ids_to_add = song_metadata_ids - existing_liked_ids
        ids_to_remove = existing_liked_ids - song_metadata_ids

        if ids_to_remove:
            self.db.query(UserLikedSong).filter(
                UserLikedSong.user_id == user.id,
                UserLikedSong.song_id.in_(ids_to_remove),
            ).delete(synchronize_session="fetch")

        if ids_to_add:
            new_likes = [
                UserLikedSong(user_id=user.id, song_id=song_id)
                for song_id in ids_to_add
            ]
            self.db.add_all(new_likes)

        # A single commit for the entire transaction is more efficient.
        self.db.commit()

    def get_collaborative_suggestions(
        self, user: User, limit: int
    ) -> List[SongMetadata]:
        """
        Generates suggestions based on collaborative filtering.
        Finds songs liked by similar users who share liked songs with the current user.
        """
        liked_song_ids = {like.song_id for like in user.likes}
        if not liked_song_ids:
            return []

        # Subquery to find users with similar tastes.
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
            .filter(SongMetadata.id.notin_(liked_song_ids))
            .group_by(SongMetadata.id)
            .order_by(func.count(SongMetadata.id).desc())
            .limit(limit)
        )
        return suggestions_query.all()


# ==============================================================================
# --- Service Layer (Business Logic) ---
# ==============================================================================


class SuggestionService:
    """Orchestrates the business logic for finding and ranking song suggestions."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @lru_cache(maxsize=512)
    def _search_youtube_for_song(self, song_name: str) -> Optional[Dict]:
        """
        Searches the YouTube API for a given song name.
        Uses LRU caching to avoid repeated API calls for the same song.
        """
        if not self.api_key:
            return None
        # Sanitize query for better search results.
        query = re.sub(r"[^\w\s]", "", song_name).lower().strip()
        q_encoded = urllib.parse.quote(query)
        search_url = (
            "https://www.googleapis.com/youtube/v3/search"
            f"?part=snippet&q={q_encoded}&type=video&videoCategoryId=10"
            f"&maxResults=1&key={self.api_key}"
        )
        try:
            resp = requests.get(search_url, timeout=5)
            resp.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            items = resp.json().get("items", [])
            if not items:
                return None

            snippet = items[0]["snippet"]
            return {
                "video_id": items[0]["id"]["videoId"],
                "title": snippet["title"],
                "artist": snippet["channelTitle"],
            }
        except requests.RequestException:
            return None

    @lru_cache(maxsize=256)
    def _get_video_details(self, video_id: str) -> Optional[Dict]:
        """
        Fetches video details from the YouTube API to get snippet information like tags.
        """
        if not self.api_key:
            return None
        url = (
            "https://www.googleapis.com/youtube/v3/videos"
            f"?part=snippet&id={video_id}&key={self.api_key}"
        )
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                return None
            return items[0]
        except requests.RequestException:
            return None

    @lru_cache(maxsize=256)
    def _get_content_based_suggestions(self, video_id: str) -> List[Dict]:
        """
        Fetches content-based suggestions using a keyword search based on the video's tags or title.
        This replaces the deprecated relatedToVideoId parameter.
        """
        details = self._get_video_details(video_id)
        if not details:
            return []
        snippet = details["snippet"]
        tags = snippet.get("tags", [])
        q = " ".join(tags[:5]) if tags else snippet["title"]
        q_encoded = urllib.parse.quote(q)
        search_url = (
            "https://www.googleapis.com/youtube/v3/search"
            f"?part=snippet&q={q_encoded}&type=video"
            f"&videoCategoryId=10&maxResults=16&key={self.api_key}"
        )
        try:
            resp = requests.get(search_url, timeout=5)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            related = [
                {
                    "video_id": item["id"]["videoId"],
                    "title": item["snippet"]["title"],
                    "artist": item["snippet"]["channelTitle"],
                }
                for item in items
                if "videoId" in item.get("id", {}) and item["id"]["videoId"] != video_id
            ]
            return related[:15]
        except requests.RequestException:
            return []

    def get_suggestions(
        self, user: User, repo: MusicRepository, num_suggestions: int = 15
    ) -> List[Dict]:
        """
        Generates a hybrid list of suggestions by combining collaborative and
        content-based results and ranking them by a simple scoring system.
        """
        collaborative_raw = repo.get_collaborative_suggestions(user, limit=30)
        content_based_raw = []
        if user.likes:
            # Use the most recently liked song to find content-based matches.
            most_recent_like = sorted(
                user.likes, key=lambda x: x.created_at, reverse=True
            )[0]
            video_id_for_content = most_recent_like.song.video_id
            content_based_raw = self._get_content_based_suggestions(
                video_id_for_content
            )

        suggestion_pool: Dict[str, Dict] = {}
        # Populate with collaborative suggestions (higher base score).
        for song in collaborative_raw:
            suggestion_pool[song.video_id] = {
                "title": song.title,
                "artist": song.artist,
                "youtube_video_id": song.video_id,
                "score": 1.0,
            }
        # Add or boost with content-based suggestions.
        for song_data in content_based_raw:
            vid = song_data["video_id"]
            if vid in suggestion_pool:
                suggestion_pool[vid]["score"] += 0.5  # Boost score if also collaborative
            else:
                suggestion_pool[vid] = {
                    "title": song_data["title"],
                    "artist": song_data["artist"],
                    "youtube_video_id": vid,
                    "score": 0.8,  # Lower base score
                }

        if not suggestion_pool:
            logger.warning(
                f"No suggestions found for user {user.user_id}. Consider a fallback."
            )
            return []

        final_suggestions = sorted(
            suggestion_pool.values(), key=lambda x: x["score"], reverse=True
        )
        return final_suggestions[:num_suggestions]


# ==============================================================================
# --- Dependency Injection ---
# ==============================================================================


def get_repo(db_session: Session = Depends(get_session)) -> MusicRepository:
    """Dependency to provide a MusicRepository instance with a database session."""
    return MusicRepository(db=db_session)


def get_suggestion_service() -> SuggestionService:
    """Dependency to provide an instance of the SuggestionService."""
    return SuggestionService(api_key=YOUTUBE_API_KEY)


# ==============================================================================
# --- API Endpoints ---
# ==============================================================================


@app.post(
    "/suggestions",
    response_model=SuggestionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Suggestions"],
    summary="Get personalized song suggestions",
)
async def post_suggestions(
    request: LikedSongsRequest,
    repo: MusicRepository = Depends(get_repo),
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
):
    """
    Accepts a user ID and a list of liked songs, then returns personalized suggestions.

    This endpoint performs the following actions:
    1.  Finds or creates metadata for each liked song via the YouTube API.
    2.  Finds or creates the user record in the database.
    3.  Persists the user's liked songs.
    4.  Generates a new list of suggestions using a hybrid recommendation model.
    """
    if not YOUTUBE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is unavailable: YouTube API key is not configured.",
        )

    try:
        # Step 1: Find or create metadata for all liked songs.
        song_metadata_ids_to_like = set()
        for song_name in request.songs:
            video_info = suggestion_service._search_youtube_for_song(song_name)
            if not video_info:
                continue

            song_meta = repo.get_song_metadata_by_video_id(video_info["video_id"])
            if not song_meta:
                song_meta = repo.create_song_metadata(video_info)

            song_metadata_ids_to_like.add(song_meta.id)

        # Step 2 & 3: Get the user and sync their liked songs in one transaction.
        # FIXED: Use the repository method which correctly handles creating new users
        # and flushing the session to get a valid ID before proceeding.
        user = repo.get_or_create_user(request.user_id)
        repo.persist_user_likes(user, song_metadata_ids_to_like)

        # Step 4: Generate new suggestions for the user.
        suggestions = suggestion_service.get_suggestions(user, repo)

        if not suggestions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Could not find personalized suggestions for the provided songs.",
            )

        return {"suggestions": suggestions}

    except requests.RequestException as e:
        # If any YouTube API call fails, return a 503 error.
        logger.error(f"A critical YouTube API error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External service (YouTube API) is currently unavailable.",
        )
    except Exception as e:
        # Catch any other unexpected server errors.
        logger.exception(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred.",
        )


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    """Provides a simple health check endpoint to verify the service is running."""
    return {"status": "healthy"}
