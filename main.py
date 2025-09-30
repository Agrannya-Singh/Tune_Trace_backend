# main.py

# --- Standard Library Imports ---
import logging
import os
import re
import json
from functools import lru_cache
from typing import Dict, List, Optional, Set

# --- Third-Party Imports ---
import redis
import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, text
from sqlalchemy.orm import Session, joinedload

# --- Local Application Imports ---
from db import SessionLocal, SongMetadata, User, UserLikedSong, get_session

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
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", 3600))  # Default to 1 hour

if not YOUTUBE_API_KEY:
    logger.critical("FATAL: YOUTUBE_API_KEY environment variable not set.")

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
    """Verify database connection on application startup."""
    logger.info("Application starting up...")
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        logger.info("Connection to the database established successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Could not connect to the database: {e}")
        raise RuntimeError(f"Database connection failed: {e}") from e
    logger.info("Application startup complete.")

# ==============================================================================
# --- Pydantic Data Models (API Contracts) ---
# ==============================================================================

class SongSuggestion(BaseModel):
    title: str
    artist: str
    youtube_video_id: str

class SuggestionResponse(BaseModel):
    suggestions: List[SongSuggestion]

class LikedSongsRequest(BaseModel):
    user_id: str = Field(..., description="User email or unique identifier from OAuth.", max_length=255)
    songs: List[str] = Field(..., min_length=1, max_length=50, description="A list of song titles the user has liked (max 50).")
    genre: Optional[str] = Field(None, description="An optional genre for fallback suggestions.", example="Rock", max_length=128)

class LikedSongResponse(BaseModel):
    video_id: str
    title: str
    artist: str
    created_at: str

# ==============================================================================
# --- Background Tasks ---
# ==============================================================================

def update_redis_user_likes(user_id: str, song_ids: set[int]):
    """
    Background task to update a user's liked songs in the Redis cache.
    This runs after the HTTP response is sent.
    """
    if not redis_client:
        logger.warning("Redis client not available. Skipping cache update for user %s.", user_id)
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
# --- Repository Layer (Data Access) ---
# ==============================================================================

class MusicRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def get_or_create_user(self, user_id: str) -> User:
        user = self.db.query(User).options(joinedload(User.likes)).filter_by(user_id=user_id).one_or_none()
        if not user:
            user = User(user_id=user_id)
            self.db.add(user)
            self.db.flush()
        return user

    def get_song_metadata_by_video_id(self, video_id: str) -> Optional[SongMetadata]:
        return self.db.query(SongMetadata).filter_by(video_id=video_id).one_or_none()

    def create_song_metadata(self, video_data: dict) -> SongMetadata:
        song = SongMetadata(
            video_id=video_data["video_id"],
            title=video_data["title"],
            artist=video_data["artist"],
        )
        self.db.add(song)
        self.db.flush()
        return song

    def persist_user_likes(self, user: User, song_metadata_ids: Set[int]):
        existing_liked_ids = {like.song_id for like in user.likes}
        ids_to_add = song_metadata_ids - existing_liked_ids
        
        if ids_to_add:
            new_likes = [UserLikedSong(user_id=user.id, song_id=song_id) for song_id in ids_to_add]
            self.db.add_all(new_likes)
        
        self.db.commit()
    
    def get_user_liked_songs(self, user_id: str) -> List[tuple]:
        """Returns list of (video_id, title, artist, created_at) for a user's liked songs."""
        user = self.db.query(User).filter_by(user_id=user_id).one_or_none()
        if not user:
            return []
        
        results = (
            self.db.query(
                SongMetadata.video_id,
                SongMetadata.title,
                SongMetadata.artist,
                UserLikedSong.created_at
            )
            .join(UserLikedSong, UserLikedSong.song_id == SongMetadata.id)
            .filter(UserLikedSong.user_id == user.id)
            .order_by(UserLikedSong.created_at.desc())
            .all()
        )
        return results
    
    def get_collaborative_suggestions(self, user: User, limit: int = 10) -> List[SongMetadata]:
        """Get song suggestions based on collaborative filtering.
        
        Finds songs liked by users with similar taste (users who liked the same songs).
        """
        if not user.likes:
            return []
        
        # Get songs liked by this user
        user_liked_song_ids = user.get_liked_song_ids()
        
        # Find other users who liked the same songs
        similar_users = (
            self.db.query(User.id)
            .join(UserLikedSong)
            .filter(UserLikedSong.song_id.in_(user_liked_song_ids))
            .filter(User.id != user.id)
            .group_by(User.id)
            .having(func.count(UserLikedSong.song_id) >= 2)  # At least 2 songs in common
            .all()
        )
        
        if not similar_users:
            return []
        
        similar_user_ids = [u[0] for u in similar_users]
        
        # Get songs liked by similar users that the current user hasn't liked
        recommendations = (
            self.db.query(SongMetadata)
            .join(UserLikedSong)
            .filter(UserLikedSong.user_id.in_(similar_user_ids))
            .filter(~SongMetadata.id.in_(user_liked_song_ids))
            .group_by(SongMetadata.id)
            .order_by(func.count(UserLikedSong.user_id).desc())  # Most popular among similar users
            .limit(limit)
            .all()
        )
        
        return recommendations

# ==============================================================================
# --- Service Layer (Business Logic) ---
# ==============================================================================

class SuggestionService:
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key

    @lru_cache(maxsize=512)
    def _search_youtube_for_song(self, song_name: str) -> Optional[Dict]:
        if not self.api_key: return None
        
        # Enhanced input sanitization - allow alphanumeric, spaces, hyphens, apostrophes
        query = re.sub(r"[^\w\s\-']", "", song_name).lower().strip()
        if not query or len(query) < 2:
            logger.warning(f"Skipping invalid/short search query from original input: '{song_name}'")
            return None
        
        # Limit query length to prevent abuse
        query = query[:200]
            
        search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video&videoCategoryId=10&maxResults=1&key={self.api_key}"
        
        try:
            resp = requests.get(search_url, timeout=8)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            
            if not items: return None
            
            snippet = items[0]["snippet"]
            return {"video_id": items[0]["id"]["videoId"], "title": snippet["title"], "artist": snippet["channelTitle"]}
        except requests.Timeout:
            logger.error(f"YouTube API timeout for query: {query[:50]}...")
            return None
        except requests.RequestException as e:
            logger.error(f"YouTube API error (sanitized): Status {getattr(e.response, 'status_code', 'N/A')}")
            return None

    def _get_fallback_suggestions(self, genre: Optional[str] = None, num_suggestions: int = 10) -> List[Dict]:
        logger.info(f"Executing fallback search for genre: {genre or 'Global Hits'}")
        if not self.api_key: return []
        
        search_term = f"Top {genre} songs" if genre else "Top Global Hits"
        search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={search_term}&type=video&videoCategoryId=10&maxResults={num_suggestions}&key={self.api_key}"
        
        resp = requests.get(search_url, timeout=5)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        
        return [
            {"title": item['snippet']['title'], "artist": item['snippet']['channelTitle'], "youtube_video_id": item['id']['videoId']}
            for item in items if 'videoId' in item.get('id', {})
        ]

    def get_suggestions(self, user: User, repo: MusicRepository, genre: Optional[str] = None, num_suggestions: int = 10) -> List[Dict]:
        collaborative_raw = repo.get_collaborative_suggestions(user, limit=num_suggestions)
        
        if not collaborative_raw:
            logger.warning(f"No personalized suggestions for user {user.user_id}. Triggering fallback.")
            return self._get_fallback_suggestions(genre=genre, num_suggestions=num_suggestions)
        
        return [
            {"title": song.title, "artist": song.artist, "youtube_video_id": song.video_id}
            for song in collaborative_raw
        ]

# ==============================================================================
# --- Dependency Injection ---
# ==============================================================================

def get_repo(db_session: Session = Depends(get_session)) -> MusicRepository:
    return MusicRepository(db=db_session)

def get_suggestion_service() -> SuggestionService:
    return SuggestionService(api_key=YOUTUBE_API_KEY)

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
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service is not configured.")
    
    try:
        # Resolve song names to YouTube video metadata
        song_metadata_ids_to_like = set()
        for song_name in request.songs:
            video_info = suggestion_service._search_youtube_for_song(song_name)
            if not video_info: continue
            
            song_meta = repo.get_song_metadata_by_video_id(video_info["video_id"])
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
        
        # 3. Generate suggestions and return the response immediately.
        suggestions = suggestion_service.get_suggestions(user, repo, genre=request.genre)
        return {"suggestions": suggestions}

    except requests.RequestException as e:
        logger.error(f"A critical YouTube API error occurred: {e}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "External service is unavailable.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "An internal server error occurred.")

@app.get("/liked-songs", response_model=List[LikedSongResponse], tags=["User Data"])
async def get_liked_songs(
    user_id: str,
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
