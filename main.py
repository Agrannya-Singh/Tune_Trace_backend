# main.py
"""
FastAPI application for the Hybrid Music Suggestion microservice.

This application provides a RESTful API for generating personalized song
suggestions. It accepts a user's liked songs and an optional genre,
persists this data, and then returns a list of new suggestions. If no
personalized suggestions are found, it falls back to popular songs from
the specified genre.
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

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# --- Environment & Configuration ---
# ==============================================================================

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

app = FastAPI(
    title="Hybrid Music Suggestion API",
    description="Generates music suggestions using a hybrid model with a genre-based fallback.",
    version="2.1.0",
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

if not YOUTUBE_API_KEY:
    logger.critical("FATAL: YOUTUBE_API_KEY environment variable not set.")

# ==============================================================================
# --- FastAPI Application Events ---
# ==============================================================================

@app.on_event("startup")
def on_startup() -> None:
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
    user_id: str = Field(..., description="Unique client-generated identifier for the user.")
    songs: List[str] = Field(..., min_length=1, description="A list of song titles the user has liked.")
    # NEW FIELD: Added an optional genre for the fallback mechanism.
    genre: Optional[str] = Field(None, description="An optional genre to use for fallback suggestions.", example="Rock")

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
        ids_to_remove = existing_liked_ids - song_metadata_ids
        if ids_to_remove:
            self.db.query(UserLikedSong).filter(
                UserLikedSong.user_id == user.id,
                UserLikedSong.song_id.in_(ids_to_remove),
            ).delete(synchronize_session="fetch")
        if ids_to_add:
            new_likes = [UserLikedSong(user_id=user.id, song_id=song_id) for song_id in ids_to_add]
            self.db.add_all(new_likes)
        self.db.commit()

    def get_collaborative_suggestions(self, user: User, limit: int) -> List[SongMetadata]:
        liked_song_ids = {like.song_id for like in user.likes}
        if not liked_song_ids:
            return []
        similar_users_subquery = self.db.query(UserLikedSong.user_id).filter(
            UserLikedSong.song_id.in_(liked_song_ids), UserLikedSong.user_id != user.id
        ).distinct()
        return self.db.query(SongMetadata).join(UserLikedSong).filter(
            UserLikedSong.user_id.in_(similar_users_subquery),
            SongMetadata.id.notin_(liked_song_ids),
        ).group_by(SongMetadata.id).order_by(func.count(SongMetadata.id).desc()).limit(limit).all()

# ==============================================================================
# --- Service Layer (Business Logic) ---
# ==============================================================================

class SuggestionService:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @lru_cache(maxsize=512)
    def _search_youtube_for_song(self, song_name: str) -> Optional[Dict]:
        if not self.api_key: return None
        query = re.sub(r"[^\w\s]", "", song_name).lower().strip()
        if not query:
            logger.warning(f"Skipping empty search query from original input: '{song_name}'")
            return None
        search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video&videoCategoryId=10&maxResults=1&key={self.api_key}"
        resp = requests.get(search_url, timeout=5)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items: return None
        snippet = items[0]["snippet"]
        return {"video_id": items[0]["id"]["videoId"], "title": snippet["title"], "artist": snippet["channelTitle"]}

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
        collaborative_raw = repo.get_collaborative_suggestions(user, limit=20)
        content_based_raw = []
        if user.likes:
            most_recent_like = sorted(user.likes, key=lambda x: x.created_at, reverse=True)[0]
            if most_recent_like.song:
                # Use a standard search as a proxy for "content-based"
                query = f"{most_recent_like.song.title} {most_recent_like.song.artist}"
                search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video&videoCategoryId=10&maxResults=15&key={self.api_key}"
                resp = requests.get(search_url, timeout=5)
                resp.raise_for_status()
                items = resp.json().get("items", [])
                content_based_raw = [
                    {"video_id": item['id']['videoId'], "title": item['snippet']['title'], "artist": item['snippet']['channelTitle']}
                    for item in items if 'videoId' in item.get('id', {}) and item['id']['videoId'] != most_recent_like.song.video_id
                ]

        suggestion_pool: Dict[str, Dict] = {}
        for song in collaborative_raw:
            suggestion_pool[song.video_id] = {"title": song.title, "artist": song.artist, "youtube_video_id": song.video_id, "score": 1.0}
        
        for song_data in content_based_raw:
            vid = song_data["video_id"]
            if vid in suggestion_pool:
                suggestion_pool[vid]["score"] += 0.5
            else:
                suggestion_pool[vid] = {"title": song_data["title"], "artist": song_data["artist"], "youtube_video_id": vid, "score": 0.8}
        
        # MODIFIED: If the pool is empty, call the fallback method.
        if not suggestion_pool:
            logger.warning(f"No personalized suggestions found for user {user.user_id}. Triggering fallback.")
            return self._get_fallback_suggestions(genre=genre, num_suggestions=num_suggestions)
        
        final_suggestions = sorted(suggestion_pool.values(), key=lambda x: x["score"], reverse=True)
        return final_suggestions[:num_suggestions]

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
    repo: MusicRepository = Depends(get_repo),
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
):
    if not YOUTUBE_API_KEY:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service unavailable: YouTube API key is not configured.")
    
    try:
        song_metadata_ids_to_like = set()
        for song_name in request.songs:
            video_info = suggestion_service._search_youtube_for_song(song_name)
            if not video_info: continue
            song_meta = repo.get_song_metadata_by_video_id(video_info["video_id"])
            if not song_meta:
                song_meta = repo.create_song_metadata(video_info)
            song_metadata_ids_to_like.add(song_meta.id)
        
        user = repo.get_or_create_user(request.user_id)
        repo.persist_user_likes(user, song_metadata_ids_to_like)
        
        # MODIFIED: Pass the genre from the request to the service layer.
        suggestions = suggestion_service.get_suggestions(user, repo, genre=request.genre)

        # The service layer now handles the "not found" case, so we just return the results.
        return {"suggestions": suggestions}

    except requests.RequestException as e:
        logger.error(f"A critical YouTube API error occurred: {e}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "External service (YouTube API) is currently unavailable.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "An internal server error occurred.")

@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    return {"status": "healthy"}
