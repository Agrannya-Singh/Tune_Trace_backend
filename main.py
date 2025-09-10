# main.py
"""
FastAPI application for the Enhanced Music Suggestion service.

Provides endpoints for managing user's liked songs and generating personalized
song suggestions using a hybrid approach (collaborative and content-based filtering).

added filtering to remove already liked songs so they do not get "suggested" to end user.
"""

import os
import requests
import logging
import re
import json
import random
from typing import List, Optional, Dict, Set, Tuple
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
import redis

from db import (
    init_db, get_read_session, get_write_sessions, 
    User, UserLikedSong, SongMetadata
)

# --- Initial Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment & Configuration ---
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "3600"))

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hybrid Music Suggestion API",
    description="Generates music suggestions using collaborative and content-based filtering.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Redis Client ---
redis_client: Optional[redis.Redis] = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        redis_client = None

if not YOUTUBE_API_KEY:
    logger.critical("FATAL: YouTube API key not found. Service will not work.")


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database tables on application startup."""
    init_db()


# --- Pydantic Models ---
class SongSuggestion(BaseModel):
    title: str
    artist: str
    youtube_video_id: str

class SuggestionResponse(BaseModel):
    suggestions: List[SongSuggestion]

class LikedSongsResponse(BaseModel):
    liked_songs: List[str]

class LikedSongsRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user.")
    songs: List[str] = Field(..., description="A list of song names the user likes.")


# --- REPOSITORY LAYER (Data Access) ---
class MusicRepository:
    """Handles all database interactions for users and songs."""
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_user(self, user_id: str) -> User:
        """Retrieves a user by their ID, creating them if they don't exist."""
        user = self.db.query(User).filter_by(user_id=user_id).one_or_none()
        if not user:
            user = User(user_id=user_id)
            self.db.add(user)
            self.db.flush() # Flush to get the user's generated ID
        return user
    
    def get_song_metadata_by_video_id(self, video_id: str) -> Optional[SongMetadata]:
        """Finds song metadata by its YouTube video ID."""
        return self.db.query(SongMetadata).filter_by(video_id=video_id).one_or_none()

    def create_song_metadata(self, video_data: dict) -> SongMetadata:
        """Creates and stores a new SongMetadata record."""
        song = SongMetadata(
            video_id=video_data["video_id"],
            title=video_data["title"],
            artist=video_data["artist"],
            tags=",".join(video_data.get("tags", []))
        )
        self.db.add(song)
        self.db.flush() # Flush to get the song's generated ID
        return song

    def get_liked_songs(self, user: User) -> List[str]:
        """Gets a list of liked song titles for a given user."""
        liked_song_records = self.db.query(UserLikedSong).options(joinedload(UserLikedSong.song)).filter_by(user_id=user.id).all()
        return [record.song.title for record in liked_song_records]

    def persist_user_likes(self, user: User, song_metadata_ids: Set[int]):
        """Synchronizes the user's liked songs with the provided set of song IDs."""
        existing_liked_ids = user.get_liked_song_ids()
        
        ids_to_add = song_metadata_ids - existing_liked_ids
        ids_to_remove = existing_liked_ids - song_metadata_ids

        if ids_to_remove:
            self.db.query(UserLikedSong).filter(
                UserLikedSong.user_id == user.id,
                UserLikedSong.song_id.in_(ids_to_remove)
            ).delete(synchronize_session='fetch')
        
        for song_id in ids_to_add:
            self.db.add(UserLikedSong(user_id=user.id, song_id=song_id))
        
        self.db.commit()

    def get_collaborative_suggestions(self, user: User, limit: int) -> List[SongMetadata]:
        """
        Implements collaborative filtering.
        Finds songs liked by 'taste neighbors' (users with overlapping likes).
        """
        liked_song_ids = user.get_liked_song_ids()
        if not liked_song_ids:
            return []

        # Find users who liked at least one same song
        similar_users_subquery = (
            self.db.query(UserLikedSong.user_id)
            .filter(UserLikedSong.song_id.in_(liked_song_ids))
            .filter(UserLikedSong.user_id != user.id)
            .distinct()
        )

        # Find songs liked by those users, excluding songs the current user already likes,
        # ordered by popularity among that group.
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


# --- SERVICE LAYER (Business Logic) ---
class SuggestionService:
    """Orchestrates the logic for finding and ranking song suggestions."""
    
    def __init__(self, api_key: str, redis: Optional[redis.Redis], ttl: int):
        self.api_key = api_key
        self.redis = redis
        self.ttl = ttl

    @lru_cache(maxsize=256)
    def _search_youtube_for_song(self, song_name: str) -> Optional[Dict]:
        """Searches YouTube for a song and returns its metadata."""
        if not self.api_key: return None
        try:
            query = re.sub(r'[^\w\s]', '', song_name).lower().strip()
            search_url = (f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video"
                          f"&videoCategoryId=10&maxResults=1&key={self.api_key}")
            resp = requests.get(search_url, timeout=5)
            resp.raise_for_status()
            items = resp.json().get('items', [])
            if not items: return None
            
            snippet = items[0]['snippet']
            return {
                "video_id": items[0]['id']['videoId'],
                "title": snippet['title'],
                "artist": snippet['channelTitle']
            }
        except requests.RequestException as e:
            logger.error(f"YouTube search API error for '{song_name}': {e}")
            return None

    @lru_cache(maxsize=128)
    def _get_content_based_suggestions(self, video_id: str) -> List[Dict]:
        """Gets 'related' videos from YouTube to use as content-based candidates."""
        if not self.api_key: return []
        try:
            related_url = (f"https://www.googleapis.com/youtube/v3/search?part=snippet&relatedToVideoId={video_id}"
                           f"&type=video&videoCategoryId=10&maxResults=15&key={self.api_key}")
            resp = requests.get(related_url, timeout=5)
            resp.raise_for_status()
            items = resp.json().get('items', [])
            
            return [
                {
                    "video_id": item['id']['videoId'],
                    "title": item['snippet']['title'],
                    "artist": item['snippet']['channelTitle']
                } 
                for item in items if 'videoId' in item.get('id', {})
            ]
        except requests.RequestException as e:
            logger.error(f"YouTube related videos API error for '{video_id}': {e}")
            return []

    def get_suggestions(self, user: User, repo: MusicRepository, num_suggestions: int = 5) -> List[Dict]:
        """
        Main suggestion generation method using a hybrid approach.
        1. Gets collaborative suggestions from taste neighbors.
        2. Gets content-based suggestions for the user's most recent like.
        3. Blends, ranks, and returns the top unique results.
        """
        collaborative_raw = repo.get_collaborative_suggestions(user, limit=10)
        
        content_based_raw = []
        if user.likes:
            # Get content suggestions based on the most recently liked song
            most_recent_like = sorted(user.likes, key=lambda x: x.created_at, reverse=True)[0]
            video_id_for_content = most_recent_like.song.video_id
            content_based_raw = self._get_content_based_suggestions(video_id_for_content)

        # Combine and rank
        suggestion_pool: Dict[str, Dict] = {}
        
        # Add collaborative suggestions with a base score
        for song in collaborative_raw:
            suggestion_pool[song.video_id] = {
                "title": song.title, "artist": song.artist, 
                "youtube_video_id": song.video_id, "score": 1.0
            }
            
        # Add content-based, boosting score if already present (hybrid boost)
        for song_data in content_based_raw:
            vid = song_data["video_id"]
            if vid in suggestion_pool:
                suggestion_pool[vid]["score"] += 0.5 # Boost for being relevant in both models
            else:
                suggestion_pool[vid] = {
                    "title": song_data["title"], "artist": song_data["artist"],
                    "youtube_video_id": vid, "score": 0.8 # Slightly lower base score for pure content
                }

        if not suggestion_pool:
            logger.warning(f"No suggestions found for user {user.user_id}. Consider a fallback.")
            return []
            
        final_suggestions = sorted(suggestion_pool.values(), key=lambda x: x['score'], reverse=True)
        return final_suggestions[:num_suggestions]


# --- DEPENDENCY INJECTION ---
def get_repo(db_session: Session = Depends(get_read_session)):
    return MusicRepository(db=db_session)

def get_write_repos(db_sessions: List[Session] = Depends(get_write_sessions)):
    return [MusicRepository(db=s) for s in db_sessions]

def get_suggestion_service():
    return SuggestionService(
        api_key=YOUTUBE_API_KEY, redis=redis_client, ttl=REDIS_TTL_SECONDS
    )


# --- API ENDPOINTS ---
@app.post("/suggestions", response_model=SuggestionResponse, summary="Get song suggestions")
async def post_suggestions(
    request: LikedSongsRequest,
    user_repos_write: List[MusicRepository] = Depends(get_write_repos),
    user_repo_read: MusicRepository = Depends(get_repo),
    suggestion_service: SuggestionService = Depends(get_suggestion_service)
):
    """
    Accepts a user's liked songs, persists them, and returns personalized suggestions.
    """
    if not YOUTUBE_API_KEY:
        raise HTTPException(status_code=503, detail="Service unavailable: YouTube API key not configured.")
    if not request.songs:
        raise HTTPException(status_code=400, detail="At least one song must be provided.")

    # --- Persist Likes ---
    # 1. Find video IDs and metadata for the provided song names
    song_metadata_ids_to_like = set()
    write_repo = user_repos_write[0] # Use the first repo for lookups/creations
    
    for song_name in request.songs:
        # First, try to find the song in our DB to avoid API calls
        video_info = suggestion_service._search_youtube_for_song(song_name)
        if not video_info: continue
        
        song_meta = write_repo.get_song_metadata_by_video_id(video_info["video_id"])
        if not song_meta:
            song_meta = write_repo.create_song_metadata(video_info)
        
        song_metadata_ids_to_like.add(song_meta.id)

    # 2. Sync likes for the user across all write-able databases
    for repo in user_repos_write:
        user = repo.get_or_create_user(request.user_id)
        repo.persist_user_likes(user, song_metadata_ids_to_like)

    # --- Generate Suggestions ---
    # Use the read repository to get fresh user data and generate suggestions
    user = user_repo_read.get_or_create_user(request.user_id)
    suggestions = suggestion_service.get_suggestions(user, user_repo_read)
    
    if not suggestions:
        # Optional: Implement a generic fallback to popular songs here
        raise HTTPException(status_code=404, detail="Could not find any personalized suggestions.")
    
    return {"suggestions": suggestions}


@app.get("/liked-songs", response_model=LikedSongsResponse, summary="Get a user's liked songs")
async def get_liked_songs_endpoint(
    user_id: str = Query(..., min_length=1),
    repo: MusicRepository = Depends(get_repo)
):
    user = repo.get_or_create_user(user_id)
    songs = repo.get_liked_songs(user)
    return {"liked_songs": songs}


@app.get("/health", summary="Health check")
async def health_check():
    return {"status": "healthy"}
