# routers/users.py
import json
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status

from repository import MusicRepository
from api_models import LikedSongResponse
from dependencies import get_repo
from auth import get_current_user
from utils.metrics import track_latency
from redis_utils import redis_client

logger = logging.getLogger(__name__)
router = APIRouter(tags=["User Data"])

@router.get("/liked-songs", response_model=List[LikedSongResponse])
async def get_liked_songs(
    user_id: str = Query(..., max_length=255, min_length=1),
    current_user: dict = Depends(get_current_user),
    repo: MusicRepository = Depends(get_repo),
):
    """Returns the list of liked songs for a given user."""
    # Enforce token identity
    if current_user and current_user.get("email"):
        user_id = current_user["email"]
    else:
        if user_id != 'anon@use.com':
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unauthenticated request for registered user.")

    # Set user context for RLS
    if user_id != 'anon@use.com':
        repo.set_app_user(user_id)

    try:
        if redis_client:
            with track_latency("Redis:Read"):
                cached_data = redis_client.get(f"user_likes:{user_id}")
            
            if cached_data:
                logger.info(f"Cache HIT for user {user_id}")
                song_ids = json.loads(cached_data)
                with track_latency("PostgreSQL:Read_Cached"):
                    liked_songs = repo.get_songs_by_ids(song_ids)
                return [
                    LikedSongResponse(
                        video_id=s.video_id,
                        title=s.title,
                        artist=s.artist,
                        created_at="from_cache"
                    )
                    for s in liked_songs
                ]

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
