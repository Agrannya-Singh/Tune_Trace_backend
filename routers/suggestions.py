# routers/suggestions.py
import asyncio
import json
import logging
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from typing import Set

from db import get_session
from ml_engine import MLEngine
from services import SuggestionService
from repository import MusicRepository
from api_models import SuggestionResponse, LikedSongsRequest, SongSuggestion
from dependencies import get_repo, get_suggestion_service
from utils.metrics import track_latency
from config import YOUTUBE_API_KEY
from redis_utils import redis_client
from tasks import update_redis_user_likes

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Suggestions"])

# Shared ML engine
from engine import ml_engine

@router.post("/suggestions", response_model=SuggestionResponse)
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
        # 1. Identify songs already in DB vs those needing YouTube search
        song_metadata_ids_to_like = set()
        songs_needing_search = []
        
        for song_str in request.songs:
            # Try to parse "Title - Artist"
            parts = song_str.split(" - ", 1)
            if len(parts) == 2:
                title, artist = parts
                existing = repo.get_song_by_title_and_artist(title.strip(), artist.strip())
                if existing:
                    song_metadata_ids_to_like.add(existing.id)
                    continue
            
            songs_needing_search.append(song_str)

        # 2. Search YouTube only for unknown songs
        if songs_needing_search:
            tasks = [
                suggestion_service._search_youtube_for_song_async(song_name)
                for song_name in songs_needing_search
            ]
            with track_latency("YouTube:Search_Parallel"):
                results = await asyncio.gather(*tasks)

            # Process results and persist new song metadata
            for video_info in results:
                if not video_info:
                    continue
                song_meta = repo.get_song_metadata_by_video_id(video_info["video_id"])
                if not song_meta:
                    song_meta = repo.create_song_metadata(video_info)
                song_metadata_ids_to_like.add(song_meta.id)

        user = repo.get_or_create_user(request.user_id)

        # 3. Persist user likes
        with track_latency("PostgreSQL:Write_Likes"):
            repo.persist_user_likes(user, song_metadata_ids_to_like)

        # 4. Fetch history for recommendations
        with track_latency("PostgreSQL:Fetch_History"):
            user_likes = repo.get_user_liked_songs_objects(user.user_id)
        
        user_liked_ids = {s.id for s in user_likes}

        # 5. Schedule background tasks
        background_tasks.add_task(update_redis_user_likes, user.user_id, user_liked_ids)

        # 6. Semantic search with exclusions
        previously_recommended: Set[str] = set()
        if redis_client:
            try:
                rec_key = f"prev_recs:{user.user_id}"
                cached_recs = redis_client.get(rec_key)
                if cached_recs:
                    previously_recommended = set(json.loads(cached_recs))
            except Exception as e:
                logger.warning("Failed to read prev_recs from Redis: %s", e)

        with track_latency("MLEngine:SemanticSearch"):
            ai_suggestions = suggestion_service.get_recommendations(
                user_history=[s.to_dict() for s in user_likes],
                repo=repo,
                top_n=10,
                excluded_video_ids=previously_recommended,
            )

        # 7. Fallback to genre-based search
        if not ai_suggestions:
            ai_suggestions = suggestion_service.get_suggestions(user, repo, genre=request.genre)

        response_suggestions = [
            SongSuggestion(
                title=s['title'],
                artist=s['artist'],
                youtube_video_id=s.get('video_id') or s.get('youtube_video_id')
            )
            for s in ai_suggestions
        ]

        # 8. Cache recommendations to avoid repeats
        if redis_client and ai_suggestions:
            try:
                new_rec_ids = [
                    s.get('video_id') or s.get('youtube_video_id')
                    for s in ai_suggestions
                    if s.get('video_id') or s.get('youtube_video_id')
                ]
                rec_key = f"prev_recs:{user.user_id}"
                merged = list(previously_recommended | set(new_rec_ids))
                redis_client.set(rec_key, json.dumps(merged), ex=60 * 60 * 24)
            except Exception as e:
                logger.warning("Failed to write prev_recs to Redis: %s", e)

        return {"suggestions": response_suggestions}

    except Exception as e:
        logger.exception(f"Error in post_suggestions: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error.")
