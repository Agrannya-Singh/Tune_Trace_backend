# utils/seed_trending_music.py
"""
Cron job script to fetch the top trending music videos from YouTube
and seed them into the TuneTrace database. This ensures the recommendation
model always has a fresh and diverse catalog of candidates.

Features:
- Fetches "mostPopular" videos in the Music category (id=10).
- Uses pagination to fetch up to a target number of songs (default 300).
- Extracts genre and tags immediately (acts as an auto-enrichment).
- Skips songs already present in the database.
- Uses exponential backoff for YouTube API reliability.
"""

import logging
import os
import time
import random
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer

import requests
from sqlalchemy.orm import Session

# Add project root to path so we can run this script directly
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db import SessionLocal, SongMetadata
from utils.enrichment import _extract_genre_from_tags, _exponential_backoff, MAX_RETRIES

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
logger = logging.getLogger("seed_trending")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_SONGS = 1000
ENRICHMENT_VERSION = "V3"
MODEL_NAME = "all-MiniLM-L6-v2"

SEARCH_QUERIES = [
    "top pop songs official video",
    "top hip hop rap hits",
    "top electronic dance edm music",
    "top rock alternative songs",
    "top r&b soul music",
    "trending latin reggaeton hits",
    "top country music songs",
    "indie alternative vibes",
    "top global viral chart",
    "best acoustic chill vibes"
]


def _search_youtube_videos(api_key: str, query: str, page_token: Optional[str] = None) -> tuple[List[str], Optional[str]]:
    """Search for music videos by query and return (video_ids, next_page_token)."""
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "id",
        "q": query,
        "type": "video",
        "videoCategoryId": "10", # Music
        "maxResults": 50,
        "key": api_key,
    }
    if page_token:
        params["pageToken"] = page_token

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                vids = [item["id"]["videoId"] for item in items if "videoId" in item.get("id", {})]
                return vids, data.get("nextPageToken")
            
            if response.status_code == 429 or response.status_code >= 500:
                sleep_time = _exponential_backoff(attempt)
                logger.warning(f"Search API {response.status_code}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue

            logger.error(f"Search API non-retryable error {response.status_code}: {response.text[:200]}")
            return [], None

        except requests.exceptions.RequestException as e:
            sleep_time = _exponential_backoff(attempt)
            logger.warning(f"Request error: {str(e)[:100]}. Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    return [], None


def _fetch_video_details(api_key: str, video_ids: List[str]) -> List[Dict]:
    """Fetch full snippets for a list of video IDs (comma separated string max 50)."""
    if not video_ids:
        return []
        
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "id": ",".join(video_ids[:50]),
        "key": api_key,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get("items", [])
                
            if response.status_code == 429 or response.status_code >= 500:
                sleep_time = _exponential_backoff(attempt)
                logger.warning(f"Videos API {response.status_code}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue

            logger.error(f"Videos API error {response.status_code}")
            return []

        except requests.exceptions.RequestException as e:
            sleep_time = _exponential_backoff(attempt)
            time.sleep(sleep_time)

    return []


def _build_text_context(title: str, artist: str, genre: str, tags_str: str) -> str:
    """Build semantic text context for embedding generation.

    Mirrors ml_engine.MLEngine.build_text_context format.
    """
    parts = [title]
    if artist:
        parts.append(f"Artist: {artist}")
    if genre:
        parts.append(f"Genre: {genre}")
    if tags_str:
        parts.append(f"Tags: {tags_str}")
    return ". ".join(parts)


def run_seeder(api_key: Optional[str] = None):
    if api_key is None:
        api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        logger.error("Enrichment skipped: YOUTUBE_API_KEY not configured.")
        return

    # Load SentenceTransformer model once for the entire run
    logger.info("Loading SentenceTransformer model '%s'...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded.")

    db = SessionLocal()
    
    try:
        existing_ids = {row[0] for row in db.query(SongMetadata.video_id).all()}
        logger.info(f"Database currently holds {len(existing_ids)} songs.")
        
        items_added = 0
        new_songs_to_insert = []
        
        for query in SEARCH_QUERIES:
            if items_added >= TARGET_SONGS:
                break
                
            logger.info(f"Executing search query: '{query}'")
            next_page_token = None
            
            # Fetch up to 5 pages per query
            for page in range(5):
                if items_added >= TARGET_SONGS:
                    break
                    
                video_ids, next_page_token = _search_youtube_videos(api_key, query, next_page_token)
                
                # Filter out ones we already have BEFORE fetching massive details payload
                new_vids = [vid for vid in video_ids if vid not in existing_ids]
                if not new_vids:
                    logger.info("  -> All videos from this query page are already in the DB.")
                else:
                    # Fetch details
                    logger.info(f"  -> Fetching snippets for {len(new_vids)} new videos (Page {page+1})...")
                    items = _fetch_video_details(api_key, new_vids)
                    
                    for item in items:
                        snippet = item.get("snippet", {})
                        tags_list = snippet.get("tags", [])
                        
                        # CRITICAL: We only want songs that actually have tags for ML!
                        if not tags_list:
                            continue
                            
                        video_id = item.get("id")
                        if not video_id or video_id in existing_ids:
                            continue
                            
                        title = snippet.get("title", "")
                        artist = snippet.get("channelTitle", "")
                        
                        # Auto-enrich genre and tags
                        genre = _extract_genre_from_tags(tags_list)
                        tags_str = ", ".join(tags_list[:20])
                            
                        new_song = SongMetadata(
                            video_id=video_id,
                            title=title,
                            artist=artist,
                            genre=genre,
                            tags=tags_str,
                            enriched=ENRICHMENT_VERSION
                        )
                        # Store text context for batch vectorization
                        new_song._text_context = _build_text_context(
                            title, artist, genre, tags_str
                        )
                        new_songs_to_insert.append(new_song)
                        existing_ids.add(video_id)
                        items_added += 1
                        
                        if items_added >= TARGET_SONGS:
                            logger.info(f"Reached target of {TARGET_SONGS} songs.")
                            break
                            
                if not next_page_token:
                    break
                    
        if new_songs_to_insert:
            # --- Batch vectorize all new songs at once ---
            logger.info(
                "Vectorizing %d new songs with SentenceTransformer...",
                len(new_songs_to_insert),
            )
            text_contexts = [s._text_context for s in new_songs_to_insert]
            embeddings = model.encode(
                text_contexts,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=64,
            )

            # Attach embeddings to ORM objects before bulk save
            # We need to use raw SQL since SQLAlchemy doesn't know about vector type
            logger.info(f"Saving {len(new_songs_to_insert)} new trending songs to database...")
            db.bulk_save_objects(new_songs_to_insert)
            db.flush()  # flush to get IDs assigned

            # Now update embeddings via raw SQL
            from sqlalchemy import text as sa_text
            for song, embedding in zip(new_songs_to_insert, embeddings):
                vector_literal = (
                    "[" + ",".join(str(float(x)) for x in embedding) + "]"
                )
                db.execute(
                    sa_text(
                        "UPDATE song_metadata SET embedding = :vec WHERE video_id = :vid"
                    ),
                    {"vec": vector_literal, "vid": song.video_id},
                )

            db.commit()
            logger.info("Database commit successful (with V3 embeddings).")
        else:
            logger.info("No new tagged songs found to add.")

    except Exception as e:
        logger.exception(f"Seeder failed with unexpected error: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Starting Trending Music Seeder...")
    run_seeder()
