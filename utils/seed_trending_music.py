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
TARGET_SONGS = 300
MAX_PAGES = 10  # Fallback safety limit (50 items per page max)
ENRICHMENT_VERSION = "V2"


def _fetch_trending_page(api_key: str, page_token: Optional[str] = None) -> Optional[Dict]:
    """Fetch one page of trending music videos from YouTube."""
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "chart": "mostPopular",
        "videoCategoryId": "10", # Music
        "maxResults": 50,
        "regionCode": "US",      # Adjust if you want global or specific regions
        "key": api_key,
    }
    if page_token:
        params["pageToken"] = page_token

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
                
            if response.status_code == 429 or response.status_code >= 500:
                sleep_time = _exponential_backoff(attempt)
                logger.warning(f"YouTube API {response.status_code}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue

            logger.error(f"YouTube API non-retryable error {response.status_code}: {response.text[:200]}")
            return None

        except requests.exceptions.RequestException as e:
            sleep_time = _exponential_backoff(attempt)
            logger.warning(f"Request error: {str(e)[:100]}. Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    logger.error("Max retries exhausted fetching trending page.")
    return None


def run_seeder(api_key: Optional[str] = None):
    if api_key is None:
        api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        logger.error("Enrichment skipped: YOUTUBE_API_KEY not configured.")
        return

    db = SessionLocal()
    
    try:
        # 1. Fetch current video IDs to avoid duplicates
        existing_ids = {row[0] for row in db.query(SongMetadata.video_id).all()}
        logger.info(f"Database currently holds {len(existing_ids)} songs.")
        
        # 2. Fetch trending from YouTube
        items_fetched = 0
        new_songs_to_insert = []
        page_token = None
        
        for page_num in range(MAX_PAGES):
            logger.info(f"Fetching trending page {page_num + 1}...")
            data = _fetch_trending_page(api_key, page_token)
            if not data:
                break
                
            items = data.get("items", [])
            for item in items:
                video_id = item.get("id")
                if not video_id or video_id in existing_ids:
                    continue
                    
                snippet = item.get("snippet", {})
                title = snippet.get("title", "")
                artist = snippet.get("channelTitle", "")
                tags_list = snippet.get("tags", [])
                
                # Auto-enrich genre and tags
                genre = None
                tags_str = None
                if tags_list:
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
                new_songs_to_insert.append(new_song)
                existing_ids.add(video_id)
                items_fetched += 1
                
                if items_fetched >= TARGET_SONGS:
                    break
            
            if items_fetched >= TARGET_SONGS:
                logger.info(f"Reached target of {TARGET_SONGS} songs.")
                break
                
            page_token = data.get("nextPageToken")
            if not page_token:
                logger.info("No more pages available from YouTube API.")
                break

        # 3. Save to database
        if new_songs_to_insert:
            logger.info(f"Saving {len(new_songs_to_insert)} new trending songs to database...")
            db.bulk_save_objects(new_songs_to_insert)
            db.commit()
            logger.info("Database commit successful.")
        else:
            logger.info("No new songs to add (all trending songs already exist in DB).")

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
