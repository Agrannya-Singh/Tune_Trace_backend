# utils/enrichment.py
"""
Startup enrichment script that fetches genre and tags from the YouTube Data API
for songs that haven't been enriched yet.

Features:
- Runs as a background thread on server startup (non-blocking).
- Processes songs in batches (YouTube API supports up to 50 IDs per call).
- Uses exponential backoff with jitter for rate-limit and transient errors.
- Marks each enriched song with `enriched = "V2"`.
- Gracefully handles all YouTube API failures without crashing the server.

Enrichment Versions:
- NULL : Raw data (title + artist only, from YouTube Search API).
- V2   : Genre + tags enriched from YouTube Videos API (categoryId → genre, tags).
"""

import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests
from sqlalchemy.orm import Session

from db import SessionLocal, SongMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known Music Genres
# Used to extract the actual genre from YouTube video tags.
# YouTube's categoryId only gives "Music" vs "Entertainment" — useless for
# music recommendations. Instead we scan the video's tags for known genres.
# Ordered roughly by specificity (sub-genres before parent genres) so
# "indie pop" matches before "pop".
# ---------------------------------------------------------------------------
KNOWN_MUSIC_GENRES: List[str] = [
    # Electronic sub-genres
    "lo-fi", "lofi", "synthwave", "vaporwave", "drum and bass", "dubstep",
    "house music", "deep house", "tech house", "progressive house",
    "electro", "trance", "techno", "edm", "electronic",
    # Hip-Hop / Rap sub-genres
    "trap", "drill", "boom bap", "conscious rap", "hip hop", "hip-hop",
    "rap", "hiphop",
    # Rock sub-genres
    "punk rock", "punk", "grunge", "hard rock", "classic rock",
    "progressive rock", "psychedelic rock", "alternative rock",
    "indie rock", "metal", "rock",
    # Pop sub-genres
    "k-pop", "kpop", "j-pop", "jpop", "synth pop", "indie pop",
    "dream pop", "electropop", "pop",
    # R&B / Soul
    "neo soul", "r&b", "rnb", "soul", "funk", "disco",
    # Latin
    "reggaeton", "bachata", "salsa", "cumbia", "latin",
    # Other
    "jazz", "blues", "gospel", "reggae", "dancehall", "ska",
    "country", "folk", "acoustic", "classical", "opera",
    "ambient", "chillout", "downtempo",
    "afrobeats", "afropop", "amapiano",
    "phonk", "hyperpop",
    "singer-songwriter", "instrumental",
    "bollywood", "desi",
]

# Pre-compute lowercase set for fast lookup, preserving display-case mapping
_GENRE_LOOKUP: Dict[str, str] = {}
for _g in KNOWN_MUSIC_GENRES:
    _GENRE_LOOKUP[_g.lower()] = _g


def _extract_genre_from_tags(tags: List[str]) -> Optional[str]:
    """Scan a video's tags and return the first matching music genre.

    Tags are checked in the order defined by KNOWN_MUSIC_GENRES (most
    specific sub-genres first), so "indie pop" matches before "pop".
    """
    # Normalize all tags to lowercase for comparison
    lower_tags = [t.lower().strip() for t in tags]

    # First pass: exact tag match (e.g., tag is literally "hip hop")
    for genre_lower, genre_display in _GENRE_LOOKUP.items():
        if genre_lower in lower_tags:
            return genre_display

    # Second pass: substring match (e.g., tag "pop music" contains "pop")
    joined = " ".join(lower_tags)
    for genre_lower, genre_display in _GENRE_LOOKUP.items():
        if genre_lower in joined:
            return genre_display

    return None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENRICHMENT_VERSION = "V2"
BATCH_SIZE = 50  # YouTube API max per request
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 64.0


def _exponential_backoff(attempt: int) -> float:
    """Calculate sleep duration with exponential backoff + jitter."""
    base = min(INITIAL_BACKOFF_SECONDS * (2 ** attempt), MAX_BACKOFF_SECONDS)
    # Full jitter: uniform random between 0 and base
    return random.uniform(0, base)


def _fetch_video_details(
    video_ids: List[str], api_key: str
) -> Optional[List[dict]]:
    """Fetch video details from YouTube Data API with exponential backoff.

    Args:
        video_ids: List of YouTube video IDs (max 50).
        api_key: YouTube Data API key.

    Returns:
        List of video item dicts, or None if all retries exhausted.
    """
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "id": ",".join(video_ids),
        "key": api_key,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=10)

            # --- Success ---
            if response.status_code == 200:
                data = response.json()
                return data.get("items", [])

            # --- Rate limited (429) or server error (5xx) → retry ---
            if response.status_code == 429 or response.status_code >= 500:
                sleep_time = _exponential_backoff(attempt)
                logger.warning(
                    "YouTube API returned %d (attempt %d/%d). "
                    "Retrying in %.1fs...",
                    response.status_code,
                    attempt + 1,
                    MAX_RETRIES,
                    sleep_time,
                )
                time.sleep(sleep_time)
                continue

            # --- Client error (4xx, not 429) → don't retry ---
            logger.error(
                "YouTube API returned non-retryable error %d: %s",
                response.status_code,
                response.text[:200],
            )
            return None

        except requests.exceptions.Timeout:
            sleep_time = _exponential_backoff(attempt)
            logger.warning(
                "YouTube API request timed out (attempt %d/%d). "
                "Retrying in %.1fs...",
                attempt + 1,
                MAX_RETRIES,
                sleep_time,
            )
            time.sleep(sleep_time)

        except requests.exceptions.ConnectionError as e:
            sleep_time = _exponential_backoff(attempt)
            logger.warning(
                "YouTube API connection error (attempt %d/%d): %s. "
                "Retrying in %.1fs...",
                attempt + 1,
                MAX_RETRIES,
                str(e)[:100],
                sleep_time,
            )
            time.sleep(sleep_time)

        except requests.exceptions.RequestException as e:
            logger.error(
                "YouTube API unexpected request error: %s", str(e)[:200]
            )
            return None

    logger.error(
        "YouTube API: all %d retries exhausted for batch of %d videos.",
        MAX_RETRIES,
        len(video_ids),
    )
    return None


def _enrich_batch(
    db: Session, songs: List[SongMetadata], api_key: str
) -> int:
    """Enrich a batch of songs with genre and tags from YouTube.

    Returns:
        Number of songs successfully enriched in this batch.
    """
    video_id_to_song = {song.video_id: song for song in songs}
    video_ids = list(video_id_to_song.keys())

    items = _fetch_video_details(video_ids, api_key)
    if items is None:
        logger.warning(
            "Skipping batch of %d songs due to YouTube API failure.",
            len(songs),
        )
        return 0

    enriched_count = 0
    for item in items:
        video_id = item.get("id")
        song = video_id_to_song.get(video_id)
        if not song:
            continue

        snippet = item.get("snippet", {})

        tags_list = snippet.get("tags", [])
        
        # --- Genre from tags (instead of generic categoryId) ---
        if tags_list:
            genre = _extract_genre_from_tags(tags_list)
            if genre:
                song.genre = genre

            # --- Tags (save raw tags) ---
            # Store as comma-separated, limit to first 20 tags to avoid bloat
            song.tags = ", ".join(tags_list[:20])

        # --- Mark as enriched ---
        song.enriched = ENRICHMENT_VERSION
        enriched_count += 1

    try:
        db.commit()
    except Exception as e:
        logger.error("Failed to commit enrichment batch: %s", e)
        db.rollback()
        return 0

    return enriched_count


def run_enrichment(api_key: Optional[str] = None) -> None:
    """Main enrichment entry point. Processes all un-enriched songs.

    Should be called from a background thread on server startup.
    Uses its own DB session (not the request session).
    """
    if api_key is None:
        api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        logger.warning(
            "Enrichment skipped: YOUTUBE_API_KEY not configured."
        )
        return

    db = SessionLocal()
    try:
        # Count total un-enriched songs
        total = (
            db.query(SongMetadata)
            .filter(SongMetadata.enriched.is_(None))
            .count()
        )

        if total == 0:
            logger.info("Enrichment: All songs are already enriched. Nothing to do.")
            return

        logger.info(
            "Enrichment: Starting %s enrichment for %d songs...",
            ENRICHMENT_VERSION,
            total,
        )

        processed = 0
        enriched = 0
        offset = 0

        while True:
            # Fetch next batch of un-enriched songs
            batch = (
                db.query(SongMetadata)
                .filter(SongMetadata.enriched.is_(None))
                .order_by(SongMetadata.id)
                .limit(BATCH_SIZE)
                .offset(0)  # Always 0: enriched songs drop out of the filter
                .all()
            )

            if not batch:
                break

            batch_enriched = _enrich_batch(db, batch, api_key)
            processed += len(batch)
            enriched += batch_enriched

            logger.info(
                "Enrichment progress: %d/%d processed, %d enriched.",
                processed,
                total,
                enriched,
            )

            # Small delay between batches to be respectful to the API
            time.sleep(0.5)

        logger.info(
            "Enrichment complete: %d/%d songs enriched to %s.",
            enriched,
            total,
            ENRICHMENT_VERSION,
        )

    except Exception as e:
        logger.exception("Enrichment failed with unexpected error: %s", e)
    finally:
        db.close()
