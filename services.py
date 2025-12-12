# services.py

import re
import logging
from typing import Dict, List, Optional

import httpx
import requests

from db import User
from repository import MusicRepository

logger = logging.getLogger(__name__)


class SuggestionService:
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=8.0)

    async def close(self):
        await self.client.aclose()

    async def _search_youtube_for_song_async(self, song_name: str) -> Optional[Dict]:
        """
        Async version of the search. Non-blocking!
        """
        if not self.api_key:
            return None

        query = re.sub(r"[^\w\s\-']", "", song_name).lower().strip()[:200]
        if not query:
            return None

        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "videoCategoryId": "10",
            "maxResults": 1,
            "key": self.api_key
        }

        try:
            resp = await self.client.get(url, params=params)
            resp.raise_for_status()
            items = resp.json().get("items", [])

            if not items:
                return None
            snippet = items[0]["snippet"]
            return {
                "video_id": items[0]["id"]["videoId"],
                "title": snippet["title"],
                "artist": snippet["channelTitle"]
            }
        except httpx.RequestError as e:
            logger.error(f"Async Search Error for query '{query}': {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during async search: {e}")
            return None

    def _get_fallback_suggestions(self, genre: Optional[str] = None, num_suggestions: int = 10) -> List[Dict]:
        logger.info(f"Executing fallback search for genre: {genre or 'Global Hits'}")
        if not self.api_key:
            return []

        search_term = f"Top {genre} songs" if genre else "Top Global Hits"
        search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={search_term}&type=video&videoCategoryId=10&maxResults={num_suggestions}&key={self.api_key}"

        try:
            response = requests.get(search_url)
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])

            return [
                {
                    "title": item['snippet']['title'],
                    "artist": item['snippet']['channelTitle'],
                    "youtube_video_id": item['id']['videoId'],
                    "score": 1.0
                }
                for item in items if 'videoId' in item.get('id', {})
            ]
        except requests.RequestException as e:
            logger.error(f"Fallback YouTube API error (sanitized): Status {getattr(e.response, 'status_code', 'N/A')}")
            return []

    def get_suggestions(self, user: User, repo: MusicRepository, genre: Optional[str] = None, num_suggestions: int = 10) -> List[Dict]:
        collaborative_raw = repo.get_collaborative_suggestions(
            user, limit=num_suggestions)

        if not collaborative_raw:
            logger.warning(
                f"No personalized suggestions for user {user.user_id}. Triggering fallback.")
            return self._get_fallback_suggestions(genre=genre, num_suggestions=num_suggestions)

        return [
            {"title": song.title, "artist": song.artist,
                "video_id": song.video_id}
            for song in collaborative_raw
        ]
