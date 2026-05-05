# services.py

import re
import logging
from typing import Dict, List, Optional, Set

import httpx
import requests

from db import User
from repository import MusicRepository
from engine import ml_engine
from config import GEMINI_API_KEY

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


class SuggestionService:
    def __init__(self, api_key: Optional[str], redis_client=None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=8.0)
        self.redis_client = redis_client

    async def close(self):
        await self.client.aclose()

    async def _search_youtube_for_song_async(self, song_name: str) -> Optional[Dict]:
        """
        Async version of the search. Non-blocking!
        """
        if not self.api_key:
            return None

        clean_query = re.sub(r"[^\w\s\-']", "", song_name).lower().strip()[:200]
        if not clean_query:
            return None

        # 1. Check Redis Cache
        if self.redis_client:
            try:
                cache_key = f"yt_search:{clean_query}"
                cached = self.redis_client.get(cache_key)
                if cached:
                    import json
                    logger.info(f"Redis Cache HIT for search: {clean_query}")
                    return json.loads(cached)
            except Exception as e:
                logger.error(f"Redis Read Error: {e}")

        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": clean_query,
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
                # Cache empty result too (to avoid repeated failed searches)
                if self.redis_client:
                     self.redis_client.setex(f"yt_search:{clean_query}", 3600 * 24, "null") 
                return None
            
            snippet = items[0]["snippet"]
            result = {
                "video_id": items[0]["id"]["videoId"],
                "title": snippet["title"],
                "artist": snippet["channelTitle"]
            }

            # 2. Write to Redis Cache
            if self.redis_client:
                try:
                    import json
                    self.redis_client.set(f"yt_search:{clean_query}", json.dumps(result), ex=3600 * 24 * 7) # 1 week cache
                except Exception as e:
                    logger.error(f"Redis Write Error: {e}")

            return result
        except httpx.RequestError as e:
            logger.error(f"Async Search Error for query '{clean_query}': {e}")
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
                    "video_id": item['id']['videoId'],  # Unified key name
                    "score": 1.0
                }
                for item in items if 'videoId' in item.get('id', {})
            ]
        except requests.RequestException as e:
            logger.error(f"Fallback YouTube API error (sanitized): Status {getattr(e.response, 'status_code', 'N/A')}")
            return []

    def get_recommendations(
        self, 
        user_history: List[Dict], 
        repo: MusicRepository, 
        top_n: int = 10, 
        excluded_video_ids: Optional[Set[str]] = None
    ) -> List[Dict]:
        """
        Coordinates between MLEngine and MusicRepository to generate recommendations.
        """
        if not user_history:
            return []

        # 1. Compute User Profile Vector
        profile_vector = ml_engine.compute_user_profile_vector(user_history)
        vector_literal = ml_engine.vector_to_literal(profile_vector)

        # 2. Build exclusion list
        user_video_ids = {s["video_id"] for s in user_history}
        all_excluded = user_video_ids | (excluded_video_ids or set())

        # 3. Fetch candidates from DB
        fetch_limit = top_n * 5
        rows = repo.get_semantic_recommendations(vector_literal, fetch_limit, all_excluded)

        if not rows:
            return []

        scored = [
            {
                "video_id": row.video_id,
                "title": row.title,
                "artist": row.artist,
                "genre": row.genre,
                "tags": row.tags,
                "enriched": row.enriched,
                "score": round(float(row.similarity), 4),
            }
            for row in rows
        ]

        # 4. Apply Diversity Logic
        final = ml_engine.apply_diversity(scored, top_n)
        
        logger.info(
            "SuggestionService: Generated %d recommendations (top_n=%d).",
            len(final), top_n
        )
        return final

    def search_semantic(self, query: str, repo: MusicRepository, top_n: int = 10) -> List[Dict]:
        """
        Coordinates between MLEngine and MusicRepository for free-text search.
        """
        if not query or not query.strip():
            return []

        # 1. Encode query
        query_vector = ml_engine.encode_single(query.strip())
        vector_literal = ml_engine.vector_to_literal(query_vector)

        # 2. Fetch from DB
        rows = repo.semantic_search(vector_literal, top_n)

        if not rows:
            return []

        results = [
            {
                "video_id": row.video_id,
                "title": row.title,
                "artist": row.artist,
                "genre": row.genre,
                "tags": row.tags,
                "score": round(float(row.similarity), 4),
            }
            for row in rows
        ]
        return results

    def get_suggestions(self, user: User, repo: MusicRepository, genre: Optional[str] = None, num_suggestions: int = 10) -> List[Dict]:
        """
        Return fallback suggestions when the semantic ML engine yields no results.
        Uses YouTube Search API to find trending songs in a genre.
        """
        return self._get_fallback_suggestions(genre=genre, num_suggestions=num_suggestions)


class ChatService:
    """
    Complete RAG (Retrieval-Augmented Generation) Service.
    Retrieves: 
      1. User Taste Profile (from PostgreSQL Likes)
      2. Semantic Knowledge (from Vector Search)
    Generates:
      Engaging conversational responses via Gemini 3 Flash.
    """
    def __init__(self, suggestion_service: SuggestionService):
        self.suggestion_service = suggestion_service
        if GEMINI_API_KEY:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                google_api_key=GEMINI_API_KEY,
                temperature=0.7
            )
        else:
            self.llm = None

    async def get_chat_response(self, user_id: Optional[str], message: str, history: List[Dict], repo: MusicRepository, top_n: int = 5) -> Dict:
        # 1. RAG Part A: Retrieve User Taste Context
        user_taste_context = "User has no recorded likes or profile yet."
        profile_recommendations = []
        
        if user_id:
            user = repo.get_user(user_id)
            if user:
                likes = repo.get_user_liked_songs_objects(user.user_id)
                if likes:
                    user_taste_context = "User's Recent Liked Songs:\n"
                    for s in likes[:10]:
                        user_taste_context += f"- {s.title} by {s.artist}\n"
                    
                    # Compute Semantic Profile Matches
                    # This uses the user's vector profile to find songs they'll like regardless of the query
                    profile_recommendations = self.suggestion_service.get_recommendations(
                        user=user,
                        repo=repo,
                        num_recommendations=5
                    )

        # 2. RAG Part B: Retrieve Query-Specific Context
        # This finds songs that semantically match the user's CURRENT chat message.
        context_songs = self.suggestion_service.search_semantic(message, repo, top_n=top_n)
        
        # Build Semantic Context string for the LLM
        semantic_context = ""
        if context_songs:
            semantic_context += "Direct matches for your query:\n"
            for s in context_songs:
                genre_str = f" [Genre: {s.get('genre')}]" if s.get('genre') else ""
                semantic_context += f"- {s['title']} by {s['artist']}{genre_str}\n"
        
        if profile_recommendations:
            semantic_context += "\nSongs matching your general music profile:\n"
            for s in profile_recommendations:
                semantic_context += f"- {s['title']} by {s['artist']} (Recommended based on taste)\n"

        if not semantic_context:
            semantic_context = "No specific semantic matches found in our library."

        # 3. Handle Missing LLM (Graceful Degradation)
        if not self.llm:
            return {
                "response": "AI generation is disabled (GEMINI_API_KEY missing), but here are some matches I found! 🔍",
                "context_songs": context_songs + profile_recommendations
            }

        # 4. LangChain Template: The "Generation" logic
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are 'TuneTrace AI', a helpful and chatty music discovery assistant. "
                "You ground your answers in the user's taste and our music library. "
                "Use the 'User Taste Context' and 'Semantic Context' to understand their preferences and suggest real songs. "
                "If the user asks for recommendations, prioritize the 'Songs matching your general music profile'. "
                "Always mention specific songs from the provided context. "
                "Be enthusiastic, use emojis, and keep the conversation musical! 🎵"
            )),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content=(
                f"### CONTEXTUAL DATA ###\n"
                f"{user_taste_context}\n\n"
                f"{semantic_context}\n\n"
                f"### USER QUERY ###\n"
                f"{message}"
            ))
        ])

        # Convert simple history dicts to LangChain Message objects
        langchain_history = []
        for h in history:
            if h["role"] == "user":
                langchain_history.append(HumanMessage(content=h["content"]))
            else:
                langchain_history.append(AIMessage(content=h["content"]))

        # Chain Construction: Prompt -> Gemini
        chain = prompt | self.llm
        
        try:
            response = await chain.ainvoke({"history": langchain_history})
            return {
                "response": response.content,
                "context_songs": context_songs + profile_recommendations
            }
        except Exception as e:
            logger.error(f"Gemini RAG Error: {e}")
            return {
                "response": "My circuits are a bit fuzzy right now. Can we try that again? 🎸",
                "context_songs": context_songs + profile_recommendations
            }
