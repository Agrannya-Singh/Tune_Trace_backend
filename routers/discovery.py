# routers/discovery.py
import logging
from fastapi import APIRouter, Depends, status

from repository import MusicRepository
from services import SuggestionService
from api_models import DiscoverRequest, DiscoverResponse, DiscoverSong
from dependencies import get_repo, get_suggestion_service, get_chat_service
from services import SuggestionService, ChatService
from utils.metrics import track_latency

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Discovery"])

@router.post("/discover", response_model=DiscoverResponse)
async def discover_music(
    request: DiscoverRequest,
    repo: MusicRepository = Depends(get_repo),
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Intelligent discovery endpoint using RAG (Retrieval-Augmented Generation).
    Retrieves semantic matches and generates a conversational response grounded in user taste.
    """
    
    with track_latency("ChatService:Discover"):
        # Convert history from Pydantic models to dicts
        history_dicts = [h.dict() for h in request.history]
        
        result = await chat_service.get_chat_response(
            user_id=request.user_id,
            message=request.query,
            history=history_dicts,
            repo=repo,
            top_n=request.limit
        )

    return DiscoverResponse(
        query=request.query,
        ai_response=result["response"],
        results=[
            DiscoverSong(
                title=s["title"],
                artist=s["artist"],
                youtube_video_id=s["video_id"],
                genre=s.get("genre"),
                score=s["score"],
            )
            for s in result["context_songs"]
        ],
    )
