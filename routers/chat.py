# routers/chat.py
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from api_models import ChatRequest, ChatResponse, SongSuggestion
from dependencies import get_repo, get_chat_service
from services import ChatService
from repository import MusicRepository

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])

@router.post("/chat", response_model=ChatResponse)
async def post_chat(
    request: ChatRequest,
    repo: MusicRepository = Depends(get_repo),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    RAG-powered chat endpoint. 
    Retrieves context from the semantic search and generates a response using Gemini.
    """
    try:
        # Convert history from Pydantic models to dicts
        history_dicts = [h.dict() for h in request.history]
        
        result = await chat_service.get_chat_response(
            user_id=request.user_id,
            message=request.message,
            history=history_dicts,
            repo=repo
        )
        
        # Prepare context songs for response
        context_songs = [
            SongSuggestion(
                title=s['title'],
                artist=s['artist'],
                youtube_video_id=s.get('video_id') or s.get('youtube_video_id')
            )
            for s in result["context_songs"]
        ]
        
        return ChatResponse(
            response=result["response"],
            context_songs=context_songs
        )
        
    except Exception as e:
        logger.exception(f"Error in post_chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat generation failed."
        )
