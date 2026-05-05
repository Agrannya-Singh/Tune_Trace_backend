# routers/discovery.py
import logging
from fastapi import APIRouter, Depends, status

from repository import MusicRepository
from services import SuggestionService
from api_models import DiscoverRequest, DiscoverResponse, DiscoverSong
from dependencies import get_repo, get_suggestion_service
from utils.metrics import track_latency

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Discovery"])

@router.post("/discover", response_model=DiscoverResponse)
async def discover_music(
    request: DiscoverRequest,
    repo: MusicRepository = Depends(get_repo),
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
):
    """Semantic discovery search using pgvector."""
    
    with track_latency("SuggestionService:Discover"):
        results = suggestion_service.search_semantic(
            query=request.query,
            repo=repo,
            top_n=request.limit,
        )

    if not results:
        logger.info("Discover returned 0 results for query: %s", request.query[:80])

    return DiscoverResponse(
        query=request.query,
        results=[
            DiscoverSong(
                title=s["title"],
                artist=s["artist"],
                youtube_video_id=s["video_id"],
                genre=s.get("genre"),
                score=s["score"],
            )
            for s in results
        ],
    )
