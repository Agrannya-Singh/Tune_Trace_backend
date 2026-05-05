# api_models.py

from typing import List, Optional

from pydantic import BaseModel, Field


class SongSuggestion(BaseModel):
    title: str
    artist: str
    youtube_video_id: str


class SuggestionResponse(BaseModel):
    suggestions: List[SongSuggestion]


class LikedSongsRequest(BaseModel):
    user_id: str = Field(...,
                         description="User email or unique identifier from OAuth.", max_length=255)
    songs: List[str] = Field(..., min_length=1, max_length=50,
                             description="A list of song titles the user has liked (max 50).")
    genre: Optional[str] = Field(
        None, description="An optional genre for fallback suggestions.", json_schema_extra={"example": "Rock"}, max_length=128)


class LikedSongResponse(BaseModel):
    video_id: str
    title: str
    artist: str
    created_at: str


class DiscoverSong(BaseModel):
    title: str
    artist: str
    youtube_video_id: str
    genre: Optional[str] = None
    score: float = Field(..., description="Semantic similarity score (0-1).")


class DiscoverRequest(BaseModel):
    user_id: str = Field(..., description="User ID for personalized context.")
    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Free-text search query — a mood, song name, description, or vibe.",
        json_schema_extra={"example": "chill lo-fi vibes for studying"},
    )
    limit: int = Field(
        10,
        ge=1,
        le=30,
        description="Number of results to return (max 30).",
    )
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Optional conversation history for multi-turn discovery."
    )


class ChatMessage(BaseModel):
    role: str # 'user' or 'assistant'
    content: str


class DiscoverResponse(BaseModel):
    query: str
    results: List[DiscoverSong]
    ai_response: Optional[str] = Field(
        None, 
        description="Conversational explanation or response from Gemini RAG pipeline."
    )
