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
        None, description="An optional genre for fallback suggestions.", example="Rock", max_length=128)


class LikedSongResponse(BaseModel):
    video_id: str
    title: str
    artist: str
    created_at: str
