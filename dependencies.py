# dependencies.py

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from db import get_session
from repository import MusicRepository
from services import SuggestionService, ChatService


def get_repo(db_session: Session = Depends(get_session)) -> MusicRepository:
    return MusicRepository(db=db_session)


def get_suggestion_service(request: Request) -> SuggestionService:
    return request.app.state.suggestion_service


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service
