# dependencies.py

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from db import get_session
from repository import MusicRepository
from services import SuggestionService


def get_repo(db_session: Session = Depends(get_session)) -> MusicRepository:
    return MusicRepository(db=db_session)


def get_suggestion_service(request: Request) -> SuggestionService:
    return request.app.state.suggestion_service
