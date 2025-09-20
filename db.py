# db.py
"""
Database models and session management for the music suggestion service.
This version is corrected to prioritize a single database connection, defaulting
to PostgreSQL when the environment variable is present.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Set

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    Session,
)

# ==============================================================================
# --- Database Configuration ---
# ==============================================================================
# This simplified logic prioritizes PostgreSQL for production environments.

# 1. Prioritize PostgreSQL using its environment variable.
DATABASE_URL = os.getenv("POSTGRES_DATABASE_URL")
connect_args = {}

# 2. Fallback to a local SQLite database ONLY if PostgreSQL is not configured.
if not DATABASE_URL:
    print("INFO: POSTGRES_DATABASE_URL not found, falling back to local SQLite database.")
    DATABASE_URL = "sqlite:///./app.db"
    connect_args = {"check_same_thread": False}

# 3. Create a single, definitive engine for the application.
engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args=connect_args
)

# 4. Create a single, definitive sessionmaker.
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, future=True
)


# ==============================================================================
# --- ORM Model Definitions ---
# ==============================================================================

class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class User(Base):
    """Represents a user of the application."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        index=True,
        comment="The unique external identifier for the user.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    likes: Mapped[List[UserLikedSong]] = relationship(
        "UserLikedSong", back_populates="user", cascade="all, delete-orphan"
    )

    def get_liked_song_ids(self) -> Set[int]:
        """Returns a set of internal DB IDs of songs liked by the user."""
        return {like.song_id for like in self.likes}


class SongMetadata(Base):
    """Stores definitive metadata for a song, identified by its YouTube video ID."""
    __tablename__ = "song_metadata"
    __table_args__ = (
        UniqueConstraint("video_id", name="uq_video_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(
        String(64), index=True, comment="YouTube video ID."
    )
    title: Mapped[str] = mapped_column(String(512))
    artist: Mapped[str] = mapped_column(
        String(256), comment="Typically the YouTube channel title."
    )
    genre: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, index=True
    )
    tags: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="Comma-separated tags from YouTube."
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class UserLikedSong(Base):
    """Association object linking a User to a SongMetadata they have liked."""
    __tablename__ = "user_liked_songs"
    __table_args__ = (
        UniqueConstraint("user_id", "song_id", name="uq_user_song"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    song_id: Mapped[int] = mapped_column(
        ForeignKey("song_metadata.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    user: Mapped[User] = relationship("User", back_populates="likes")
    song: Mapped[SongMetadata] = relationship("SongMetadata")


# ==============================================================================
# --- Session Management ---
# ==============================================================================

def get_session() -> Iterator[Session]:
    """Provides a single database session for a request as a FastAPI dependency."""
    db: Optional[Session] = None
    try:
        db = SessionLocal()
        yield db
    finally:
        if db:
            db.close()
