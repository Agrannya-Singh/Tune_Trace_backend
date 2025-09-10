# db.py
"""
Database models and session management for the music suggestion service.

This module defines the SQLAlchemy ORM models for Users, Songs, and their relationships.
It supports both PostgreSQL and SQLite for flexibility in development and production,
with a read/write session management system.
"""

from __future__ import annotations
import os
from datetime import datetime
from typing import Optional, Dict, List, Iterator, Set
from sqlalchemy import (
    create_engine, String, Integer, DateTime, ForeignKey, Text, 
    UniqueConstraint, func
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session
)
from contextlib import contextmanager

# --- Database Configuration ---
# Establishes connections to PostgreSQL and/or SQLite based on environment variables.
LEGACY_DB_URL = os.getenv("DATABASE_URL")
SQLITE_DATABASE_URL = os.getenv("SQLITE_DATABASE_URL", "sqlite:///app.db")
POSTGRES_DATABASE_URL = os.getenv("POSTGRES_DATABASE_URL") or (
    LEGACY_DB_URL if (LEGACY_DB_URL or "").startswith("postgres") else None
)
DB_READ_PREFERENCE = os.getenv("DB_READ_PREFERENCE", "postgres").lower()

engines: Dict[str, object] = {}
sessions: Dict[str, sessionmaker] = {}

if SQLITE_DATABASE_URL:
    sqlite_connect_args = {"check_same_thread": False} if SQLITE_DATABASE_URL.startswith("sqlite") else {}
    engines["sqlite"] = create_engine(SQLITE_DATABASE_URL, echo=False, future=True, connect_args=sqlite_connect_args)
    sessions["sqlite"] = sessionmaker(bind=engines["sqlite"], autoflush=False, autocommit=False, future=True)

if POSTGRES_DATABASE_URL:
    engines["postgres"] = create_engine(POSTGRES_DATABASE_URL, echo=False, future=True)
    sessions["postgres"] = sessionmaker(bind=engines["postgres"], autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# --- ORM Models ---

class User(Base):
    """Represents a user of the application."""
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), unique=True, index=True, comment="The unique identifier for the user.")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationship to liked songs via the association object UserLikedSong
    likes: Mapped[List[UserLikedSong]] = relationship("UserLikedSong", back_populates="user", cascade="all, delete-orphan")

    def get_liked_song_ids(self) -> Set[int]:
        """Returns a set of IDs of songs liked by the user."""
        return {like.song_id for like in self.likes}


class SongMetadata(Base):
    """
    Stores definitive metadata for a song, identified by its YouTube video ID.
    This acts as a central repository to avoid data duplication.
    """
    __tablename__ = "song_metadata"
    __table_args__ = (
        UniqueConstraint("video_id", name="uq_video_id"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(String(64), index=True, comment="YouTube video ID.")
    title: Mapped[str] = mapped_column(String(512))
    artist: Mapped[str] = mapped_column(String(256), comment="Typically the YouTube channel title.")
    genre: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="Comma-separated tags from YouTube.")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserLikedSong(Base):
    """
    Association table linking a User to a SongMetadata they have liked.
    This is the core of user preferences.
    """
    __tablename__ = "user_liked_songs"
    __table_args__ = (
        UniqueConstraint("user_id", "song_id", name="uq_user_song"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    song_id: Mapped[int] = mapped_column(ForeignKey("song_metadata.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships to parent tables
    user: Mapped[User] = relationship("User", back_populates="likes")
    song: Mapped[SongMetadata] = relationship("SongMetadata")


# --- Database Initialization and Session Management ---

def init_db() -> None:
    """Creates all database tables defined in the models if they don't exist."""
    print("Initializing database tables...")
    for eng in engines.values():
        Base.metadata.create_all(bind=eng)
    print("Database initialization complete.")

@contextmanager
def get_read_session() -> Iterator[Session]:
    """Provides a database session for read operations, respecting the read preference."""
    session = None
    db_key = DB_READ_PREFERENCE if DB_READ_PREFERENCE in sessions else next(iter(sessions.keys()), None)
    if not db_key:
        raise ConnectionError("No database is configured.")
    try:
        session = sessions[db_key]()
        yield session
    finally:
        if session:
            session.close()

@contextmanager
def get_write_sessions() -> Iterator[List[Session]]:
    """Provides a list of all configured database sessions for write operations."""
    sessions_list = []
    try:
        sessions_list = [s() for s in sessions.values()]
        yield sessions_list
    finally:
        for s in sessions_list:
            s.close()
