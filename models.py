# models.py
"""
SQLAlchemy ORM models for the TuneTrace backend.
"""

from __future__ import annotations
from datetime import datetime
from typing import List, Optional, Set

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class User(Base):
    """Represents a user of the application."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        comment="User email or unique identifier from OAuth (e.g., Google email).",
    )
    name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="User's display name from OAuth provider."
    )
    email: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="User's email address from OAuth provider."
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
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
    enriched: Mapped[Optional[str]] = mapped_column(
        String(16),
        nullable=True,
        default=None,
        index=True,
        comment="Enrichment version. NULL=raw, 'V2'=genre/tags enriched, etc.",
    )
    embedding = mapped_column(
        Vector(384),
        nullable=True,
        comment="384-d semantic embedding from all-MiniLM-L6-v2"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def to_dict(self) -> dict:
        """Converts the SQLAlchemy model to a dictionary for the ML engine."""
        emb = self.embedding
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
            
        return {
            "id": self.id,
            "video_id": self.video_id,
            "title": self.title,
            "artist": self.artist,
            "genre": self.genre,
            "tags": self.tags,
            "enriched": self.enriched,
            "embedding": emb,
        }


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
