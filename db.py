# db.py
"""
Database models and session management for the music suggestion service.

This module defines the SQLAlchemy ORM models, establishes database connections,
and provides a robust session management system for FastAPI dependencies.
It is designed to support multiple database backends (PostgreSQL/SQLite)
for flexibility across different environments.
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
# This section reads database connection strings from environment variables, a best
# practice for production applications to avoid hardcoding secrets.

# Support for a legacy environment variable name.
LEGACY_DB_URL = os.getenv("DATABASE_URL")

# Define primary connection URLs for different database backends.
SQLITE_DATABASE_URL = os.getenv("SQLITE_DATABASE_URL", "sqlite:///app.db")
POSTGRES_DATABASE_URL = os.getenv("POSTGRES_DATABASE_URL") or (
    LEGACY_DB_URL if (LEGACY_DB_URL or "").startswith("postgres") else None
)

# Allows specifying a preferred database for read operations (e.g., a read replica).
DB_READ_PREFERENCE = os.getenv("DB_READ_PREFERENCE", "postgres").lower()

# Dictionaries to hold the engine and sessionmaker objects for each configured DB.
engines: Dict[str, object] = {}
sessions: Dict[str, sessionmaker] = {}

if SQLITE_DATABASE_URL:
    # `check_same_thread: False` is a specific requirement for SQLite when used
    # in a multi-threaded context like a web server. It's not needed for PostgreSQL.
    sqlite_connect_args = (
        {"check_same_thread": False}
        if SQLITE_DATABASE_URL.startswith("sqlite")
        else {}
    )
    # `future=True` enables SQLAlchemy 2.0-style usage, which is more explicit
    # and performant. `echo=False` prevents logging of every SQL statement.
    engines["sqlite"] = create_engine(
        SQLITE_DATABASE_URL,
        echo=False,
        future=True,
        connect_args=sqlite_connect_args,
    )
    sessions["sqlite"] = sessionmaker(
        bind=engines["sqlite"], autoflush=False, autocommit=False, future=True
    )

if POSTGRES_DATABASE_URL:
    engines["postgres"] = create_engine(
        POSTGRES_DATABASE_URL, echo=False, future=True
    )
    sessions["postgres"] = sessionmaker(
        bind=engines["postgres"], autoflush=False, autocommit=False, future=True
    )


# ==============================================================================
# --- ORM Model Definitions ---
# ==============================================================================


class Base(DeclarativeBase):
    """Base class for all ORM models.

    SQLAlchemy's ORM uses a declarative base class to associate model definitions
    with a metadata catalog, which holds information about the database schema.
    """

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

    # This one-to-many relationship links a User to their liked songs.
    # `back_populates` creates a bi-directional link to the UserLikedSong.user field.
    # `cascade="all, delete-orphan"` ensures that when a User is deleted, all their
    # associated likes are also automatically deleted from the database.
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
        # Ensures that every YouTube video ID in our database is unique.
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
        # A user can only like a specific song once.
        UniqueConstraint("user_id", "song_id", name="uq_user_song"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # `ondelete="CASCADE"` ensures that if a user or song is deleted, this
    # linking record is also removed at the database level.
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    song_id: Mapped[int] = mapped_column(
        ForeignKey("song_metadata.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    # Relationships to the parent tables for easy access in code, e.g., `like.user`.
    user: Mapped[User] = relationship("User", back_populates="likes")
    song: Mapped[SongMetadata] = relationship("SongMetadata")


# ==============================================================================
# --- Database Initialization & Session Management ---
# ==============================================================================


def init_db() -> None:
    """Creates all database tables defined in the models.

    ############################################################################
    # PRODUCTION NOTE: This function is for initial setup or development ONLY. #
    # It CANNOT handle schema migrations (e.g., adding a new column).           #
    # In a production CI/CD pipeline, this should be replaced by a proper      #
    # migration tool like Alembic (`alembic upgrade head`).                    #
    ############################################################################
    """
    print("Initializing database tables...")
    for eng in engines.values():
        # The `checkfirst=True` parameter is the IMMEDIATE FIX.
        # It prevents the "relation already exists" error on startup by issuing
        # a check to the database before running a CREATE TABLE statement.
        Base.metadata.create_all(bind=eng, checkfirst=True)
    print("Database initialization complete.")


def get_read_session() -> Iterator[Session]:
    """Provides a database session for read operations as a FastAPI dependency."""
    db_key = (
        DB_READ_PREFERENCE
        if DB_READ_PREFERENCE in sessions
        else next(iter(sessions.keys()), None)
    )
    if not db_key:
        raise ConnectionError("No database is configured.")

    session: Optional[Session] = None
    try:
        session = sessions[db_key]()
        # `yield` passes the session object to the endpoint function.
        # The code execution pauses here until the endpoint is finished.
        yield session
    finally:
        # This block is executed after the endpoint has returned a response.
        # It guarantees that the session is closed, releasing the connection
        # back to the pool and preventing resource leaks.
        if session:
            session.close()


def get_write_sessions() -> Iterator[List[Session]]:
    """Provides a list of all configured DB sessions for write operations."""
    sessions_list: List[Session] = []
    try:
        # This ensures that write operations are sent to ALL configured databases,
        # keeping them in sync.
        sessions_list = [s() for s in sessions.values()]
        yield sessions_list
    finally:
        for s in sessions_list:
            s.close()
