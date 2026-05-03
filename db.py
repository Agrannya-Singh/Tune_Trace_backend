# db.py
"""
Database configuration and session management for the TuneTrace backend.
"""

import os
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models import Base  # Import models to ensure they are registered with Base

# ==============================================================================
# --- Database Configuration ---
# ==============================================================================

DATABASE_URL = os.getenv("POSTGRES_DATABASE_URL")
connect_args = {}

# Fallback to a local SQLite database ONLY if PostgreSQL is not configured.
if not DATABASE_URL:
    print("INFO: POSTGRES_DATABASE_URL not found, falling back to local SQLite database.")
    DATABASE_URL = "sqlite:///./app.db"
    connect_args = {"check_same_thread": False}

# Create a single, definitive engine for the application.
engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args=connect_args
)

# Create a single, definitive sessionmaker.
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, future=True
)


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

# Re-exporting models for convenience and backwards compatibility
from models import User, SongMetadata, UserLikedSong
