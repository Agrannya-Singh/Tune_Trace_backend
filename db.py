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

def normalize_database_url(url: Optional[str]) -> Optional[str]:
    """
    Normalizes PostgreSQL connection strings and ensures the appropriate DBAPI driver
    (psycopg vs psycopg2) is matched against installed packages, preventing
    ModuleNotFoundError at runtime.
    """
    if not url:
        return url

    # Normalize deprecated postgres:// prefix to postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Driver availability check
    has_psycopg = False
    try:
        import psycopg  # noqa: F401
        has_psycopg = True
    except ImportError:
        pass

    has_psycopg2 = False
    try:
        import psycopg2  # noqa: F401
        has_psycopg2 = True
    except ImportError:
        pass

    # Resolve dialect if psycopg/psycopg2 is specified or default
    if url.startswith("postgresql+psycopg://"):
        if not has_psycopg and has_psycopg2:
            url = url.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql+psycopg2://"):
        if not has_psycopg2 and has_psycopg:
            url = url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        # Default postgresql:// in SQLAlchemy defaults to psycopg2.
        # If psycopg2 is absent but psycopg (v3) is present, route to postgresql+psycopg://
        if not has_psycopg2 and has_psycopg:
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    return url


DATABASE_URL = normalize_database_url(os.getenv("POSTGRES_DATABASE_URL"))
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
