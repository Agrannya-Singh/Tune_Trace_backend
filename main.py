# main.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from config import YOUTUBE_API_KEY
from redis_utils import redis_client
from db import SessionLocal
from services import SuggestionService
from engine import ml_engine
from routers import suggestions, users, discovery

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manages startup and shutdown for the application."""
    # --- Startup ---
    application.state.suggestion_service = SuggestionService(api_key=YOUTUBE_API_KEY)
    logger.info("Application starting up...")
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        logger.info("Connection to the database established successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Could not connect to the database: {e}")
        raise RuntimeError(f"Database connection failed: {e}") from e

    # --- Pre-load SentenceTransformer model into RAM (~90 MB) ---
    ml_engine.load_model()
    logger.info("Application startup complete.")

    yield  # --- Application runs here ---

    # --- Shutdown ---
    logger.info("Application shutting down...")
    service = getattr(application.state, "suggestion_service", None)
    if service:
        await service.close()
        logger.info("SuggestionService client closed.")


app = FastAPI(
    title="TuneTrace Semantic Music API",
    description="Generates music suggestions using semantic vector search (pgvector) with genre-based fallback.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Include Routers ---
app.include_router(suggestions.router)
app.include_router(users.router)
app.include_router(discovery.router)

@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    """A simple endpoint to confirm the service is running."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
