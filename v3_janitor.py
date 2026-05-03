# v3_janitor.py
"""
V3 Backfill Daemon — migrates song_metadata from V2 (lexical) to V3 (semantic).

Generates 384-d dense vectors via SentenceTransformer('all-MiniLM-L6-v2')
and batch-updates rows in Supabase/PostgreSQL.

Operational constraints:
- Must run OUTSIDE the FastAPI main thread (GIL contention, CPU starvation).
- Execute as: `python v3_janitor.py` or via GitHub Actions.
- Model weights are loaded once into RAM (~90 MB footprint).
"""

import logging
import os
import sys
import time

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from db import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s [%(name)s] %(message)s",
)
logger = logging.getLogger("v3_janitor")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 100
MODEL_PATH = os.getenv("MODEL_PATH", MODEL_NAME)


def build_text_context(row) -> str:
    """Build semantic text context from a song row.

    Format: "{title}. Artist: {artist}. Genre: {genre}. Tags: {tags}"
    """
    parts = [row.title or ""]
    if row.artist:
        parts.append(f"Artist: {row.artist}")
    if row.genre:
        parts.append(f"Genre: {row.genre}")
    if row.tags:
        parts.append(f"Tags: {row.tags}")
    return ". ".join(parts)


def run_backfill():
    """Main backfill loop: vectorize V2 rows → V3."""
    load_dotenv()

    logger.info("Loading SentenceTransformer model from '%s'...", MODEL_PATH)
    t0 = time.time()
    model = SentenceTransformer(MODEL_PATH)
    logger.info("Model loaded in %.1fs.", time.time() - t0)

    db = SessionLocal()
    total_updated = 0

    try:
        while True:
            # Fetch a batch of un-vectorized rows
            rows = db.execute(
                text("""
                    SELECT id, title, artist, genre, tags
                    FROM song_metadata
                    WHERE embedding IS NULL
                    LIMIT :batch_size
                """),
                {"batch_size": BATCH_SIZE},
            ).fetchall()

            if not rows:
                logger.info("No more rows to backfill. Done!")
                break

            # Build text contexts and encode
            texts = [build_text_context(row) for row in rows]
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=BATCH_SIZE,
            )

            # Batch update
            for row, embedding in zip(rows, embeddings):
                vector_literal = (
                    "[" + ",".join(str(float(x)) for x in embedding) + "]"
                )
                db.execute(
                    text("""
                        UPDATE song_metadata
                        SET embedding = :vec, enriched = 'V3'
                        WHERE id = :id
                    """),
                    {"vec": vector_literal, "id": row.id},
                )

            db.commit()
            total_updated += len(rows)
            logger.info(
                "Batch complete: %d rows vectorized (total: %d).",
                len(rows),
                total_updated,
            )

    except Exception as e:
        logger.exception("Backfill failed: %s", e)
        db.rollback()
    finally:
        db.close()

    logger.info(
        "V3 backfill finished. Total rows updated: %d.", total_updated
    )


if __name__ == "__main__":
    run_backfill()
