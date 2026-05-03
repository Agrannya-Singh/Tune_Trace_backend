# ml_engine.py
"""
Semantic recommendation engine using SentenceTransformer (all-MiniLM-L6-v2)
dense vector embeddings with pgvector cosine distance retrieval.

Replaces the legacy TF-IDF engine with:
- 384-d dense vector encoding via SentenceTransformer
- Direct pgvector `<=>` cosine distance search on Supabase
- Recency-decay weighted user profile vector
- Diversity-aware selection to avoid echo chambers
"""

import logging
from typing import Dict, List, Optional, Set

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — can be overridden via constructor
# ---------------------------------------------------------------------------
DEFAULT_DIVERSITY_RATIO = 0.4  # 40% of final results from diverse sampling
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
# Local path priority (e.g. /app/models/all-MiniLM-L6-v2)
MODEL_PATH = os.getenv("MODEL_PATH", MODEL_NAME)


class MLEngine:
    def __init__(
        self,
        diversity_ratio: float = DEFAULT_DIVERSITY_RATIO,
    ):
        self.diversity_ratio = diversity_ratio
        self._model: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------
    # Model Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Pre-load the SentenceTransformer model into RAM.

        Called during FastAPI lifespan startup to avoid cold-start latency
        on the first request (~90 MB constant footprint).
        """
        if self._model is None:
            logger.info("Loading SentenceTransformer model from '%s'...", MODEL_PATH)
            self._model = SentenceTransformer(MODEL_PATH)
            logger.info("Model loaded successfully (%d-d vectors).", EMBEDDING_DIM)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self.load_model()
        return self._model

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    @staticmethod
    def build_text_context(song: Dict) -> str:
        """Build a natural-language text context for semantic encoding.

        Format: "{title}. Artist: {artist}. Genre: {genre}. Tags: {tags}"
        This structure gives the transformer model clear semantic cues
        about each metadata field's role.
        """
        title = song.get("title") or ""
        artist = song.get("artist") or ""
        genre = song.get("genre") or ""
        tags = song.get("tags") or ""

        parts = [title]
        if artist:
            parts.append(f"Artist: {artist}")
        if genre:
            parts.append(f"Genre: {genre}")
        if tags:
            parts.append(f"Tags: {tags}")
        return ". ".join(parts)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings into 384-d float32 vectors.

        Returns:
            np.ndarray of shape (len(texts), 384).
        """
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string. Returns shape (384,)."""
        return self.encode([text])[0]

    # ------------------------------------------------------------------
    # Recommendation via pgvector
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_history: List[Dict],
        db_session: Session,
        top_n: int = 10,
        excluded_video_ids: Optional[Set[str]] = None,
    ) -> List[Dict]:
        """Generate content-based recommendations using pgvector cosine search.

        Pipeline:
        1. Encode user liked songs into text contexts → 384-d vectors.
        2. Compute a recency-decay-weighted user-profile vector.
        3. Query pgvector for nearest neighbours via `<=>` cosine distance.
        4. Filter already-liked and previously-recommended songs.
        5. Select results with diversity injection.

        Args:
            user_history: List of song dicts from user's liked history
                          (ordered newest-first).
            db_session:   Active SQLAlchemy session for pgvector queries.
            top_n:        Number of recommendations to return.
            excluded_video_ids: Video IDs to skip (previously recommended).

        Returns:
            List of song dicts with 'score' field (higher = more similar).
        """
        if not user_history:
            return []

        _excluded = excluded_video_ids or set()

        # --- Build user profile vector with recency decay ----------------
        user_texts = [self.build_text_context(s) for s in user_history]
        user_vectors = self.encode(user_texts)  # (N, 384)

        n = len(user_vectors)
        if n > 1:
            # Most recent like → weight ≈ 1.0, oldest → weight ≈ 0.37 (e^-1)
            decay = np.exp(-np.linspace(0, 1, n))
            weights = (decay / decay.sum()).reshape(-1, 1)
            profile_vector = (user_vectors * weights).sum(axis=0)
        else:
            profile_vector = user_vectors[0]

        # Normalize the profile vector for cosine distance
        norm = np.linalg.norm(profile_vector)
        if norm > 0:
            profile_vector = profile_vector / norm

        # --- Query pgvector for nearest neighbours -----------------------
        # Fetch more candidates than needed to allow filtering + diversity
        fetch_limit = top_n * 5

        # Build the exclusion list: user's own liked video IDs + prev recs
        user_video_ids = {s["video_id"] for s in user_history}
        all_excluded = user_video_ids | _excluded

        # Convert profile vector to PostgreSQL array literal
        vector_literal = "[" + ",".join(str(float(x)) for x in profile_vector) + "]"

        # Build SQL with optional exclusion filter
        if all_excluded:
            placeholders = ", ".join(f":exc_{i}" for i in range(len(all_excluded)))
            sql = text(f"""
                SELECT video_id, title, artist, genre, tags, enriched,
                       1 - (embedding <=> :query_vec) AS similarity
                FROM song_metadata
                WHERE embedding IS NOT NULL
                  AND video_id NOT IN ({placeholders})
                ORDER BY embedding <=> :query_vec
                LIMIT :k
            """)
            params = {"query_vec": vector_literal, "k": fetch_limit}
            for i, vid in enumerate(all_excluded):
                params[f"exc_{i}"] = vid
        else:
            sql = text("""
                SELECT video_id, title, artist, genre, tags, enriched,
                       1 - (embedding <=> :query_vec) AS similarity
                FROM song_metadata
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :query_vec
                LIMIT :k
            """)
            params = {"query_vec": vector_literal, "k": fetch_limit}

        try:
            result = db_session.execute(sql, params)
            rows = result.fetchall()
        except Exception as e:
            logger.error("pgvector query failed: %s", e)
            return []

        if not rows:
            logger.info("No vectorized candidates found in the database.")
            return []

        scored = [
            {
                "video_id": row.video_id,
                "title": row.title,
                "artist": row.artist,
                "genre": row.genre,
                "tags": row.tags,
                "enriched": row.enriched,
                "score": round(float(row.similarity), 4),
            }
            for row in rows
        ]

        # --- Diversity-aware selection --------------------------------
        n_top = max(1, int(top_n * (1 - self.diversity_ratio)))
        n_diverse = top_n - n_top

        top_picks = scored[:n_top]
        remaining = scored[n_top:]

        diverse_picks: list = []
        if remaining and n_diverse > 0:
            rng = np.random.default_rng()
            sample_size = min(n_diverse, len(remaining))
            chosen_indices = rng.choice(
                len(remaining), size=sample_size, replace=False
            )
            diverse_picks = [remaining[i] for i in chosen_indices]

        final = top_picks + diverse_picks

        logger.info(
            "MLEngine: %d top + %d diverse = %d results (from %d pgvector candidates).",
            len(top_picks),
            len(diverse_picks),
            len(final),
            len(scored),
        )

        return final

    # ------------------------------------------------------------------
    # Free-Text Semantic Search (for /discover)
    # ------------------------------------------------------------------

    def search_by_text(
        self,
        query: str,
        db_session: Session,
        top_n: int = 10,
    ) -> List[Dict]:
        """Encode a free-text query and return the closest songs via pgvector.

        Use cases:
        - Mood search:  "chill lo-fi vibes for studying"
        - Song lookup:  "Blinding Lights by The Weeknd"
        - Genre browse: "upbeat 90s hip-hop"

        Args:
            query:      Raw user input string (mood / song name / description).
            db_session:  Active SQLAlchemy session for pgvector queries.
            top_n:       Number of results to return.

        Returns:
            List of song dicts sorted by semantic similarity (descending).
        """
        if not query or not query.strip():
            return []

        query_vector = self.encode_single(query.strip())

        # Normalize (encode already normalizes, but belt-and-suspenders)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        vector_literal = "[" + ",".join(str(float(x)) for x in query_vector) + "]"

        sql = text("""
            SELECT video_id, title, artist, genre, tags, enriched,
                   1 - (embedding <=> :query_vec) AS similarity
            FROM song_metadata
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> :query_vec
            LIMIT :k
        """)

        try:
            result = db_session.execute(
                sql, {"query_vec": vector_literal, "k": top_n}
            )
            rows = result.fetchall()
        except Exception as e:
            logger.error("pgvector search_by_text query failed: %s", e)
            return []

        if not rows:
            logger.info("search_by_text: no vectorized candidates found.")
            return []

        results = [
            {
                "video_id": row.video_id,
                "title": row.title,
                "artist": row.artist,
                "genre": row.genre,
                "tags": row.tags,
                "score": round(float(row.similarity), 4),
            }
            for row in rows
        ]

        logger.info(
            "search_by_text: query=%r → %d results (top score=%.4f).",
            query[:50],
            len(results),
            results[0]["score"] if results else 0.0,
        )

        return results
