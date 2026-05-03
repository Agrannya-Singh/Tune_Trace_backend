# ml_engine.py
"""
Semantic recommendation engine using SentenceTransformer (all-MiniLM-L6-v2)
dense vector embeddings with pgvector cosine distance retrieval.

Replaces the legacy TF-IDF engine with:
- 384-d dense vector encoding via SentenceTransformer
- Recency-decay weighted user profile vector
- Diversity-aware selection to avoid echo chambers
"""

import logging
from typing import Dict, List, Optional, Set

import os
import numpy as np
from sentence_transformers import SentenceTransformer

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
        """Pre-load the SentenceTransformer model into RAM."""
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
        """Build a natural-language text context for semantic encoding."""
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
        """Encode a list of text strings into 384-d float32 vectors."""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string. Returns shape (384,)."""
        return self.encode([text])[0]

    # ------------------------------------------------------------------
    # Mathematical Logic for Recommendations
    # ------------------------------------------------------------------

    def compute_user_profile_vector(self, user_history: List[Dict]) -> np.ndarray:
        """Computes a recency-decay-weighted user-profile vector."""
        if not user_history:
            return np.zeros(EMBEDDING_DIM)

        user_texts = [self.build_text_context(s) for s in user_history]
        user_vectors = self.encode(user_texts)  # (N, 384)

        n = len(user_vectors)
        if n > 1:
            decay = np.exp(-np.linspace(0, 1, n))
            weights = (decay / decay.sum()).reshape(-1, 1)
            profile_vector = (user_vectors * weights).sum(axis=0)
        else:
            profile_vector = user_vectors[0]

        # Normalize the profile vector for cosine distance
        norm = np.linalg.norm(profile_vector)
        if norm > 0:
            profile_vector = profile_vector / norm
        
        return profile_vector

    @staticmethod
    def vector_to_literal(vector: np.ndarray) -> str:
        """Converts a numpy vector to a PostgreSQL array literal string."""
        return "[" + ",".join(str(float(x)) for x in vector) + "]"

    def apply_diversity(self, scored_results: List[Dict], top_n: int) -> List[Dict]:
        """Selects results with diversity injection."""
        if not scored_results:
            return []
            
        n_top = max(1, int(top_n * (1 - self.diversity_ratio)))
        n_diverse = top_n - n_top

        top_picks = scored_results[:n_top]
        remaining = scored_results[n_top:]

        diverse_picks: list = []
        if remaining and n_diverse > 0:
            rng = np.random.default_rng()
            sample_size = min(n_diverse, len(remaining))
            chosen_indices = rng.choice(
                len(remaining), size=sample_size, replace=False
            )
            diverse_picks = [remaining[i] for i in chosen_indices]

        final = top_picks + diverse_picks
        return final
