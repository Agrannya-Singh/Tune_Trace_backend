# ml_engine.py
"""
Content-based recommendation engine using TF-IDF vectorization with:
- Correct token-level weighting (artist 2x, genre 3x)
- Recency-decay weighted user profile
- Minimum similarity threshold to filter noise
- Diversity-aware selection to avoid echo chambers
"""

import logging
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — can be overridden via constructor
# ---------------------------------------------------------------------------
DEFAULT_MIN_SCORE = 0.05
DEFAULT_DIVERSITY_RATIO = 0.4  # 40% of final results from diverse sampling


class MLEngine:
    def __init__(
        self,
        min_score: float = DEFAULT_MIN_SCORE,
        diversity_ratio: float = DEFAULT_DIVERSITY_RATIO,
    ):
        self.min_score = min_score
        self.diversity_ratio = diversity_ratio

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _build_feature_text(song: Dict) -> str:
        """Build a weighted text-feature string for a single song.

        Weighting is achieved by *repeating space-separated tokens*, NOT
        by repeating the raw string.  ``"Drake" * 2`` would produce the
        single nonsense token ``"DrakeDrake"``; instead we produce
        ``"Drake Drake"`` so TF-IDF counts two genuine occurrences.
        """
        title = song.get("title") or ""
        artist = song.get("artist") or ""
        genre = song.get("genre") or ""
        tags = song.get("tags") or ""

        parts = [
            title,                                    # 1× weight
            " ".join([artist] * 2) if artist else "",  # 2× weight
            " ".join([genre] * 3) if genre else "",    # 3× weight
            tags,                                      # 1× weight
        ]
        return " ".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_history: List[Dict],
        all_songs: List[Dict],
        top_n: int = 10,
    ) -> List[Dict]:
        """Generate content-based recommendations.

        Pipeline:
        1. Build TF-IDF feature text for user history & candidates.
        2. Compute a recency-decay-weighted user-profile vector.
        3. Score every candidate via cosine similarity.
        4. Filter below ``min_score`` and already-liked songs.
        5. Select results with diversity injection.
        """
        if not user_history or not all_songs:
            return []

        user_texts = [self._build_feature_text(s) for s in user_history]
        candidate_texts = [self._build_feature_text(s) for s in all_songs]

        # --- Vectorize (new instance per call → thread-safe) -----------
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf = vectorizer.fit_transform(user_texts + candidate_texts)
        except ValueError:
            # Happens when vocabulary is empty (all stop words, etc.)
            logger.warning("TF-IDF vectorization produced an empty vocabulary.")
            return []

        user_matrix = tfidf[: len(user_history)]
        candidate_matrix = tfidf[len(user_history) :]

        # --- Recency-decay weighted user profile ----------------------
        #   Most recent like → weight ≈ 1.0
        #   Oldest like       → weight ≈ 0.37  (e^-1)
        #   Assumes user_history is ordered newest-first.
        n = user_matrix.shape[0]
        if n > 1:
            decay = np.exp(-np.linspace(0, 1, n))
            weights = (decay / decay.sum()).reshape(-1, 1)
            user_profile = np.asarray(user_matrix.T @ weights).T  # (1, V)
        else:
            user_profile = np.asarray(user_matrix.todense())

        # --- Score candidates -----------------------------------------
        scores = cosine_similarity(user_profile, candidate_matrix)[0]

        # --- Filter ---------------------------------------------------
        user_video_ids = {s["video_id"] for s in user_history}
        scored = []
        for idx, score in enumerate(scores):
            if score < self.min_score:
                continue
            if idx >= len(all_songs):
                continue
            candidate = all_songs[idx]
            if candidate["video_id"] in user_video_ids:
                continue
            scored.append((idx, float(score), candidate))

        if not scored:
            logger.info(
                "No candidates passed the similarity threshold (%.3f).",
                self.min_score,
            )
            return []

        # --- Diversity-aware selection --------------------------------
        scored.sort(key=lambda x: x[1], reverse=True)

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
            "MLEngine: %d top + %d diverse = %d results (from %d scored candidates).",
            len(top_picks),
            len(diverse_picks),
            len(final),
            len(scored),
        )

        return [
            {**candidate, "score": round(score, 4)}
            for _, score, candidate in final
        ]
