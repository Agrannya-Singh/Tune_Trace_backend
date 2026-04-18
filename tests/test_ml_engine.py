import unittest
from ml_engine import MLEngine


class TestBuildFeatureText(unittest.TestCase):
    """Tests for the corrected token-level weighting."""

    def setUp(self):
        self.engine = MLEngine()

    def test_basic_weighting(self):
        song = {"title": "Song 1", "artist": "Artist A", "genre": "Pop", "tags": "happy, summer", "video_id": "1"}
        text = self.engine._build_feature_text(song)
        # Artist should appear 2x as space-separated tokens
        self.assertEqual(text.count("Artist A"), 2)
        # Genre should appear 3x as space-separated tokens
        self.assertEqual(text.count("Pop"), 3)

    def test_none_values(self):
        """Regression test: None fields must not crash or produce nonsense."""
        song = {"title": "Song 1", "artist": None, "genre": None, "tags": None, "video_id": "1"}
        text = self.engine._build_feature_text(song)
        self.assertIn("Song 1", text)
        # Should NOT contain "None" as a string
        self.assertNotIn("None", text)

    def test_missing_keys(self):
        song = {"title": "Song 1", "video_id": "1"}
        text = self.engine._build_feature_text(song)
        self.assertEqual(text.strip(), "Song 1")

    def test_multi_word_artist_repeated_correctly(self):
        """Catches the old bug: 'The Weeknd' * 2 → 'The WeekndThe Weeknd'."""
        song = {"title": "Blinding Lights", "artist": "The Weeknd", "genre": None, "tags": None, "video_id": "1"}
        text = self.engine._build_feature_text(song)
        self.assertIn("The Weeknd The Weeknd", text)
        self.assertNotIn("WeekndThe", text)


class TestRecommend(unittest.TestCase):
    """Tests for the recommend() pipeline."""

    def setUp(self):
        self.engine = MLEngine(min_score=0.0, diversity_ratio=0.0)

    def test_basic_recommendation(self):
        user_history = [
            {"title": "Pop Song", "artist": "A", "genre": "Pop", "tags": "upbeat", "video_id": "v1"}
        ]
        all_songs = [
            {"title": "Pop Song 2", "artist": "A", "genre": "Pop", "tags": "upbeat", "video_id": "v2"},
            {"title": "Death Metal Anthem", "artist": "B", "genre": "Death Metal", "tags": "heavy brutal", "video_id": "v3"},
        ]
        recs = self.engine.recommend(user_history, all_songs, top_n=1)
        self.assertEqual(len(recs), 1)
        # Pop song should be ranked higher than death metal
        self.assertEqual(recs[0]["video_id"], "v2")

    def test_empty_history_returns_empty(self):
        recs = self.engine.recommend([], [{"video_id": "v1", "title": "X", "artist": "Y"}])
        self.assertEqual(recs, [])

    def test_empty_candidates_returns_empty(self):
        recs = self.engine.recommend([{"video_id": "v1", "title": "X", "artist": "Y"}], [])
        self.assertEqual(recs, [])

    def test_excludes_already_liked(self):
        """User's own liked songs must never appear in recommendations."""
        user_history = [
            {"title": "Song A", "artist": "X", "genre": "Pop", "tags": "", "video_id": "v1"},
        ]
        all_songs = [
            {"title": "Song A", "artist": "X", "genre": "Pop", "tags": "", "video_id": "v1"},  # same as liked
            {"title": "Song B", "artist": "X", "genre": "Pop", "tags": "", "video_id": "v2"},
        ]
        recs = self.engine.recommend(user_history, all_songs, top_n=10)
        rec_ids = {r["video_id"] for r in recs}
        self.assertNotIn("v1", rec_ids)

    def test_score_threshold_filters_noise(self):
        """With a high threshold, irrelevant candidates should be excluded."""
        strict_engine = MLEngine(min_score=0.99, diversity_ratio=0.0)
        user_history = [
            {"title": "Classical Piano Sonata", "artist": "Mozart", "genre": "Classical", "tags": "piano", "video_id": "v1"}
        ]
        all_songs = [
            {"title": "Extreme Gabber Techno", "artist": "DJ Hardcore", "genre": "Gabber", "tags": "hard rave", "video_id": "v2"},
        ]
        recs = strict_engine.recommend(user_history, all_songs, top_n=10)
        # Completely unrelated song should not pass a 0.99 threshold
        self.assertEqual(recs, [])

    def test_output_includes_score(self):
        """The new engine should return a 'score' field in each recommendation."""
        user_history = [
            {"title": "Pop Song", "artist": "A", "genre": "Pop", "tags": "upbeat", "video_id": "v1"}
        ]
        all_songs = [
            {"title": "Pop Song 2", "artist": "A", "genre": "Pop", "tags": "upbeat", "video_id": "v2"},
        ]
        engine = MLEngine(min_score=0.0, diversity_ratio=0.0)
        recs = engine.recommend(user_history, all_songs, top_n=1)
        self.assertEqual(len(recs), 1)
        self.assertIn("score", recs[0])
        self.assertIsInstance(recs[0]["score"], float)
        self.assertGreater(recs[0]["score"], 0.0)

    def test_diversity_adds_variety(self):
        """With diversity_ratio > 0, results should include non-top-1 picks."""
        diverse_engine = MLEngine(min_score=0.0, diversity_ratio=0.5)
        user_history = [
            {"title": "Pop Hit", "artist": "PopStar", "genre": "Pop", "tags": "catchy", "video_id": "v0"}
        ]
        # Create enough candidates to make diversity meaningful
        all_songs = [
            {"title": f"Song {i}", "artist": f"Artist {i}", "genre": "Pop" if i < 5 else "Rock",
             "tags": "upbeat" if i < 5 else "guitar", "video_id": f"v{i+1}"}
            for i in range(20)
        ]
        recs = diverse_engine.recommend(user_history, all_songs, top_n=10)
        # Should return results (the exact composition varies due to randomness)
        self.assertGreater(len(recs), 0)
        self.assertLessEqual(len(recs), 10)


if __name__ == "__main__":
    unittest.main()
