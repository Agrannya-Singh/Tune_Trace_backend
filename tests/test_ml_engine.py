import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from ml_engine import MLEngine, EMBEDDING_DIM


class TestBuildTextContext(unittest.TestCase):
    """Tests for the semantic text context builder."""

    def test_full_metadata(self):
        song = {"title": "Blinding Lights", "artist": "The Weeknd", "genre": "Pop", "tags": "synth, retro", "video_id": "1"}
        text = MLEngine.build_text_context(song)
        self.assertIn("Blinding Lights", text)
        self.assertIn("Artist: The Weeknd", text)
        self.assertIn("Genre: Pop", text)
        self.assertIn("Tags: synth, retro", text)

    def test_none_values(self):
        """Regression test: None fields must not crash or produce nonsense."""
        song = {"title": "Song 1", "artist": None, "genre": None, "tags": None, "video_id": "1"}
        text = MLEngine.build_text_context(song)
        self.assertIn("Song 1", text)
        self.assertNotIn("None", text)
        self.assertNotIn("Artist:", text)
        self.assertNotIn("Genre:", text)

    def test_missing_keys(self):
        song = {"title": "Song 1", "video_id": "1"}
        text = MLEngine.build_text_context(song)
        self.assertEqual(text.strip(), "Song 1")


class TestMLEngineLogic(unittest.TestCase):
    """Tests for MLEngine mathematical logic (profile computation, diversity)."""

    @patch("ml_engine.SentenceTransformer")
    def test_compute_user_profile_vector(self, MockST):
        mock_model = MockST.return_value
        # Mock encode to return normalized vectors
        def mock_encode(texts, **kwargs):
            return np.ones((len(texts), EMBEDDING_DIM), dtype=np.float32) / np.sqrt(EMBEDDING_DIM)
        
        mock_model.encode.side_effect = mock_encode
        
        engine = MLEngine()
        engine._model = mock_model
        
        user_history = [
            {"title": "Song 1", "video_id": "v1"},
            {"title": "Song 2", "video_id": "v2"}
        ]
        
        profile = engine.compute_user_profile_vector(user_history)
        self.assertEqual(profile.shape, (EMBEDDING_DIM,))
        # Check normalization
        self.assertAlmostEqual(np.linalg.norm(profile), 1.0, places=5)

    def test_compute_user_profile_empty(self):
        engine = MLEngine()
        profile = engine.compute_user_profile_vector([])
        self.assertTrue(np.all(profile == 0))

    def test_apply_diversity(self):
        engine = MLEngine(diversity_ratio=0.5)
        scored_results = [
            {"video_id": f"v{i}", "score": 1.0 - i/100} for i in range(20)
        ]
        
        top_n = 10
        # With 0.5 ratio, 5 should be top, 5 diverse
        final = engine.apply_diversity(scored_results, top_n)
        
        self.assertEqual(len(final), top_n)
        # The first 5 should be the absolute top 5
        for i in range(5):
            self.assertEqual(final[i]["video_id"], f"v{i}")
        
        # The next 5 should be from the remaining 15
        remaining_ids = {f"v{i}" for i in range(5, 20)}
        for i in range(5, 10):
            self.assertIn(final[i]["video_id"], remaining_ids)

    def test_vector_to_literal(self):
        engine = MLEngine()
        vec = np.array([0.1, 0.2, 0.3])
        literal = engine.vector_to_literal(vec)
        self.assertEqual(literal, "[0.1,0.2,0.3]")


if __name__ == "__main__":
    unittest.main()
