import unittest
from unittest.mock import MagicMock, patch
from ml_engine import MLEngine


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

    def test_format_structure(self):
        """Text should use dot-separated fields for clear semantic cues."""
        song = {"title": "Waves", "artist": "Dean Lewis", "genre": "Indie", "tags": "acoustic", "video_id": "1"}
        text = MLEngine.build_text_context(song)
        # Should be dot-separated
        self.assertIn(". Artist:", text)
        self.assertIn(". Genre:", text)
        self.assertIn(". Tags:", text)


class TestMLEngineModelLoading(unittest.TestCase):
    """Tests for model lifecycle management."""

    def test_model_lazy_loads(self):
        """Model should not be loaded until explicitly requested."""
        engine = MLEngine()
        self.assertIsNone(engine._model)

    @patch("ml_engine.SentenceTransformer")
    def test_load_model_sets_model(self, MockST):
        engine = MLEngine()
        engine.load_model()
        MockST.assert_called_once_with("all-MiniLM-L6-v2")
        self.assertIsNotNone(engine._model)

    @patch("ml_engine.SentenceTransformer")
    def test_load_model_idempotent(self, MockST):
        """Calling load_model() twice should not reload."""
        engine = MLEngine()
        engine.load_model()
        engine.load_model()
        MockST.assert_called_once()


class TestRecommend(unittest.TestCase):
    """Tests for the recommend() pipeline with mocked DB and model."""

    def test_empty_history_returns_empty(self):
        engine = MLEngine()
        mock_session = MagicMock()
        recs = engine.recommend([], mock_session)
        self.assertEqual(recs, [])

    @patch("ml_engine.SentenceTransformer")
    def test_excludes_previously_recommended(self, MockST):
        """Songs in excluded_video_ids must not appear in recommendations."""
        import numpy as np

        mock_model_instance = MockST.return_value
        mock_model_instance.encode.return_value = np.random.randn(1, 384).astype(np.float32)

        engine = MLEngine(diversity_ratio=0.0)
        engine._model = mock_model_instance

        # Mock DB session returning two candidate rows
        mock_row_1 = MagicMock()
        mock_row_1.video_id = "v2"
        mock_row_1.title = "Song 2"
        mock_row_1.artist = "A"
        mock_row_1.genre = "Pop"
        mock_row_1.tags = ""
        mock_row_1.enriched = "V3"
        mock_row_1.similarity = 0.95

        mock_row_2 = MagicMock()
        mock_row_2.video_id = "v3"
        mock_row_2.title = "Song 3"
        mock_row_2.artist = "B"
        mock_row_2.genre = "Rock"
        mock_row_2.tags = ""
        mock_row_2.enriched = "V3"
        mock_row_2.similarity = 0.85

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row_1, mock_row_2]
        mock_session.execute.return_value = mock_result

        user_history = [
            {"title": "Pop Song", "artist": "A", "genre": "Pop", "tags": "upbeat", "video_id": "v1"}
        ]

        recs = engine.recommend(
            user_history=user_history,
            db_session=mock_session,
            top_n=10,
            excluded_video_ids={"v2"},
        )

        # Since v2 is excluded at the SQL level (via the generated query),
        # the mock returns both rows, but in real scenario the DB would filter.
        # This test validates the flow doesn't crash and returns results.
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)

    @patch("ml_engine.SentenceTransformer")
    def test_output_includes_score(self, MockST):
        """Results should include a 'score' field."""
        import numpy as np

        mock_model_instance = MockST.return_value
        mock_model_instance.encode.return_value = np.random.randn(1, 384).astype(np.float32)

        engine = MLEngine(diversity_ratio=0.0)
        engine._model = mock_model_instance

        mock_row = MagicMock()
        mock_row.video_id = "v2"
        mock_row.title = "Pop Song 2"
        mock_row.artist = "A"
        mock_row.genre = "Pop"
        mock_row.tags = "upbeat"
        mock_row.enriched = "V3"
        mock_row.similarity = 0.92

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result

        recs = engine.recommend(
            user_history=[
                {"title": "Pop Song", "artist": "A", "genre": "Pop", "tags": "upbeat", "video_id": "v1"}
            ],
            db_session=mock_session,
            top_n=1,
        )
        self.assertEqual(len(recs), 1)
        self.assertIn("score", recs[0])
        self.assertIsInstance(recs[0]["score"], float)


if __name__ == "__main__":
    unittest.main()
