import unittest
from ml_engine import MLEngine

class TestMLEngine(unittest.TestCase):
    def setUp(self):
        self.engine = MLEngine()

    def test_create_metadata_soup_basic(self):
        songs = [
            {"title": "Song 1", "artist": "Artist A", "genre": "Pop", "tags": "happy, summer", "video_id": "1"},
            {"title": "Song 2", "artist": "Artist B", "genre": "Rock", "tags": "energetic", "video_id": "2"}
        ]
        soup = self.engine._create_metadata_soup(songs)
        self.assertEqual(len(soup), 2)
        self.assertIn("Song 1", soup[0])
        self.assertIn("Artist A Artist A", soup[0])
        self.assertIn("Pop Pop Pop", soup[0])

    def test_create_metadata_soup_none_values(self):
        # Regression test for NoneType multiplication bug
        songs = [
            {"title": "Song 1", "artist": None, "genre": None, "tags": None, "video_id": "1"}
        ]
        soup = self.engine._create_metadata_soup(songs)
        self.assertEqual(len(soup), 1)
        self.assertEqual(soup[0].strip(), "Song 1")

    def test_create_metadata_soup_missing_keys(self):
        songs = [
            {"title": "Song 1", "video_id": "1"}
        ]
        soup = self.engine._create_metadata_soup(songs)
        self.assertEqual(len(soup), 1)
        self.assertEqual(soup[0].strip(), "Song 1")

    def test_recommend_basic(self):
        user_history = [{"title": "Pop Song", "artist": "A", "genre": "Pop", "tags": "upbeat", "video_id": "v1"}]
        all_songs = [
            {"title": "Pop Song 2", "artist": "A", "genre": "Pop", "tags": "upbeat", "video_id": "v2"},
            {"title": "Rock Song", "artist": "B", "genre": "Rock", "tags": "heavy", "video_id": "v3"}
        ]
        recs = self.engine.recommend(user_history, all_songs, top_n=1)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]['video_id'], "v2")

    def test_recommend_empty_history(self):
        recs = self.engine.recommend([], [{"video_id": "v1"}])
        self.assertEqual(recs, [])

    def test_recommend_empty_candidates(self):
        recs = self.engine.recommend([{"video_id": "v1"}], [])
        self.assertEqual(recs, [])

if __name__ == "__main__":
    unittest.main()
