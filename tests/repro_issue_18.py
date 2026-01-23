
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import Base, User, SongMetadata, UserLikedSong
from repository import MusicRepository

class TestIssue18(unittest.TestCase):
    def setUp(self):
        # Use in-memory SQLite for speed and isolation
        self.engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.repo = MusicRepository(self.session)

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)

    def test_stale_user_likes_caching(self):
        # 1. Setup initial data
        user_id = "test_user_1"
        user = self.repo.get_or_create_user(user_id)
        
        # Create a song
        song_data = {"video_id": "v1", "title": "Song 1", "artist": "Artist 1"}
        song = self.repo.create_song_metadata(song_data)
        
        # 2. Simulate the flow: Load user (caches empty likes)
        # calling get_user_liked_songs_objects calls get_or_create_user internally or queries User
        # Let's ensure the user object is in the session and loaded
        # The bug is that 'user' variable is held or the session has cached 'User' with no likes.
        
        # Verify initially empty
        initial_likes = self.repo.get_user_liked_songs_objects(user_id)
        self.assertEqual(len(initial_likes), 0)
        
        # 3. Add a new like
        # We use persist_user_likes which commits.
        self.repo.persist_user_likes(user, {song.id})
        
        # 4. Retrieve liked songs again
        # This is where the bug manifests: currently it uses 'user.likes' from the cached user object
        updated_likes = self.repo.get_user_liked_songs_objects(user_id)
        
        # 5. Assert we found the new like
        # Fails if the repo uses formatted user.likes without refresh
        self.assertEqual(len(updated_likes), 1, "Should find 1 liked song after persisting")
        self.assertEqual(updated_likes[0].video_id, "v1")

if __name__ == '__main__':
    unittest.main()
