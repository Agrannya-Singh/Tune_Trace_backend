# repository.py

from typing import List, Optional, Set

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func

from db import SongMetadata, User, UserLikedSong


class MusicRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_user(self, user_id: str) -> User:
        """
        Retrieves a user by ID or creates a new one if not found.
        
        Implements a database-agnostic 'get-or-create' pattern using optimistic
        insertion with rollback to handle concurrency race conditions safely.
        """
        user = self.db.query(User).options(joinedload(
            User.likes)).filter_by(user_id=user_id).one_or_none()
        
        if not user:
            user = User(user_id=user_id)
            self.db.add(user)
            try:
                self.db.flush()
            except Exception:
                self.db.rollback()
                user = self.db.query(User).options(joinedload(
                    User.likes)).filter_by(user_id=user_id).one()
        return user

    def get_song_metadata_by_video_id(self, video_id: str) -> Optional[SongMetadata]:
        return self.db.query(SongMetadata).filter_by(video_id=video_id).one_or_none()

    def create_song_metadata(self, video_data: dict) -> SongMetadata:
        song = SongMetadata(
            video_id=video_data["video_id"],
            title=video_data["title"],
            artist=video_data["artist"],
        )
        self.db.add(song)
        self.db.flush()
        return song

    def persist_user_likes(self, user: User, song_metadata_ids: Set[int]):
        existing_liked_ids = {like.song_id for like in user.likes}
        ids_to_add = song_metadata_ids - existing_liked_ids

        if ids_to_add:
            new_likes = [UserLikedSong(user_id=user.id, song_id=song_id)
                         for song_id in ids_to_add]
            self.db.add_all(new_likes)

        self.db.commit()

    def get_user_liked_songs(self, user_id: str) -> List[tuple]:
        """Returns list of (video_id, title, artist, created_at) for a user's liked songs."""
        user = self.db.query(User).filter_by(user_id=user_id).one_or_none()
        if not user:
            return []

        results = (
            self.db.query(
                SongMetadata.video_id,
                SongMetadata.title,
                SongMetadata.artist,
                UserLikedSong.created_at
            )
            .join(UserLikedSong, UserLikedSong.song_id == SongMetadata.id)
            .filter(UserLikedSong.user_id == user.id)
            .order_by(UserLikedSong.created_at.desc())
            .all()
        )
        return results

    def get_user_liked_songs_objects(self, user_id: str) -> List[SongMetadata]:
        """Returns a list of SongMetadata objects for a user's liked songs."""
        user = self.db.query(User).filter_by(user_id=user_id).one_or_none()
        if not user:
            return []

        return [
            liked_song.song for liked_song in user.likes
        ]

    def get_candidate_songs(self, limit: int = 1000) -> List[SongMetadata]:
        """Returns a list of candidate songs for recommendation."""
        # Simple strategy: get the most recently updated songs
        return (
            self.db.query(SongMetadata)
            .order_by(SongMetadata.updated_at.desc())
            .limit(limit)
            .all()
        )

    def get_collaborative_suggestions(self, user: User, limit: int = 10) -> List[SongMetadata]:
        """Get song suggestions based on collaborative filtering.

        Finds songs liked by users with similar taste (users who liked the same songs).
        """
        if not user.likes:
            return []

        # Get songs liked by this user
        user_liked_song_ids = user.get_liked_song_ids()

        # Find other users who liked the same songs
        similar_users = (
            self.db.query(User.id)
            .join(UserLikedSong)
            .filter(UserLikedSong.song_id.in_(user_liked_song_ids))
            .filter(User.id != user.id)
            .group_by(User.id)
            .having(func.count(UserLikedSong.song_id) >= 2)  # At least 2 songs in common
            .all()
        )

        if not similar_users:
            return []

        similar_user_ids = [u[0] for u in similar_users]

        # Get songs liked by similar users that the current user hasn't liked
        recommendations = (
            self.db.query(SongMetadata)
            .join(UserLikedSong)
            .filter(UserLikedSong.user_id.in_(similar_user_ids))
            .filter(~SongMetadata.id.in_(user_liked_song_ids))
            .group_by(SongMetadata.id)
            .order_by(func.count(UserLikedSong.user_id).desc())  # Most popular among similar users
            .limit(limit)
            .all()
        )

        return recommendations
    def get_songs_by_ids(self, song_ids: List[int]) -> List[SongMetadata]:
        """Returns a list of SongMetadata objects for the given IDs."""
        return self.db.query(SongMetadata).filter(SongMetadata.id.in_(song_ids)).all()
