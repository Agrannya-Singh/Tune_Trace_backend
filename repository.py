# repository.py

from typing import List, Optional, Set
import logging

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, text

from models import SongMetadata, User, UserLikedSong


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
        self.db.refresh(user)

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
        """Returns a list of SongMetadata objects for a user's liked songs.

        Uses a direct JOIN query instead of relationship lazy-loading to
        avoid stale identity-map data after persist_user_likes() commits.
        """
        return (
            self.db.query(SongMetadata)
            .join(UserLikedSong, UserLikedSong.song_id == SongMetadata.id)
            .join(User, User.id == UserLikedSong.user_id)
            .filter(User.user_id == user_id)
            .all()
        )

    def get_candidate_songs(self, exclude_song_ids: Optional[Set[int]] = None, limit: int = 1000) -> List[SongMetadata]:
        """Returns candidate songs ordered by metadata richness for best TF-IDF quality.

        Priority:
          1. Songs with both genre AND tags populated  (richest vectors)
          2. Songs with genre only
          3. Remaining songs (title/artist only)
        Within each tier, most-recently-updated songs appear first.
        """
        query = self.db.query(SongMetadata)
        if exclude_song_ids:
            query = query.filter(~SongMetadata.id.in_(exclude_song_ids))
        return (
            query
            .order_by(
                # Richest metadata first: both genre and tags present
                (
                    (SongMetadata.genre.isnot(None)) &
                    (SongMetadata.genre != "") &
                    (SongMetadata.tags.isnot(None)) &
                    (SongMetadata.tags != "")
                ).desc(),
                # Second tier: genre present
                (
                    (SongMetadata.genre.isnot(None)) &
                    (SongMetadata.genre != "")
                ).desc(),
                SongMetadata.updated_at.desc(),
            )
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

    def get_semantic_recommendations(self, vector_literal: str, fetch_limit: int, exclude_video_ids: Set[str]) -> List[tuple]:
        """
        Executes pgvector cosine similarity search.
        Returns a list of tuples containing (video_id, title, artist, genre, tags, enriched, similarity).
        """
        if exclude_video_ids:
            placeholders = ", ".join(f":exc_{i}" for i in range(len(exclude_video_ids)))
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
            for i, vid in enumerate(exclude_video_ids):
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

        result = self.db.execute(sql, params)
        return result.fetchall()

    def semantic_search(self, vector_literal: str, limit: int) -> List[tuple]:
        """
        Executes pgvector cosine similarity search for free-text queries.
        """
        sql = text("""
            SELECT video_id, title, artist, genre, tags, enriched,
                   1 - (embedding <=> :query_vec) AS similarity
            FROM song_metadata
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> :query_vec
            LIMIT :k
        """)
        result = self.db.execute(sql, {"query_vec": vector_literal, "k": limit})
        return result.fetchall()
