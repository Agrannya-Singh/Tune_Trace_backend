# ml_engine.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class MLEngine:
    def __init__(self):
        # Stop words remove common filler words so we focus on genres/tags
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def _generate_text_features(self, songs: List[Dict]) -> List[str]:
        """Aggegating metadata (title, artist, genre, tags) into a text feature string for vectorization."""
        feature_vectors = []
        for song in songs:
            # Weighted feature engineering: Artist(2x), Genre(3x)
            title = song.get('title') or ''
            artist = (song.get('artist') or '') * 2
            genre = (song.get('genre') or '') * 3
            tags = song.get('tags') or ''
            
            features = [title, artist, genre, tags]
            feature_vectors.append(" ".join(features))
        return feature_vectors

    def recommend(self, user_history: List[Dict], all_songs: List[Dict], top_n: int = 10) -> List[Dict]:
        """Generates content-based recommendations using TF-IDF vectorization and Cosine Similarity."""
        if not user_history or not all_songs:
            return []

        # Generate feature strings
        user_features = self._generate_text_features(user_history)
        candidate_features = self._generate_text_features(all_songs)

        try:
             # Fit-transform on combined corpus to ensure consistent vocabulary
            tfidf_matrix = self.vectorizer.fit_transform(user_features + candidate_features)
        except ValueError:
            return []
        
        # Split matrices
        user_matrix = tfidf_matrix[:len(user_history)]
        candidate_matrix = tfidf_matrix[len(user_history):]

        # Calculate User Profile (Mean Vector) and Similarity Scores
        user_profile = np.asarray(np.mean(user_matrix, axis=0))
        scores = cosine_similarity(user_profile, candidate_matrix)

        if scores.shape[0] == 0:
            return []
            
        # Rank by similarity score (descending)
        indices = scores.argsort()[0][::-1]
        
        recommendations = []
        user_video_ids = {s['video_id'] for s in user_history}

        for idx in indices:
            if idx < len(all_songs):
                candidate = all_songs[idx]
                if candidate['video_id'] not in user_video_ids:
                    recommendations.append(candidate)
                    if len(recommendations) >= top_n:
                        break
        
        return recommendations
