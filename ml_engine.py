# ml_engine.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class MLEngine:
    def __init__(self):
        # Stop words remove common filler words so we focus on genres/tags
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def _create_metadata_soup(self, songs: List[Dict]) -> List[str]:
        """
        Combines title, artist, genre, and tags into a single string 
        for each song to analyze.
        """
        soup_list = []
        for song in songs:
            # We weight the artist and genre heavily by repeating them
            # This makes the AI 'care' more about them than random tags
            title = song.get('title') or ''
            artist = (song.get('artist') or '') * 2
            genre = (song.get('genre') or '') * 3
            tags = song.get('tags') or ''
            
            features = [title, artist, genre, tags]
            soup_list.append(" ".join(features))
        return soup_list

    def recommend(self, user_history: List[Dict], all_songs: List[Dict], top_n: int = 10):
        """
        1. Vectors user history into a 'User Profile'
        2. Vectors all candidate songs
        3. Finds candidates closest to the User Profile
        """
        if not user_history or not all_songs:
            return []

        # 1. Prepare data
        user_soup = self._create_metadata_soup(user_history)
        candidate_soup = self._create_metadata_soup(all_songs)

        # 2. Fit and Transform
        # We learn the vocabulary from ALL songs (history + candidates)
        try:
            tfidf_matrix = self.vectorizer.fit_transform(user_soup + candidate_soup)
        except ValueError:
            # This can happen if all documents are empty or only contain stop words
            return []
        
        # Split back into user profile and candidates
        user_matrix = tfidf_matrix[:len(user_history)]
        candidate_matrix = tfidf_matrix[len(user_history):]

        # 3. Create User Profile Vector
        # Average the vectors of all songs the user likes to get one "Taste Vector"
        user_profile = np.asarray(np.mean(user_matrix, axis=0))

        # 4. Calculate Cosine Similarity
        # Result is a list of scores between 0 (not similar) and 1 (identical)
        scores = cosine_similarity(user_profile, candidate_matrix)

        # 5. Rank and Filter
        # Flatten scores array and sort indices by score descending
        if scores.shape[0] == 0:
            return []
            
        indices = scores.argsort()[0][::-1]
        
        recommendations = []
        user_video_ids = {s['video_id'] for s in user_history}

        for idx in indices:
            if idx < len(all_songs):
                candidate = all_songs[idx]
                # Don't recommend songs they already liked
                if candidate['video_id'] not in user_video_ids:
                    recommendations.append(candidate)
                    if len(recommendations) >= top_n:
                        break
        
        return recommendations
