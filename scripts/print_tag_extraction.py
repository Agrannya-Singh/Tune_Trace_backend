import sys
import os

# Add the project root (parent of scripts/) to sys.path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.enrichment import _extract_genre_from_tags
from ml_engine import MLEngine


def run_demo():
    print("==========================================================")
    print("   Tag Extraction & Semantic Feature Construction Demo")
    print("==========================================================")
    
    samples = [
        {
            "id": "1",
            "title": "Summer Vibes 2024",
            "artist": "DJ Chill",
            "raw_tags": ["summer music", "electronic", "house music", "chillout", "ibiza 2024"]
        },
        {
            "id": "2",
            "title": "Midnight Drive",
            "artist": "The Midnight",
            "raw_tags": ["synthwave", "retro", "80s", "neon", "drive"]
        },
        {
            "id": "3",
            "title": "Street Poetry",
            "artist": "MC Bars",
            "raw_tags": ["rap", "underground hip hop", "boom bap", "beats"]
        },
        {
            "id": "4",
            "title": "Acoustic Sunset",
            "artist": "Jane Doe",
            "raw_tags": ["live performance", "acoustic guitar", "indie folk", "singer-songwriter"]
        },
        {
            "id": "5",
            "title": "Generic Vlog Video",
            "artist": "Vlogger Bob",
            "raw_tags": ["vlog", "daily life", "funny", "family friendly"]
        }
    ]

    for sample in samples:
        print(f"\n--- Video: {sample['title']} by {sample['artist']} ---")
        print(f"1. Raw Tags from YouTube: {sample['raw_tags']}")
        
        # Simulate enrichment: find genre from tags
        genre = _extract_genre_from_tags(sample['raw_tags'])
        print(f"2. Extracted Genre:       {genre}")
        
        # Simulate DB: limit to 20 tags joined by comma
        tags_str = ", ".join(sample['raw_tags'][:20])
        
        # Simulate ML Engine input: to_dict() provides genre and tags
        song_dict = {
            "title": sample['title'],
            "artist": sample['artist'],
            "genre": genre,
            "tags": tags_str,
            "video_id": sample['id']
        }
        
        # This is what the SentenceTransformer encoder sees
        feature_text = MLEngine.build_text_context(song_dict)
        
        print("3. Final Semantic Context Text:")
        print(f"   => {feature_text}")


if __name__ == "__main__":
    run_demo()

