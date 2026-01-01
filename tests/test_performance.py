import time
import logging
import random
from ml_engine import MLEngine
from utils.metrics import track_latency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_performance_test():
    engine = MLEngine()
    
    # Large data set simulation
    all_songs = [
        {"title": f"Song {i}", "artist": f"Artist {i%10}", "genre": "Pop", "video_id": f"v{i}"}
        for i in range(1000)
    ]
    user_history = all_songs[:50]
    
    logger.info("Starting ML Engine Recommendation Performance Test...")
    
    with track_latency("MLEngine:Recommend_1000_items"):
        recs = engine.recommend(user_history, all_songs, top_n=10)
    
    logger.info(f"Generated {len(recs)} recommendations.")

if __name__ == "__main__":
    run_performance_test()
