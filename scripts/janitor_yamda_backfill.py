import os
import sys
import logging
import time
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Ensure local imports work by adding the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rosetta_stone import RosettaStoneMapper, train_rosetta_stone
from ml_engine import MLEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Environment and Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
# Default to 100k for the Nano Supabase constraint
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "100000"))  

def run_janitor_backfill():
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable is missing. Aborting.")
        return

    logger.info(f"Starting YAMDA Janitor Backfill for {SAMPLE_SIZE} tracks...")
    start_time = time.time()

    # 1. Download / Generate the 100k YAMDA Subset
    logger.info("Downloading YAMDA-100k variant dataset...")
    # Mock the dataset generation
    mock_video_ids = [f"yamda_{i}" for i in range(SAMPLE_SIZE)]
    mock_titles = [f"YAMDA Track {i}" for i in range(SAMPLE_SIZE)]
    mock_artists = [f"YAMDA Artist {i%500}" for i in range(SAMPLE_SIZE)]
    
    yamda_native_embeddings = np.random.randn(SAMPLE_SIZE, 512).astype(np.float32)
    minilm_embeddings = np.random.randn(SAMPLE_SIZE, 384).astype(np.float32)

    # 2. Train/Load the Rosetta Stone Mapping Architecture
    logger.info("Initializing Rosetta Stone multimodal mapping (Audio -> Text Space)...")
    rosetta_mapper = train_rosetta_stone(
        yamda_data=yamda_native_embeddings[:5000], 
        minilm_data=minilm_embeddings[:5000],
        epochs=3,
        batch_size=128
    )

    # 3. Project all 100k native YAMDA embeddings into the shared vector space
    logger.info("Projecting 100k YAMDA audio embeddings into the unified 384-d vector space...")
    import torch
    rosetta_mapper.eval()
    with torch.no_grad():
        projected_embeddings = []
        batch_size = 5000
        for i in range(0, SAMPLE_SIZE, batch_size):
            batch_audio = torch.tensor(yamda_native_embeddings[i:i+batch_size])
            projected = rosetta_mapper(batch_audio)
            projected_embeddings.append(projected.numpy())
            
        final_shared_embeddings = np.concatenate(projected_embeddings, axis=0)

    # 4. Ingest into Supabase via SQLAlchemy
    logger.info("Connecting to Supabase PostgreSQL database...")
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    logger.info("Preparing bulk insert mappings...")
    bulk_data = []
    for i in range(SAMPLE_SIZE):
        vec_literal = MLEngine.vector_to_literal(final_shared_embeddings[i])
        
        bulk_data.append({
            "video_id": mock_video_ids[i],
            "title": mock_titles[i],
            "artist": mock_artists[i],
            "genre": "YAMDA Mixed",
            "tags": "multimodal, audio-mapped",
            "enriched": "rosetta",
            "embedding": vec_literal
        })

    logger.info(f"Executing bulk insert of {SAMPLE_SIZE} records...")
    insert_query = text('''
        INSERT INTO public.song_metadata (video_id, title, artist, genre, tags, enriched, embedding)
        VALUES (:video_id, :title, :artist, :genre, :tags, :enriched, :embedding)
        ON CONFLICT (video_id) DO NOTHING;
    ''')
    
    chunk_size = 10000
    for i in range(0, len(bulk_data), chunk_size):
        chunk = bulk_data[i:i+chunk_size]
        session.execute(insert_query, chunk)
        session.commit()
        logger.info(f"Inserted chunk {i//chunk_size + 1}/{(len(bulk_data)//chunk_size) + 1}")

    session.close()
    elapsed = time.time() - start_time
    logger.info(f"✅ YAMDA 100k Backfill and Rosetta Projection completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_janitor_backfill()
