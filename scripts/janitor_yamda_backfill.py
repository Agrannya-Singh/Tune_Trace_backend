import os
import sys
import logging
import time
from urllib.parse import urlparse
import numpy as np
import psycopg2
import psycopg2.extras

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

from typing import Literal

class YambdaDataset:
    INTERACTIONS = frozenset([
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ])

    def __init__(
        self,
        dataset_type: Literal["flat", "sequential"] = "flat",
        dataset_size: Literal["50m", "500m", "5b"] = "50m"
    ):
        assert dataset_type in {"flat", "sequential"}
        assert dataset_size in {"50m", "500m", "5b"}
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

    def interaction(self, event_type: Literal[
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ]):
        assert event_type in YambdaDataset.INTERACTIONS
        return self._download(f"{self.dataset_type}/{self.dataset_size}", event_type)

    def audio_embeddings(self):
        return self._download("", "embeddings")

    def album_item_mapping(self):
        return self._download("", "album_item_mapping")

    def artist_item_mapping(self):
        return self._download("", "artist_item_mapping")

    @staticmethod
    def _download(data_dir: str, file: str):
        from datasets import load_dataset
        # We explicitly set streaming=True so GitHub Actions doesn't download the massive 50GB file to disk
        data = load_dataset("yandex/yambda", data_dir=data_dir, data_files=f"{file}.parquet", streaming=True)
        return data["train"]

def run_janitor_backfill():
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable is missing. Aborting.")
        return

    logger.info(f"Starting YAMDA Janitor Backfill for {SAMPLE_SIZE} tracks...")
    start_time = time.time()

    # 1. Download the Real 100k YAMDA Subset via HuggingFace
    logger.info("Downloading actual Yambda-50M variant dataset from HuggingFace...")
    try:
        yambda_loader = YambdaDataset("flat", "50m")
        embeddings_dataset = yambda_loader.audio_embeddings()
        
        mock_video_ids = []
        mock_titles = []
        mock_artists = []
        yamda_native_embeddings_list = []
        
        # We also need the original text representations to compute the MiniLM targets
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        logger.info(f"Extracting first {SAMPLE_SIZE} records from Yambda dataset...")
        count = 0
        for row in embeddings_dataset:
            if count >= SAMPLE_SIZE:
                break
            
            item_id = str(row.get("item_id", row.get("item", f"yamda_id_{count}")))
            mock_video_ids.append(item_id)
            mock_titles.append(f"Yambda Track {item_id}")
            mock_artists.append(f"Yambda Artist")
            
            # Dynamically find the embedding column, prioritizing normalized_embed
            vec_col = next((col for col in ["normalized_embed", "embed", "embedding", "features", "audio_embedding", "vector"] if col in row), None)
            if not vec_col:
                raise ValueError(f"Could not find vector column in row. Keys found: {list(row.keys())}")
                
            yamda_native_embeddings_list.append(row[vec_col]) 
            count += 1
            
        yamda_native_embeddings = np.array(yamda_native_embeddings_list, dtype=np.float32)
        # We ONLY need MiniLM target embeddings for the Rosetta Stone training subset (e.g., 5000)
        # NOT the entire 100k dataset. The rest will just be projected by the trained model.
        training_subset_size = 5000
        text_contexts = [f"{mock_titles[i]} by {mock_artists[i]}" for i in range(training_subset_size)]
        
        logger.info(f"Computing MiniLM target embeddings for only the {training_subset_size} training tracks...")
        minilm_embeddings = encoder.encode(text_contexts, show_progress_bar=False, normalize_embeddings=True)
        
    except Exception as e:
        logger.error(f"Failed to load Yambda dataset: {e}. Ensure 'datasets' is installed.")
        sys.exit(1)

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

    # 4. Ingest into Supabase using psycopg2 directly with execute_values
    # This is far faster than SQLAlchemy for bulk inserts — single network roundtrip per chunk.
    logger.info("Connecting to Supabase PostgreSQL database...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    logger.info("Preparing bulk insert tuples...")
    # Build list of tuples matching the INSERT column order
    bulk_tuples = [
        (
            mock_video_ids[i],
            mock_titles[i],
            mock_artists[i],
            "YAMDA Mixed",
            "multimodal, audio-mapped",
            "rosetta",
            "[" + ",".join(f"{float(x):.6f}" for x in final_shared_embeddings[i]) + "]",
        )
        for i in range(len(mock_video_ids))
    ]

    INSERT_SQL = """
        INSERT INTO public.song_metadata (video_id, title, artist, genre, tags, enriched, embedding)
        VALUES %s
        ON CONFLICT (video_id) DO NOTHING;
    """

    chunk_size = 1000
    total_chunks = (len(bulk_tuples) + chunk_size - 1) // chunk_size
    logger.info(f"Executing bulk insert of {len(bulk_tuples)} records in {total_chunks} chunks of {chunk_size}...")

    for i in range(0, len(bulk_tuples), chunk_size):
        chunk = bulk_tuples[i:i + chunk_size]
        psycopg2.extras.execute_values(cur, INSERT_SQL, chunk, page_size=chunk_size)
        conn.commit()
        logger.info(f"Inserted chunk {i // chunk_size + 1}/{total_chunks}")

    cur.close()
    conn.close()
    elapsed = time.time() - start_time
    logger.info(f"✅ YAMDA Backfill and Rosetta Projection completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_janitor_backfill()
