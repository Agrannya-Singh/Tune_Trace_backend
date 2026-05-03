import os
import sys
from sentence_transformers import SentenceTransformer

def download_model(model_name: str, output_path: str):
    """Download and save a SentenceTransformer model locally."""
    print(f"Downloading model '{model_name}' to '{output_path}'...")
    model = SentenceTransformer(model_name)
    model.save(output_path)
    print(f"Model saved successfully to {output_path}")

if __name__ == "__main__":
    # Default to the project's model and local models directory
    model_name = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
    output_dir = os.path.join(os.getcwd(), "models", model_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        download_model(model_name, output_dir)
    else:
        print(f"Model already exists at {output_dir}. Skipping download.")
