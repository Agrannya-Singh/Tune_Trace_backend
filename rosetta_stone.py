import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from typing import Union

logger = logging.getLogger(__name__)

# Production hardware detection
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

class YamdaDataset(Dataset):
    """
    Dataset loader for the YAMDA (limited model dataset).
    Contains pairs of (all-MiniLM-L6-v2 embeddings, native YAMDA embeddings).
    """
    def __init__(self, minilm_embeddings: np.ndarray, yamda_embeddings: np.ndarray):
        """
        Args:
            minilm_embeddings (np.ndarray): The 384-d embeddings from all-MiniLM-L6-v2.
            yamda_embeddings (np.ndarray): The native embeddings from the YAMDA dataset.
        """
        assert len(minilm_embeddings) == len(yamda_embeddings), "Embedding counts must match."
        self.minilm = torch.tensor(minilm_embeddings, dtype=torch.float32)
        self.yamda = torch.tensor(yamda_embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.minilm)

    def __getitem__(self, idx):
        return self.minilm[idx], self.yamda[idx]

class RosettaStoneMapper(nn.Module):
    """
    Rosetta Stone Architecture:
    A neural mapping network that projects embeddings from the text space
    (all-MiniLM-L6-v2) to the multimodal/audio space (native YAMDA embeddings).
    """
    def __init__(self, input_dim: int = 384, output_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project MiniLM embeddings into YAMDA space.
        Optionally L2-normalize if the target space requires cosine similarity.
        """
        # Ensure input has batch dimension even for single vector inference
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = self.network(x)
        # L2 normalization helps align directional representations like embeddings
        out = nn.functional.normalize(out, p=2, dim=1)
        return out

    def save_weights(self, filepath: str) -> None:
        """Serialize model weights to disk."""
        torch.save(self.state_dict(), filepath)
        logger.info(f"RosettaStone weights saved successfully to {filepath}")

    def load_weights(self, filepath: str) -> None:
        """Load serialized model weights from disk."""
        self.load_state_dict(torch.load(filepath, map_location=DEVICE))
        logger.info(f"RosettaStone weights loaded successfully from {filepath}")

    def project_embedding(self, minilm_vector: np.ndarray) -> np.ndarray:
        """
        Production-grade inference wrapper. Converts NumPy input to PyTorch,
        executes the forward pass on the correct device, and returns a NumPy array.
        
        Args:
            minilm_vector (np.ndarray): 1D or 2D array of MiniLM embeddings.
            
        Returns:
            np.ndarray: Projected YAMDA space embeddings.
        """
        self.eval()
        is_1d = minilm_vector.ndim == 1
        tensor_in = torch.tensor(minilm_vector, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            tensor_out = self.forward(tensor_in)
            
        res = tensor_out.cpu().numpy()
        return res[0] if is_1d else res

def train_rosetta_stone(
    yamda_data: np.ndarray, 
    minilm_data: np.ndarray, 
    epochs: int = 20, 
    batch_size: int = 64, 
    lr: float = 1e-3,
    device: str = DEVICE
) -> RosettaStoneMapper:
    """
    Train the Rosetta Stone mapping architecture.
    """
    dataset = YamdaDataset(minilm_data, yamda_data)
    # drop_last=True prevents BatchNorm1d from crashing on single-item remainder batches
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    input_dim = yamda_data.shape[1]   # Audio (e.g. 512)
    output_dim = minilm_data.shape[1] # Text (384)
    
    model = RosettaStoneMapper(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # We use CosineEmbeddingLoss because we care about the angular mapping 
    # of the semantic space rather than raw euclidean distance.
    criterion = nn.CosineEmbeddingLoss()
    target_tensor = torch.ones(batch_size).to(device)  # 1 means 'make them similar'
    
    logger.info(f"Starting Rosetta Stone training mapping {input_dim}-d -> {output_dim}-d on device: {device}")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_minilm, batch_yamda in loader:
            batch_minilm = batch_minilm.to(device)
            batch_yamda = batch_yamda.to(device)
            
            # Handle last batch size just in case, though drop_last=True makes it constant
            current_batch_size = batch_minilm.size(0)
            batch_target = target_tensor[:current_batch_size]

            optimizer.zero_grad()
            # Predict the MiniLM (384) from the YAMDA (512)
            pred_minilm = model(batch_yamda)
            
            loss = criterion(pred_minilm, batch_minilm, batch_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    return model

if __name__ == "__main__":
    # Configure basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Generate mock 'limited model dataset' for the YAMDA integration test
    print("Initializing mock YAMDA dataset (limited model subset)...")
    num_samples = 1000
    
    # 1. Text space embeddings from all-MiniLM-L6-v2 (dim 384)
    mock_minilm = np.random.randn(num_samples, 384).astype(np.float32)
    # 2. Multimodal target space embeddings from YAMDA (assume dim 512 for example)
    mock_yamda = np.random.randn(num_samples, 512).astype(np.float32)
    
    # Normalize mocks to mimic real-world embeddings
    mock_minilm /= np.linalg.norm(mock_minilm, axis=1, keepdims=True)
    mock_yamda /= np.linalg.norm(mock_yamda, axis=1, keepdims=True)
    
    print(f"Training Rosetta Stone architecture on {DEVICE} for multimodal integration...")
    trained_model = train_rosetta_stone(mock_yamda, mock_minilm, epochs=5)
    
    print("Training complete! Testing save/load utility...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_name = tmp.name
    
    trained_model.save_weights(tmp_name)
    
    new_model = RosettaStoneMapper(input_dim=512, output_dim=384)
    new_model.load_weights(tmp_name)
    
    test_vector = np.random.randn(512).astype(np.float32)
    test_vector /= np.linalg.norm(test_vector)
    projected = new_model.project_embedding(test_vector)
    print(f"Projected output shape: {projected.shape}")
    print("Inference verification complete!")
