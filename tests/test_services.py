import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from services import SuggestionService

@pytest.fixture
def mock_repo():
    return MagicMock()

@pytest.fixture
def mock_ml_engine():
    with patch("services.ml_engine") as mock:
        yield mock

@pytest.fixture
def service():
    return SuggestionService(api_key="test_key")

def test_get_recommendations_flow(service, mock_repo, mock_ml_engine):
    # Setup mock behaviors
    mock_ml_engine.compute_user_profile_vector.return_value = np.zeros(384)
    mock_ml_engine.vector_to_literal.return_value = "[0,0,...]"
    
    # Mock repository returning some rows
    row1 = MagicMock(video_id="v1", title="S1", artist="A1", genre="G1", tags="T1", enriched="V3", similarity=0.9)
    row2 = MagicMock(video_id="v2", title="S2", artist="A2", genre="G2", tags="T2", enriched="V3", similarity=0.8)
    mock_repo.get_semantic_recommendations.return_value = [row1, row2]
    
    # Mock diversity to just return what it got
    mock_ml_engine.apply_diversity.side_effect = lambda scored, top_n: scored[:top_n]
    
    user_history = [{"video_id": "vh1", "title": "History"}]
    
    recs = service.get_recommendations(user_history, mock_repo, top_n=2)
    
    assert len(recs) == 2
    assert recs[0]["video_id"] == "v1"
    assert recs[0]["score"] == 0.9
    mock_repo.get_semantic_recommendations.assert_called_once()

def test_get_recommendations_empty_history(service, mock_repo):
    recs = service.get_recommendations([], mock_repo)
    assert recs == []
    mock_repo.get_semantic_recommendations.assert_not_called()

def test_search_semantic_flow(service, mock_repo, mock_ml_engine):
    mock_ml_engine.encode_single.return_value = np.zeros(384)
    mock_ml_engine.vector_to_literal.return_value = "[0,0,...]"
    
    row = MagicMock(video_id="v1", title="Result", artist="Artist", genre="G", tags="T", similarity=0.95)
    mock_repo.semantic_search.return_value = [row]
    
    results = service.search_semantic("test query", mock_repo)
    
    assert len(results) == 1
    assert results[0]["title"] == "Result"
    assert results[0]["score"] == 0.95
    mock_repo.semantic_search.assert_called_once()
