import pytest
from fastapi.testclient import TestClient
from main import app
from dependencies import get_repo, get_suggestion_service
from unittest.mock import MagicMock

@pytest.fixture
def mock_repo():
    return MagicMock()

@pytest.fixture
def mock_service():
    return MagicMock()

def test_discover_endpoint_success(mock_repo, mock_service):
    # Setup mock return value
    mock_service.search_semantic.return_value = [
        {
            "video_id": "vid123",
            "title": "Discovery Song",
            "artist": "Artist X",
            "genre": "Rock",
            "score": 0.99
        }
    ]
    
    # Override dependencies
    app.dependency_overrides[get_repo] = lambda: mock_repo
    app.dependency_overrides[get_suggestion_service] = lambda: mock_service
    
    client = TestClient(app)
    response = client.post("/discover", json={"query": "rock music", "limit": 5})
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "rock music"
    assert len(data["results"]) == 1
    assert data["results"][0]["youtube_video_id"] == "vid123"
    
    # Cleanup
    app.dependency_overrides.clear()

def test_discover_endpoint_empty(mock_repo, mock_service):
    mock_service.search_semantic.return_value = []
    
    app.dependency_overrides[get_repo] = lambda: mock_repo
    app.dependency_overrides[get_suggestion_service] = lambda: mock_service
    
    client = TestClient(app)
    response = client.post("/discover", json={"query": "nonexistent", "limit": 5})
    
    assert response.status_code == 200
    assert len(response.json()["results"]) == 0
    
    app.dependency_overrides.clear()
