import pytest
from fastapi.testclient import TestClient
from main import app
from dependencies import get_suggestion_service, get_repo
from auth import get_current_user
from unittest.mock import MagicMock, AsyncMock, patch

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status") == "healthy"

@patch("routers.suggestions.YOUTUBE_API_KEY", "fake_key")
def test_suggestions_valid_payload(client):
    # Mocking the service
    mock_service = MagicMock()
    # _search_youtube_for_song_async is async
    mock_service._search_youtube_for_song_async = AsyncMock(return_value={
        "video_id": "vid1", "title": "S1", "artist": "A1"
    })
    mock_service.get_recommendations.return_value = [
        {"video_id": "v1", "title": "S1", "artist": "A1", "genre": "G1", "tags": "T1", "score": 0.9}
    ]
    
    mock_repo = MagicMock()
    # Mocking repository methods called in suggestions.py
    mock_repo.get_song_metadata_by_video_id.return_value = None
    mock_song = MagicMock()
    mock_song.id = 1
    mock_repo.create_song_metadata.return_value = mock_song
    mock_repo.get_or_create_user.return_value = MagicMock(user_id="test_user")
    mock_repo.get_user_liked_songs_objects.return_value = []

    app.dependency_overrides[get_suggestion_service] = lambda: mock_service
    app.dependency_overrides[get_repo] = lambda: mock_repo
    app.dependency_overrides[get_current_user] = lambda: {"email": "test_user", "uid": "test_uid"}
    
    payload = {
        "user_id": "test_user",
        "songs": ["Song 1"],
        "genre": "Pop"
    }
    response = client.post("/suggestions", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "suggestions" in data
    assert len(data["suggestions"]) == 1
    assert data["suggestions"][0]["youtube_video_id"] == "v1"
    
    app.dependency_overrides.clear()

def test_suggestions_invalid_payload(client):
    # Empty songs list should trigger 422
    response = client.post("/suggestions", json={"user_id": "test_user", "songs": []})
    assert response.status_code == 422
