from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "TuneTrace API" in response.json().get("message", "")

def test_suggestions_empty_payload():
    # Test with empty payload to see handling
    response = client.post("/suggestions", json={})
    # Depending on model validation, this might be 422
    assert response.status_code in [422, 500, 400]

def test_suggestions_valid_payload(mocker):
    # Mocking MLEngine or services might be needed for a pure unit test, 
    # but here we test the endpoint integration.
    # We would need to mock the Supabase/DB part in services.py
    pass
