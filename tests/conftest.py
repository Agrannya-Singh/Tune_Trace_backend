import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from models import Base
from db import get_session
from main import app
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Globally mock MLEngine and SessionLocal check to prevent heavy loading/IO
@pytest.fixture(autouse=True)
def mock_global_dependencies():
    with patch("engine.ml_engine") as mock_ml, \
         patch("main.SessionLocal") as mock_db:
        mock_ml.load_model = MagicMock()
        mock_ml.model = MagicMock()
        
        # Mock the context manager for SessionLocal
        mock_session = MagicMock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        yield mock_ml

# Mock pgvector for SQLite tests if needed, or just use a standard type
# Since we are using SQLite for unit testing, we override the Vector type
# in the models specifically for tests if necessary, but usually, 
# for unit tests of logic, we mock the repository.

@pytest.fixture(scope="session")
def test_db():
    # Use in-memory SQLite for tests
    SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # We might need to handle the 'Vector' type in models.py 
    # if we were to call Base.metadata.create_all(bind=engine)
    # But since we want to test ML logic/services, we will mostly mock repo.
    
    # Base.metadata.create_all(bind=engine) # This would fail due to Vector
    
    return TestingSessionLocal

@pytest.fixture(scope="function")
def db_session(test_db):
    session = test_db()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
