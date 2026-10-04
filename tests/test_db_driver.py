import sys
from unittest.mock import patch
import pytest
from sqlalchemy import create_engine
from db import normalize_database_url


class TestDatabaseDriverResolution:
    """Test suite for PostgreSQL DBAPI driver dialect normalization and resilient fallbacks."""

    def test_empty_or_none_url(self):
        assert normalize_database_url(None) is None
        assert normalize_database_url("") == ""

    def test_sqlite_url_unchanged(self):
        url = "sqlite:///./app.db"
        assert normalize_database_url(url) == url

    def test_legacy_postgres_prefix_rewritten(self):
        url = "postgres://user:pass@localhost:5432/tunetrace"
        normalized = normalize_database_url(url)
        assert not normalized.startswith("postgres://")
        assert normalized.startswith("postgresql")

    def test_psycopg3_dialect_resolution(self):
        url = "postgresql+psycopg://user:pass@localhost:5432/tunetrace"
        normalized = normalize_database_url(url)
        # Should stay psycopg (if installed) or fallback to psycopg2
        assert normalized in (
            "postgresql+psycopg://user:pass@localhost:5432/tunetrace",
            "postgresql+psycopg2://user:pass@localhost:5432/tunetrace",
        )
        engine = create_engine(normalized)
        assert engine.dialect.name == "postgresql"
        assert engine.dialect.driver in ("psycopg", "psycopg2")

    def test_psycopg2_dialect_resolution(self):
        url = "postgresql+psycopg2://user:pass@localhost:5432/tunetrace"
        normalized = normalize_database_url(url)
        assert normalized in (
            "postgresql+psycopg://user:pass@localhost:5432/tunetrace",
            "postgresql+psycopg2://user:pass@localhost:5432/tunetrace",
        )
        engine = create_engine(normalized)
        assert engine.dialect.name == "postgresql"
        assert engine.dialect.driver in ("psycopg", "psycopg2")

    def test_default_postgresql_dialect_resolution(self):
        url = "postgresql://user:pass@localhost:5432/tunetrace"
        normalized = normalize_database_url(url)
        engine = create_engine(normalized)
        assert engine.dialect.name == "postgresql"
        assert engine.dialect.driver in ("psycopg", "psycopg2")

    def test_fallback_when_psycopg3_missing(self):
        """Simulate environment where psycopg (v3) is missing but psycopg2 is present."""
        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "psycopg":
                raise ImportError("No module named 'psycopg'")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            url = "postgresql+psycopg://user:pass@localhost:5432/tunetrace"
            normalized = normalize_database_url(url)
            assert normalized == "postgresql+psycopg2://user:pass@localhost:5432/tunetrace"
            engine = create_engine(normalized)
            assert engine.dialect.driver == "psycopg2"

    def test_fallback_when_psycopg2_missing(self):
        """Simulate environment where psycopg2 is missing but psycopg (v3) is present."""
        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "psycopg2":
                raise ImportError("No module named 'psycopg2'")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            url = "postgresql+psycopg2://user:pass@localhost:5432/tunetrace"
            normalized = normalize_database_url(url)
            assert normalized == "postgresql+psycopg://user:pass@localhost:5432/tunetrace"
            engine = create_engine(normalized)
            assert engine.dialect.driver == "psycopg"
