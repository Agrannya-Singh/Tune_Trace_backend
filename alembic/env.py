# alembic/env.py

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from dotenv import load_dotenv

from alembic import context

# --- SETUP ---
# This makes sure Alembic can find your models in the 'db.py' file.
# We add the parent directory (your project root) to the Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db import Base

# Load environment variables from a .env file for local development.
load_dotenv()

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# --- CENTRAL DATABASE CONFIGURATION ---
# This is the single source of truth for the database URL.
# It prioritizes PostgreSQL and falls back to SQLite if the URL is not set.

DATABASE_URL = os.getenv("POSTGRES_DATABASE_URL")
if DATABASE_URL is None:
    # This is used for local development when running `alembic` commands
    # without a production database environment variable.
    print("INFO: POSTGRES_DATABASE_URL not found, Alembic is using local SQLite database.")
    DATABASE_URL = "sqlite:///./app.db"

# Set the URL in Alembic's config so the rest of the script can use it.
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# --- METADATA SETUP ---
# Point Alembic to your models' metadata so it can detect changes.
target_metadata = Base.metadata

# ==============================================================================

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.
    This generates SQL scripts without connecting to a database.
    """
    # This now correctly uses the URL determined in the central config section.
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.
    This connects to the database to apply migrations.
    """
    # The 'engine_from_config' function will automatically use the
    # 'sqlalchemy.url' we set in the central config section.
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


# This determines whether to run in online or offline mode.
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()