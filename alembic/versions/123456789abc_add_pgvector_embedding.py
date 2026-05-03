"""add pgvector embedding

Revision ID: 123456789abc
Revises: add_user_oauth_fields
Create Date: 2026-05-02 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision = '123456789abc'
down_revision = 'b2c3d4e5f6g7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.add_column('song_metadata', sa.Column('embedding', pgvector.sqlalchemy.Vector(384), nullable=True))
    op.execute("CREATE INDEX IF NOT EXISTS idx_song_embedding_hnsw ON song_metadata USING hnsw (embedding vector_cosine_ops);")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_song_embedding_hnsw;")
    op.drop_column('song_metadata', 'embedding')
