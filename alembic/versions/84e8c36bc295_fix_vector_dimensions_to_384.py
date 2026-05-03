"""fix_vector_dimensions_to_384

Revision ID: 84e8c36bc295
Revises: b2c3d4e5f6g7
Create Date: 2026-05-02 19:37:29.769642

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '84e8c36bc295'
down_revision: Union[str, Sequence[str], None] = '123456789abc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER TABLE song_metadata ALTER COLUMN embedding TYPE vector(384);")
    op.execute("DROP INDEX IF EXISTS idx_song_embedding_hnsw;")
    op.execute("CREATE INDEX idx_song_embedding_hnsw ON song_metadata USING hnsw (embedding vector_cosine_ops);")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS idx_song_embedding_hnsw;")
    op.execute("ALTER TABLE song_metadata ALTER COLUMN embedding TYPE vector(768);")
    op.execute("CREATE INDEX idx_song_embedding_hnsw ON song_metadata USING hnsw (embedding vector_cosine_ops);")
