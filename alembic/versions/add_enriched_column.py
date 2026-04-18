"""Add enriched column to song_metadata

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-19 00:12:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add 'enriched' column to track metadata enrichment version."""
    op.add_column(
        'song_metadata',
        sa.Column(
            'enriched',
            sa.String(length=16),
            nullable=True,
            comment="Enrichment version. NULL=raw, 'V2'=genre/tags enriched, etc.",
        ),
    )
    op.create_index(
        op.f('ix_song_metadata_enriched'),
        'song_metadata',
        ['enriched'],
        unique=False,
    )


def downgrade() -> None:
    """Remove 'enriched' column."""
    op.drop_index(op.f('ix_song_metadata_enriched'), table_name='song_metadata')
    op.drop_column('song_metadata', 'enriched')
