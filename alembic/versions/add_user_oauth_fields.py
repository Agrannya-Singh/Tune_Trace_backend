"""Add user OAuth fields

Revision ID: a1b2c3d4e5f6
Revises: 999777df0f65
Create Date: 2025-09-30 19:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '999777df0f65'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add OAuth fields to users table."""
    # Add new columns
    op.add_column('users', sa.Column('name', sa.String(length=255), nullable=True, comment="User's display name from OAuth provider."))
    op.add_column('users', sa.Column('email', sa.String(length=255), nullable=True, comment="User's email address from OAuth provider."))
    op.add_column('users', sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()))
    
    # Alter user_id column to support longer OAuth identifiers (emails)
    op.alter_column('users', 'user_id',
                    existing_type=sa.String(length=128),
                    type_=sa.String(length=255),
                    existing_nullable=False,
                    comment='User email or unique identifier from OAuth (e.g., Google email).')
    
    # Create index on email field
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)


def downgrade() -> None:
    """Remove OAuth fields from users table."""
    op.drop_index(op.f('ix_users_email'), table_name='users')
    
    op.alter_column('users', 'user_id',
                    existing_type=sa.String(length=255),
                    type_=sa.String(length=128),
                    existing_nullable=False,
                    comment='The unique external identifier for the user.')
    
    op.drop_column('users', 'updated_at')
    op.drop_column('users', 'email')
    op.drop_column('users', 'name')
