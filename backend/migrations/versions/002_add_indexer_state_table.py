"""add_indexer_state_table

Revision ID: 002_add_indexer_state_table
Revises: e5c7a29c8f46
Create Date: 2026-07-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002_add_indexer_state_table'
down_revision: Union[str, None] = 'e5c7a29c8f46'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create indexer_state table for persistent indexer block tracking
    op.create_table(
        'indexer_state',
        sa.Column('key', sa.String(length=100), primary_key=True, nullable=False),
        sa.Column('value', sa.Integer(), nullable=False),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            onupdate=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
    )

    # 2. Seed initial state record with default block 0
    # The indexer will overwrite this upon initialization
    indexer_state_table = sa.table(
        'indexer_state',
        sa.column('key', sa.String),
        sa.column('value', sa.Integer),
    )

    op.bulk_insert(
        indexer_state_table,
        [
            {
                'key': 'escrow_last_processed_block',
                'value': 0,  # Replace with ESCROW_START_BLOCK for production
            }
        ],
    )


def downgrade() -> None:
    op.drop_table('indexer_state')
