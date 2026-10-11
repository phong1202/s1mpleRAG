"""query traces

Revision ID: 62562f52f127
Revises: 4af5845c2429
Create Date: 2026-10-10 19:17:00.266121

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '62562f52f127'
down_revision: Union[str, Sequence[str], None] = '4af5845c2429'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Autogenerate also proposed dropping the HNSW index, child_chunks.tsv and
    # the indexes and check constraint the database owns but the models do not
    # map. Those ops were removed by hand: this revision adds one table.
    op.create_table('query_traces',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('question', sa.Text(), nullable=False),
    sa.Column('standalone', sa.Text(), nullable=True),
    sa.Column('strategy', sa.Text(), nullable=True),
    sa.Column('iterations', sa.Integer(), nullable=False),
    sa.Column('total_ms', sa.Integer(), nullable=False),
    sa.Column('total_tokens', sa.Integer(), nullable=True),
    sa.Column('citation_count', sa.Integer(), nullable=False),
    sa.Column('nodes', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('query_traces')
