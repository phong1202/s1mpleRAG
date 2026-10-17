"""users and conversation owners

Revision ID: a2c10a08e2a2
Revises: 351b23ad1fd7
Create Date: 2026-10-17 14:49:55.438502

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a2c10a08e2a2'
down_revision: Union[str, Sequence[str], None] = '351b23ad1fd7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


ADMIN_ID = "00000000-0000-4000-8000-000000000001"  # app/auth.py: ADMIN_ID


def upgrade() -> None:
    # Written by hand: autogenerate also proposes dropping the HNSW index and
    # the tsv column, which the models do not map on purpose.
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("username", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )
    # Every request is this user until login exists (AUTH_MODE=auto_admin).
    op.execute(
        f"INSERT INTO users (id, username, display_name, is_admin) "
        f"VALUES ('{ADMIN_ID}', 'admin', 'Admin', true)"
    )

    # Owners: added nullable, backfilled to the admin, then required.
    op.add_column("conversations", sa.Column("user_id", sa.UUID(), nullable=True))
    op.execute(f"UPDATE conversations SET user_id = '{ADMIN_ID}'")
    op.alter_column("conversations", "user_id", nullable=False)
    op.create_foreign_key(
        "fk_conversations_user_id", "conversations", "users", ["user_id"], ["id"], ondelete="CASCADE"
    )
    op.add_column("conversations", sa.Column("title", sa.Text(), nullable=True))
    op.add_column(
        "conversations",
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    # Older conversations: titled by their first question, dated by their last message.
    op.execute(
        """
        UPDATE conversations c
        SET    title = (SELECT left(m.text, 120) FROM messages m
                        WHERE m.conversation_id = c.id AND m.role = 'user' ORDER BY m.id LIMIT 1),
               updated_at = coalesce((SELECT max(m.created_at) FROM messages m
                                      WHERE m.conversation_id = c.id), c.created_at)
        """
    )
    op.create_index("ix_conversations_user_id_updated_at", "conversations", ["user_id", "updated_at"])
    op.add_column("messages", sa.Column("citations", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "citations")
    op.drop_index("ix_conversations_user_id_updated_at", table_name="conversations")
    op.drop_column("conversations", "updated_at")
    op.drop_column("conversations", "title")
    op.drop_constraint("fk_conversations_user_id", "conversations", type_="foreignkey")
    op.drop_column("conversations", "user_id")
    op.drop_table("users")
