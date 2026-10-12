"""split codes in the keyword index

Revision ID: a0454bfe1fe8
Revises: 62562f52f127
Create Date: 2026-10-11 18:55:31.276730

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a0454bfe1fe8'
down_revision: Union[str, Sequence[str], None] = '62562f52f127'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _rebuild(text_expression: str) -> None:
    # A generated column's expression cannot be altered in place: the column
    # goes and comes back, and Postgres recomputes every row as it is added.
    # Nothing is re-ingested, and no stage writes this column.
    op.drop_index("ix_child_chunks_tsv", table_name="child_chunks")
    op.execute("ALTER TABLE child_chunks DROP COLUMN tsv")
    op.execute(
        f"""
        ALTER TABLE child_chunks
        ADD COLUMN tsv tsvector
        GENERATED ALWAYS AS (
            to_tsvector(
                CASE WHEN language = 'en' THEN 'english'::regconfig
                     ELSE 'simple'::regconfig END,
                {text_expression}
            )
        ) STORED
        """
    )
    op.create_index("ix_child_chunks_tsv", "child_chunks", ["tsv"], postgresql_using="gin")


def upgrade() -> None:
    # Postgres' parser takes "168/2024/NĐ-CP" for a file path and stops at the
    # first non-ASCII letter: the index held '168/2024/n', 'đ-cp', 'đ', 'cp',
    # and a question saying "nghị định 168" found nothing. With "/" and "-"
    # turned into spaces the code is indexed as its parts. translate() is
    # IMMUTABLE, which a generated column requires. The search query applies
    # the same translate to the question (app/repositories/chunk_repository.py):
    # the two must change together.
    _rebuild("translate(content, '/-', '  ')")


def downgrade() -> None:
    _rebuild("content")
