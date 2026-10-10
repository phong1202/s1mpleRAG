"""retrieval readiness

Revision ID: 4af5845c2429
Revises: f0efc554ed36
Create Date: 2026-09-08 17:15:46.242531

Five nullable columns Phase 2 needs for keyword search, a bilingual corpus,
and document structure browsing. All nullable because nothing in Phase 1a
reads them, and S1/S2 (tasks 13/14) have not been written yet -- the stage
chain stays green whether they are filled or not.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4af5845c2429'
down_revision: Union[str, Sequence[str], None] = 'f0efc554ed36'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("documents", sa.Column("title", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("language", sa.Text(), nullable=True))
    op.add_column("parent_chunks", sa.Column("heading_path", sa.Text(), nullable=True))
    op.add_column("parent_chunks", sa.Column("language", sa.Text(), nullable=True))
    op.add_column("child_chunks", sa.Column("language", sa.Text(), nullable=True))

    # A GENERATED column, not a trigger: Postgres recomputes it whenever
    # `content` or `language` changes, so no stage can forget to.
    #
    # The two-argument to_tsvector(regconfig, text) is IMMUTABLE. The
    # one-argument form is only STABLE -- it reads
    # default_text_search_config at run time -- and Postgres refuses that in
    # a generated column. Passing the config explicitly is not a style
    # choice; it is what makes this column legal.
    op.execute(
        """
        ALTER TABLE child_chunks
        ADD COLUMN tsv tsvector
        GENERATED ALWAYS AS (
            to_tsvector(
                CASE WHEN language = 'en' THEN 'english'::regconfig
                     ELSE 'simple'::regconfig END,
                content
            )
        ) STORED
        """
    )
    op.create_index("ix_child_chunks_tsv", "child_chunks", ["tsv"], postgresql_using="gin")

    # `content`, NOT `contextualized`: the context sentence is the LLM's own
    # gloss. It belongs to the embedding, which searches by meaning. A
    # keyword index built on the gloss would return chunks for words the
    # document never wrote, with a citation pointing at a passage that does
    # not contain the word just searched for.
    op.execute(
        """
        CREATE INDEX ix_documents_title_fts ON documents USING gin (
            to_tsvector('simple', coalesce(title, '') || ' ' || filename)
        )
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_documents_title_fts")
    op.drop_index("ix_child_chunks_tsv", table_name="child_chunks")
    op.execute("ALTER TABLE child_chunks DROP COLUMN tsv")

    op.drop_column("child_chunks", "language")
    op.drop_column("parent_chunks", "language")
    op.drop_column("parent_chunks", "heading_path")
    op.drop_column("documents", "language")
    op.drop_column("documents", "title")
