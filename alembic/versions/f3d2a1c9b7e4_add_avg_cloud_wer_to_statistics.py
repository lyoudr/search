"""add avg cloud wer to statistics

Revision ID: f3d2a1c9b7e4
Revises: 9b8a02b2fe79
Create Date: 2026-02-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f3d2a1c9b7e4"
down_revision: Union[str, Sequence[str], None] = "9b8a02b2fe79"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("wer_statistics", sa.Column("avg_google_wer", sa.Float(), nullable=True))
    op.add_column("wer_statistics", sa.Column("avg_aws_wer", sa.Float(), nullable=True))
    op.add_column("wer_statistics", sa.Column("avg_dr_ai_wer", sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("wer_statistics", "avg_dr_ai_wer")
    op.drop_column("wer_statistics", "avg_aws_wer")
    op.drop_column("wer_statistics", "avg_google_wer")

