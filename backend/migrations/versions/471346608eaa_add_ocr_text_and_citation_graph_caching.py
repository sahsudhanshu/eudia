"""Add OCR text and citation graph caching

Revision ID: 471346608eaa
Revises: b6f7a16cbff2
Create Date: 2025-11-08 15:13:05.014131

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '471346608eaa'
down_revision = 'b6f7a16cbff2'
branch_labels = None
depends_on = None


def upgrade():
    # Add new columns to documents table
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.add_column(sa.Column('ocr_text', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('ocr_metadata', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('citation_graph', sa.JSON(), nullable=True))


def downgrade():
    # Remove columns from documents table
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_column('citation_graph')
        batch_op.drop_column('ocr_metadata')
        batch_op.drop_column('ocr_text')
