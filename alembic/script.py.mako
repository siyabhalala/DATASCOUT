"""${message}

Revision ID: ${up_revision}
Revises:     ${down_revision | comma,n}
Create Date: ${create_date}

"""
from __future__ import annotations

from typing import Sequence, Union

import alembic.op as op
import sqlalchemy as sa
${imports if imports else ""}

# ── Alembic revision identifiers (used by Alembic internally) ─────────────────
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    """Apply the forward migration.

    Place all schema-changing operations here.  Use ``op.create_table()``,
    ``op.add_column()``, ``op.drop_column()``, ``op.create_index()``, etc.
    This function is called when running ``alembic upgrade``.
    """
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """Reverse the migration performed by :func:`upgrade`.

    Place all rollback operations here.  This function is called when running
    ``alembic downgrade``.  Every ``op.*`` call in :func:`upgrade` should have
    a corresponding inverse here.
    """
    ${downgrades if downgrades else "pass"}
