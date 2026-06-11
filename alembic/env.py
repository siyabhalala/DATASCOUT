"""
Alembic environment configuration for DataScout.

This file is invoked by Alembic when running ``alembic upgrade``,
``alembic downgrade``, or ``alembic revision``.  It wires the SQLAlchemy
``Base.metadata`` to Alembic's autogenerate machinery so that migration
scripts can be generated automatically from model changes.

Async note
──────────
DataScout uses ``aiosqlite`` / ``asyncpg`` in production, but Alembic's
migration runner is synchronous.  We strip the ``+aiosqlite`` / ``+asyncpg``
dialect suffix from the connection URL when running migrations so that
Alembic uses the standard sync driver.
"""

from __future__ import annotations

import os
import re
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Target metadata ───────────────────────────────────────────────────────────
# Import the declarative Base from contracts so Alembic can inspect all models.
# Degrade gracefully if the import fails so that alembic commands still work
# even in a partially-initialised environment.
try:
    from datascout.contracts.models import Base  # type: ignore

    target_metadata = Base.metadata
except Exception:  # pragma: no cover
    target_metadata = None  # type: ignore

# ── Alembic Config object ─────────────────────────────────────────────────────
config = context.config

# Interpret the config file's logging settings.
if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except Exception:  # pragma: no cover
        pass

# ── Resolve database URL ──────────────────────────────────────────────────────
# Priority:
#   1. URL injected by alembic.ini (via --config or programmatic use)
#   2. DATABASE_URL environment variable
#   3. Default SQLite path
_db_url: str = (
    config.get_main_option("sqlalchemy.url")
    or os.environ.get("DATABASE_URL", "")
    or "sqlite:///./datascout.db"
)

# Strip async dialect suffixes — Alembic needs the sync driver.
_db_url = re.sub(r"\+aiosqlite", "", _db_url, flags=re.IGNORECASE)
_db_url = re.sub(r"\+asyncpg", "", _db_url, flags=re.IGNORECASE)
_db_url = re.sub(r"\+aiomysql", "", _db_url, flags=re.IGNORECASE)

config.set_main_option("sqlalchemy.url", _db_url)


# ── Migration helpers ─────────────────────────────────────────────────────────


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    In offline mode Alembic does not need a live database connection; it
    generates SQL scripts that can be reviewed and applied manually.  This is
    useful in CI pipelines and environments where the database is not
    accessible at migration-generation time.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In online mode Alembic opens a real connection, begins a transaction, and
    applies the pending migrations directly.  This is the mode used by the
    ``alembic upgrade head`` command in production deployments.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section) or {},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


# ── Entry point ───────────────────────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()