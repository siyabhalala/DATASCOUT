"""datascout.api.routes — FastAPI route modules."""

# FIX: Previously this did 'from datascout.api.routes import admin, datasets, health'
# which is a circular self-import — this file IS datascout.api.routes.
# Routes are imported directly by api/main.py so this __init__ just needs to exist.

__all__ = ["admin", "datasets", "health"]