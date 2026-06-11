"""DataScout entrypoint. Run with: uvicorn datascout.main:app"""
from datascout.api.main import app

__all__ = ["app"]
