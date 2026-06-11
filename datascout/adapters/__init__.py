"""
datascout.adapters
─────────────────────────────────────────────────────
Public surface for the adapters package.
"""

from .base import BaseAdapter, AdapterHealth
from .kaggle_adapter import KaggleAdapter
from .huggingface_adapter import HuggingFaceAdapter
from .openml_adapter import OpenMLAdapter
from .registry import AdapterRegistry, adapter_registry

__all__ = [
    "BaseAdapter",
    "AdapterHealth",
    "KaggleAdapter",
    "HuggingFaceAdapter",
    "OpenMLAdapter",
    "AdapterRegistry",
    "adapter_registry",
]