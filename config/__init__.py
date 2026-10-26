"""Medica pipeline configuration: Kaggle-aware path resolution.

Imports are lazy so that `python3 -m config.paths` runs the module as a script
without importing it twice. Use `from config.paths import ...` directly.
"""

__all__ = ["Paths", "check", "resolve"]
