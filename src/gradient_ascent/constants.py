"""Shared experiment constants (avoid circular imports between pipeline modules)."""

ALGORITHM_ORDER: tuple[str, ...] = ("ga", "ssd", "salun", "certified", "scrub")
SIMILARITY_REFERENCES: tuple[str, ...] = ("retrained", "original")
