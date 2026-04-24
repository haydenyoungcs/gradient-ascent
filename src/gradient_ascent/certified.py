"""Compatibility shim for certified unlearning.

The implementation now lives under ``gradient_ascent.unlearning.certified`` so
all unlearning baselines share one package namespace. This module re-exports
the old symbols to avoid breaking existing imports.
"""

from .unlearning.certified import CertifiedConfig, run_certified_snapshots, run_certified_unlearning

__all__ = ["CertifiedConfig", "run_certified_unlearning", "run_certified_snapshots"]
