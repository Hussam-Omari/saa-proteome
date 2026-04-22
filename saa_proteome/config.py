"""
Library-wide configuration constants and amino-acid reference profiles.

This module centralizes definitions used throughout the package,
including canonical amino-acid ordering, reference essential
amino-acid (EAA) profiles for reporting, and default parameters
for sliding-window analyses.

All definitions provided here represent analytical conventions
used for reproducible computation and comparison. They do not
imply universal biological essentiality and may be adapted
depending on organism or research context.
"""

from __future__ import annotations

# Canonical 20 standard amino acids (one-letter codes).
# The order is fixed to ensure reproducible column ordering in tables and plots.
CANONICAL_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
CANONICAL_AA_SET = set(CANONICAL_AA_ORDER)

# Reference essential amino-acid (EAA) profiles.
# Essentiality depends on organism, life stage, and physiological context.
# These profiles are provided for analytical comparison and reporting.
EAA_PROFILES = {
    "human_classic": set("HILKMFTWV"),
    "human_plus_arg": set("HILKMFTWVR"),
}

# Default sliding-window parameters for residue-enrichment analyses.
# These serve as methodological defaults and can be overridden by users.
DEFAULT_WINDOW_SIZE = 100
DEFAULT_WINDOW_STEP = 1