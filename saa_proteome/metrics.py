"""
Single-sequence amino-acid metrics.

This module defines per-protein feature functions that operate on a single input
sequence (string) and return Python scalars and dictionaries (no DataFrames).
Batch and proteome-level wrappers that aggregate across many sequences are
implemented in `proteome.py`.

Terminology and scale conventions
---------------------------------
Residue (amino acid): one-letter amino-acid code.
freq: proportion in the adjusted sequence (0—1).
pct: percentage in the adjusted sequence (0—100).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence, Set, Union

import numpy as np

from .sequence import clean_sequence, remove_nterm_m
from .config import (
    CANONICAL_AA_ORDER,
    CANONICAL_AA_SET,
    EAA_PROFILES,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_STEP,
)


def aa_profile(
    seq: str,
    remove_start_m: bool = True,
    canonical_only: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-residue counts and composition metrics for canonical amino acids.

    The input is first cleaned using `clean_sequence()` (optionally filtering to
    canonical residues). If `remove_start_m=True`, an N-terminal methionine is
    removed from the cleaned sequence to produce an adjusted sequence.

    Returns
    -------
    dict
        Nested dictionary containing:
        - count: per-residue counts (float) on the adjusted sequence
        - freq: per-residue frequencies (0—1) on the adjusted sequence
        - pct: per-residue percentages (0—100) on the adjusted sequence
        - length: aa_count_cleaned and aa_count_adjusted
        - flags: start_m_removed
    """
    cleaned = clean_sequence(seq, canonical_only=canonical_only)
    adjusted, removed = remove_nterm_m(cleaned, enabled=remove_start_m)

    n_cleaned = len(cleaned)
    n_adj = len(adjusted)

    counts = Counter(adjusted) if n_adj > 0 else Counter()

    # Ensure all canonical amino acids exist as keys
    counts_full = {aa: float(counts.get(aa, 0)) for aa in CANONICAL_AA_ORDER}

    if n_adj == 0:
        freq_full = {aa: np.nan for aa in CANONICAL_AA_ORDER}
        pct_full = {aa: np.nan for aa in CANONICAL_AA_ORDER}
    else:
        freq_full = {aa: counts_full[aa] / n_adj for aa in CANONICAL_AA_ORDER}
        pct_full = {aa: (counts_full[aa] / n_adj) * 100.0 for aa in CANONICAL_AA_ORDER}

    return {
        "count": counts_full,
        "freq": freq_full,
        "pct": pct_full,
        "length": {"aa_count_cleaned": n_cleaned, "aa_count_adjusted": n_adj},
        "flags": {"start_m_removed": bool(removed)},
    }


def group_stats(
    seq: str,
    group: Union[str, Sequence[str], Set[str]],
    remove_start_m: bool = True,
    canonical_only: bool = True,
) -> Dict[str, float]:
    """
    Compute composition statistics for a user-defined residue group.

    The sequence is cleaned and optionally N-terminal methionine is removed,
    then group composition is computed on the adjusted sequence.

    Parameters
    ----------
    group : str or sequence of str
        Either a string such as "MC" (interpreted as {'M','C'}) or an iterable
        of one-letter residue codes.

    Returns
    -------
    dict
        - group_count, group_freq (0—1), group_pct (0—100)
        - aa_count_adjusted
        - start_m_removed
    """
    if isinstance(group, str):
        residues = set(list(group.upper()))
    else:
        residues = set([str(x).upper() for x in group])

    # # Validate that all requested residues are canonical one-letter amino acids
    bad = sorted([r for r in residues if r not in CANONICAL_AA_SET])
    if bad:
        raise ValueError(f"Non-canonical residues in group: {bad}. Allowed: {sorted(CANONICAL_AA_SET)}")

    cleaned = clean_sequence(seq, canonical_only=canonical_only)
    adjusted, removed = remove_nterm_m(cleaned, enabled=remove_start_m)
    n = len(adjusted)

    if n == 0:
        return {
            "group_count": np.nan,
            "group_freq": np.nan,
            "group_pct": np.nan,
            "aa_count_adjusted": 0,
            "start_m_removed": bool(removed),
        }

    c = Counter(adjusted)
    group_count = float(sum(c.get(r, 0) for r in residues))
    return {
        "group_count": group_count,
        "group_freq": group_count / n,
        "group_pct": (group_count / n) * 100.0,
        "aa_count_adjusted": n,
        "start_m_removed": bool(removed),
    }


def essential_aa_set(profile: str = "human_classic") -> Set[str]:
    """
    Return a reference essential-amino-acid (EAA) residue set for a named profile.

    The available profiles are defined in `config.EAA_PROFILES`.
    """
    if profile not in EAA_PROFILES:
        raise ValueError(f"Unknown EAA profile '{profile}'. Available: {sorted(EAA_PROFILES.keys())}")
    return set(EAA_PROFILES[profile])


def essential_aa_stats(
    seq: str,
    profile: str = "human_classic",
    remove_start_m: bool = True,
    canonical_only: bool = True,
    include_per_aa: bool = True,
) -> Dict[str, float]:
    """
    Compute reference essential amino acid (EAA) composition metrics for one sequence.

    Returns
    -------
    dict
        Always includes:
        - eaa_count, eaa_freq (0—1), eaa_pct (0—100)
        - aa_count_adjusted, start_m_removed, eaa_profile

        If `include_per_aa=True`, also includes per-residue EAA percentages as keys:
        - eaa_pct_H, eaa_pct_I, ...
    """
    eaa = essential_aa_set(profile=profile)

    cleaned = clean_sequence(seq, canonical_only=canonical_only)
    adjusted, removed = remove_nterm_m(cleaned, enabled=remove_start_m)
    n = len(adjusted)

    if n == 0:
        out = {
            "eaa_count": np.nan,
            "eaa_freq": np.nan,
            "eaa_pct": np.nan,
            "aa_count_adjusted": 0,
            "start_m_removed": bool(removed),
            "eaa_profile": profile,
        }
        if include_per_aa:
            for aa in sorted(eaa):
                out[f"eaa_pct_{aa}"] = np.nan
        return out

    c = Counter(adjusted)
    eaa_count = float(sum(c.get(aa, 0) for aa in eaa))
    out = {
        "eaa_count": eaa_count,
        "eaa_freq": eaa_count / n,
        "eaa_pct": (eaa_count / n) * 100.0,
        "aa_count_adjusted": n,
        "start_m_removed": bool(removed),
        "eaa_profile": profile,
    }

    if include_per_aa:
        for aa in sorted(eaa):
            out[f"eaa_pct_{aa}"] = (c.get(aa, 0) / n) * 100.0

    return out


def max_residue_pct_per_window(
    seq: str,
    residue: str = "M",
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_step: int = DEFAULT_WINDOW_STEP,
    remove_start_m: bool = True,
    canonical_only: bool = True,
) -> float:
    """
    Compute the maximum percentage of a residue within any sliding window.

    The sequence is cleaned and optionally adjusted (N-terminal Met removed),
    then windows of length `window_size` are scanned with stride `window_step`.

    Behavior
    --------
    - If adjusted length is 0: returns np.nan.
    - If adjusted length < window_size: returns the global residue percentage
      on the adjusted sequence.
    """
    residue = str(residue).upper()
    if residue not in CANONICAL_AA_SET:
        raise ValueError(f"Residue '{residue}' is not canonical.")

    if window_size <= 0:
        raise ValueError("window must be > 0")
    if window_step <= 0:
        raise ValueError("step must be > 0")

    cleaned = clean_sequence(seq, canonical_only=canonical_only)
    adjusted, _ = remove_nterm_m(cleaned, enabled=remove_start_m)
    n = len(adjusted)

    if n == 0:
        return np.nan

    if n < window_size:
        return (adjusted.count(residue) / n) * 100.0

    # # Efficient scanning via cumulative sums (O(n) preprocessing; O(#windows) queries)
    arr = np.fromiter((1 if ch == residue else 0 for ch in adjusted), dtype=int, count=n)
    csum = np.concatenate(([0], np.cumsum(arr)))

    max_pct = 0.0
    # window count for start i: csum[i+window] - csum[i]
    for i in range(0, n - window_size + 1, window_step):
        w_count = csum[i + window_size] - csum[i]
        pct = (w_count / window_size) * 100.0
        if pct > max_pct:
            max_pct = float(pct)
    return float(max_pct)


def protein_metrics(
    seq: str,
    remove_start_m: bool = True,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_step: int = DEFAULT_WINDOW_STEP,
    canonical_only: bool = True,
) -> Dict[str, object]:
    """
    Compute primary protein-level metrics used across the library.

    All metrics are computed after sequence cleaning. Composition metrics are
    computed on the adjusted sequence (i.e., after optional N-terminal Met removal).

    Returns
    -------
    dict
        - aa_count_cleaned, aa_count_adjusted
        - canonical_only, remove_start_m, start_m_removed
        - met_pct, cys_pct, saa_pct  (percentages on the adjusted sequence)
        - window_size, window_step
        - max_met_per_window (maximum Met% in any sliding window under the same settings)
    """
    aa_count_raw = len(seq)

    prof = aa_profile(seq, remove_start_m=remove_start_m, canonical_only=canonical_only)

    n_cleaned = int(prof["length"]["aa_count_cleaned"])
    n_adj = int(prof["length"]["aa_count_adjusted"])
    removed = bool(prof["flags"]["start_m_removed"])

    met_pct = prof["pct"]["M"]
    cys_pct = prof["pct"]["C"]
    saa_pct = np.nan if (np.isnan(met_pct) or np.isnan(cys_pct)) else float(met_pct + cys_pct)

    max_met = max_residue_pct_per_window(
        seq,
        residue="M",
        window_size=window_size,
        window_step=window_step,
        remove_start_m=remove_start_m,
        canonical_only=canonical_only,
    )

    return {
        "aa_count_raw": aa_count_raw, 
        "canonical_only": bool(canonical_only),
        "aa_count_cleaned": n_cleaned,
        "remove_start_m": bool(remove_start_m),
        "start_m_removed": removed,
        "aa_count_adjusted": n_adj,
        "met_pct": met_pct,
        "cys_pct": cys_pct,
        "saa_pct": saa_pct,
        "window_size": int(window_size),
        "window_step": int(window_step),
        "max_met_per_window": max_met,
    }

