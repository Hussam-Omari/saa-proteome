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

from .sequence import prepare_sequence
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

    The input is normalized, and N-terminal methionine removal is determined
    from the original normalized sequence before optional canonical-residue
    filtering. Composition metrics are calculated from the resulting adjusted
    sequence.

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

    cleaned, adjusted, removed = prepare_sequence(
    seq,
    canonical_only=canonical_only,
    remove_start_m=remove_start_m,
)

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

    The input sequence is normalized first. If `remove_start_m=True`,
    N-terminal methionine removal is determined from the original normalized
    sequence before optional canonical-residue filtering. Group composition
    is then calculated from the resulting adjusted sequence.

    Parameters
    ----------
    group : str or sequence of str
        Either a string such as "MC" (interpreted as {'M', 'C'}) or an
        iterable of one-letter residue codes.

    Returns
    -------
    dict
        Dictionary containing:
        - group_count
        - group_freq (0 to 1)
        - group_pct (0 to 100)
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

    cleaned, adjusted, removed = prepare_sequence(
    seq,
    canonical_only=canonical_only,
    remove_start_m=remove_start_m,
    )

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
    Compute reference essential-amino-acid composition metrics for one sequence.

    The input sequence is normalized first. If `remove_start_m=True`,
    N-terminal methionine removal is determined from the original normalized
    sequence before optional canonical-residue filtering. Essential-amino-acid
    statistics are then calculated from the resulting adjusted sequence.

    Returns
    -------
    dict
        Always includes:
        - eaa_count
        - eaa_freq (0 to 1)
        - eaa_pct (0 to 100)
        - aa_count_adjusted
        - start_m_removed
        - eaa_profile

        If `include_per_aa=True`, also includes per-residue EAA percentages
        as keys such as:
        - eaa_pct_H
        - eaa_pct_I
        - eaa_pct_K
    """

    eaa = essential_aa_set(profile=profile)

    cleaned, adjusted, removed = prepare_sequence(
    seq,
    canonical_only=canonical_only,
    remove_start_m=remove_start_m,
    )

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


# ----------------------------------------------------------------------
# Sliding-window enrichment metrics
# ----------------------------------------------------------------------

def max_group_pct_per_window(
    seq: str,
    group: Union[str, Sequence[str], Set[str]],
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_step: int = DEFAULT_WINDOW_STEP,
    remove_start_m: bool = True,
    canonical_only: bool = True,
) -> float:
    """
    Compute the maximum percentage of a residue group within any sliding window.

    Examples
    --------
    group="M"  -> localized methionine percentage
    group="C"  -> localized cysteine percentage
    group="MC" -> localized total sulfur amino-acid percentage

    Behavior
    --------
    - If adjusted sequence length is zero, return np.nan.
    - If adjusted sequence length is shorter than window_size, calculate the
      group percentage using the full adjusted sequence.
    - For longer sequences, scan overlapping windows and return the maximum
      percentage observed in a single window.

    Notes
    -----
    `group` may contain one or more canonical one-letter amino-acid codes.
    For example, "M" evaluates methionine, "MC" evaluates the combined
    percentage of methionine and cysteine, and "KR" evaluates the combined
    percentage of lysine and arginine.

    The input is interpreted as a residue set, not as a sequence motif.
    Therefore, "MC" counts all M and C residues within each window,
    regardless of their order or adjacency. The inputs "MC" and "CM"
    produce the same result.
    """
    if isinstance(group, str):
        residues = set(group.upper())
    else:
        residues = {str(residue).upper() for residue in group}

    if not residues:
        raise ValueError("group must contain at least one residue.")

    invalid = sorted(residues - CANONICAL_AA_SET)
    if invalid:
        raise ValueError(
            f"Non-canonical residues in group: {invalid}. "
            f"Allowed residues: {sorted(CANONICAL_AA_SET)}"
        )

    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if window_step <= 0:
        raise ValueError("window_step must be > 0")

    _, adjusted, _ = prepare_sequence(
    seq,
    canonical_only=canonical_only,
    remove_start_m=remove_start_m,
    )


    n = len(adjusted)

    if n == 0:
        return np.nan

    group_array = np.fromiter(
        (1 if residue in residues else 0 for residue in adjusted),
        dtype=int,
        count=n,
    )

    if n < window_size:
        return float((group_array.sum() / n) * 100.0)

    cumulative_sum = np.concatenate(([0], np.cumsum(group_array)))
    max_count = 0

    for start in range(0, n - window_size + 1, window_step):
        end = start + window_size
        window_count = cumulative_sum[end] - cumulative_sum[start]

        if window_count > max_count:
            max_count = int(window_count)

    return float((max_count / window_size) * 100.0)



def max_residue_pct_per_window(
    seq: str,
    residue: str = "M",
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_step: int = DEFAULT_WINDOW_STEP,
    remove_start_m: bool = True,
    canonical_only: bool = True,
) -> float:
    """
    Compute the maximum percentage of one residue within any sliding window.

    This function is retained as a backward-compatible wrapper around
    max_group_pct_per_window().
    """
    residue = str(residue).upper()

    if len(residue) != 1 or residue not in CANONICAL_AA_SET:
        raise ValueError(
            f"Residue '{residue}' must be one canonical amino-acid code."
        )

    return max_group_pct_per_window(
        seq=seq,
        group=residue,
        window_size=window_size,
        window_step=window_step,
        remove_start_m=remove_start_m,
        canonical_only=canonical_only,
    )


# ----------------------------------------------------------------------
# Primary protein-level metrics
# ----------------------------------------------------------------------

def protein_metrics(
    seq: str,
    remove_start_m: bool = True,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_step: int = DEFAULT_WINDOW_STEP,
    canonical_only: bool = True,
) -> Dict[str, object]:
    """
    Compute primary protein-level metrics used across the library.
    
    The input sequence is normalized first. If `remove_start_m=True`,
    N-terminal methionine removal is determined from the original normalized
    sequence before optional canonical-residue filtering. All composition and
    sliding-window metrics are calculated from the resulting adjusted sequence.

    Returns
    -------
    dict
        Protein-level metrics containing:
        - aa_count_raw
        - aa_count_cleaned
        - aa_count_adjusted
        - canonical_only
        - remove_start_m
        - start_m_removed
        - met_pct
        - cys_pct
        - saa_pct
        - window_size
        - window_step
        - max_met_per_window
        - max_cys_per_window
        - max_saa_per_window

    Notes
    -----
    `max_saa_per_window` is calculated directly from methionine and cysteine
    occurring within the same sliding window. It is not calculated by adding
    `max_met_per_window` and `max_cys_per_window`.
    """
    aa_count_raw = len(seq)

    prof = aa_profile(
        seq,
        remove_start_m=remove_start_m,
        canonical_only=canonical_only,
    )

    n_cleaned = int(prof["length"]["aa_count_cleaned"])
    n_adj = int(prof["length"]["aa_count_adjusted"])
    removed = bool(prof["flags"]["start_m_removed"])

    met_pct = prof["pct"]["M"]
    cys_pct = prof["pct"]["C"]

    if np.isnan(met_pct) or np.isnan(cys_pct):
        saa_pct = np.nan
    else:
        saa_pct = float(met_pct + cys_pct)

    # Maximum localized methionine percentage
    max_met = max_residue_pct_per_window(
        seq=seq,
        residue="M",
        window_size=window_size,
        window_step=window_step,
        remove_start_m=remove_start_m,
        canonical_only=canonical_only,
    )

    # Maximum localized cysteine percentage
    max_cys = max_residue_pct_per_window(
        seq=seq,
        residue="C",
        window_size=window_size,
        window_step=window_step,
        remove_start_m=remove_start_m,
        canonical_only=canonical_only,
    )

    # Maximum localized total sulfur amino-acid percentage
    max_saa = max_group_pct_per_window(
        seq=seq,
        group="MC",
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
        "max_cys_per_window": max_cys,
        "max_saa_per_window": max_saa,
    }
