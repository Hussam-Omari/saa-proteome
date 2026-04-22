"""
Sequence preprocessing and FASTA identifier normalization.

This module standardizes protein sequences (case normalization and optional
canonical filtering), supports optional N-terminal methionine removal, and
provides FASTA record identifier normalization for consistent matching.

"""

from __future__ import annotations

from typing import Tuple, Literal, Optional

from .config import CANONICAL_AA_SET


def clean_sequence(seq: str, canonical_only: bool = True) -> str:
    """
    Standardize a protein sequence:
    - convert to uppercase
    - remove stop character '*'
    - optionally filter to the 20 canonical amino acids

    Parameters
    ----------
    seq : str
        Raw protein sequence (may include '*', X, etc.)
    canonical_only : bool
        If True, remove all non-canonical letters.

    Returns
    -------
    str
        Cleaned sequence.
    """
    s = str(seq).upper().replace("*", "")
    if not canonical_only:
        return s
    return "".join([aa for aa in s if aa in CANONICAL_AA_SET])



def remove_nterm_m(seq: str, enabled: bool = True) -> Tuple[str, bool]:
    """
    Optionally remove an N-terminal methionine.

    Parameters
    ----------
    seq : str
        Input sequence (typically already cleaned).
    enabled : bool, default True
        If True, remove the first residue only when it is 'M'.

    Returns
    -------
    tuple (str, bool)
        (adjusted_sequence, removed_flag), where removed_flag indicates whether an
        N-terminal 'M' was removed.
    """
    if not enabled:
        return seq, False
    if not seq:
        return seq, False
    if seq[0] == "M":
        return seq[1:], True
    return seq, False


def normalize_protein_id(rec_id: str, mode: str = "auto") -> str:
    """
    Normalize a FASTA record identifier for consistent matching.

    Common UniProt forms:
    - 'sp|Q9XXXX|NAME' or 'tr|A0A...|NAME'  -> accession (Q9XXXX or A0A...)
    - otherwise returns the raw rec_id

    Parameters
    ----------
    rec_id : str
        FASTA record id field.
    mode : str
        'auto' -> extract accession when pipe-delimited form exists
        'raw'  -> return rec_id as is
        'accession' -> force accession extraction when possible

    Returns
    -------
    str
        Normalized identifier.
    """
    VALID_MODES = {"auto", "accession", "raw"}

    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of {sorted(VALID_MODES)}."
        )

    rid = str(rec_id)
    if mode == "raw":
        return rid

    parts = rid.split("|")
    if len(parts) >= 2:
        # UniProt-like
        accession = parts[1]
        return accession

    # fallback: no pipes
    if mode == "accession":
        return rid
    return rid