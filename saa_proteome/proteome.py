"""
Proteome-level tabular outputs (FASTA → pandas DataFrames).

This module provides batch wrappers that parse a proteome FASTA file and return
analysis-ready DataFrames at the protein level and proteome-aggregated level.
Single-sequence feature extraction is implemented in `metrics.py`.

"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Union, Optional
from .io import read_fasta
from .metrics import aa_profile, essential_aa_stats, protein_metrics
from .sequence import normalize_protein_id
from .config import (
    CANONICAL_AA_ORDER,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_STEP,
)


# Load fasta file as metrics pd
def load_proteome_metrics(
    fasta_path: Union[str, Path],
    *,
    species: Optional[str] = None,
    canonical_only: bool = True,
    remove_start_m: bool = True,
    window_size: int = DEFAULT_WINDOW_SIZE,           
    window_step: int = DEFAULT_WINDOW_STEP,
    include_aa_freq: bool = True,
    aa_freq_prefix: str = "aa_",
    id_mode: str = "auto",
) -> pd.DataFrame:
    """
    Load a proteome FASTA file and construct a protein-level metrics DataFrame.

    Parameters
    ----------
    fasta_path : str or Path
        Path to the proteome FASTA file.
    species : str, optional
        Species label added to the output table.
    canonical_only : bool, default True
        If True, restrict sequences to the 20 canonical amino acids.
    remove_start_m : bool, default True
        If True, remove N-terminal methionine prior to metric calculation.
    window_size : int
        Sliding-window size used for window-based metrics.
    window_step : int
        Step size (stride) for sliding-window scanning.
    include_aa_freq : bool, default True
        If True, include per-residue frequency columns (aa_X_freq).
    aa_freq_prefix : str, default "aa_"
        Prefix used for amino-acid frequency columns.
    id_mode : str, default "auto"
        Mode used to normalize FASTA record identifiers.

    Returns
    -------
    pd.DataFrame
        One row per protein with computed metrics and metadata.

    Notes
    -----
    All derived columns are computed from the same adjusted sequence defined
    by (canonical_only, remove_start_m).
    """
    records = read_fasta(str(fasta_path))
    rows: list[dict] = []

    for rec in records:
        pid = normalize_protein_id(str(rec.id), mode=id_mode)
        seq_raw = str(rec.seq)

        metrics = protein_metrics(
            seq_raw,
            canonical_only=canonical_only,
            remove_start_m=remove_start_m,
            window_size=int(window_size),
            window_step=int(window_step),
        )

        row = dict(metrics)

        # Deterministic metadata
        row["protein_id"] = pid                    # normalized internal key
        row["protein_id_raw"] = rec.id             # exact FASTA ID token
        row["description"] = rec.description       # full FASTA description/header text
        row["canonical_only"] = bool(canonical_only)
        row["remove_start_m"] = bool(remove_start_m)
        if species is not None:
            row["species"] = species

        # AA frequencies: reuse aa_profile() as the single authority
        if include_aa_freq:
            prof = aa_profile(
                seq_raw,
                remove_start_m=remove_start_m,
                canonical_only=canonical_only,
            )
            # prof["freq"][aa] is NaN if adjusted length is 0
            for aa in CANONICAL_AA_ORDER:
                row[f"{aa_freq_prefix}{aa}_freq"] = float(prof["freq"][aa])

        rows.append(row)

    df = pd.DataFrame(rows)
    
    # Reorder columns for identifier columns to appear first
    front_cols = [
        "species",
        "protein_id",
        "protein_id_raw",
        "description",
    ]

    front_cols = [c for c in front_cols if c in df.columns]

    df = df.loc[:, front_cols + [c for c in df.columns if c not in front_cols]]

    return df


# ----------------------------------------------------------------------
# Backward-compatible alias (deprecated name)
# ----------------------------------------------------------------------
def proteome_to_df(*args, **kwargs) -> pd.DataFrame:
    """
    Deprecated alias for load_proteome_metrics().
    """
    return load_proteome_metrics(*args, **kwargs)


# Amino acids composition in protein-level or species proteome 
def aa_composition_df(
    fasta_path: Union[str, Path],
    species: str,
    level: str = "proteome",
    canonical_only: bool = True,
    remove_start_m: bool = True,
    value_scale: str = "pct",    # {"count", "freq", "pct"}
    id_mode: str = "auto",
    extended: bool = True,
    protein_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute canonical amino-acid composition at protein or proteome level.

    Parameters
    ----------
    fasta_path : str or Path
        Path to the proteome FASTA file.
    species : str
        Species label added to the output table.
    level : {"protein", "proteome"}, default "proteome"
        - "protein": returns one row per protein (or a single row if protein_id is specified).
        - "proteome": returns one aggregated row over all proteins.
    canonical_only : bool, default True
        If True, restrict sequences to the 20 canonical amino acids.
    remove_start_m : bool, default True
        If True, remove N-terminal methionine prior to metric calculation.
    value_scale : {"count", "freq", "pct"}, default "pct"
        Output scale for amino-acid composition:
        - "count": absolute residue counts
        - "freq": proportion (0—1)
        - "pct": percentage (0—100)
    id_mode : str, default "auto"
        Mode used to normalize FASTA record identifiers.
    extended : bool, default True
        If True, include metadata columns.
        If False, return only amino-acid composition columns
        (plus protein_id for protein-level output).
    protein_id : str, optional
        Only used when level="protein".
        If provided, returns a one-row table for the matching protein
        after identifier normalization.

    Returns
    -------
    pd.DataFrame
        Wide-format table with columns aa_A … aa_V in the requested scale.

    Notes
    -----
    All statistics are derived from the adjusted sequence defined by
    (canonical_only, remove_start_m).
    Proteins with zero adjusted sequence length are excluded.
    """
    if level not in {"protein", "proteome"}:
        raise ValueError("level must be 'protein' or 'proteome'")
    if value_scale not in {"count", "freq", "pct"}:
        raise ValueError("value_scale must be 'count', 'freq', or 'pct'")

    records = list(read_fasta(str(fasta_path)))
    total_proteins = len(records)

    aa_cols = [f"aa_{aa}" for aa in CANONICAL_AA_ORDER]

    # ---------------------------
    # Protein-level composition
    # ---------------------------
    if level == "protein":
        rows = []

        protein_id_norm = None
        if protein_id is not None:
            protein_id_norm = normalize_protein_id(protein_id, mode=id_mode)

        for rec in records:
            pid = normalize_protein_id(str(rec.id), mode=id_mode)

            if protein_id_norm is not None and pid != protein_id_norm:
                continue

            prof = aa_profile(
                str(rec.seq),
                remove_start_m=remove_start_m,
                canonical_only=canonical_only,
            )

            if prof["length"]["aa_count_cleaned"] == 0:
                continue

            # Exclude proteins that become empty after adjustment (e.g., single 'M' with remove_start_m=True)
            n_adj = int(prof["length"]["aa_count_adjusted"])
            if n_adj == 0:
                continue

            row = {
                "species": species,
                "level": "protein",  # Added for consistency with proteome-level output
                "protein_id": pid,
                "protein_id_raw": rec.id,
                "description": rec.description,
                "canonical_only": bool(canonical_only),
                "aa_count_cleaned": int(prof["length"]["aa_count_cleaned"]),
                "remove_start_m": bool(remove_start_m),
                "start_m_removed": bool(prof["flags"]["start_m_removed"]),
                "aa_count_adjusted": n_adj,
                "aa_value_scale": value_scale,
            }
            row.update({f"aa_{aa}": prof[value_scale][aa] for aa in CANONICAL_AA_ORDER})
            rows.append(row)

        df = pd.DataFrame(rows)

        if protein_id is not None:
            if df.empty:
                raise ValueError(
                    f"protein_id='{protein_id}' not found in FASTA "
                    f"(after id normalization with mode='{id_mode}')."
                )
            if len(df) > 1:
                raise ValueError(
                    f"protein_id='{protein_id}' matched multiple records. "
                    f"Check FASTA IDs or set id_mode='raw'."
                )

        if not extended:
            keep = ["protein_id"] + [c for c in aa_cols if c in df.columns]
            return df.loc[:, keep].copy()

        return df

    # ---------------------------
    # Proteome-level composition
    # ---------------------------
    total_counts = {aa: 0.0 for aa in CANONICAL_AA_ORDER}
    total_len_adjusted = 0
    proteins_used = 0
    proteins_dropped_empty = 0

    for rec in records:
        prof = aa_profile(
            str(rec.seq),
            remove_start_m=remove_start_m,
            canonical_only=canonical_only,
        )

        if prof["length"]["aa_count_cleaned"] == 0:
            proteins_dropped_empty += 1
            continue

        n_adj = int(prof["length"]["aa_count_adjusted"])
        if n_adj == 0:
            proteins_dropped_empty += 1
            continue

        proteins_used += 1
        total_len_adjusted += n_adj
        for aa in CANONICAL_AA_ORDER:
            total_counts[aa] += float(prof["count"][aa])

    if total_len_adjusted == 0:
        out = {f"aa_{aa}": float("nan") for aa in CANONICAL_AA_ORDER}
    else:
        if value_scale == "count":
            out = {f"aa_{aa}": total_counts[aa] for aa in CANONICAL_AA_ORDER}
        elif value_scale == "freq":
            out = {f"aa_{aa}": total_counts[aa] / total_len_adjusted for aa in CANONICAL_AA_ORDER}
        else:
            out = {f"aa_{aa}": (total_counts[aa] / total_len_adjusted) * 100.0 for aa in CANONICAL_AA_ORDER}

    df = pd.DataFrame([{
        "species": species,
        "level": "proteome",
        "total_proteins": total_proteins,
        "canonical_only": bool(canonical_only),
        "remove_start_m": bool(remove_start_m),
        "proteins_dropped_empty": proteins_dropped_empty,
        "total_proteins_used": proteins_used,
        "total_aa_adjusted": total_len_adjusted,
        "aa_value_scale": value_scale,
        **out,
    }])

    if not extended:
        keep = [c for c in aa_cols if c in df.columns]
        return df.loc[:, keep].copy()

    return df


# AA summary stat function
def eaa_summary(
    fasta_path: Union[str, Path],
    species: str,
    profile: str = "human_classic",
    remove_start_m: bool = True,
    canonical_only: bool = True,
    include_per_aa: bool = True,
    id_mode: str = "auto",
) -> pd.DataFrame:
    """
    Compute per-protein essential amino-acid (EAA) statistics for a proteome.

    Each FASTA record is processed independently, and essential amino-acid
    composition metrics are calculated on the adjusted sequence
    (after optional N-terminal methionine removal).

    Parameters
    ----------
    fasta_path : str or Path
        Path to the proteome FASTA file.
    species : str
        Species label added to the output table.
    profile : str, default "human_classic"
        Named essential-amino-acid profile defined in config.EAA_PROFILES.
    remove_start_m : bool, default True
        If True, remove N-terminal methionine prior to metric calculation.
    canonical_only : bool, default True
        If True, restrict sequences to the 20 canonical amino acids.
    include_per_aa : bool, default True
        If True, include per-residue EAA percentage columns (eaa_pct_X).
    id_mode : str, default "auto"
        Mode used to normalize FASTA record identifiers.

    Returns
    -------
    pd.DataFrame
        One row per protein containing:
        - species and protein identifiers
        - EAA count, frequency, and percentage
        - adjusted sequence length
        - N-terminal removal flag
        - optional per-residue EAA percentages

    Notes
    -----
    Proteins with zero adjusted sequence length are excluded from the output.
    All statistics are derived from the adjusted sequence defined by
    (canonical_only, remove_start_m).
    """
    records = read_fasta(str(fasta_path))

    rows = []
    for rec in records:
        pid = normalize_protein_id(str(rec.id), mode=id_mode)
        e = essential_aa_stats(
            str(rec.seq),
            profile=profile,
            canonical_only=canonical_only,
            remove_start_m=remove_start_m,
            include_per_aa=include_per_aa,
        )

        if e.get("aa_count_adjusted", 0) == 0:
            continue

        rows.append({
            "species": species,
            "protein_id": pid,         # normalized ID (stable internal key)
            "protein_id_raw": rec.id,  # exact FASTA header ID (unmodified)
            "description": rec.description,
            "canonical_only": bool(canonical_only),
            "remove_start_m": bool(remove_start_m),
            **e
        })

    return pd.DataFrame(rows)



def saa_composition_df(
    fasta_path: Union[str, Path],
    species: str,
    level: str = "proteome",
    canonical_only: bool = True,
    remove_start_m: bool = True,
    value_scale: str = "pct",  # {"count", "freq", "pct"}
    id_mode: str = "auto",
    extended: bool = True,
    protein_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute sulfur amino-acid (SAA) composition derived from aa_composition_df().

    This function extracts methionine (M), cysteine (C), and their sum (SAA = M + C)
    from canonical amino-acid composition tables and returns them in the requested scale.

    Parameters
    ----------
    fasta_path : str or Path
        Path to the proteome FASTA file.
    species : str
        Species label added to the output table.
    level : {"protein", "proteome"}, default "proteome"
        - "protein": returns one row per protein (or a single row if protein_id is specified).
        - "proteome": returns one aggregated row over all proteins.
    canonical_only : bool, default True
        If True, restrict sequences to the 20 canonical amino acids.
    remove_start_m : bool, default True
        If True, remove N-terminal methionine prior to metric calculation.
    value_scale : {"count", "freq", "pct"}, default "pct"
        Output scale:
            - "count": absolute residue counts
            - "freq": proportion (0—1)
            - "pct": percentage (0—100)
    id_mode : str, default "auto"
        Mode used to normalize FASTA record identifiers.
    extended : bool, default True
        If True, include metadata columns.
        If False, return only sulfur values (plus protein_id for protein-level output).
    protein_id : str, optional
        Only used when level="protein".
        If provided, returns a one-row table for the matching protein.

    Returns
    -------
    pd.DataFrame
        Wide-format table containing:
        - met
        - cys
        - saa (met + cys)
        plus metadata columns when extended=True.

    Notes
    -----
    All statistics are derived from the adjusted sequence defined by
    (canonical_only, remove_start_m).
    Proteins with zero adjusted sequence length are excluded.
    """
    df = aa_composition_df(
        fasta_path=fasta_path,
        species=species,
        level=level,
        canonical_only=canonical_only,
        remove_start_m=remove_start_m,
        value_scale=value_scale,
        id_mode=id_mode,
        extended=True,  # we will decide what to keep below
        protein_id=protein_id,
    )

    # Expect these from aa_composition_df
    if "aa_M" not in df.columns or "aa_C" not in df.columns:
        raise ValueError("Expected columns 'aa_M' and 'aa_C' not found in aa_composition_df output.")

    # Build SAA columns (scale-consistent)
    df = df.copy()
    df["saa"] = pd.to_numeric(df["aa_M"], errors="coerce") + pd.to_numeric(df["aa_C"], errors="coerce")

    # Rename for clarity (optional but recommended)
    df = df.rename(columns={
        "aa_M": "met",
        "aa_C": "cys",
        "aa_value_scale": "saa_value_scale",  # keep consistent naming
    })

    # Decide which metadata columns to keep
    if level == "proteome":
        meta_cols = [
            "species", "level", "total_proteins",
            "canonical_only", "remove_start_m",
            "proteins_dropped_empty", "total_proteins_used",
            "total_aa_adjusted", "saa_value_scale",
        ]
    else:
        meta_cols = [
            "species", "protein_id", "description",
            "canonical_only", "remove_start_m",
            "start_m_removed",
            "aa_count_cleaned", "aa_count_adjusted",
            "saa_value_scale",
        ]

    keep = [c for c in meta_cols if c in df.columns] + ["met", "cys", "saa"]

    df = df.loc[:, keep].copy()

    if not extended:
        # Return only sulfur values (plus identifiers if protein-level)
        if level == "proteome":
            return df.loc[:, ["met", "cys", "saa"]].copy()
        else:
            base = ["protein_id"]
            return df.loc[:, base + ["met", "cys", "saa"]].copy()

    return df
