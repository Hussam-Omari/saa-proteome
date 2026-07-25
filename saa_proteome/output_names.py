"""
Column naming utilities for presentation and export.

This module provides a centralized mapping from stable, machine-friendly
internal column names to human-readable, publication-ready labels.

Design principle
---------------
Analytical functions should return stable internal column names to ensure
reproducibility. Renaming should be applied only at the presentation layer
(e.g., before exporting tables or rendering manuscript figures). A small
data-dictionary helper is included for interactive notebook use.

This module is intentionally dependency-light and does not import analysis code.
"""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd


# Central mapping (single source of truth).
# Unmatched columns are left unchanged by the helper functions below.
COLUMN_RENAME_MAP: dict[str, str] = {
    # -----------------------------
    # Metadata
    # -----------------------------
    "species": "Species",
    "protein_id": "Protein ID",
    "protein_id_raw": "Protein ID (Raw FASTA Token)",
    "description": "Protein Description",
    "path": "FASTA File Path",
    "first_id": "First Protein ID",
    "rank": "Rank",
    "rank_metric": "Ranking Metric",
    "level": "Analysis Level",

    # -----------------------------
    # Summary counts
    # -----------------------------
    "total_proteins": "Total Proteins",
    "proteins": "Total Proteins",
    "proteins_in_df": "Proteins in DataFrame",
    "records_found_probe": "Records Found (Probe)",
    "error_message": "Error Message",
    "readable": "File Readable",
    "parseable": "File Parseable",
    "looks_like_fasta": "Valid FASTA Format",

    # Proteome-level bookkeeping emitted by aa_composition_df / summaries
    "proteins_dropped_empty": "Proteins Dropped (Empty After Processing)",
    "total_proteins_used": "Total Proteins Used",
    "total_aa_adjusted": "Total Amino Acids (Adjusted)",

    # -----------------------------
    # Sequence Length Metrics
    # -----------------------------
    # Protein-level length columns
    "aa_count_raw": "Sequence Length (Raw)",
    "aa_count_cleaned": "Sequence Length (Cleaned)",
    "aa_count_adjusted": "Sequence Length (Adjusted)",

    # QC/summary length columns (used by qc_report)
    "mean_length": "Mean Sequence Length",
    "median_length": "Median Sequence Length",
    "min_length": "Minimum Sequence Length",
    "max_length": "Maximum Sequence Length",

    # -----------------------------
    # Sulfur Amino Acids (percent columns are always %)
    # -----------------------------
    "met_pct": "Methionine (%)",
    "met_pct_mean": "Mean Methionine (%)",
    "met_pct_median": "Median Methionine (%)",
    "met_pct_q50": "Methionine 50th Percentile (%)",
    "met_pct_q90": "Methionine 90th Percentile (%)",
    "met_pct_q99": "Methionine 99th Percentile (%)",
    "met_pct_q25": "Methionine 25th Percentile (%)",
    "met_pct_q95": "Methionine 95th Percentile (%)",
    "cys_pct_q25": "Cysteine 25th Percentile (%)",
    "cys_pct_q95": "Cysteine 95th Percentile (%)",
    "saa_pct_q25": "Total Sulfur Amino Acids 25th Percentile (%)",
    "saa_pct_q95": "Total Sulfur Amino Acids 95th Percentile (%)",

    "cys_pct": "Cysteine (%)",
    "cys_pct_mean": "Mean Cysteine (%)",
    "cys_pct_median": "Median Cysteine (%)",
    "cys_pct_q50": "Cysteine 50th Percentile (%)",
    "cys_pct_q90": "Cysteine 90th Percentile (%)",
    "cys_pct_q99": "Cysteine 99th Percentile (%)",

    "saa_pct": "Total Sulfur Amino Acids (%)",
    "saa_pct_mean": "Mean Total Sulfur Amino Acids (%)",
    "saa_pct_median": "Median Total Sulfur Amino Acids (%)",
    "saa_pct_q50": "Total Sulfur Amino Acids 50th Percentile (%)",
    "saa_pct_q90": "Total Sulfur Amino Acids 90th Percentile (%)",
    "saa_pct_q99": "Total Sulfur Amino Acids 99th Percentile (%)",

    # Scale-dependent sulfur outputs: do NOT hard-code units here
    "met": "Methionine",
    "cys": "Cysteine",
    "saa": "Total Sulfur Amino Acids",

    # -----------------------------
    # Sliding Window Metrics
    # -----------------------------
    "max_met_per_window": "Maximum Methionine per Sliding Window (%)",
    "max_cys_per_window": "Maximum Cysteine per Sliding Window (%)",
    "max_saa_per_window": "Maximum Total Sulfur Amino Acids per Sliding Window (%)",

    "max_met_per_window_mean": "Mean Maximum Methionine per Sliding Window (%)",
    "max_cys_per_window_mean": "Mean Maximum Cysteine per Sliding Window (%)",
    "max_saa_per_window_mean": "Mean Maximum Total Sulfur Amino Acids per Sliding Window (%)",

    "max_met_per_window_median": "Median Maximum Methionine per Sliding Window (%)",
    "max_cys_per_window_median": "Median Maximum Cysteine per Sliding Window (%)",
    "max_saa_per_window_median": "Median Maximum Total Sulfur Amino Acids per Sliding Window (%)",

    "max_met_per_window_q50": "Maximum Methionine per Sliding Window (50th Percentile, %)",
    "max_cys_per_window_q50": "Maximum Cysteine per Sliding Window (50th Percentile, %)",
    "max_saa_per_window_q50": "Maximum Total Sulfur Amino Acids per Sliding Window (50th Percentile, %)",

    "max_met_per_window_q90": "Maximum Methionine per Sliding Window (90th Percentile, %)",
    "max_cys_per_window_q90": "Maximum Cysteine per Sliding Window (90th Percentile, %)",
    "max_saa_per_window_q90": "Maximum Total Sulfur Amino Acids per Sliding Window (90th Percentile, %)",

    "max_met_per_window_q99": "Maximum Methionine per Sliding Window (99th Percentile, %)",
    "max_cys_per_window_q99": "Maximum Cysteine per Sliding Window (99th Percentile, %)",
    "max_saa_per_window_q99": "Maximum Total Sulfur Amino Acids per Sliding Window (99th Percentile, %)",

    # Window parameters (new canonical + legacy aliases)
    "window_size": "Sliding Window Size (Amino Acids)",
    "window_step": "Sliding Window Step (Amino Acids)",

    # -----------------------------
    # Essential Amino Acids (EAA)
    # -----------------------------
    "eaa_count": "Essential Amino Acid Count",
    "eaa_freq": "Essential Amino Acid Frequency",
    "eaa_pct": "Essential Amino Acids (%)",
    "eaa_profile": "Essential Amino Acid Profile",

    "eaa_pct_I": "Isoleucine (%)",
    "eaa_pct_L": "Leucine (%)",
    "eaa_pct_K": "Lysine (%)",
    "eaa_pct_M": "Methionine (Essential, %)",
    "eaa_pct_F": "Phenylalanine (%)",
    "eaa_pct_T": "Threonine (%)",
    "eaa_pct_V": "Valine (%)",
    "eaa_pct_H": "Histidine (%)",
    "eaa_pct_W": "Tryptophan (%)",

    # -----------------------------
    # Amino Acid Columns (aa_X) from aa_composition_df (scale-dependent)
    # -----------------------------
    "aa_A": "Alanine",
    "aa_R": "Arginine",
    "aa_N": "Asparagine",
    "aa_D": "Aspartic Acid",
    "aa_C": "Cysteine",
    "aa_Q": "Glutamine",
    "aa_E": "Glutamic Acid",
    "aa_G": "Glycine",
    "aa_H": "Histidine",
    "aa_I": "Isoleucine",
    "aa_L": "Leucine",
    "aa_K": "Lysine",
    "aa_M": "Methionine",
    "aa_F": "Phenylalanine",
    "aa_P": "Proline",
    "aa_S": "Serine",
    "aa_T": "Threonine",
    "aa_W": "Tryptophan",
    "aa_Y": "Tyrosine",
    "aa_V": "Valine",

    # -----------------------------
    # Single-letter columns (raw counts)
    # -----------------------------
    "A": "Alanine",
    "R": "Arginine",
    "N": "Asparagine",
    "D": "Aspartic Acid",
    "C": "Cysteine",
    "Q": "Glutamine",
    "E": "Glutamic Acid",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "L": "Leucine",
    "K": "Lysine",
    "M": "Methionine",
    "F": "Phenylalanine",
    "P": "Proline",
    "S": "Serine",
    "T": "Threonine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
    "V": "Valine",

    # -----------------------------
    # Flags and processing options
    # -----------------------------
      # Canonical flag + legacy alias
    "remove_start_m": "Start-Met Removal Enabled",
    "start_m_removed": "N-terminal Met Removed",
    "removed_start_m_pct": "Proteins with N-terminal Met Removed (%)",
    "canonical_only": "Canonical Amino Acids Only",

    # Value scale / kind
    "value_scale": "Value Scale",
    "aa_value_scale": "Amino Acid Value Scale",
    "saa_value_scale": "Sulfur Amino Acid Value Scale",
    "aa_kind": "Amino Acid Category",
}


# Full amino acid names (used for derived column auto-mapping)
_AA_FULL_NAMES: dict[str, str] = {
    "A": "Alanine",
    "R": "Arginine",
    "N": "Asparagine",
    "D": "Aspartic Acid",
    "C": "Cysteine",
    "Q": "Glutamine",
    "E": "Glutamic Acid",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "L": "Leucine",
    "K": "Lysine",
    "M": "Methionine",
    "F": "Phenylalanine",
    "P": "Proline",
    "S": "Serine",
    "T": "Threonine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
    "V": "Valine",
}


# Derived columns sometimes emitted by rankings/metrics tables
for _aa, _name in _AA_FULL_NAMES.items():
    # Count-style derived columns
    COLUMN_RENAME_MAP.setdefault(f"aa_{_aa}_count", f"{_name} (Count)")
    # Frequency (0–1)
    COLUMN_RENAME_MAP.setdefault(f"aa_{_aa}_freq", f"{_name} (Frequency)")
    # Percent (0–100)
    COLUMN_RENAME_MAP.setdefault(f"aa_{_aa}_pct", f"{_name} (%)")




def apply_friendly_column_names(
    df: pd.DataFrame,
    mapping: Optional[Mapping[str, str]] = None,
    *,
    inplace: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Rename internal computational column names to publication-ready column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (any output table produced by the project).
    mapping : Mapping[str, str], optional
        Custom mapping to override/extend COLUMN_RENAME_MAP.
        If None, COLUMN_RENAME_MAP is used.
    inplace : bool, default False
        If True, rename columns in-place and return the same object.
        If False, returns a renamed copy.
    strict : bool, default False
        If True and `mapping` is provided, raise KeyError when `mapping` contains keys
        not present in df.columns. This does NOT validate the entire global mapping.

    Returns
    -------
    pd.DataFrame
        Renamed dataframe (copy unless inplace=True).
    """
    m = dict(COLUMN_RENAME_MAP)

    if mapping:
        user_map = dict(mapping)
        if strict:
            missing = [k for k in user_map.keys() if k not in df.columns]
            if missing:
                raise KeyError(f"Columns not found in df (strict mapping): {missing}")
        m.update(user_map)

    m_apply = {k: v for k, v in m.items() if k in df.columns}

    if inplace:
        df.rename(columns=m_apply, inplace=True)
        return df

    return df.rename(columns=m_apply)



def data_dictionary_df(*, mapping: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    """
    Return a tidy data dictionary dataframe based on the column rename mapping.
    """
    m = dict(COLUMN_RENAME_MAP) if mapping is None else dict(mapping)

    rows = [{"column": k, "label": v} for k, v in m.items()]

    # Always define columns explicitly to avoid empty-DataFrame column loss
    df = pd.DataFrame(rows, columns=["column", "label"])

    if df.empty:
        return df  # already has correct columns

    return df.sort_values("column").reset_index(drop=True)



def preview_data_dictionary(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Preview the data dictionary.

    Parameters
    ----------
    df : pd.DataFrame, optional
        If provided, returns dictionary rows only for df.columns (preserves df column order).
        Unknown columns are included with empty labels.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'column', 'label'.
    """
    dd = data_dictionary_df()

    if df is None:
        return dd

    cols = list(df.columns)
    dd2 = dd.set_index("column").reindex(cols).reset_index()
    dd2.rename(columns={"index": "column"}, inplace=True)
    dd2["label"] = dd2["label"].fillna("")
    return dd2

