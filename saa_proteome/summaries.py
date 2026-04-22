"""
Quality-control and summary tables (pandas DataFrames).

Functions in this module summarize protein-level outputs (e.g., from
load_proteome_metrics) into species-level QC and descriptive statistics.

"""

from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import pandas as pd


def qc_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a species-level preprocessing quality-control (QC) summary.

    Parameters
    ----------
    df : pd.DataFrame
        Protein-level metrics DataFrame returned by load_proteome_metrics().

    Returns
    -------
    pd.DataFrame
        One row per species with summary statistics of sequence length used in analysis
        and the percentage of proteins with N-terminal methionine removed.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required = {"species", "aa_count_cleaned", "aa_count_adjusted", "start_m_removed"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"qc_report: missing columns {missing}")

    out_rows = []
    for species, sub in df.groupby("species", dropna=False):
        n = len(sub)

        # Robust numeric handling (prevents warnings on all-NaN groups)
        x = pd.to_numeric(sub["aa_count_adjusted"], errors="coerce")
        val = x.dropna()

        mean_len = float(val.mean()) if len(val) else np.nan
        median_len = float(val.median()) if len(val) else np.nan
        min_len = int(val.min()) if len(val) else np.nan
        max_len = int(val.max()) if len(val) else np.nan

        removed_pct = (
            float(np.nanmean(pd.to_numeric(sub["start_m_removed"], errors="coerce")) * 100.0)
            if n
            else np.nan
        )

        out_rows.append(
            {
                "species": species,
                "total_proteins": n,
                "mean_length": mean_len,
                "median_length": median_len,
                "min_length": min_len,
                "max_length": max_len,
                "removed_start_m_pct": removed_pct,
            }
        )

    df_out = pd.DataFrame(out_rows)

    # Nullable integers for min/max sequence length
    df_out["min_length"] = df_out["min_length"].astype("Int64")
    df_out["max_length"] = df_out["max_length"].astype("Int64")

    return df_out



def saa_summary(
    df_all: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    quantiles: Sequence[float] = (0.5, 0.9, 0.99),
) -> pd.DataFrame:
    """
    Create a species-level summary table for selected metrics.

    Parameters
    ----------
    df_all : pd.DataFrame
        Protein-level metrics DataFrame containing a 'species' column and the
        requested metric columns.
    metrics : sequence of str, optional
        Metrics to summarize. If None, defaults to ['met_pct', 'cys_pct', 'saa_pct'].
    quantiles : sequence of float, default (0.5, 0.9, 0.99)
        Quantiles to compute for each metric.

    Returns
    -------
    pd.DataFrame
        One row per species with mean, median, and requested quantiles for each metric.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    if metrics is None:
        metrics = ["met_pct", "cys_pct", "saa_pct"]

    required = {"species"} | set(metrics)
    missing = sorted(list(required - set(df_all.columns)))
    if missing:
        raise ValueError(f"saa_summary: missing columns {missing}")

    rows = []
    for species, sub in df_all.groupby("species", dropna=False):
        row = {
            "species": species,
            "total_proteins": len(sub),
        }
    
        for m in metrics:
            x = pd.to_numeric(sub[m], errors="coerce")
            row[f"{m}_mean"] = float(x.mean(skipna=True)) if len(x) else np.nan
            row[f"{m}_median"] = float(x.median(skipna=True)) if len(x) else np.nan
            for q in quantiles:
                row[f"{m}_q{int(q*100)}"] = float(x.quantile(q)) if len(x) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)



def length_class_summary(
    df: pd.DataFrame,
    bins=(0, 100, 300, 1000, np.inf),
    labels=("<100 aa", "100–300 aa", "300–1000 aa", ">1000 aa"),
    metrics=("met_pct", "cys_pct", "saa_pct"),
) -> pd.DataFrame:
    """
    Summarize mean metric values by sequence-length classes.

    Parameters
    ----------
    df : pd.DataFrame
        Protein-level metrics DataFrame containing 'species', 'aa_count_adjusted',
        and the requested metric columns.
    bins : sequence, default (0, 100, 300, 1000, np.inf)
        Bin edges for adjusted sequence length.
    labels : sequence of str
        Labels corresponding to the bins.
    metrics : sequence of str
        Metric columns to summarize within each length class.

    Returns
    -------
    pd.DataFrame
        Table with columns ['species', 'length_class', ...metrics] containing mean values.
    """
    needed = {"species", "aa_count_adjusted"} | set(metrics)
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise ValueError(f"length_class_summary: missing columns {missing}")

    df2 = df.copy()
    df2["length_class"] = pd.cut(df2["aa_count_adjusted"], bins=bins, labels=labels, include_lowest=True)

    out = (
        df2.groupby(["species", "length_class"], observed=False)[list(metrics)]
        .mean(numeric_only=True)
        .reset_index()
    )
    return out