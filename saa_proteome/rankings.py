"""
Ranking utilities for producing top-N protein tables.

Functions in this module operate on protein-level DataFrames (e.g., outputs
from load_proteome_metrics) and return ranked subsets suitable for reporting.

"""

from __future__ import annotations
from typing import List, Optional, Literal
import pandas as pd


ValueScale = Literal["pct", "freq", "count"]
AAValueScale = Literal["auto", "keep", "pct", "freq", "count"]


def _is_aa_code(s: str) -> bool:
    """
    Return True if s looks like a one-letter amino-acid code.
    """
    return isinstance(s, str) and len(s) == 1 and s.isalpha()


def _resolve_aa_freq_col(df: pd.DataFrame, aa_code: str, aa_freq_prefix: str) -> str:
    """
    Resolve the amino-acid frequency column name for a given AA code.

    Supports either uppercase (aa_M_freq) or lowercase (aa_m_freq) AA token styles.
    """
    aa_u = aa_code.upper()
    aa_l = aa_code.lower()

    candidates = [
        f"{aa_freq_prefix}{aa_u}_freq",
        f"{aa_freq_prefix}{aa_l}_freq",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not find AA frequency column for '{aa_code}'. "
        f"Tried: {candidates}. Ensure load_proteome_metrics(include_aa_freq=True) was used."
    )

# Top n protein based on metric (e.g., aa column)
def top_n_proteins(
    df: pd.DataFrame,
    metric: str = "saa_pct",
    n: int = 10,
    ascending: bool = False,
    columns: Optional[List[str]] = None,
    *,
    value_scale: ValueScale = "pct",
    aa_freq_prefix: str = "aa_",
    length_col: str = "aa_count_adjusted",
    include_aa_freq: bool = True,
    aa_value_scale: AAValueScale = "auto",
) -> pd.DataFrame:
    """
    Return a top-N ranked protein table from a protein-level metrics DataFrame.

    The ranking metric can be provided in two ways:
    1) A column name present in `df` (e.g., "saa_pct", "max_met_per_window").
    2) A one-letter amino-acid code (e.g., "M", "C"). In this case, the function
    ranks using the corresponding amino-acid frequency column (e.g., aa_M_freq),
    optionally converted to percent or counts according to `value_scale`.

    Parameters
    ----------
    df : pd.DataFrame
        Protein-level metrics DataFrame (typically from load_proteome_metrics).
    metric : str, default "saa_pct"
        Ranking target. Either an existing column name in `df` or a one-letter
        amino-acid code to derive ranking from AA frequency columns.
    n : int, default 10
        Number of top-ranked proteins to return.
    ascending : bool, default False
        Sort order. Use False for "highest values first".
    columns : list of str, optional
        Columns to keep in the output. If None, a default set of identifiers and
        key metrics is used. Missing columns are ignored.

    value_scale : {"pct", "freq", "count"}, default "pct"
        Applied only when `metric` is a one-letter amino-acid code:
        - "freq": rank by amino-acid frequency (0—1)
        - "pct": rank by percentage (0—100)
        - "count": rank by estimated residue count (freq * length_col)
    aa_freq_prefix : str, default "aa_"
        Prefix used for amino-acid frequency columns (e.g., "aa_M_freq").
    length_col : str, default "aa_count_adjusted"
        Sequence-length column required when value_scale="count".
    include_aa_freq : bool, default True
        If True and `columns` is None, include all amino-acid frequency columns.
    aa_value_scale : {"auto", "keep", "pct", "freq", "count"}, default "auto"
        Post-processing scale for included amino-acid columns:
        - "auto": use `value_scale` for display conversion
        - "keep": keep frequency columns unchanged
        - "pct": convert aa_*_freq to aa_*_pct
        - "count": convert aa_*_freq to aa_*_count using length_col

    Returns
    -------
    pd.DataFrame
        Ranked table with a 1-based 'rank' column and a 'rank_metric' label.
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    derived_metric_name: Optional[str] = None

    if metric in df.columns:
        sort_key = metric

    elif _is_aa_code(metric):
        freq_col = _resolve_aa_freq_col(df, metric, aa_freq_prefix)
        base = df[freq_col].astype(float)

        if value_scale == "freq":
            rank_series = base
            derived_metric_name = f"{metric.upper()}_freq"
        elif value_scale == "pct":
            rank_series = base * 100.0
            derived_metric_name = f"{metric.upper()}_pct"
        elif value_scale == "count":
            if length_col not in df.columns:
                raise ValueError(
                    f"length_col '{length_col}' not found; required for value_scale='count'."
                )
            rank_series = base * df[length_col].astype(float)
            derived_metric_name = f"{metric.upper()}_count"
        else:
            raise ValueError("value_scale must be one of: 'pct', 'freq', 'count'.")

        sort_key = "__rank_value__"
        df = df.copy()
        df[sort_key] = rank_series

    else:
        raise ValueError(
            f"Metric '{metric}' not found and is not a one-letter AA code."
        )

    if columns is None:
        columns = [
            "species",
            "protein_id",
            "description",
            "aa_count_adjusted",
            "met_pct",
            "cys_pct",
            "saa_pct",
            "max_met_per_window",
            "max_cys_per_window",
            "max_saa_per_window",
            "start_m_removed",
        ]

    

        if include_aa_freq:
            aa_cols_all = [
                c for c in df.columns
                if c.startswith(aa_freq_prefix) and c.endswith("_freq")
            ]
            columns.extend(aa_cols_all)

        columns = [c for c in columns if c in df.columns]

    out = (
        df.sort_values(sort_key, ascending=ascending)
        .head(n)
        .loc[:, columns]
        .reset_index(drop=True)
    )

    out.insert(0, "rank", range(1, len(out) + 1))
    rank_metric_label = derived_metric_name if derived_metric_name else metric
    out.insert(1, "rank_metric", rank_metric_label)

    # Apply AA display scaling
    if include_aa_freq:
        aa_cols = [
            c for c in out.columns
            if c.startswith(aa_freq_prefix) and c.endswith("_freq")
        ]

        effective_aa_scale = value_scale if aa_value_scale == "auto" else aa_value_scale

        if effective_aa_scale == "pct":
            out.loc[:, aa_cols] = out.loc[:, aa_cols].astype(float) * 100.0
            out = out.rename(columns={c: c.replace("_freq", "_pct") for c in aa_cols})

        elif effective_aa_scale == "count":
            if length_col not in out.columns:
                raise ValueError(
                    f"Cannot convert AA frequencies to counts because '{length_col}' "
                    f"is not in the output."
                )
            out.loc[:, aa_cols] = out.loc[:, aa_cols].astype(float).mul(
                out[length_col].astype(float), axis=0
            )
            out = out.rename(columns={c: c.replace("_freq", "_count") for c in aa_cols})

    if "__rank_value__" in out.columns:
        out = out.drop(columns="__rank_value__")

    return out



def top_saa_proteins(
    df: pd.DataFrame,
    n: int = 10,
    *,
    value_scale: Literal["pct", "freq", "count"] = "pct",
    aa_freq_prefix: str = "aa_",
    ascending: bool = False,
    length_col: str = "aa_count_adjusted",
) -> pd.DataFrame:
    """
    Return the top-N proteins ranked by total sulfur amino acids (SAA = Met + Cys).

    This function ranks proteins based on the sum of methionine and cysteine
    composition derived from amino-acid frequency columns produced by
    load_proteome_metrics(include_aa_freq=True).

    Parameters
    ----------
    df : pd.DataFrame
        Protein-level metrics DataFrame containing amino-acid frequency columns.
    n : int, default 10
        Number of top-ranked proteins to return.
    value_scale : {"pct", "freq", "count"}, default "pct"
        Scale used for ranking:
        - "freq": sum of Met and Cys frequencies (0—1)
        - "pct": percentage (0—100)
        - "count": estimated residue count (requires sequence length column)
    aa_freq_prefix : str, default "aa_"
        Prefix used for amino-acid frequency columns (e.g., "aa_M_freq").
    ascending : bool, default False
        Sort order. False returns highest SAA values first.
    length_col : str, default "aa_count_adjusted"
        Sequence-length column required when value_scale="count".

    Returns
    -------
    pd.DataFrame
        Ranked table with:
        - rank (1-based)
        - rank_metric (e.g., "saa_pct")
        - selected identifier and metric columns

    Notes
    -----
    - The function resolves Met and Cys frequency columns automatically
    (supports uppercase and lowercase AA tokens).
    - If value_scale="count", the length column must be present in df.
    - Internally delegates final ranking to top_n_proteins().
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    m_col = _resolve_aa_freq_col(df, "M", aa_freq_prefix)
    c_col = _resolve_aa_freq_col(df, "C", aa_freq_prefix)

    if value_scale == "count" and length_col not in df.columns:
        raise ValueError(
            f"value_scale='count' requires length_col='{length_col}' in df."
        )

    out = df.copy()
    saa_freq = (
        pd.to_numeric(out[m_col], errors="coerce")
        + pd.to_numeric(out[c_col], errors="coerce")
    )

    if value_scale == "pct":
        out["__rank_value__"] = saa_freq * 100.0
    elif value_scale == "freq":
        out["__rank_value__"] = saa_freq
    else:
        out["__rank_value__"] = saa_freq * pd.to_numeric(out[length_col], errors="coerce")

    ranked = top_n_proteins(
        out,
        metric="__rank_value__",
        n=n,
        ascending=ascending,
        value_scale=value_scale,
        include_aa_freq=False,
        aa_value_scale="keep",  # defensive: prevent any AA scaling if toggled later
    )

    ranked["rank_metric"] = f"saa_{value_scale}"
    return ranked