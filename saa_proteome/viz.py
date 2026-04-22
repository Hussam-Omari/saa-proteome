"""
Visualization utilities for proteome amino-acid analyses.

This module provides helper functions to (i) construct plot-ready tables
(e.g., heatmap matrices and long-format amino-acid profiles) and (ii) render
standard publication-oriented plots using Matplotlib.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Optional, Tuple, Union, Literal, cast

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.colors import Colormap

from .io import read_fasta
from .sequence import clean_sequence, remove_nterm_m, normalize_protein_id
from .config import CANONICAL_AA_ORDER
# from .output_names import COLUMN_RENAME_MAP


# Heatmap matrix df
def aa_heatmap_matrix(
    df_top: pd.DataFrame,
    fasta_path: str,
    aa_order: Sequence[str] = CANONICAL_AA_ORDER,
    remove_start_m: bool = True,
    id_mode: str = "auto",
    canonical_only: bool = True,
) -> pd.DataFrame:
    """
    Build an amino-acid frequency matrix for a set of proteins.

    FASTA records are indexed by normalized identifiers (id_mode), and incoming
    df_top['protein_id'] values are normalized using the same mode before lookup.

    Parameters
    ----------
    df_top : pd.DataFrame
        Must contain a 'protein_id' column.
    fasta_path : str or Path
        Path to the proteome FASTA file.
    aa_order : sequence of str
        Amino-acid column order.
    remove_start_m : bool, default True
        If True, remove N-terminal methionine prior to frequency calculation.
    id_mode : str, default "auto"
        Identifier normalization mode used for FASTA lookup.
    canonical_only : bool, default True
        If True, restrict sequences to canonical amino acids.

    Returns
    -------
    pd.DataFrame
        Rows are proteins and columns are amino acids. Values are frequencies (0—1).
    """
    if "protein_id" not in df_top.columns:
        raise ValueError("df_top must contain a 'protein_id' column.")

    records = list(read_fasta(fasta_path))

    # FASTA lookup keyed by normalized ID
    id_to_record = {normalize_protein_id(str(r.id), mode=id_mode): r for r in records}

    # Normalize incoming IDs (handles raw FASTA tokens like 'sp|P12345|...')
    df_ids = (
        df_top["protein_id"]
        .astype(str)
        .map(lambda x: normalize_protein_id(x, mode=id_mode))
        .tolist()
    )

    matrix = []
    index = []
    missing = 0

    for pid in df_ids:
        rec = id_to_record.get(pid)
        if rec is None:
            missing += 1
            continue

        seq = clean_sequence(str(rec.seq), canonical_only=canonical_only)
        seq, _ = remove_nterm_m(seq, enabled=remove_start_m)

        L = len(seq)
        if L == 0:
            continue

        counts = Counter(seq)
        row = [counts.get(aa, 0) / L for aa in aa_order]

        matrix.append(row)
        index.append(pid)

    # Give warning when some requested protein IDs are not found after normalization.
    if missing > 0:
        import warnings
        warnings.warn(
            f"{missing} protein_id values were not found in FASTA after normalization "
            f"(id_mode='{id_mode}'). They were skipped.",
            RuntimeWarning,
        )

    return pd.DataFrame(matrix, index=index, columns=list(aa_order))



# Plot heatmap matrix
def plot_aa_heatmap(
    aa_df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    xlabel: str = "Amino Acid",
    ylabel: str = "Protein ID",
    figsize: Tuple[float, float] = (12, 7),
    font_scale: float = 1.0,
    cmap: Optional[Union[str, Colormap]] = "Purples",
    vmin: float = 0.0,
    vmax: Optional[float] = 0.30,
    absent_mask: Optional[pd.DataFrame] = None,
    absent_color: str = "#A9A9A9",
    linewidths: float = 0.2,
    linecolor: str = "white",
    cbar: bool = True,
    cbar_label: str = "Amino Acid Density",
    xtick_rotation: float = 0.0,
    ytick_rotation: float = 0.0,
    xtick_ha: str = "center",
    ytick_ha: str = "right",
    ax: Optional[Axes] = None,
    tight_layout: bool = True,
    save_plot: bool = False,
    save_filename: Optional[str] = None,
    dpi: int = 300,
) -> Axes:
    """
    Plot an amino-acid heatmap from a numeric matrix.

    Parameters
    ----------
    aa_df : pd.DataFrame
        Matrix to plot (rows=proteins, columns=amino acids). Values must be numeric.
    absent_mask : pd.DataFrame, optional
        Boolean mask indicating absent cells to overlay in gray. If None, cells
        equal to zero are treated as absent.
    save_plot : bool, default False
        If True, saves the figure to `save_filename` ('.png' appended if missing).

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the rendered heatmap.
    """
    if not isinstance(aa_df, pd.DataFrame):
        raise TypeError("aa_df must be a pandas DataFrame.")
    if aa_df.shape[0] == 0 or aa_df.shape[1] == 0:
        raise ValueError("aa_df must have at least 1 row and 1 column.")

    df_plot = aa_df.copy().apply(pd.to_numeric, errors="coerce")

    if absent_mask is None:
        absent_mask = df_plot.eq(0)
    else:
        if not isinstance(absent_mask, pd.DataFrame):
            raise TypeError("absent_mask must be a pandas DataFrame (same shape as aa_df).")
        if absent_mask.shape != df_plot.shape:
            raise ValueError("absent_mask must have the same shape as aa_df.")
        absent_mask = absent_mask.astype(bool)

    if vmax is None:
        arr = df_plot.to_numpy()
        finite_vals = arr[~np.isnan(arr)]
        vmax = float(finite_vals.max()) if finite_vals.size else 0.0

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    base_label = 12 * font_scale
    base_tick = 11 * font_scale
    base_title = 13 * font_scale

    # Main heatmap (mask absent cells)
    im = ax.imshow(
        df_plot.mask(absent_mask).to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    # Overlay absent cells in gray
    overlay = np.where(absent_mask.to_numpy(), 1.0, np.nan)
    ax.imshow(
        overlay,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([absent_color]),
        vmin=0,
        vmax=1,
    )
    # Cell borders (gridlines)
    if linewidths and linewidths > 0:
        ax.set_xticks(np.arange(-0.5, df_plot.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, df_plot.shape[0], 1), minor=True)
        ax.grid(which="minor", color=linecolor, linestyle="-", linewidth=linewidths)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Ticks and labels
    ax.set_xticks(np.arange(df_plot.shape[1]))
    ax.set_yticks(np.arange(df_plot.shape[0]))

    ax.set_xticklabels(list(df_plot.columns), fontsize=base_tick, rotation=xtick_rotation)
    plt.setp(ax.get_xticklabels(), ha=xtick_ha)

    ax.set_yticklabels(list(df_plot.index), fontsize=base_tick, rotation=ytick_rotation)
    plt.setp(ax.get_yticklabels(), ha=ytick_ha)

    ax.set_xlabel(xlabel, fontsize=base_label)
    ax.set_ylabel(ylabel, fontsize=base_label)

    if title:
        ax.set_title(title, fontsize=base_title, pad=10)

    # Colorbar
    if cbar:
        cbar_obj = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar_obj.set_label(cbar_label, fontsize=base_label)
        cbar_obj.ax.tick_params(labelsize=base_tick)

    if tight_layout and created_fig:
        plt.tight_layout()

    # Optional saving
    if save_plot:
        if not save_filename:
            raise ValueError("save_filename must be provided when save_plot=True.")

        filename = save_filename.strip()
        if not filename.lower().endswith(".png"):
            filename += ".png"

        out = Path(filename)
        if out.parent and str(out.parent) != ".":
            out.parent.mkdir(parents=True, exist_ok=True)

        fig = cast(Figure, ax.figure)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    return ax




# Multipurpose bar chart plot function
def plot_category_bars(
    data: Union[pd.DataFrame, pd.Series, Mapping[str, float]],
    *,
    category_col: str = "category",
    value_col: str = "value",
    sort: bool = True,
    ascending: bool = False,
    keep_order: Optional[Sequence[str]] = None,
    drop_zeros: bool = False,
    title: Optional[str] = None,
    xlabel: str = "Category",
    ylabel: str = "Value",
    figsize: Tuple[float, float] = (8, 5),
    font_scale: float = 1.0,
    rotation: float = 45,
    ha: str = "right",
    grid: bool = True,
    grid_alpha: float = 0.25,
    horizontal: bool = False,
    annotate: bool = False,
    annotate_fmt: str = "{:.3g}",
    ax: Optional[Axes] = None,
    show: bool = True,
) -> Axes:
    """
    Multipurpose categorical bar plot.

    Accepts:
    - DataFrame with [category_col, value_col]
    - Series (index=category, values=value)
    - dict-like Mapping (keys=category, values=value)

    Notes
    -----
    Any top-N selection should be done by the caller (explicit slicing),
    which keeps this plotting function purely visual.
    """
    if isinstance(data, pd.Series):
        df_plot = data.rename(value_col).reset_index()
        df_plot.columns = [category_col, value_col]
    elif isinstance(data, Mapping):
        df_plot = pd.DataFrame({category_col: list(data.keys()), value_col: list(data.values())})
    elif isinstance(data, pd.DataFrame):
        if category_col not in data.columns or value_col not in data.columns:
            raise ValueError(f"DataFrame must contain '{category_col}' and '{value_col}'.")
        df_plot = data[[category_col, value_col]].copy()
    else:
        raise TypeError("data must be a DataFrame, Series, or dict-like mapping.")

    df_plot[value_col] = pd.to_numeric(df_plot[value_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[value_col])

    if drop_zeros:
        df_plot = df_plot[df_plot[value_col] != 0]

    if df_plot.empty:
        raise ValueError("No data to plot after filtering.")

    if keep_order is not None:
        keep_order = list(keep_order)
        df_plot[category_col] = df_plot[category_col].astype(str)
        df_plot = (
            df_plot.set_index(category_col)
            .reindex(keep_order)
            .dropna(subset=[value_col])
            .reset_index()
        )
    else:
        if sort:
            df_plot = df_plot.sort_values(value_col, ascending=ascending)

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    base_label = 12 * font_scale
    base_tick = 11 * font_scale
    base_title = 13 * font_scale

    cats = df_plot[category_col].astype(str).tolist()
    vals = df_plot[value_col].astype(float).tolist()

    if horizontal:
        ax.barh(cats, vals, edgecolor="black", linewidth=0.8)
        ax.set_xlabel(ylabel, fontsize=base_label)
        ax.set_ylabel(xlabel, fontsize=base_label)
        if grid:
            ax.grid(True, axis="x", alpha=grid_alpha)
            ax.set_axisbelow(True)
        if annotate:
            for y, v in enumerate(vals):
                ax.text(v, y, " " + annotate_fmt.format(v), va="center", fontsize=base_tick)
    else:
        ax.bar(cats, vals, edgecolor="black", linewidth=0.8)
        ax.set_xlabel(xlabel, fontsize=base_label)
        ax.set_ylabel(ylabel, fontsize=base_label)

        ax.tick_params(axis="x", labelrotation=rotation, labelsize=base_tick)
        plt.setp(ax.get_xticklabels(), ha=ha)

        ax.tick_params(axis="y", labelsize=base_tick)

        if grid:
            ax.grid(True, axis="y", alpha=grid_alpha)
            ax.set_axisbelow(True)

        if annotate:
            ymax = float(np.nanmax(vals)) if len(vals) else 0.0
            bump = 0.01 * ymax if ymax else 0.0
            for x, v in enumerate(vals):
                ax.text(x, v + bump, annotate_fmt.format(v), ha="center", va="bottom", fontsize=base_tick)

    if title:
        ax.set_title(title, fontsize=base_title)

    if created_fig:
        plt.tight_layout()

    if show:
        plt.show()

    return ax



# Plot bar chart
def plot_bar(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
    font_scale: float = 1.0,
    color: Optional[Union[str, Sequence[str]]] = "#c0504d",
    cmap: Optional[str] = None,
    edgecolor: str = "black",
    linewidth: float = 0.8,
    alpha: float = 1.0,
    xtick_rotation: float = 0,
    xtick_ha: str = "center",
    grid: bool = True,
    grid_axis: Literal["x", "y", "both"] = "y",
    grid_alpha: float = 0.25,
    ax: Optional[Axes] = None,
    tight_layout: bool = True,
    save_plot: bool = False,
    save_filename: Optional[str] = None,
    dpi: int = 300,
) -> Axes:
    """
    General-purpose bar chart plotting function.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least columns `x` and `y`.
    x : str
        Column name used for categorical x-axis values.
    y : str
        Column name used for numeric bar heights.
    hue : str, optional
        If provided, bars are grouped by this column.
        Duplicate (hue, x) pairs are aggregated using mean.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label. Defaults to `x` if None.
    ylabel : str, optional
        Y-axis label. Defaults to `y` if None.
    figsize : tuple of float, default (8, 5)
        Figure size in inches.
    font_scale : float, default 1.0
        Scaling factor for axis labels, ticks, and title.
    color : str or sequence of str, optional
        Bar color(s) when `hue` is None and `cmap` is not used.
    cmap : str, optional
        Matplotlib colormap name. If provided, overrides `color`.
    edgecolor : str, default "black"
        Bar edge color.
    linewidth : float, default 0.8
        Bar edge width.
    alpha : float, default 1.0
        Bar transparency.
    xtick_rotation : float, default 0
        Rotation angle for x-axis tick labels.
    xtick_ha : str, default "center"
        Horizontal alignment for x-axis tick labels.
    grid : bool, default False
        Whether to draw grid lines.
    grid_axis : {"x", "y", "both"}, default "y"
        Axis along which grid lines are drawn.
    grid_alpha : float, default 0.25
        Grid transparency.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure and axes are created.
    tight_layout : bool, default True
        Apply tight layout when a new figure is created.
    save_plot : bool, default False
        If True, save the figure to `save_filename`.
    save_filename : str, optional
        Output filename ('.png' appended if missing).
    dpi : int, default 300
        Resolution used when saving the figure.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the rendered bar plot.

    Notes
    -----
    - If `hue` is None, one bar is drawn per row in `df`.
    - If `hue` is provided, bars are grouped by `x` across categories.
    - When `cmap` is provided, it takes precedence over `color`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columns '{x}' and/or '{y}' not found in DataFrame.")

    df_plot = df.copy()
    df_plot[x] = df_plot[x].astype(str)
    df_plot[y] = pd.to_numeric(df_plot[y], errors="coerce")
    df_plot = df_plot.dropna(subset=[y])

    if hue is not None:
        if hue not in df_plot.columns:
            raise ValueError(f"Column '{hue}' not found in DataFrame.")
        df_plot[hue] = df_plot[hue].astype(str)

    if df_plot.empty:
        raise ValueError("No data to plot after cleaning (all y values are NaN?).")

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    base_label = 12 * font_scale
    base_tick = 11 * font_scale
    base_title = 13 * font_scale

    if hue is None:
        categories = df_plot[x].tolist()
        values = df_plot[y].astype(float).tolist()

        if cmap:
            cm = plt.get_cmap(cmap)
            colors = cm(np.linspace(0.3, 0.8, len(values)))
        else:
            if isinstance(color, (list, tuple, np.ndarray)):
                colors = list(color)
            else:
                colors = color if color else "#4C72B0"

        ax.bar(
            categories,
            values,
            color=colors,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )

    else:
        categories = pd.unique(df_plot[x]).tolist()
        groups = pd.unique(df_plot[hue]).tolist()

        width = 0.8 / max(len(groups), 1)
        x_positions = np.arange(len(categories))

        # Aggregate duplicates per (hue, x) using mean (robust default)
        agg = df_plot.groupby([hue, x], as_index=False)[y].mean()

        for i, g in enumerate(groups):
            sub = agg[agg[hue] == g]

            y_map = sub.set_index(x)[y].to_dict()
            values = [float(y_map.get(cat, 0.0)) for cat in categories]

            if cmap:
                cm = plt.get_cmap(cmap)
                color_i = cm(0.3 + 0.6 * i / max(len(groups) - 1, 1))
            else:
                color_i = None

            ax.bar(
                x_positions + i * width,
                values,
                width=width,
                label=str(g),
                color=color_i,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
            )

        ax.set_xticks(x_positions + width * (len(groups) - 1) / 2)
        ax.set_xticklabels(categories)

    # Common formatting (must run for BOTH cases)
    ax.set_xlabel(xlabel if xlabel else x, fontsize=base_label)
    ax.set_ylabel(ylabel if ylabel else y, fontsize=base_label)

    ax.tick_params(axis="x", labelrotation=xtick_rotation, labelsize=base_tick)
    plt.setp(ax.get_xticklabels(), ha=xtick_ha)
    ax.tick_params(axis="y", labelsize=base_tick)

    if title:
        ax.set_title(title, fontsize=base_title)

    if hue:
        ax.legend(title=hue, fontsize=base_tick)

    if grid:
        ax.grid(True, axis=grid_axis, alpha=grid_alpha)
        ax.set_axisbelow(True)

    if tight_layout and created_fig:
        plt.tight_layout()

    # Optional saving (works for both cases)
    if save_plot:
        if not save_filename:
            raise ValueError("save_filename must be provided when save_plot=True.")

        filename = save_filename.strip()
        if not filename.lower().endswith(".png"):
            filename += ".png"

        from pathlib import Path
        out = Path(filename)
        if out.parent and str(out.parent) != ".":
            out.parent.mkdir(parents=True, exist_ok=True)

        from matplotlib.figure import Figure
        from typing import cast
        fig = cast(Figure, ax.figure)

        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    return ax



# Convert output tables from wide to long for visualization purposes
def aa_profile_table(
    df_aa: pd.DataFrame,
    *,
    aa_prefix: str = "aa_",
    value_name: str = "Percentage",
    aa_colname: str = "AA",
    canonical_order: Sequence[str] = CANONICAL_AA_ORDER,
    sort: bool = False,
    row: int = 0,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Convert a wide-format amino-acid profile table (aa_* columns) into a long table.

    Input (wide):
        protein_id, aa_A, aa_C, ..., aa_Y   (typically 1 row)

    Output (long):
        AA, Percentage
        A,  6.68
        C,  1.82
        ...

    Notes
    -----
    This function ONLY prepares a plot-ready table.
    Plotting is handled by plot_bar().
    """
    if not isinstance(df_aa, pd.DataFrame):
        raise TypeError("df_aa must be a pandas DataFrame.")

    if df_aa.empty:
        raise ValueError("df_aa is empty.")

    aa_cols = [f"{aa_prefix}{aa}"
               for aa in CANONICAL_AA_ORDER
               if f"{aa_prefix}{aa}" in df_aa.columns
               ]
    if not aa_cols:
        raise ValueError(f"No columns starting with '{aa_prefix}' were found.")

    if row < 0 or row >= len(df_aa):
        raise IndexError(f"row must be between 0 and {len(df_aa) - 1}.")

    # Use selected row (default: first row)
    s = df_aa.iloc[row][aa_cols]

    # Robust conversion (works across pandas versions)
    df_long = s.to_frame(value_name).reset_index()
    df_long.columns = [aa_colname, value_name]

    # Clean AA labels and ensure numeric values
    df_long[aa_colname] = df_long[aa_colname].astype(str).str.replace(aa_prefix, "", regex=False)
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors="coerce")

    if dropna:
        df_long = df_long.dropna(subset=[value_name])

    # Order handling
    if not sort:
        canonical = list(canonical_order)
        df_long[aa_colname] = pd.Categorical(df_long[aa_colname], categories=canonical, ordered=True)
        df_long = df_long.sort_values(aa_colname)
    else:
        df_long = df_long.sort_values(value_name, ascending=False)

    return df_long.reset_index(drop=True)


