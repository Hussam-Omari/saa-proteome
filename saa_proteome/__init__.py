"""
saa_proteome

A small, reproducible library for proteome-wide amino-acid composition analysis
with special support for sulfur amino acids (Met, Cys) and localized enrichment
of individual residues and user-defined amino-acid groups.

Core features:
- FASTA validation and reading
- Canonical 20-AA sequence cleaning
- Optional N-terminal methionine removal (noStartM)
- Per-protein metrics: Met%, Cys%, SAA%, and maximum localized Met%, Cys%, and SAA%
- Sliding-window enrichment analysis for individual residues and user-defined amino-acid groups
- Full canonical AA composition tables (counts, frequency, percentage)
- Essential amino acid (EAA) statistics
- User-defined amino acid group statistics
- Top-N ranking tables
- QC and summary tables as pandas DataFrames
- Friendly column naming utilities
- Interactive data dictionary preview

All batch outputs are DataFrames by design.
"""

from .config import CANONICAL_AA_ORDER, CANONICAL_AA_SET, EAA_PROFILES
from .io import is_fasta, validate_fasta, read_fasta, save_output

from .sequence import (
    clean_sequence,
    remove_nterm_m,
    prepare_sequence,
    normalize_protein_id,
)

from .metrics import (
    aa_profile,
    group_stats,
    essential_aa_set,
    essential_aa_stats,
    max_group_pct_per_window,
    max_residue_pct_per_window,
    protein_metrics,
)

from .proteome import (
    load_proteome_metrics,
    aa_composition_df,
    eaa_summary,
    saa_composition_df,
    proteome_to_df
)

from .summaries import qc_report, saa_summary, length_class_summary
from .rankings import top_n_proteins, top_saa_proteins
from .viz import aa_heatmap_matrix, plot_aa_heatmap, plot_bar, aa_profile_table


# Friendly naming & data dictionary utilities
from .output_names import (
    apply_friendly_column_names,
    data_dictionary_df,
    preview_data_dictionary,
)


__all__ = [
    "CANONICAL_AA_ORDER",
    "CANONICAL_AA_SET",
    "EAA_PROFILES",
    "is_fasta",
    "validate_fasta",
    "read_fasta",
    "clean_sequence",
    "remove_nterm_m",
    "prepare_sequence",
    "normalize_protein_id",
    "aa_profile",
    "group_stats",
    "proteome_to_df",
    "load_proteome_metrics",
    "eaa_summary",
    "essential_aa_set",
    "essential_aa_stats",
    "max_group_pct_per_window",
    "max_residue_pct_per_window",
    "protein_metrics",
    "aa_composition_df",
    "qc_report",
    "saa_summary",
    "length_class_summary",
    "top_n_proteins",
    "top_saa_proteins",
    "aa_heatmap_matrix",
    "apply_friendly_column_names",
    "data_dictionary_df",
    "preview_data_dictionary",
    "save_output",
    "plot_aa_heatmap",
    "plot_bar",
    "aa_profile_table",
    "saa_composition_df",
    ]

__version__ = "0.4.1"