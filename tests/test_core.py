import numpy as np
import pandas as pd

from saa_proteome.rankings import top_saa_proteins
from saa_proteome.metrics import (
    aa_profile,
    max_group_pct_per_window,
    protein_metrics,
)


def test_saa_equals_met_plus_cys():
    seq = "MMMC"
    m = protein_metrics(seq, remove_start_m=False)
    assert np.isclose(float(m["saa_pct"]), float(m["met_pct"]) + float(m["cys_pct"]), equal_nan=True)  # type: ignore

def test_remove_start_m():
    seq = "MAAA"
    m1 = protein_metrics(seq, remove_start_m=False)
    m2 = protein_metrics(seq, remove_start_m=True)
    assert m1["aa_count_adjusted"] == 4
    assert m2["aa_count_adjusted"] == 3

def test_zero_length_after_removal():
    seq = "M"
    prof = aa_profile(seq, remove_start_m=True)
    assert prof["length"]["aa_count_adjusted"] == 0


def test_top_saa_scale_propagation():
    df = pd.DataFrame({
        "protein_id": ["p1", "p2"],
        "aa_M_freq": [0.1, 0.2],
        "aa_C_freq": [0.1, 0.1],
        "aa_count_adjusted": [100, 100],
    })

    out_pct = top_saa_proteins(df, n=1, value_scale="pct")
    out_freq = top_saa_proteins(df, n=1, value_scale="freq")
    out_count = top_saa_proteins(df, n=1, value_scale="count")

    assert out_pct["rank_metric"].iloc[0] == "saa_pct"
    assert out_freq["rank_metric"].iloc[0] == "saa_freq"
    assert out_count["rank_metric"].iloc[0] == "saa_count"


def test_localized_met_cys_and_saa():
    seq = "M" * 20 + "C" * 30 + "A" * 50

    result = protein_metrics(
        seq,
        remove_start_m=False,
        window_size=100,
        window_step=1,
    )

    assert np.isclose(result["max_met_per_window"], 20.0)
    assert np.isclose(result["max_cys_per_window"], 30.0)
    assert np.isclose(result["max_saa_per_window"], 50.0)


def test_short_sequence_uses_full_sequence():
    seq = "MMCCAAAAAA"

    result = max_group_pct_per_window(
        seq,
        group="MC",
        window_size=100,
        remove_start_m=False,
    )

    assert np.isclose(result, 40.0)


def test_max_saa_is_calculated_within_same_window():
    seq = "M" * 30 + "A" * 70 + "C" * 40 + "A" * 60

    max_met = max_group_pct_per_window(
        seq,
        group="M",
        window_size=100,
        remove_start_m=False,
    )

    max_cys = max_group_pct_per_window(
        seq,
        group="C",
        window_size=100,
        remove_start_m=False,
    )

    max_saa = max_group_pct_per_window(
        seq,
        group="MC",
        window_size=100,
        remove_start_m=False,
    )

    assert np.isclose(max_met, 30.0)
    assert np.isclose(max_cys, 40.0)
    assert np.isclose(max_saa, 40.0)
    assert not np.isclose(max_saa, max_met + max_cys)


def test_empty_sequence_window_metrics():
    result = protein_metrics("")

    assert np.isnan(result["max_met_per_window"])
    assert np.isnan(result["max_cys_per_window"])
    assert np.isnan(result["max_saa_per_window"])