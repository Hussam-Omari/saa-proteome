import numpy as np
from saa_proteome.metrics import protein_metrics, aa_profile
from saa_proteome.rankings import top_saa_proteins
import pandas as pd

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