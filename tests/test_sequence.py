from saa_proteome.sequence import prepare_sequence


def test_remove_true_nterminal_methionine():
    cleaned, adjusted, removed = prepare_sequence(
        "MACDX",
        canonical_only=True,
        remove_start_m=True,
    )

    assert cleaned == "MACD"
    assert adjusted == "ACD"
    assert removed is True


def test_do_not_remove_m_after_leading_noncanonical_residue():
    cleaned, adjusted, removed = prepare_sequence(
        "XMACD",
        canonical_only=True,
        remove_start_m=True,
    )

    assert cleaned == "MACD"
    assert adjusted == "MACD"
    assert removed is False


def test_retained_nterminal_methionine():
    cleaned, adjusted, removed = prepare_sequence(
        "MACD",
        canonical_only=True,
        remove_start_m=False,
    )

    assert cleaned == "MACD"
    assert adjusted == "MACD"
    assert removed is False


def test_empty_sequence():
    cleaned, adjusted, removed = prepare_sequence(
        "",
        canonical_only=True,
        remove_start_m=True,
    )

    assert cleaned == ""
    assert adjusted == ""
    assert removed is False


def test_single_methionine_sequence():
    cleaned, adjusted, removed = prepare_sequence(
        "M",
        canonical_only=True,
        remove_start_m=True,
    )

    assert cleaned == "M"
    assert adjusted == ""
    assert removed is True