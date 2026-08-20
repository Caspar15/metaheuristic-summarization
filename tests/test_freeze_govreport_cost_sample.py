import pytest

from scripts.audit.freeze_govreport_cost_sample import select_cost_rows


def test_cost_sample_has_three_disjoint_deterministic_strata():
    counts = {f"doc-{index:02d}": index for index in range(1, 61)}
    rows, targets = select_cost_rows(counts, per_stratum=10)

    assert len(rows) == 30
    assert len({row["id"] for row in rows}) == 30
    assert [row["stratum"] for row in rows].count("q10") == 10
    assert [row["stratum"] for row in rows].count("q50") == 10
    assert [row["stratum"] for row in rows].count("q90") == 10
    assert set(targets) == {"q10", "q50", "q90"}
    assert rows == select_cost_rows(dict(reversed(list(counts.items()))), per_stratum=10)[0]


def test_cost_sample_fails_loud_on_invalid_inputs():
    with pytest.raises(ValueError, match="not enough rows"):
        select_cost_rows({"a": 1}, per_stratum=1)
    with pytest.raises(ValueError, match="invalid sentence count"):
        select_cost_rows({f"doc-{i}": i for i in range(30)} | {"bad": -1})
