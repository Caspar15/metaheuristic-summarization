from scripts.audit.analyze_d3a_router_fusion_screen import select_full_dev_finalists


def _candidates():
    return [
        {"id": "R00_anchor", "family": "anchor"},
        {"id": "C1", "family": "capacity"},
        {"id": "C2", "family": "capacity"},
        {"id": "L1", "family": "lexical"},
        {"id": "L2", "family": "lexical"},
        {"id": "W1", "family": "weight"},
        {"id": "W2", "family": "weight"},
        {"id": "W3", "family": "weight"},
    ]


def test_finalist_rule_keeps_family_winners_then_best_threshold_candidate():
    scores = {
        "R00_anchor": 0.40,
        "C1": 0.399,
        "C2": 0.401,
        "L1": 0.404,
        "L2": 0.403,
        "W1": 0.410,
        "W2": 0.408,
        "W3": 0.401,
    }
    assert select_full_dev_finalists(_candidates(), scores) == [
        "R00_anchor", "C2", "L1", "W1", "W2"
    ]


def test_finalist_tie_uses_lower_candidate_id():
    scores = {row["id"]: 0.40 for row in _candidates()}
    assert select_full_dev_finalists(_candidates(), scores) == [
        "R00_anchor", "C1", "L1", "W1"
    ]
