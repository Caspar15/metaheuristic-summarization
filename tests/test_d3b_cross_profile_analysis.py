from scripts.audit.analyze_d3b_cross_profile_combination import promotion_decision


def _endpoint(**overrides):
    value = {
        "ci_lower": 0.001,
        "ci_upper": 0.01,
        "p_value_holm_8": 0.04,
        "p_value_selection_bonferroni_340": 0.04,
    }
    value.update(overrides)
    return value


def test_d3b_promotion_requires_every_frozen_check():
    comparison = {
        "rouge1": _endpoint(),
        "rouge2": _endpoint(),
        "rougeLsum": _endpoint(),
        "macro_rouge": _endpoint(),
    }
    assert promotion_decision(comparison, macro_point_delta=0.002)["eligible"] is True
    comparison["macro_rouge"] = _endpoint(
        p_value_selection_bonferroni_340=0.051
    )
    result = promotion_decision(comparison, macro_point_delta=0.002)
    assert result["eligible"] is False
    assert result["checks"]["macro_selection_bonferroni_340_le_005"] is False


def test_d3b_promotion_rejects_nonpositive_macro_or_negative_component():
    comparison = {
        "rouge1": _endpoint(),
        "rouge2": _endpoint(ci_lower=-0.02, ci_upper=-0.001),
        "rougeLsum": _endpoint(),
        "macro_rouge": _endpoint(),
    }
    result = promotion_decision(comparison, macro_point_delta=0.0)
    assert result["eligible"] is False
    assert result["checks"]["macro_point_positive"] is False
    assert result["checks"]["no_component_ci_entirely_negative"] is False
