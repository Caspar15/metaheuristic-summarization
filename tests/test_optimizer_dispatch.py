"""Regression tests for optimizer dispatch semantics."""

import numpy as np
import pytest

import src.pipeline.optimizer_dispatch as dispatch


def _call(method, cfg=None, sim=None):
    return dispatch.dispatch_optimizer(
        method=method,
        sub_sentences=["a", "b"],
        sub_scores=[1.0, 0.5],
        sub_sim=sim,
        max_tokens=2,
        cfg=cfg or {},
        alpha=0.7,
        unit="tokens",
        max_sents=2,
    )


def test_nsga2_forwards_effective_parameters(monkeypatch):
    captured = {}

    def fake_nsga(*args, **kwargs):
        captured.update(kwargs)
        return [0]

    monkeypatch.setattr(dispatch, "nsga2_select", fake_nsga)
    result = _call(
        "nsga2",
        cfg={"optimizer": {"pop_size": 17, "n_gen": 23}, "seed": 41},
        sim=np.eye(2),
    )

    assert result == [0]
    assert captured["pop_size"] == 17
    assert captured["n_gen"] == 23
    assert captured["seed"] == 41
    assert captured["evaluator"].weights.salience == pytest.approx(1.0)
    assert captured["evaluator"].weights.facility_coverage == pytest.approx(0.8)
    assert captured["evaluator"].weights.redundancy == pytest.approx(0.7)


def test_nsga2_without_similarity_fails_loudly():
    with pytest.raises(ValueError, match="requires a similarity matrix"):
        _call("nsga2")


def test_mmr_forwards_lambda_and_shared_evaluator(monkeypatch):
    captured = {}

    def fake_mmr(*args, **kwargs):
        captured.update(kwargs)
        return [0]

    monkeypatch.setattr(dispatch, "mmr_select", fake_mmr)
    result = _call(
        "mmr",
        cfg={"optimizer": {"lambda_relevance": 0.63}},
        sim=np.eye(2),
    )
    assert result == [0]
    assert captured["lambda_relevance"] == pytest.approx(0.63)
    assert captured["evaluator"].weights.salience == pytest.approx(1.0)


def test_mmr_without_similarity_fails_loudly():
    with pytest.raises(ValueError, match="mmr requires a similarity matrix"):
        _call("mmr")


def test_unknown_method_fails_loudly():
    with pytest.raises(ValueError, match="Unknown optimizer method"):
        _call("typo-that-must-not-run-greedy")


def test_missing_nsga_dependency_is_not_silently_replaced(monkeypatch):
    def missing_dependency(*args, **kwargs):
        raise ImportError("pymoo is absent")

    monkeypatch.setattr(dispatch, "nsga2_select", missing_dependency)
    with pytest.raises(ImportError, match="pymoo is absent"):
        _call("nsga2", sim=np.eye(2))
