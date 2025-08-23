from __future__ import annotations

from src.features.tf import tf_score
from src.features.tf_isf import compute_isf, tf_isf_score
from src.features.position import position_scores


def test_tf_and_tfisf_non_empty():
    sents = [["apple", "launches", "iphone"], ["apple", "prices", "rise"]]
    isf = compute_isf(sents, smooth=1.0)
    tf1 = tf_score(sents[0], norm="log")
    tf2 = tf_score(sents[1], norm="log")
    tfisf1 = tf_isf_score(sents[0], isf, norm="log")
    tfisf2 = tf_isf_score(sents[1], isf, norm="log")
    assert tf1 > 0 and tf2 > 0
    assert tfisf1 != 0 or tfisf2 != 0


def test_position_lead_bonus():
    scores = position_scores(3, lead_bonus=0.2)
    assert scores[0] == 0.2
    assert scores[1] == 0.0 and scores[2] == 0.0

