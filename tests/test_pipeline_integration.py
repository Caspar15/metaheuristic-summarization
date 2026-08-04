"""Integration tests for the full pipeline."""

import pytest

from src.data.schemas import build_document_example
import json

from src.pipeline.select_sentences import (
    build_feasibility_report,
    summarize_jsonl,
    summarize_one,
    validate_requested_split,
)


@pytest.fixture
def sample_doc():
    return {
        "id": "test_001",
        "sentences": [
            "This is the first sentence about artificial intelligence.",
            "Machine learning is a subset of AI.",
            "Deep learning has revolutionized many fields.",
            "Natural language processing is important.",
            "Computer vision has many applications.",
        ],
        "highlights": "AI and machine learning are transforming technology.",
    }


@pytest.fixture
def base_config():
    return {
        "optimizer": {"method": "greedy"},
        "length_control": {"unit": "tokens", "max_tokens": 30},
        "redundancy": {"lambda": 0.7},
        "representations": {"use": True, "method": "tfidf"},
        "candidates": {"use": False},
    }


class TestPipelineGreedy:
    def test_basic(self, sample_doc, base_config):
        result = summarize_one(sample_doc, base_config)
        assert "summary" in result
        assert "selected_indices" in result
        assert "id" in result
        assert result["id"] == "test_001"
        assert len(result["summary"]) > 0

    def test_indices_valid(self, sample_doc, base_config):
        result = summarize_one(sample_doc, base_config)
        n = len(sample_doc["sentences"])
        assert all(0 <= i < n for i in result["selected_indices"])

    def test_indices_sorted(self, sample_doc, base_config):
        result = summarize_one(sample_doc, base_config)
        assert result["selected_indices"] == sorted(result["selected_indices"])

    def test_summary_matches_indices(self, sample_doc, base_config):
        result = summarize_one(sample_doc, base_config)
        expected = "\n".join(sample_doc["sentences"][i] for i in result["selected_indices"])
        assert result["summary"] == expected
        assert result["summary_sentences"] == [
            sample_doc["sentences"][i] for i in result["selected_indices"]
        ]


class TestPipelineGrasp:
    def test_grasp(self, sample_doc, base_config):
        base_config["optimizer"]["method"] = "grasp"
        base_config["seed"] = 42
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0

    def test_grasp_no_feasible_solution_is_recorded(self, base_config, monkeypatch):
        doc = {
            "id": "grasp-min-shortfall",
            "sentences": [
                "aa bb",
                "one two three four five six seven eight nine",
            ],
            "highlights": "Reference.",
        }
        base_config["optimizer"] = {"method": "grasp"}
        base_config["grasp"] = {"iters": 5, "rcl_ratio": 0.5}
        base_config["seed"] = 0
        base_config["length_control"] = {
            "unit": "words",
            "max_words": 10,
            "min_words": 8,
        }
        base_config["redundancy"] = {"lambda": 0.0}
        base_config["objectives"] = {
            "importance_aggregation": "sum",
            "coverage_method": "max",
            "lambda_importance": 1.0,
            "lambda_coverage": 0.0,
            "lambda_redundancy": 0.0,
        }
        monkeypatch.setattr(
            "src.pipeline.select_sentences.build_base_scores",
            lambda *args, **kwargs: [10.0, 1.0],
        )

        result = summarize_one(doc, base_config)

        assert result["selected_indices"] == [0]
        assert result["feasible"] is False
        assert result["infeasible_code"] == "optimizer_no_feasible_solution"
        assert result["violations"]["min_words"] == 6.0


class TestPipelineWithV2Features:
    def test_v2_tf_isf(self, sample_doc, base_config):
        base_config["features"] = {
            "tf_isf": {"version": "v2", "use_stopwords": True},
        }
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0


class TestPipelineNsga2:
    def test_pareto_front_is_preserved_in_artifact(self, sample_doc, base_config):
        pytest.importorskip("pymoo")
        base_config["optimizer"] = {"method": "nsga2", "pop_size": 12, "n_gen": 6}
        base_config["seed"] = 2024
        result = summarize_one(sample_doc, base_config)

        diagnostics = result["optimizer_diagnostics"]
        assert diagnostics["method"] == "nsga2"
        assert diagnostics["pareto_size"] == len(diagnostics["pareto_front"])
        assert diagnostics["pareto_size"] > 0
        selected_solution = diagnostics["pareto_front"][
            diagnostics["selected_pareto_row"]
        ]
        assert selected_solution["selected_indices"] == result["selected_indices"]
        assert selected_solution["feasible"] is True

    def test_nsga2_impossible_candidate_floor_is_recorded(
        self, base_config, monkeypatch
    ):
        pytest.importorskip("pymoo")
        doc = {
            "id": "nsga2-candidate-shortfall",
            "sentences": [
                "aa bb",
                "one two three four five six seven eight nine",
            ],
            "highlights": "Reference.",
        }
        base_config["optimizer"] = {"method": "nsga2", "pop_size": 8, "n_gen": 3}
        base_config["seed"] = 0
        base_config["length_control"] = {
            "unit": "words",
            "max_words": 10,
            "min_words": 8,
        }
        base_config["candidates"] = {
            "use": True,
            "mode": "hard",
            "sources": ["score"],
        }
        base_config["candidate_budget"] = {"route_top_k": 1, "total": 1}
        monkeypatch.setattr(
            "src.pipeline.select_sentences.build_base_scores",
            lambda *args, **kwargs: [10.0, 1.0],
        )

        result = summarize_one(doc, base_config)

        assert result["feasible"] is False
        assert result["infeasible_code"] == "candidate_capacity_shortfall"
        assert result["violations"]["min_words"] > 0
    def test_v2_position(self, sample_doc, base_config):
        base_config["features"] = {
            "position": {"version": "v2", "method": "inverse"},
        }
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0

    def test_v2_fusion(self, sample_doc, base_config):
        base_config["features"] = {
            "fusion": {"version": "v2"},
            "weights": {"importance": 0.8, "length": 0.2, "position": 0.3},
        }
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0

    def test_semantic_features(self, sample_doc, base_config):
        base_config["features"] = {
            "weights": {
                "importance": 0.6,
                "length": 0.1,
                "position": 0.2,
                "centrality": 0.3,
                "novelty": 0.2,
            },
        }
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0


class TestCandidateBudgetContract:
    def test_pipeline_reports_separate_candidate_budget(self, sample_doc, base_config):
        base_config["candidates"] = {
            "use": True,
            "mode": "hard",
            "sources": ["score"],
        }
        base_config["candidate_budget"] = {
            "route_top_k": 4,
            "min_per_route": 1,
            "total": 2,
        }
        result = summarize_one(sample_doc, base_config)
        assert result["candidate_pool"]["route_top_k"] == 4
        assert result["candidate_pool"]["min_per_route"] == 1
        assert result["candidate_pool"]["total_cap"] == 2
        assert result["candidate_pool"]["actual_size"] == 2

    def test_fixed_compute_budget_controls_enabled_routes(
        self, sample_doc, base_config
    ):
        base_config["candidates"] = {
            "use": True,
            "mode": "hard",
            "sources": ["imaginary_route"],
        }
        base_config["compute_budget"] = {
            "mode": "fixed",
            "enabled_routes": ["lexical"],
        }
        base_config["candidate_budget"] = {"route_top_k": 2, "total": 2}
        result = summarize_one(sample_doc, base_config)
        assert result["candidate_pool"]["configured_sources"] == ["lexical"]

    def test_rrf_provenance_is_the_score_received_by_selector(
        self, sample_doc, base_config, monkeypatch
    ):
        captured = {}

        def fake_encoder_route_scores(sentences, **kwargs):
            return [0.1, 0.2, 0.3, 0.8, 0.9], {
                "model_name": kwargs["model_name"],
                "model_revision": "fake-semantic-revision",
                "max_model_tokens": kwargs["max_model_tokens"],
                "effective_max_model_tokens": kwargs["max_model_tokens"],
                "batch_size": kwargs["batch_size"],
                "estimated_cost": {
                    "encoded_sentences": len(sentences),
                    "input_tokens_before_truncation": 25,
                },
                "truncated_sentences": 0,
                "truncation_rate": 0.0,
            }

        def fake_dispatch(*args, **kwargs):
            captured["scores"] = list(args[2])
            captured["coverage_shape"] = args[-1].shape
            return [0]

        monkeypatch.setattr(
            "src.pipeline.select_sentences.dispatch_optimizer", fake_dispatch
        )
        monkeypatch.setattr(
            "src.pipeline.candidate_builder.encoder_route_scores",
            fake_encoder_route_scores,
        )
        base_config["candidates"] = {
            "use": True,
            "mode": "hard",
            "sources": ["lexical", "semantic"],
        }
        base_config["candidate_budget"] = {
            "route_top_k": 3,
            "min_per_route": 1,
            "total": 4,
        }
        base_config["routes"] = {
            "semantic": {
                "model_name": "sentence-transformers/fake",
                "max_model_tokens": 64,
            }
        }
        base_config["selector"] = {"salience_source": "rrf_fusion"}
        result = summarize_one(sample_doc, base_config)
        expected = [
            candidate["fusion_normalized"]
            for candidate in result["candidate_records"]
        ]
        lexical = [
            candidate["route_scores"]["lexical"]["raw"]
            for candidate in result["candidate_records"]
        ]
        assert captured["scores"] == pytest.approx(expected)
        assert captured["scores"] != pytest.approx(lexical)
        assert captured["coverage_shape"] == (len(sample_doc["sentences"]), 4)
        assert all(
            candidate["selector_salience_source"] == "rrf_fusion"
            for candidate in result["candidate_records"]
        )
        assert result["selected_sentences"][0]["selection_evidence"][
            "selector_salience_source"
        ] == "rrf_fusion"

    def test_unimplemented_adaptive_router_fails_loudly(
        self, sample_doc, base_config
    ):
        base_config["candidates"] = {"use": True}
        base_config["compute_budget"] = {"mode": "adaptive"}
        with pytest.raises(ValueError, match="adaptive routing"):
            summarize_one(sample_doc, base_config)

    def test_word_budget_still_dispatches_configured_selector(
        self, sample_doc, base_config, monkeypatch
    ):
        captured = {}

        def fake_dispatch(*args, **kwargs):
            captured["max_budget"] = args[4]
            captured["unit"] = args[7]
            return [0]

        monkeypatch.setattr(
            "src.pipeline.select_sentences.dispatch_optimizer", fake_dispatch
        )
        base_config["length_control"] = {"unit": "words", "max_words": 17}
        result = summarize_one(sample_doc, base_config)
        assert result["selected_indices"] == [0]
        assert captured == {"max_budget": 17, "unit": "words"}


class TestPipelineEdgeCases:
    def test_empty_sentences(self, base_config):
        doc = {"id": "empty", "sentences": [], "highlights": ""}
        result = summarize_one(doc, base_config)
        assert result["selected_indices"] == []
        assert result["summary"] == ""
        assert result["feasible"] is False
        assert result["infeasible_code"] == "empty_source"
        assert result["violations"]["nonempty"] > 0

    def test_single_sentence(self, base_config):
        doc = {"id": "single", "sentences": ["Hello world."], "highlights": "Hello."}
        result = summarize_one(doc, base_config)
        assert result["selected_indices"] == [0]

    def test_no_sentence_fits_is_recorded_not_raised(self, base_config):
        doc = {
            "id": "all-oversized",
            "sentences": ["one two three four five six"],
            "highlights": "Reference.",
        }
        base_config["length_control"] = {
            "unit": "words",
            "max_words": 5,
            "min_words": 0,
        }

        result = summarize_one(doc, base_config)

        assert result["selected_indices"] == []
        assert result["feasible"] is False
        assert result["infeasible_code"] == "source_no_eligible_sentence"
        assert result["violations"]["nonempty"] > 0

    def test_canonical_document_is_supported(self, base_config):
        doc = build_document_example(
            example_id="canonical",
            split="test",
            documents=[["Hello world.", "A second sentence."]],
            references=["First reference.", "Second reference."],
            input_mode="single_document",
            output_mode="single_sentence",
        )
        result = summarize_one(doc, base_config)
        assert result["id"] == "canonical"
        assert "references" not in result
        assert len(result["selected_indices"]) == 1
        assert result["objective_spec"]["active"] == ["salience"]
        assert all(item["document_id"] is not None for item in result["selected_sentences"])

    def test_single_sentence_task_rejects_nsga2(self, base_config):
        doc = build_document_example(
            example_id="single-rank-only",
            split="validation",
            documents=[["One sentence.", "Another sentence."]],
            references=["Reference."],
            input_mode="single_document",
            output_mode="single_sentence",
        )
        base_config["optimizer"]["method"] = "nsga2"
        with pytest.raises(ValueError, match="deterministic ranking"):
            summarize_one(doc, base_config)

    def test_profiled_multi_sentence_defaults_to_mean_importance(self, base_config):
        doc = build_document_example(
            example_id="multi-objectives",
            split="validation",
            documents=[["One sentence.", "Another sentence."]],
            references=["Reference."],
            input_mode="single_document",
            output_mode="multi_sentence",
        )
        result = summarize_one(doc, base_config)
        assert result["objective_spec"]["importance_aggregation"] == "mean"
        assert result["objective_spec"]["active"] == [
            "salience",
            "facility_coverage",
            "redundancy",
        ]
        assert result["selection_evaluation"]["feasible"] is True
        assert result["selection_evaluation"]["selected_indices"] == [0]
        assert result["selection_evaluation"]["coverage_universe_size"] == 2
        assert result["objective_spec"]["coverage_scope"] == "full_source_sentences"

    def test_source_intrinsic_minimum_shortfall_is_relaxed_and_recorded(
        self, base_config
    ):
        doc = build_document_example(
            example_id="impossible-min",
            split="validation",
            documents=[["Only two words."]],
            references=["Reference."],
            input_mode="single_document",
            output_mode="multi_sentence",
        )
        base_config["length_control"] = {
            "unit": "words",
            "max_words": 10,
            "min_words": 5,
        }
        result = summarize_one(doc, base_config)
        budget = result["output_budget"]
        assert result["selected_indices"] == [0]
        assert budget["requested_min_words"] == 5
        assert budget["effective_min_words"] == 3
        assert budget["source_capacity_words"] == 3
        assert budget["candidate_capacity_words"] == 3
        assert budget["min_words_relaxed"] is True
        assert budget["relaxation_reason"] == "source_intrinsic_capacity"

    def test_candidate_pool_cannot_hide_a_feasible_source(self, base_config):
        doc = build_document_example(
            example_id="candidate-infeasible",
            split="validation",
            documents=[["one two three", "four five six seven eight six"]],
            references=["Reference."],
            input_mode="single_document",
            output_mode="multi_sentence",
        )
        base_config["length_control"] = {
            "unit": "words",
            "max_words": 10,
            "min_words": 6,
        }
        base_config["candidates"] = {
            "use": True,
            "mode": "hard",
            "sources": ["score"],
        }
        base_config["candidate_budget"] = {"route_top_k": 1, "total": 1}
        base_config["features"] = {
            "weights": {
                "importance": 0.0,
                "length": 0.0,
                "position": 1.0,
                "graph": 0.0,
                "centrality": 0.0,
                "novelty": 0.0,
            }
        }
        result = summarize_one(doc, base_config)
        assert result["feasible"] is False
        assert result["infeasible_code"] == "candidate_capacity_shortfall"
        assert result["output_budget"]["candidate_capacity_words"] == 3
        assert result["output_budget"]["effective_min_words"] == 6

    def test_min_words_only_shortfall_is_recorded_not_raised(self, base_config):
        """F-17 option 1: greedy has no backtracking, so it can commit to a
        short high-utility sentence, then be unable to add the only other
        sentence without exceeding max_words -- even though that other
        sentence alone would have reached min_words. The document must not
        abort the run; it must come back marked infeasible."""
        doc = {
            "id": "min-words-shortfall",
            "sentences": ["aa bb", "cc dd ee ff gg hh ii jj kk"],
        }
        base_config["length_control"] = {
            "unit": "words",
            "max_words": 10,
            "min_words": 8,
        }
        result = summarize_one(doc, base_config)
        assert result["selected_indices"] == [0]
        assert result["feasible"] is False
        assert "effective_min_words" in result["infeasible_reason"]
        assert result["violations"]["min_words"] > 0
        assert result["violations"]["max_length"] <= 0
        assert result["violations"]["max_sentences"] <= 0
        assert result["violations"]["nonempty"] <= 0
        assert result["selection_evaluation"]["feasible"] is False

    def test_non_min_words_violation_from_optimizer_still_aborts(
        self, sample_doc, base_config, monkeypatch
    ):
        """Only an unreachable min_words floor is downgraded to a recorded
        row. Any other violation reaching this point means an upper bound the
        selector is supposed to enforce by construction was broken -- a real
        bug, not a packing limitation -- and must still abort the whole run."""
        from src.objectives.evaluator import InfeasibleSelectionError, SelectionEvaluation

        def fake_dispatch(*args, **kwargs):
            raise InfeasibleSelectionError(
                "selector returned an infeasible summary: {'max_length': 1.0}",
                SelectionEvaluation(
                    selected_indices=[0],
                    salience=0.0,
                    facility_coverage=0.0,
                    redundancy=0.0,
                    scalar_utility=0.0,
                    selected_words=5,
                    selected_sentences=1,
                    coverage_universe_size=0,
                    feasible=False,
                    violations={
                        "nonempty": 0.0,
                        "min_words": 0.0,
                        "max_length": 1.0,
                        "max_sentences": 0.0,
                    },
                ),
            )

        monkeypatch.setattr(
            "src.pipeline.select_sentences.dispatch_optimizer", fake_dispatch
        )
        with pytest.raises(InfeasibleSelectionError):
            summarize_one(sample_doc, base_config)

    def test_over_budget_sentence_does_not_consume_candidate_quota(
        self, base_config
    ):
        doc = build_document_example(
            example_id="oversized-sentence",
            split="validation",
            documents=[["oversized " * 11, "short usable sentence"]],
            references=["Reference."],
            input_mode="single_document",
            output_mode="multi_sentence",
        )
        base_config["length_control"] = {
            "unit": "words",
            "max_words": 10,
            "min_words": 0,
        }
        base_config["candidates"] = {
            "use": True,
            "mode": "hard",
            "sources": ["score"],
        }
        base_config["candidate_budget"] = {"route_top_k": 1, "total": 1}
        result = summarize_one(doc, base_config)
        assert [item["original_index"] for item in result["candidate_records"]] == [1]
        assert result["candidate_pool"]["selection_ineligible_sentences"] == [
            {
                "sentence_id": "oversized-sentence:d000:s000000",
                "original_index": 0,
                "word_count": 11,
                "reason": "exceeds_active_output_budget",
            }
        ]

    def test_profiled_multi_sentence_rejects_sum_importance(self, base_config):
        doc = build_document_example(
            example_id="biased-sum",
            split="validation",
            documents=[["One sentence.", "Another sentence."]],
            references=["Reference."],
            input_mode="single_document",
            output_mode="multi_sentence",
        )
        base_config["objectives"] = {"importance_aggregation": "sum"}
        with pytest.raises(ValueError, match="cardinality bias"):
            summarize_one(doc, base_config)

    def test_gold_reference_cannot_change_selection(self, sample_doc, base_config):
        first = dict(sample_doc, highlights="Completely unrelated gold text.")
        second = dict(sample_doc, highlights="Machine learning AI deep learning.")
        assert summarize_one(first, base_config)["selected_indices"] == summarize_one(
            second, base_config
        )["selected_indices"]

    def test_reference_dependent_candidate_sizing_is_forbidden(self, sample_doc, base_config):
        base_config["candidates"] = {"use": True, "recall_target": 0.9}
        with pytest.raises(ValueError, match="gold references"):
            summarize_one(sample_doc, base_config)

    def test_canonical_split_mismatch_is_forbidden(self):
        doc = build_document_example(
            example_id="validation-row",
            split="validation",
            documents=[["One sentence."]],
            references=["Reference."],
            input_mode="single_document",
            output_mode="single_sentence",
        )
        with pytest.raises(ValueError, match="belongs to split 'validation'"):
            validate_requested_split(doc, "test")

    def test_jsonl_runner_streams_into_atomic_writer(
        self, sample_doc, base_config, monkeypatch
    ):
        captured = {}
        monkeypatch.setattr(
            "src.pipeline.select_sentences.read_jsonl", lambda _path: iter([sample_doc])
        )

        def capture_writer(path, rows):
            captured["path"] = path
            captured["rows"] = list(rows)

        monkeypatch.setattr(
            "src.pipeline.select_sentences.write_jsonl_atomic", capture_writer
        )
        count = summarize_jsonl(
            "input.jsonl", "predictions.jsonl", base_config, "test"
        )
        assert count == 1
        assert captured["path"] == "predictions.jsonl"
        assert len(captured["rows"]) == 1

    def test_jsonl_runner_continues_past_an_infeasible_document(
        self, sample_doc, base_config, monkeypatch
    ):
        """F-17: one infeasible document must not throw away the whole run.
        predictions.jsonl must still get a row for every input document, with
        the infeasible one marked rather than silently dropped or aborting."""
        infeasible_doc = {
            "id": "min-words-shortfall",
            "sentences": ["aa bb", "cc dd ee ff gg hh ii jj kk"],
        }
        docs = [sample_doc, infeasible_doc]
        monkeypatch.setattr(
            "src.pipeline.select_sentences.read_jsonl", lambda _path: iter(docs)
        )
        captured = {}

        def capture_writer(path, rows):
            captured["path"] = path
            captured["rows"] = list(rows)

        monkeypatch.setattr(
            "src.pipeline.select_sentences.write_jsonl_atomic", capture_writer
        )
        base_config["length_control"] = {
            "unit": "words",
            "max_words": 10,
            "min_words": 8,
        }
        count = summarize_jsonl(
            "input.jsonl", "predictions.jsonl", base_config, "test"
        )
        assert count == 2
        rows_by_id = {row["id"]: row for row in captured["rows"]}
        assert set(rows_by_id) == {sample_doc["id"], "min-words-shortfall"}
        assert rows_by_id["min-words-shortfall"]["feasible"] is False
        assert rows_by_id["min-words-shortfall"]["selected_indices"] == [0]

    def test_build_feasibility_report_summarizes_predictions_jsonl(self, tmp_path):
        preds_path = tmp_path / "predictions.jsonl"
        rows = [
            {
                "id": "feasible-doc",
                "feasible": True,
                "infeasible_reason": None,
                "violations": {"nonempty": 0.0, "min_words": -2.0,
                               "max_length": -3.0, "max_sentences": 0.0},
                "selection_evaluation": {"selected_words": 10},
                "output_budget": {"effective_min_words": 8},
            },
            {
                "id": "min-words-shortfall",
                "feasible": False,
                "infeasible_code": "selector_min_words_shortfall",
                "infeasible_reason": "selector could not reach effective_min_words",
                "violations": {"nonempty": 0.0, "min_words": 6.0,
                               "max_length": -8.0, "max_sentences": 0.0},
                "selection_evaluation": {"selected_words": 2},
                "output_budget": {"effective_min_words": 8},
            },
        ]
        with open(preds_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        report = build_feasibility_report(str(preds_path))
        assert report["total_count"] == 2
        assert report["feasible_count"] == 1
        assert report["infeasible_count"] == 1
        assert report["infeasible_ids"] == [
            {
                "id": "min-words-shortfall",
                "infeasible_code": "selector_min_words_shortfall",
                "violations": {"nonempty": 0.0, "min_words": 6.0,
                               "max_length": -8.0, "max_sentences": 0.0},
                "infeasible_reason": "selector could not reach effective_min_words",
                "selected_words": 2,
                "effective_min_words": 8,
            }
        ]

    def test_governed_jsonl_runner_requires_dataset_preflight(
        self, sample_doc, base_config, monkeypatch
    ):
        captured = {}
        governed = dict(base_config)
        governed["experiment"] = {
            "status": "validation_pilot_only",
            "dataset": "fixture",
        }
        governed["data_policy"] = {
            "policy_path": "policy.json",
            "policy_sha256": "0" * 64,
            "analysis": "main",
        }
        sample_doc = dict(sample_doc, split="validation")
        monkeypatch.setattr(
            "src.pipeline.select_sentences.read_jsonl",
            lambda _path: iter([sample_doc]),
        )

        def capture_preflight(cfg, input_path, split):
            captured["preflight"] = (cfg, input_path, split)
            return {"valid": True}

        monkeypatch.setattr(
            "src.pipeline.select_sentences.validate_dataset_policy_request",
            capture_preflight,
        )
        monkeypatch.setattr(
            "src.pipeline.select_sentences.write_jsonl_atomic",
            lambda _path, rows: list(rows),
        )
        monkeypatch.setattr(
            "src.pipeline.select_sentences.validate_experiment_document",
            lambda _cfg, _doc: None,
        )

        assert summarize_jsonl(
            "input.jsonl", "predictions.jsonl", governed, "validation"
        ) == 1
        assert captured["preflight"][1:] == ("input.jsonl", "validation")

    def test_jsonl_runner_rejects_empty_input(self, base_config, monkeypatch):
        monkeypatch.setattr(
            "src.pipeline.select_sentences.read_jsonl", lambda _path: iter(())
        )
        monkeypatch.setattr(
            "src.pipeline.select_sentences.write_jsonl_atomic",
            lambda _path, rows: list(rows),
        )
        with pytest.raises(ValueError, match="input dataset is empty"):
            summarize_jsonl("empty.jsonl", "predictions.jsonl", base_config, "test")
