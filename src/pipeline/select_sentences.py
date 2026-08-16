"""Main sentence-selection pipeline.

Orchestrates feature building, candidate-pool construction, and
optimizer dispatch — each delegated to its own module.
"""

import argparse
import hashlib
import os
import re
import time
from typing import Dict, List, Mapping

import numpy as np
from tqdm import tqdm

from src.utils.io import (
    load_yaml,
    ensure_dir,
    now_stamp,
    read_jsonl,
    write_jsonl_atomic,
    set_global_seed,
)
from src.representations.sent_vectors import SentenceVectors
from src.representations.similarity import cosine_similarity_matrix
from src.data.schemas import flatten_sentence_records, validate_candidate_record
from src.data.policy import validate_dataset_policy_request
from src.data.partitions import (
    iter_partition_rows,
    partition_report_for_artifact,
    resolve_experiment_partition,
)
from src.eval.feasibility import classify_feasibility_row

from src.pipeline.feature_builder import build_base_scores
from src.pipeline.candidate_builder import build_candidate_pool
from src.pipeline.optimizer_dispatch import dispatch_optimizer
from src.models.extractive.encoder_rank import (
    centroid_scores_from_embeddings,
    cosine_matrix_from_embeddings,
    encoder_document_embeddings,
)
from src.objectives.factory import build_objective_spec, validate_selector_for_task
from src.objectives.evaluator import (
    InfeasibleSelectionError,
    maximum_feasible_words,
    objective_from_spec,
    resolve_effective_min_words,
    resolve_selection_eligibility,
)


# ------------------------------------------------------------------ #
#  Core per-document summarisation                                     #
# ------------------------------------------------------------------ #

def validate_requested_split(doc: Dict, requested_split: str) -> None:
    """Prevent a canonical row from being run under the wrong split label."""

    if "schema_version" in doc and doc.get("split") != requested_split:
        raise ValueError(
            f"canonical row {doc.get('id')!r} belongs to split {doc.get('split')!r}, "
            f"but --split is {requested_split!r}"
        )


def validate_experiment_request(cfg: Mapping, requested_split: str) -> None:
    """Enforce config-level data-access gates before a run starts."""

    experiment = cfg.get("experiment")
    if experiment is None:
        # Legacy reproduction configs predate the experiment contract. They
        # remain runnable, but cannot acquire a formal status implicitly.
        return
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment configuration must be an object")
    status = experiment.get("status")
    allowed_split = {
        "validation_pilot_only": "validation",
        "final_test_only": "test",
    }.get(status)
    if allowed_split is None:
        raise ValueError(
            f"unknown experiment.status {status!r}; choose one of "
            "'validation_pilot_only' or 'final_test_only'"
        )
    if requested_split != allowed_split:
        raise ValueError(
            f"experiment.status={status!r} may only access the "
            f"{allowed_split} split, not {requested_split!r}"
        )
    data_policy = cfg.get("data_policy")
    if not isinstance(data_policy, Mapping):
        raise ValueError("governed experiments require a data_policy object")
    if not isinstance(data_policy.get("policy_path"), str):
        raise ValueError("data_policy.policy_path must be declared")
    if not isinstance(data_policy.get("policy_sha256"), str):
        raise ValueError("data_policy.policy_sha256 must be declared")
    if not isinstance(data_policy.get("analysis"), str):
        raise ValueError("data_policy.analysis must be declared")
    if status == "final_test_only":
        dataset_name = _normalized_dataset_name(experiment.get("dataset"))
        frozen_test_policies = {
            "govreport": "configs/data_policies/govreport_test_v1.json",
            "multinews": "configs/data_policies/multinews_test_v1.json",
        }
        expected_policy = frozen_test_policies.get(dataset_name)
        if expected_policy is None:
            raise ValueError(
                "final_test_only is frozen only for GovReport and Multi-News"
            )
        if data_policy.get("policy_path") != expected_policy:
            raise ValueError(
                f"final_test_only requires the frozen {experiment.get('dataset')} "
                "test policy"
            )
        if cfg.get("experiment_partition") is not None:
            raise ValueError("final_test_only must evaluate the complete official test policy")


def _normalized_dataset_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def validate_experiment_document(cfg: Mapping, doc: Mapping) -> None:
    """Bind a governed experiment config to canonical split/dataset metadata."""

    experiment = cfg.get("experiment")
    if experiment is None:
        return
    if "schema_version" not in doc:
        raise ValueError(
            "governed experiments require canonical rows with split and dataset provenance"
        )
    split = doc.get("split")
    validate_experiment_request(cfg, str(split or ""))
    expected_dataset = experiment.get("dataset")
    if not isinstance(expected_dataset, str) or not expected_dataset.strip():
        raise ValueError("governed experiments require experiment.dataset")
    actual_dataset = doc.get("dataset_name")
    if _normalized_dataset_name(actual_dataset) != _normalized_dataset_name(
        expected_dataset
    ):
        raise ValueError(
            f"experiment dataset {expected_dataset!r} does not match canonical "
            f"row dataset_name {actual_dataset!r}"
        )


def attach_selector_salience(
    candidate_records: List[Dict],
    base_scores: List[float],
    source: str,
) -> List[float]:
    """Resolve the exact auditable score passed from candidates to selector."""

    normalized_source = (source or "base_score").strip().lower()
    selector_scores: List[float] = []
    for candidate in candidate_records:
        index = candidate["original_index"]
        if normalized_source in {"base_score", "membership_only"}:
            value = float(base_scores[index])
        elif normalized_source == "rrf_fusion":
            value = float(candidate["fusion_normalized"])
        elif normalized_source.endswith("_percentile"):
            route = normalized_source[: -len("_percentile")]
            if route not in candidate["route_scores"]:
                raise ValueError(
                    f"selector salience source {source!r} requires route {route!r}"
                )
            value = float(candidate["route_scores"][route]["percentile"])
        elif normalized_source.endswith("_raw"):
            route = normalized_source[: -len("_raw")]
            if route not in candidate["route_scores"]:
                raise ValueError(
                    f"selector salience source {source!r} requires route {route!r}"
                )
            value = float(candidate["route_scores"][route]["raw"])
        else:
            raise ValueError(f"unknown selector.salience_source: {source!r}")
        candidate["selector_salience"] = value
        candidate["selector_salience_source"] = normalized_source
        validate_candidate_record(candidate)
        selector_scores.append(value)
    return selector_scores


def _numeric_sha256(values, dtype: str) -> str | None:
    """Hash a numeric selector input with explicit little-endian encoding."""

    if values is None:
        return None
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def summarize_one(
    doc: Dict,
    cfg: Dict,
    *,
    fixed_candidate_original_indices: List[int] | None = None,
    audit_route_weights: Dict[str, float] | None = None,
) -> Dict:
    """Summarize one canonical row.

    The two keyword-only overrides exist solely for preregistered selector-
    provenance ablations.  They are intentionally unavailable from the
    production CLI/config schema: the normal inference path must construct its
    own pool.  When supplied, route scores are still recomputed over the full
    eligible source, then the selector is restricted to the exact frozen pool.
    This isolates selector evidence without accidentally changing membership.
    """

    if audit_route_weights is not None and fixed_candidate_original_indices is None:
        raise ValueError(
            "audit_route_weights requires fixed_candidate_original_indices"
        )
    validate_experiment_document(cfg, doc)
    sentence_records = flatten_sentence_records(doc)
    sentences: List[str] = [record["text"] for record in sentence_records]

    cand_cfg = cfg.get("candidates", {})
    # A gold-reference-dependent candidate size is an oracle diagnostic, not
    # a deployable inference rule.  Refuse old configs before doing any model
    # or feature work.
    if cand_cfg.get("recall_target") is not None:
        raise ValueError(
            "candidates.recall_target is forbidden in the production selection "
            "pipeline because it requires gold references; run oracle/candidate "
            "recall analysis as a separate diagnostic"
        )

    # 1. Similarity matrix (needed by graph features, candidates, NSGA-II)
    rep_cfg = cfg.get("representations", {})
    sim = None
    if bool(rep_cfg.get("use", True)) and len(sentences) > 0:
        method = rep_cfg.get("method", "tfidf")
        vec = SentenceVectors(method=method)
        X = vec.fit_transform(sentences)
        sim = cosine_similarity_matrix(X)

    # 2. Feature scores
    base_scores = build_base_scores(
        sentences,
        cfg,
        similarity_matrix=sim,
        sentence_records=sentence_records,
    )

    # 3. Length / redundancy parameters
    lc = cfg.get("length_control", {})
    unit = (lc.get("unit", "tokens") or "tokens").lower()
    if unit not in {"words", "tokens", "sentences"}:
        raise ValueError("length_control.unit must be words, tokens, or sentences")
    max_tokens = int(lc.get("max_tokens", 100))
    max_words = int(lc.get("max_words", 400))
    selector_budget = max_words if unit == "words" else max_tokens
    max_sents_limit = lc.get("max_sentences", None)
    max_sents = int(max_sents_limit) if (max_sents_limit is not None) else None
    requested_min_words = int(lc.get("min_words", 0))
    require_nonempty = bool(lc.get("require_nonempty", True))
    if requested_min_words < 0:
        raise ValueError("length_control.min_words cannot be negative")
    if unit in {"words", "tokens"} and requested_min_words > selector_budget:
        raise ValueError(
            "length_control.min_words cannot exceed the active maximum length"
        )
    alpha = float(cfg.get("redundancy", {}).get("lambda", 0.7))
    objective_spec = build_objective_spec(doc.get("task_profile"), cfg)
    method_opt = cfg.get("optimizer", {}).get("method", "greedy").lower()
    validate_selector_for_task(objective_spec, method_opt)
    required_max_sentences = objective_spec.get("required_max_sentences")
    if required_max_sentences is not None:
        if max_sents is not None and max_sents != required_max_sentences:
            raise ValueError(
                f"task profile requires max_sentences={required_max_sentences}, "
                f"but config declares {max_sents}"
            )
        max_sents = int(required_max_sentences)

    min_words_resolution = resolve_effective_min_words(
        sentences,
        requested_min_words=requested_min_words,
        max_length=selector_budget,
        length_unit=unit,
        max_sentences=max_sents,
    )
    source_capacity_words = min_words_resolution.source_capacity_words
    effective_min_words = min_words_resolution.effective_min_words
    min_words_relaxed = min_words_resolution.min_words_relaxed
    relaxation_reason = min_words_resolution.relaxation_reason

    eligibility = resolve_selection_eligibility(
        sentences,
        sentence_records,
        max_length=selector_budget,
        length_unit=unit,
        require_nonempty=require_nonempty,
        document_id=doc.get("id"),
    )
    selection_eligible_indices = eligibility.eligible_indices
    ineligible_sentences = eligibility.ineligible_sentences

    # A selector representation is independent of the lexical representation
    # used by handcrafted features and optional graph routing.  In the matched
    # selector study all selectors receive this exact matrix.  When SBERT is
    # requested, encode the full source once, reuse the eligible slice for the
    # semantic candidate route, and retain the full-source rows for facility
    # coverage.
    selector_cfg = cfg.get("selector", {}) or {}
    selector_similarity_source = str(
        selector_cfg.get("similarity_source", "pipeline_similarity")
    ).lower()
    selector_full_sim = sim
    selector_representation_metadata: Dict = {
        "source": "pipeline_similarity",
        "method": rep_cfg.get("method", "tfidf"),
    }
    precomputed_route_data: Dict = {}
    if selector_similarity_source == "sbert":
        semantic_cfg = (cfg.get("routes", {}) or {}).get("semantic", {})
        model_name = semantic_cfg.get("model_name")
        revision = semantic_cfg.get("revision")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "selector.similarity_source='sbert' requires "
                "routes.semantic.model_name"
            )
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(
                "selector.similarity_source='sbert' requires a pinned "
                "routes.semantic.revision"
            )
        full_embeddings, semantic_metadata = encoder_document_embeddings(
            sentences,
            model_name=model_name,
            device=semantic_cfg.get("device"),
            batch_size=int(semantic_cfg.get("batch_size", 16)),
            max_model_tokens=int(semantic_cfg.get("max_model_tokens", 256)),
            revision=revision,
        )
        selector_full_sim = cosine_matrix_from_embeddings(full_embeddings)
        eligible_embeddings = full_embeddings[selection_eligible_indices]
        eligible_semantic_scores = centroid_scores_from_embeddings(
            eligible_embeddings
        )
        precomputed_route_data["semantic"] = {
            "values": eligible_semantic_scores,
            "metadata": semantic_metadata,
        }
        selector_representation_metadata = {
            "source": "sbert",
            **semantic_metadata,
        }
    elif selector_similarity_source not in {
        "pipeline_similarity",
        "tfidf",
    }:
        raise ValueError(
            "selector.similarity_source must be 'pipeline_similarity', "
            "'tfidf', or 'sbert'"
        )

    # 4. Candidate pool. Per-route quota and final selector budget are
    # deliberately separate from the output-length budget above.
    budget_cfg = cfg.get("candidate_budget", {})
    if isinstance(budget_cfg, dict):
        route_top_k_budget = budget_cfg.get(
            "route_top_k", budget_cfg.get("per_route", cand_cfg.get("k"))
        )
        min_per_route = budget_cfg.get("min_per_route", 0)
        total_candidate_budget = budget_cfg.get(
            "total", cand_cfg.get("total_budget")
        )
    elif budget_cfg in (None, {}):
        route_top_k_budget = cand_cfg.get("k")
        min_per_route = 0
        total_candidate_budget = cand_cfg.get("total_budget")
    else:
        route_top_k_budget = cand_cfg.get("k")
        min_per_route = 0
        total_candidate_budget = budget_cfg
    k = int(
        min(15, len(sentences))
        if route_top_k_budget is None
        else route_top_k_budget
    )
    total_candidate_budget = (
        int(total_candidate_budget) if total_candidate_budget is not None else None
    )
    use_cand = bool(cand_cfg.get("use", True))
    mode = (cand_cfg.get("mode", "hard") or "hard").lower()
    compute_cfg = cfg.get("compute_budget", {}) or {}
    compute_mode = str(compute_cfg.get("mode", "fixed")).lower()
    if compute_mode != "fixed":
        raise ValueError(
            "only compute_budget.mode='fixed' is implemented; adaptive routing "
            "must not be claimed before its validation-frozen policy exists"
        )
    sources = (
        compute_cfg.get("enabled_routes")
        or cand_cfg.get("sources", ["score"])
        or ["score"]
    )
    soft_boost = float(cand_cfg.get("soft_boost", 1.05))

    g_thresh = float(cfg.get("graph_params", {}).get("threshold", 0.0))
    eligible_records = [sentence_records[index] for index in selection_eligible_indices]
    eligible_scores = [base_scores[index] for index in selection_eligible_indices]
    eligible_sim = (
        sim[np.ix_(selection_eligible_indices, selection_eligible_indices)]
        if sim is not None
        else None
    )
    audit_fixed_pool = fixed_candidate_original_indices is not None
    build_k = len(eligible_records) if audit_fixed_pool else k
    build_total_budget = None if audit_fixed_pool else total_candidate_budget
    build_min_per_route = 0 if audit_fixed_pool else min_per_route
    build_coverage_guard = {} if audit_fixed_pool else (cfg.get("coverage_guard", {}) or {})
    build_route_weights = (
        audit_route_weights
        if audit_route_weights is not None
        else cand_cfg.get("route_weights")
    )
    candidate_pool_result = (
        build_candidate_pool(
            eligible_records,
            eligible_scores,
            build_k,
            sources,
            sim_matrix=eligible_sim,
            threshold=g_thresh,
            total_budget=build_total_budget,
            min_per_route=build_min_per_route,
            route_config=cfg.get("routes", {}) or {},
            coverage_guard=build_coverage_guard,
            rrf_constant=int(cand_cfg.get("rrf_constant", 60)),
            route_weights=build_route_weights,
            precomputed_route_data=precomputed_route_data,
        )
        if use_cand
        else {
            "records": [],
            "route_proposals": {},
            "allocation": {"actual_size": 0},
        }
    )
    if audit_fixed_pool:
        requested = list(fixed_candidate_original_indices or [])
        if len(requested) != len(set(requested)):
            raise ValueError("fixed candidate pool contains duplicate indices")
        eligible_set = set(selection_eligible_indices)
        invalid = sorted(set(requested) - eligible_set)
        if invalid:
            raise ValueError(
                f"fixed candidate pool contains ineligible indices: {invalid[:5]}"
            )
        all_records = {
            int(record["original_index"]): record
            for record in candidate_pool_result["records"]
        }
        missing = sorted(set(requested) - set(all_records))
        if missing:
            raise ValueError(
                f"full-route reconstruction missed fixed pool indices: {missing[:5]}"
            )
        fixed_records = [all_records[index] for index in sorted(requested)]
        for record in fixed_records:
            selected_routes = [
                route
                for route, score in record["route_scores"].items()
                if int(score["rank"]) <= k
            ]
            record["selected_by_routes"] = selected_routes
            record["route_agreement"] = len(selected_routes)
            record["inclusion_reasons"] = ["audit:fixed_C01_candidate_pool"]
        candidate_pool_result["records"] = fixed_records
        requested_set = set(requested)
        candidate_pool_result["route_proposals"] = {
            route: [
                {
                    **proposal,
                    "selected_in_final_pool": (
                        int(proposal["original_index"]) in requested_set
                    ),
                }
                for proposal in proposals[:k]
            ]
            for route, proposals in candidate_pool_result["route_proposals"].items()
        }
        candidate_pool_result["allocation"] = {
            **candidate_pool_result["allocation"],
            "actual_size": len(requested),
            "audit_fixed_candidate_pool": True,
            "audit_fixed_candidate_original_indices_sha256": _numeric_sha256(
                sorted(requested), "<i8"
            ),
            "audit_route_weights": build_route_weights,
            "route_scores_recomputed_over_full_eligible_source": True,
        }
    candidate_records = candidate_pool_result["records"]
    cand_idx = [record["original_index"] for record in candidate_records]
    salience_source = str(selector_cfg.get("salience_source", "base_score"))

    # 5. Apply candidate mode
    if use_cand and mode == "hard":
        sub_original_indices = list(cand_idx)
        sub_sentences = [sentences[i] for i in cand_idx]
        sub_scores = attach_selector_salience(
            candidate_records, base_scores, salience_source
        )
        sub_sim = (
            selector_full_sim[np.ix_(cand_idx, cand_idx)]
            if selector_full_sim is not None
            else None
        )
        sub_coverage = (
            selector_full_sim[:, cand_idx]
            if selector_full_sim is not None
            else None
        )
    elif use_cand and cand_idx:
        if mode != "hard":
            if salience_source.lower() not in {"base_score", "membership_only"}:
                raise ValueError(
                    "provenance-aware selector salience requires candidates.mode='hard'"
                )
            sub_original_indices = list(selection_eligible_indices)
            eligible_position = {
                original: relative
                for relative, original in enumerate(sub_original_indices)
            }
            sub_sentences = [sentences[i] for i in sub_original_indices]
            sub_scores = [float(base_scores[i]) for i in sub_original_indices]
            for original in cand_idx:
                relative = eligible_position[original]
                sub_scores[relative] = float(sub_scores[relative]) * soft_boost
            sub_sim = (
                selector_full_sim[
                    np.ix_(sub_original_indices, sub_original_indices)
                ]
                if selector_full_sim is not None
                else None
            )
            sub_coverage = (
                selector_full_sim[:, sub_original_indices]
                if selector_full_sim is not None
                else None
            )
    else:
        # Eligibility is independent of whether candidate routing is enabled.
        # Individually over-budget sentences must never re-enter through the
        # no-candidate or empty-soft-candidate path.
        sub_original_indices = list(selection_eligible_indices)
        sub_sentences = [sentences[i] for i in sub_original_indices]
        sub_scores = [float(base_scores[i]) for i in sub_original_indices]
        sub_sim = (
            selector_full_sim[np.ix_(sub_original_indices, sub_original_indices)]
            if selector_full_sim is not None
            else None
        )
        sub_coverage = (
            selector_full_sim[:, sub_original_indices]
            if selector_full_sim is not None
            else None
        )

    candidate_capacity_words = maximum_feasible_words(
        sub_sentences,
        max_length=selector_budget,
        length_unit=unit,
        max_sentences=max_sents,
    )
    candidate_capacity_shortfall = (
        use_cand
        and mode == "hard"
        and candidate_capacity_words < effective_min_words
    )

    # 6. Optimizer dispatch. A word budget no longer bypasses the configured
    # selector; it is simply the selector's independent output constraint.
    #
    # F-17 option 1: document-level lower-bound infeasibility is data, not a
    # batch-level exception.  Keep the optimizer's actual attempted output and
    # record why it was infeasible; do not backfill or silently re-search.
    # Upper-bound violations still indicate a selector bug and remain fatal.
    optimizer_diagnostics: Dict = {}
    evaluator_similarity = (
        np.zeros((0, 0), dtype=float)
        if not sub_sentences and sub_sim is None
        else sub_sim
    )
    evaluator_coverage = (
        np.zeros((0, 0), dtype=float)
        if not sub_sentences and sub_coverage is None
        else sub_coverage
    )
    evaluator = objective_from_spec(
        sub_sentences,
        sub_scores,
        evaluator_similarity,
        objective_spec,
        max_length=selector_budget,
        length_unit=unit,
        max_sentences=max_sents,
        min_words=effective_min_words,
        require_nonempty=require_nonempty,
        coverage_matrix=evaluator_coverage,
    )
    selector_input_fingerprints = {
        "candidate_original_indices_sha256": _numeric_sha256(
            sub_original_indices, "<i8"
        ),
        "salience_sha256": _numeric_sha256(sub_scores, "<f8"),
        "similarity_sha256": _numeric_sha256(sub_sim, "<f8"),
        "coverage_sha256": _numeric_sha256(sub_coverage, "<f8"),
    }
    caught_infeasible_error = None
    infeasible_code = None
    infeasible_reason = None
    if not sub_sentences:
        picked_sub = []
        evaluation = evaluator.evaluate([])
    else:
        try:
            picked_sub = dispatch_optimizer(
                method_opt,
                sub_sentences,
                sub_scores,
                sub_sim,
                selector_budget,
                cfg,
                alpha,
                unit,
                max_sents,
                objective_spec,
                effective_min_words,
                require_nonempty,
                optimizer_diagnostics,
                sub_coverage,
            )
            evaluation = evaluator.evaluate(picked_sub)
        except InfeasibleSelectionError as exc:
            caught_infeasible_error = exc
            picked_sub = exc.evaluation.selected_indices
            evaluation = exc.evaluation

    positive_violations = {
        key: value for key, value in evaluation.violations.items() if value > 0
    }
    upper_bound_violations = set(positive_violations) & {
        "max_length",
        "max_sentences",
    }
    if upper_bound_violations:
        if caught_infeasible_error is not None:
            raise caught_infeasible_error
        raise InfeasibleSelectionError(
            f"selector returned an infeasible summary: {positive_violations}",
            evaluation,
        )
    unexpected_violations = set(positive_violations) - {"min_words", "nonempty"}
    if unexpected_violations:
        raise InfeasibleSelectionError(
            f"selector returned an infeasible summary: {positive_violations}",
            evaluation,
        )

    if positive_violations:
        if not sub_sentences:
            if sentences and not selection_eligible_indices:
                infeasible_code = "source_no_eligible_sentence"
                infeasible_reason = (
                    "source has no sentence eligible under the active output "
                    f"budget ({unit}={selector_budget}); empty selection recorded"
                )
            elif use_cand and mode == "hard":
                infeasible_code = "candidate_pool_empty"
                infeasible_reason = (
                    "hard candidate routing produced an empty pool; empty "
                    "selection recorded"
                )
            else:
                infeasible_code = "empty_source"
                infeasible_reason = "source contains no selectable sentence"
        elif candidate_capacity_shortfall:
            infeasible_code = "candidate_capacity_shortfall"
            infeasible_reason = (
                "hard candidate pool cannot reach effective_min_words "
                f"(candidate_capacity_words={candidate_capacity_words}, "
                f"effective_min_words={effective_min_words}, "
                f"source_capacity_words={source_capacity_words}); selector's "
                "attempted output recorded as-is"
            )
        elif (
            caught_infeasible_error is not None
            and caught_infeasible_error.reason_code
            == "optimizer_no_feasible_solution"
        ):
            infeasible_code = "optimizer_no_feasible_solution"
            infeasible_reason = (
                f"{method_opt} found no feasible solution; its least-violating "
                "attempted output was recorded as-is"
            )
        elif "min_words" in positive_violations:
            infeasible_code = "selector_min_words_shortfall"
            infeasible_reason = (
                "selector could not reach effective_min_words for this document "
                f"(shortfall={positive_violations['min_words']:.0f} words); "
                "selection kept as-is"
            )
        else:
            infeasible_code = "selector_nonempty_shortfall"
            infeasible_reason = (
                "selector returned an empty summary despite require_nonempty=true; "
                "selection kept as-is"
            )

    optimizer_diagnostics.setdefault("method", method_opt)
    optimizer_diagnostics["selector_input_fingerprints"] = dict(
        selector_input_fingerprints
    )
    optimizer_diagnostics["selector_representation"] = dict(
        selector_representation_metadata
    )

    selection_evaluation = evaluation.to_dict()

    # 7. Map back to original indices
    selected = sorted(sub_original_indices[i] for i in picked_sub)

    selected.sort()
    if selection_evaluation is not None:
        selection_evaluation["candidate_relative_indices"] = list(
            selection_evaluation["selected_indices"]
        )
        selection_evaluation["selected_indices"] = list(selected)
    if optimizer_diagnostics.get("pareto_front"):
        for solution in optimizer_diagnostics["pareto_front"]:
            relative = list(solution["selected_indices"])
            solution["candidate_relative_indices"] = relative
            solution["selected_indices"] = sorted(
                sub_original_indices[index] for index in relative
            )
    summary_sentences = [sentences[i] for i in selected]
    summary = "\n".join(summary_sentences)
    candidate_by_index = {
        candidate["original_index"]: candidate for candidate in candidate_records
    }
    selected_sentences = [
        {
            **sentence_records[index],
            "selection_order": order,
            "selection_evidence": (
                {
                    "selector_salience": candidate_by_index[index].get(
                        "selector_salience"
                    ),
                    "selector_salience_source": candidate_by_index[index].get(
                        "selector_salience_source"
                    ),
                    "fusion_score": candidate_by_index[index]["fusion_score"],
                    "fusion_normalized": candidate_by_index[index][
                        "fusion_normalized"
                    ],
                    "fused_rank": candidate_by_index[index]["fused_rank"],
                    "route_agreement": candidate_by_index[index]["route_agreement"],
                }
                if index in candidate_by_index
                else None
            ),
        }
        for order, index in enumerate(selected)
    ]
    return {
        "id": doc.get("id"),
        "selected_indices": selected,
        "selected_sentences": selected_sentences,
        "summary_sentences": summary_sentences,
        "summary": summary,
        "candidate_records": candidate_records,
        "candidate_pool": {
            "enabled": use_cand,
            "configured_sources": list(sources) if use_cand else [],
            "route_top_k": k if use_cand else None,
            "min_per_route": min_per_route if use_cand else None,
            "total_cap": total_candidate_budget if use_cand else None,
            "actual_size": len(candidate_records),
            "coverage_guard": dict(cfg.get("coverage_guard", {}) or {}),
            "selector_salience_source": salience_source,
            "route_proposals": candidate_pool_result["route_proposals"],
            "allocation": candidate_pool_result["allocation"],
            "selection_ineligible_sentences": ineligible_sentences,
        },
        "objective_spec": objective_spec,
        "selection_evaluation": selection_evaluation,
        "feasible": (
            selection_evaluation["feasible"]
        ),
        "infeasible_code": infeasible_code,
        "infeasible_reason": infeasible_reason,
        "violations": selection_evaluation["violations"],
        "optimizer_diagnostics": optimizer_diagnostics or None,
        "selector_inputs": {
            "candidate_count": len(sub_original_indices),
            "coverage_universe_size": (
                0 if sub_coverage is None else int(sub_coverage.shape[0])
            ),
            "representation": dict(selector_representation_metadata),
            **selector_input_fingerprints,
        },
        "output_budget": {
            "unit": unit,
            "max_words": max_words if unit == "words" else None,
            "max_tokens": max_tokens if unit == "tokens" else None,
            "max_sentences": max_sents,
            "min_words": effective_min_words,
            "requested_min_words": requested_min_words,
            "effective_min_words": effective_min_words,
            "source_capacity_words": source_capacity_words,
            "candidate_capacity_words": candidate_capacity_words,
            "min_words_relaxed": min_words_relaxed,
            "relaxation_reason": relaxation_reason,
            "require_nonempty": require_nonempty,
        },
        "task_profile": doc.get("task_profile"),
    }


# ------------------------------------------------------------------ #
#  CLI entry-point                                                     #
# ------------------------------------------------------------------ #

def build_feasibility_report(predictions_path: str) -> Dict:
    """Summarize predictions.jsonl's feasible/infeasible rows (F-17).

    A summary index over the artifact, not a second source of truth: every
    field here is re-read from the rows already written, the same way any
    downstream consumer would filter them.
    """

    feasible_count = 0
    infeasible_ids = []
    for row in read_jsonl(predictions_path):
        feasible, _ = classify_feasibility_row(
            row, assume_legacy_feasible=False
        )
        if not feasible:
            infeasible_ids.append(
                {
                    "id": row.get("id"),
                    "infeasible_code": row.get("infeasible_code"),
                    "violations": row.get("violations"),
                    "infeasible_reason": row.get("infeasible_reason"),
                    "selected_words": (
                        row.get("selection_evaluation", {}) or {}
                    ).get("selected_words"),
                    "effective_min_words": (
                        row.get("output_budget", {}) or {}
                    ).get("effective_min_words"),
                }
            )
        else:
            feasible_count += 1
    return {
        "total_count": feasible_count + len(infeasible_ids),
        "feasible_count": feasible_count,
        "infeasible_count": len(infeasible_ids),
        "infeasible_ids": infeasible_ids,
    }


def summarize_jsonl(
    input_path: str,
    predictions_path: str,
    cfg: Dict,
    requested_split: str,
    dataset_preflight: Dict | None = None,
    partition_preflight: Dict | None = None,
) -> int:
    """Stream one dataset into an atomic prediction artifact."""

    if dataset_preflight is None:
        dataset_preflight = validate_dataset_policy_request(
            cfg, input_path, requested_split
        )
    if partition_preflight is None:
        partition_preflight = resolve_experiment_partition(cfg, dataset_preflight)

    processed = 0

    def prediction_rows():
        nonlocal processed
        rows = iter_partition_rows(read_jsonl(input_path), partition_preflight)
        for doc in tqdm(rows, desc="Summarizing"):
            validate_requested_split(doc, requested_split)
            result = summarize_one(doc, cfg)
            processed += 1
            yield result
        if processed == 0:
            raise ValueError("input dataset is empty; refusing to write an empty run")

    write_jsonl_atomic(predictions_path, prediction_rows())
    return processed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config yaml")
    ap.add_argument("--split", required=True, help="dataset split name")
    ap.add_argument("--input", required=True, help="processed jsonl path")
    ap.add_argument("--run_dir", default="runs", help="runs output root")
    ap.add_argument("--stamp", default=None, help="optional fixed stamp for output dir")
    ap.add_argument("--optimizer", default=None, help="override optimizer.method in config")
    ap.add_argument(
        "--data_policy_analysis",
        default=None,
        help="override data_policy.analysis with a name declared by the frozen policy",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.optimizer:
        cfg.setdefault("optimizer", {})
        cfg["optimizer"]["method"] = args.optimizer
    if args.data_policy_analysis:
        if not isinstance(cfg.get("data_policy"), dict):
            raise ValueError("--data_policy_analysis requires config data_policy")
        cfg["data_policy"]["analysis"] = args.data_policy_analysis

    validate_experiment_request(cfg, args.split)
    dataset_preflight = validate_dataset_policy_request(cfg, args.input, args.split)
    partition_preflight = resolve_experiment_partition(cfg, dataset_preflight)

    # Guard: Stage2 union input should use fast (non-BERT) optimizers only
    method_opt = (cfg.get("optimizer", {}).get("method") or "").lower()
    in_path = str(args.input)
    is_stage2_union = ("stage2" in in_path and "union" in in_path)
    if is_stage2_union and method_opt in ("bert", "roberta", "xlnet", "fused"):
        raise RuntimeError(
            f"Stage2 union input detected ({in_path}). Please use non-BERT optimizers: fast | fast_grasp | fast_nsga2. "
            f"Current optimizer '{method_opt}' is not allowed for Stage2."
        )

    set_global_seed(cfg.get("seed"))
    stamp = args.stamp or now_stamp()
    out_dir = os.path.join(args.run_dir, stamp)
    ensure_dir(out_dir)

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    t0 = time.perf_counter()
    summarize_jsonl(
        args.input,
        preds_path,
        cfg,
        args.split,
        dataset_preflight=dataset_preflight,
        partition_preflight=partition_preflight,
    )
    t1 = time.perf_counter()

    # dump the config used
    import json
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    if dataset_preflight is not None:
        with open(
            os.path.join(out_dir, "dataset_preflight.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(dataset_preflight, f, ensure_ascii=False, indent=2)
    partition_report = partition_report_for_artifact(partition_preflight)
    if partition_report is not None:
        with open(
            os.path.join(out_dir, "partition_preflight.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(partition_report, f, ensure_ascii=False, indent=2)

    feasibility_report = build_feasibility_report(preds_path)
    with open(
        os.path.join(out_dir, "feasibility_report.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(feasibility_report, f, ensure_ascii=False, indent=2)

    # A formal run is incomplete if its timing artifact cannot be written.
    with open(os.path.join(out_dir, "time_select_seconds.txt"), "w", encoding="utf-8") as f:
        f.write(f"{t1 - t0:.6f}")
    print(f"Wrote predictions to {preds_path}")


if __name__ == "__main__":
    main()
