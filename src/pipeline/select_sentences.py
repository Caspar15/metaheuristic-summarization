"""Main sentence-selection pipeline.

Orchestrates feature building, candidate-pool construction, and
optimizer dispatch — each delegated to its own module.
"""

import argparse
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

from src.pipeline.feature_builder import build_base_scores
from src.pipeline.candidate_builder import build_candidate_pool
from src.pipeline.optimizer_dispatch import dispatch_optimizer
from src.objectives.factory import build_objective_spec, validate_selector_for_task
from src.objectives.evaluator import (
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
    if status != "validation_pilot_only":
        raise ValueError(
            f"unknown experiment.status {status!r}; only "
            "'validation_pilot_only' is currently implemented"
        )
    if requested_split != "validation":
        raise ValueError(
            "experiment.status='validation_pilot_only' may only access the "
            f"validation split, not {requested_split!r}"
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
        else:
            raise ValueError(f"unknown selector.salience_source: {source!r}")
        candidate["selector_salience"] = value
        candidate["selector_salience_source"] = normalized_source
        validate_candidate_record(candidate)
        selector_scores.append(value)
    return selector_scores


def summarize_one(doc: Dict, cfg: Dict) -> Dict:
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
    base_scores = build_base_scores(sentences, cfg, similarity_matrix=sim)

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
    candidate_pool_result = (
        build_candidate_pool(
            eligible_records,
            eligible_scores,
            k,
            sources,
            sim_matrix=eligible_sim,
            threshold=g_thresh,
            total_budget=total_candidate_budget,
            min_per_route=min_per_route,
            route_config=cfg.get("routes", {}) or {},
            coverage_guard=cfg.get("coverage_guard", {}) or {},
            rrf_constant=int(cand_cfg.get("rrf_constant", 60)),
        )
        if use_cand
        else {
            "records": [],
            "route_proposals": {},
            "allocation": {"actual_size": 0},
        }
    )
    candidate_records = candidate_pool_result["records"]
    cand_idx = [record["original_index"] for record in candidate_records]
    selector_cfg = cfg.get("selector", {}) or {}
    salience_source = str(selector_cfg.get("salience_source", "base_score"))

    # 5. Apply candidate mode
    if use_cand and mode == "hard":
        sub_sentences = [sentences[i] for i in cand_idx]
        sub_scores = attach_selector_salience(
            candidate_records, base_scores, salience_source
        )
        sub_sim = sim[np.ix_(cand_idx, cand_idx)] if sim is not None else None
        sub_coverage = sim[:, cand_idx] if sim is not None else None
    elif use_cand and cand_idx:
        if mode != "hard":
            if salience_source.lower() not in {"base_score", "membership_only"}:
                raise ValueError(
                    "provenance-aware selector salience requires candidates.mode='hard'"
                )
            sub_sentences = sentences
            sub_scores = base_scores[:]
            for i in cand_idx:
                sub_scores[i] = float(sub_scores[i]) * soft_boost
            sub_sim = sim
            sub_coverage = sim
    else:
        sub_sentences = sentences
        sub_scores = base_scores
        sub_sim = sim
        sub_coverage = sim

    candidate_capacity_words = maximum_feasible_words(
        sub_sentences,
        max_length=selector_budget,
        length_unit=unit,
        max_sentences=max_sents,
    )
    if use_cand and mode == "hard" and candidate_capacity_words < effective_min_words:
        raise ValueError(
            f"candidate pool for document {doc.get('id')!r} cannot satisfy the "
            f"effective minimum: candidate_capacity_words={candidate_capacity_words}, "
            f"effective_min_words={effective_min_words}, "
            f"source_capacity_words={source_capacity_words}"
        )

    # 6. Optimizer dispatch. A word budget no longer bypasses the configured
    # selector; it is simply the selector's independent output constraint.
    optimizer_diagnostics: Dict = {}
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

    selection_evaluation = None
    if sub_sentences:
        evaluator = objective_from_spec(
            sub_sentences,
            sub_scores,
            sub_sim,
            objective_spec,
            max_length=selector_budget,
            length_unit=unit,
            max_sentences=max_sents,
            min_words=effective_min_words,
            require_nonempty=require_nonempty,
            coverage_matrix=sub_coverage,
        )
        selection_evaluation = evaluator.assert_feasible(picked_sub).to_dict()

    # 7. Map back to original indices
    if use_cand and cand_idx and mode == "hard":
        selected = sorted(cand_idx[i] for i in picked_sub)
    else:
        selected = sorted(picked_sub)

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
            solution["selected_indices"] = (
                sorted(cand_idx[index] for index in relative)
                if use_cand and cand_idx and mode == "hard"
                else relative
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
        "optimizer_diagnostics": optimizer_diagnostics or None,
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

def summarize_jsonl(
    input_path: str,
    predictions_path: str,
    cfg: Dict,
    requested_split: str,
    dataset_preflight: Dict | None = None,
) -> int:
    """Stream one dataset into an atomic prediction artifact."""

    if dataset_preflight is None:
        dataset_preflight = validate_dataset_policy_request(
            cfg, input_path, requested_split
        )

    processed = 0

    def prediction_rows():
        nonlocal processed
        for doc in tqdm(read_jsonl(input_path), desc="Summarizing"):
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
    # A formal run is incomplete if its timing artifact cannot be written.
    with open(os.path.join(out_dir, "time_select_seconds.txt"), "w", encoding="utf-8") as f:
        f.write(f"{t1 - t0:.6f}")
    print(f"Wrote predictions to {preds_path}")


if __name__ == "__main__":
    main()
