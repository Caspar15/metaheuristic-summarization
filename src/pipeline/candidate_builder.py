"""Candidate routing, provenance fusion, and coverage guards.

Every enabled route scores the complete input before its quota is applied.  A
route failure is a failed run: this module never replaces missing scores with
zeros and never silently switches to another route.
"""

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from src.data.schemas import validate_candidate_record
from src.features.graph import (
    compute_textrank_scores,
    sparse_tfidf_knn_textrank_scores,
)
from src.features.position import document_position_scores
from src.models.extractive.encoder_rank import encoder_route_scores
from src.representations.tfidf_helper import tfidf_scores_and_sim
from src.utils.tokenizer import count_tokens


RouteMetadata = Dict[str, Any]


def _document_aware_position_scores(sentence_records: List[Dict]) -> List[float]:
    return document_position_scores(sentence_records, version="v1")


def _canonical_route_name(raw_source: str) -> str:
    name = (raw_source or "").strip().lower()
    aliases = {
        "score": "lexical",
        "lexical": "lexical",
        "centrality": "tfidf_centroid",
        "tfidf": "tfidf_centroid",
        "tfidf_centroid": "tfidf_centroid",
        "graph": "graph",
        "textrank": "graph",
        "plm": "semantic",
        "semantic": "semantic",
        # Compatibility only. Position is labelled as a guard signal in the
        # artifact and should be configured through coverage_guard in new runs.
        "position": "position_guard",
        "position_guard": "position_guard",
    }
    if name not in aliases:
        raise ValueError(f"unknown candidate route: {raw_source!r}")
    return aliases[name]


def _config_for_route(
    route_config: Mapping[str, Mapping[str, Any]],
    canonical_name: str,
    raw_source: str,
) -> Mapping[str, Any]:
    value = route_config.get(canonical_name, route_config.get(raw_source, {}))
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration for route {canonical_name!r} must be an object")
    return value


def _score_route(
    route: str,
    sentences: List[str],
    sentence_records: List[Dict],
    base_scores: List[float],
    sim_matrix,
    threshold: float,
    config: Mapping[str, Any],
    precomputed: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[float], RouteMetadata]:
    n = len(sentences)
    if route == "lexical":
        values = [float(value) for value in base_scores]
        metadata: RouteMetadata = {
            "route_type": "lexical_handcrafted",
            "model_revision": str(config.get("revision", "base_scores:v1")),
            "estimated_cost": {"scored_sentences": n},
        }
    elif route == "position_guard":
        values = _document_aware_position_scores(sentence_records)
        metadata = {
            "route_type": "coverage_guard_signal",
            "model_revision": "document_position:v1",
            "estimated_cost": {"scored_sentences": n},
        }
    elif route == "tfidf_centroid":
        values, _ = tfidf_scores_and_sim(
            sentences,
            sublinear_tf=bool(config.get("sublinear_tf", False)),
            ngram_range=tuple(config.get("ngram_range", (1, 1))),
        )
        metadata = {
            "route_type": "lexical_tfidf_diagnostic",
            "model_revision": "sklearn_tfidf_centroid:v1",
            "estimated_cost": {"vectorized_sentences": n},
        }
    elif route == "graph":
        implementation = str(config.get("implementation", "sparse_knn")).lower()
        if implementation == "sparse_knn":
            values, graph_facts = sparse_tfidf_knn_textrank_scores(
                sentences,
                n_neighbors=int(config.get("n_neighbors", 8)),
                min_similarity=float(config.get("min_similarity", threshold or 0.05)),
                alpha=float(config.get("alpha", 0.85)),
                max_iter=int(config.get("max_iter", 100)),
                tol=float(config.get("tol", 1e-6)),
            )
            metadata = {
                "route_type": "graph_centrality",
                "model_revision": "sparse_tfidf_knn_textrank:v1",
                "estimated_cost": graph_facts,
                "representation": "tfidf_cosine_knn",
                "bounded_sparse": True,
            }
        elif implementation == "dense_legacy":
            if sim_matrix is None:
                _, graph_sim = tfidf_scores_and_sim(sentences)
                representation = "tfidf_internal"
            else:
                graph_sim = sim_matrix
                representation = str(
                    config.get("representation", "pipeline_similarity")
                )
            values = [
                float(value)
                for value in compute_textrank_scores(graph_sim, threshold=threshold)
            ]
            metadata = {
                "route_type": "graph_centrality",
                "model_revision": "dense_textrank_legacy:v1",
                "estimated_cost": {
                    "nodes": n,
                    "dense_similarity_entries": n * n,
                },
                "representation": representation,
                "threshold": threshold,
                "bounded_sparse": False,
            }
        else:
            raise ValueError(
                "routes.graph.implementation must be 'sparse_knn' or 'dense_legacy'"
            )
    elif route == "semantic":
        model_name = config.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "semantic candidate route requires routes.semantic.model_name; "
                "use an explicit sentence-similarity checkpoint"
            )
        if precomputed is None:
            values, semantic_metadata = encoder_route_scores(
                sentences,
                model_name=model_name,
                device=config.get("device"),
                batch_size=int(config.get("batch_size", 16)),
                max_model_tokens=int(config.get("max_model_tokens", 256)),
                revision=config.get("revision"),
            )
        else:
            values = list(precomputed.get("values", []))
            semantic_metadata = dict(precomputed.get("metadata", {}))
            if semantic_metadata.get("model_name") != model_name:
                raise ValueError(
                    "precomputed semantic route model does not match "
                    "routes.semantic.model_name"
                )
            configured_revision = config.get("revision")
            if configured_revision is not None and str(
                semantic_metadata.get("model_revision")
            ) != str(configured_revision):
                raise ValueError(
                    "precomputed semantic route revision does not match "
                    "routes.semantic.revision"
                )
        metadata = {
            "route_type": "semantic_sentence_encoder",
            **semantic_metadata,
        }
    else:  # pragma: no cover - guarded by _canonical_route_name
        raise ValueError(f"unsupported candidate route: {route!r}")

    if len(values) != n:
        raise RuntimeError(
            f"candidate route {route!r} returned {len(values)} scores for {n} rows"
        )
    return [float(value) for value in values], metadata


def _rank(values: List[float]) -> Tuple[List[int], Dict[int, int]]:
    ranking = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return ranking, {index: rank for rank, index in enumerate(ranking, start=1)}


def _coverage_guard_indices(
    sentence_records: List[Dict],
    fused_ranks: Mapping[int, int],
    guard_config: Mapping[str, Any],
) -> Dict[int, List[str]]:
    """Return deterministic guard nominees and their auditable reasons."""

    if not guard_config or not bool(guard_config.get("enabled", False)):
        return {}

    reasons: Dict[int, List[str]] = {}

    def add_best_by(field: str, reason: str) -> None:
        groups: Dict[object, List[int]] = {}
        for index, record in enumerate(sentence_records):
            key = record.get(field)
            if key is not None:
                groups.setdefault(key, []).append(index)
        for indices in groups.values():
            best = min(indices, key=lambda index: (fused_ranks[index], index))
            reasons.setdefault(best, []).append(reason)

    if bool(guard_config.get("document", True)):
        add_best_by("document_id", "guard:document")
    if bool(guard_config.get("section", False)):
        add_best_by("section_id", "guard:section")
    if bool(guard_config.get("position", False)):
        groups: Dict[object, List[int]] = {}
        for index, record in enumerate(sentence_records):
            key = record.get("document_id") or "__legacy_flat__"
            groups.setdefault(key, []).append(index)
        for indices in groups.values():
            lead = min(
                indices,
                key=lambda index: (
                    int(sentence_records[index].get("document_position", index)),
                    index,
                ),
            )
            reasons.setdefault(lead, []).append("guard:position")

    max_items = guard_config.get("max_items")
    if max_items is not None:
        limit = max(0, int(max_items))
        ordered = sorted(reasons, key=lambda index: (fused_ranks[index], index))[:limit]
        reasons = {index: reasons[index] for index in ordered}
    return reasons


def _resolve_min_per_route(
    value: int | Mapping[str, int],
    routes: List[str],
    configured_route_top_k: int,
) -> Dict[str, int]:
    if isinstance(value, Mapping):
        resolved = {route: 0 for route in routes}
        for raw_route, raw_minimum in value.items():
            route = _canonical_route_name(str(raw_route))
            if route not in resolved:
                raise ValueError(
                    f"minimum reservation configured for disabled route {raw_route!r}"
                )
            resolved[route] = int(raw_minimum)
    else:
        resolved = {route: int(value) for route in routes}
    for route, minimum in resolved.items():
        if minimum < 0:
            raise ValueError("min_per_route must be non-negative")
        if minimum > configured_route_top_k:
            raise ValueError(
                f"min_per_route for {route!r} ({minimum}) exceeds "
                f"configured route_top_k ({configured_route_top_k})"
            )
    return resolved


def build_candidate_pool(
    sentence_records: List[Dict],
    base_scores: List[float],
    k: int,
    sources: List[str],
    sim_matrix=None,
    threshold: float = 0.0,
    *,
    total_budget: Optional[int] = None,
    min_per_route: int | Mapping[str, int] = 0,
    route_config: Optional[Mapping[str, Mapping[str, Any]]] = None,
    coverage_guard: Optional[Mapping[str, Any]] = None,
    rrf_constant: int = 60,
    precomputed_route_data: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build route proposals, reservations, guards, and one capped final pool.

    ``k`` is the proposal depth (``route_top_k``), not a guarantee that every
    proposal survives the final cap. ``min_per_route`` is the explicit route
    reservation. RRF may fill remaining slots only from the proposal union or
    explicit coverage guards; sentences outside that universe cannot enter.
    """

    n = len(sentence_records)
    if n == 0:
        return {
            "records": [],
            "route_proposals": {},
            "allocation": {
                "route_top_k": 0,
                "requested_route_top_k": int(k),
                "effective_route_top_k": 0,
                "min_per_route": {},
                "requested_min_per_route": {},
                "effective_min_per_route": {},
                "reservation_shortfall_by_route": {},
                "total_cap": total_budget,
                "proposal_union_size": 0,
                "candidate_universe_size": 0,
                "actual_size": 0,
                "underfilled_by": int(total_budget or 0),
            },
        }
    if len(base_scores) != n:
        raise ValueError("base_scores length must match sentence_records")
    if isinstance(sources, (str, bytes)) or not sources:
        raise ValueError("at least one candidate route is required")
    if rrf_constant < 1:
        raise ValueError("rrf_constant must be positive")

    configured_quota = int(k)
    if configured_quota < 1:
        raise ValueError("per-route candidate quota must be positive")
    quota = min(configured_quota, n)
    if total_budget is not None:
        total_budget = int(total_budget)
        if total_budget < 1:
            raise ValueError("total candidate budget must be positive")
        total_budget = min(total_budget, n)
    sentences = [record["text"] for record in sentence_records]
    configs = route_config or {}
    precomputed_routes = precomputed_route_data or {}

    route_values: Dict[str, List[float]] = {}
    route_metadata: Dict[str, RouteMetadata] = {}
    for raw_source in sources:
        route = _canonical_route_name(raw_source)
        if route in route_values:
            continue
        config = _config_for_route(configs, route, raw_source)
        try:
            values, metadata = _score_route(
                route,
                sentences,
                sentence_records,
                base_scores,
                sim_matrix,
                threshold,
                config,
                precomputed_routes.get(route),
            )
        except Exception as exc:
            raise RuntimeError(f"candidate route {route!r} failed: {exc}") from exc
        route_values[route] = values
        route_metadata[route] = metadata

    route_rankings: Dict[str, List[int]] = {}
    route_ranks: Dict[str, Dict[int, int]] = {}
    route_proposal_sets: Dict[str, Set[int]] = {}
    union_indices: Set[int] = set()
    for route, values in route_values.items():
        ranking, ranks = _rank(values)
        route_rankings[route] = ranking
        route_ranks[route] = ranks
        proposals = set(ranking[:quota])
        route_proposal_sets[route] = proposals
        union_indices.update(proposals)

    requested_route_minimums = _resolve_min_per_route(
        min_per_route, list(route_values), configured_quota
    )
    # A reservation is a guarantee over evidence that actually exists, not a
    # requirement that every document contain at least the configured number
    # of sentences. Keep configuration validation tied to the configured
    # route_top_k, then clamp each row to the proposals that route can supply.
    route_minimums = {
        route: min(requested_route_minimums[route], len(proposals))
        for route, proposals in route_proposal_sets.items()
    }
    reservation_shortfalls = {
        route: requested_route_minimums[route] - route_minimums[route]
        for route in route_values
    }

    fusion_scores = {
        index: sum(
            1.0 / (rrf_constant + route_ranks[route][index])
            for route in route_values
        )
        for index in range(n)
    }
    fused_order = sorted(range(n), key=lambda i: (-fusion_scores[i], i))
    fused_ranks = {index: rank for rank, index in enumerate(fused_order, start=1)}
    fusion_min = min(fusion_scores.values())
    fusion_max = max(fusion_scores.values())
    fusion_normalized = {
        index: (
            1.0
            if fusion_max == fusion_min
            else (fusion_scores[index] - fusion_min) / (fusion_max - fusion_min)
        )
        for index in range(n)
    }
    guard_reasons = _coverage_guard_indices(
        sentence_records, fused_ranks, coverage_guard or {}
    )

    inclusion_reasons: Dict[int, List[str]] = {
        index: [
            f"route:{route}"
            for route, ranking in route_rankings.items()
            if index in ranking[:quota]
        ]
        for index in union_indices
    }
    for index, reasons in guard_reasons.items():
        inclusion_reasons.setdefault(index, []).extend(reasons)

    # Reserve route-specific evidence before consensus fill. Exclusive
    # proposals are preferred so the mechanism does not erase the unique
    # contribution it is intended to measure.
    proposal_memberships = {
        index: sum(index in proposals for proposals in route_proposal_sets.values())
        for index in union_indices
    }
    reserved_by_route: Dict[str, List[int]] = {}
    for route, ranking in route_rankings.items():
        proposal_order = [index for index in ranking if index in route_proposal_sets[route]]
        exclusive = [
            index for index in proposal_order if proposal_memberships[index] == 1
        ]
        shared = [index for index in proposal_order if proposal_memberships[index] > 1]
        reserved = (exclusive + shared)[: route_minimums[route]]
        reserved_by_route[route] = reserved
        for index in reserved:
            inclusion_reasons.setdefault(index, []).append(f"reserve:{route}")

    candidate_universe = set(union_indices) | set(guard_reasons)
    mandatory = set(guard_reasons)
    for reserved in reserved_by_route.values():
        mandatory.update(reserved)

    effective_cap = total_budget
    if effective_cap is not None:
        if len(mandatory) > effective_cap:
            raise ValueError(
                f"candidate total cap {effective_cap} cannot fit "
                f"{len(mandatory)} mandatory route/guard reservations"
            )
        chosen = set(mandatory)
        for index in fused_order:
            if len(chosen) >= effective_cap:
                break
            if index not in candidate_universe or index in chosen:
                continue
            chosen.add(index)
            inclusion_reasons.setdefault(index, []).append("rrf_fill")
        selected_indices = chosen
    else:
        selected_indices = candidate_universe

    candidates = []
    for index in sorted(selected_indices):
        sentence = sentence_records[index]
        candidate = {
            "sentence_id": sentence["sentence_id"],
            "text": sentence["text"],
            "original_index": sentence["original_index"],
            "document_id": sentence.get("document_id"),
            "section_id": sentence.get("section_id"),
            "word_count": len(sentence["text"].split()),
            "route_scores": {},
            "selected_by_routes": [],
            "fusion_score": fusion_scores[index],
            "fusion_normalized": fusion_normalized[index],
            "fused_rank": fused_ranks[index],
            "inclusion_reasons": inclusion_reasons.get(index, []),
        }
        for route, values in route_values.items():
            rank = route_ranks[route][index]
            metadata = route_metadata[route]
            candidate["route_scores"][route] = {
                "raw": values[index],
                "rank": rank,
                "percentile": 1.0 if n == 1 else 1.0 - ((rank - 1) / (n - 1)),
                "route_type": metadata["route_type"],
                "model_revision": metadata.get("model_revision"),
                "estimated_cost": metadata.get("estimated_cost"),
                "metadata": {
                    key: value
                    for key, value in metadata.items()
                    if key
                    not in {"route_type", "model_revision", "estimated_cost"}
                },
            }
            if rank <= quota:
                candidate["selected_by_routes"].append(route)
        candidate["route_agreement"] = len(candidate["selected_by_routes"])
        validate_candidate_record(candidate)
        candidates.append(candidate)

    route_proposals = {
        route: [
            {
                "sentence_id": sentence_records[index]["sentence_id"],
                "original_index": sentence_records[index]["original_index"],
                "rank": route_ranks[route][index],
                "raw": route_values[route][index],
                "selected_in_final_pool": index in selected_indices,
                "reserved": index in reserved_by_route[route],
            }
            for index in route_rankings[route][:quota]
        ]
        for route in route_values
    }
    dropped_by_route = {
        route: sum(index not in selected_indices for index in proposals)
        for route, proposals in route_proposal_sets.items()
    }
    reservation_counts = {
        route: sum(index in selected_indices for index in proposals)
        for route, proposals in route_proposal_sets.items()
    }
    underfilled_by = (
        max(0, effective_cap - len(selected_indices))
        if effective_cap is not None
        else 0
    )
    return {
        "records": candidates,
        "route_proposals": route_proposals,
        "allocation": {
            "route_top_k": quota,
            "requested_route_top_k": configured_quota,
            "effective_route_top_k": quota,
            "min_per_route": route_minimums,
            "requested_min_per_route": requested_route_minimums,
            "effective_min_per_route": route_minimums,
            "reservation_shortfall_by_route": reservation_shortfalls,
            "total_cap": effective_cap,
            "proposal_union_size": len(union_indices),
            "coverage_guard_size": len(guard_reasons),
            "candidate_universe_size": len(candidate_universe),
            "mandatory_size": len(mandatory),
            "actual_size": len(candidates),
            "underfilled_by": underfilled_by,
            "selected_proposals_by_route": reservation_counts,
            "dropped_proposals_by_route": dropped_by_route,
        },
    }


def build_candidate_records(
    sentence_records: List[Dict],
    base_scores: List[float],
    k: int,
    sources: List[str],
    sim_matrix=None,
    threshold: float = 0.0,
    *,
    total_budget: Optional[int] = None,
    min_per_route: int | Mapping[str, int] = 0,
    route_config: Optional[Mapping[str, Mapping[str, Any]]] = None,
    coverage_guard: Optional[Mapping[str, Any]] = None,
    rrf_constant: int = 60,
    precomputed_route_data: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[Dict]:
    """Backward-compatible record-only view of :func:`build_candidate_pool`."""

    return build_candidate_pool(
        sentence_records,
        base_scores,
        k,
        sources,
        sim_matrix=sim_matrix,
        threshold=threshold,
        total_budget=total_budget,
        min_per_route=min_per_route,
        route_config=route_config,
        coverage_guard=coverage_guard,
        rrf_constant=rrf_constant,
        precomputed_route_data=precomputed_route_data,
    )["records"]


def build_candidate_union(
    sentences: List[str],
    base_scores: List[float],
    k: int,
    sources: List[str],
    sim_matrix=None,
    threshold: float = 0.0,
) -> List[int]:
    """Backward-compatible strict wrapper returning candidate indices only."""

    records = [
        {
            "sentence_id": f"legacy:s{index:06d}",
            "text": sentence,
            "original_index": index,
            "document_id": None,
            "section_id": None,
            "document_position": index,
        }
        for index, sentence in enumerate(sentences)
    ]
    return [
        record["original_index"]
        for record in build_candidate_records(
            records,
            base_scores,
            k,
            sources,
            sim_matrix=sim_matrix,
            threshold=threshold,
        )
    ]


def greedy_oracle_indices(
    sentences: List[str], reference: str, max_tokens: int
) -> List[int]:
    """Greedy reference oracle for offline diagnostic analysis only."""
    try:
        from rouge_score import rouge_scorer
    except Exception as exc:
        raise RuntimeError(
            "greedy oracle diagnostics require the 'rouge-score' package"
        ) from exc
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    selected: List[int] = []
    cur_summary = ""
    cur_tokens = 0
    best_f = 0.0
    n = len(sentences)
    for _ in range(n):
        best_i = None
        best_gain = 0.0
        for i in range(n):
            if i in selected:
                continue
            t = count_tokens(sentences[i])
            if cur_tokens + t > max_tokens:
                continue
            cand = (cur_summary + " " + sentences[i]).strip()
            f = scorer.score(reference or "", cand)["rouge1"].fmeasure
            gain = f - best_f
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        cur_summary = (cur_summary + " " + sentences[best_i]).strip()
        cur_tokens += count_tokens(sentences[best_i])
        best_f = scorer.score(reference or "", cur_summary)["rouge1"].fmeasure
    return sorted(selected)
