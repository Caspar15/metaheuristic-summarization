"""Measure preregistered Multi-News E2 cold/warm runtime, memory, and scaling."""

from collections.abc import Mapping
from typing import Any, Callable

from scripts.audit import run_govreport_e2_cost as implementation
from scripts.audit.run_greedy_sensitivity import REPO_ROOT
from src.baselines.centrality import summarize_one_lexrank, summarize_one_textrank
from src.baselines.lead import summarize_one_lead
from src.baselines.pacsum import summarize_one_pacsum_sbert, summarize_one_pacsum_tfidf
from src.baselines.semantic import summarize_one_sbert_centroid, summarize_one_sbert_mmr
from src.pipeline.select_sentences import summarize_one


implementation.ADDENDUM = REPO_ROOT / "configs/preregistrations/multinews_cost_addendum_v1.json"
implementation.CACHE_ERRATUM = REPO_ROOT / "configs/preregistrations/multinews_e2_cache_classification_v1.json"
implementation.SAMPLE = REPO_ROOT / "configs/pilot_manifests/multinews_cost_scaling_sample_v1.json"
implementation.CANONICAL = REPO_ROOT / "data/processed/multi_news_validation_canonical.jsonl"
implementation.OUTPUT_ROOT = REPO_ROOT / "runs_v2/multinews_cost_scaling_v1"
implementation.DATASET_LABEL = "Multi-News"
implementation.STUDY_ID = "multinews-cost-scaling-v1"
implementation.WORKER_MODULE = "scripts.audit.run_multinews_e2_cost"
implementation.EXCLUDED_FAILED_ATTEMPTS = []
implementation.PLM_SYSTEMS = {
    "frozen_C01_proposed",
    "pacsum_sbert_P03",
    "full_source_sbert_mmr_lambda_0.7",
    "sbert_centroid",
    "matched_nsga2_tfidf",
}
implementation.SYSTEM_ORDER = (
    "frozen_C01_proposed",
    "pacsum_tfidf_P08",
    "pacsum_sbert_P03",
    "full_source_sbert_mmr_lambda_0.7",
    "sbert_centroid",
    "matched_nsga2_tfidf",
    "lead",
    "textrank",
    "lexrank",
)


def _runner(system: str) -> Callable[[Mapping[str, Any], Mapping[str, Any]], dict]:
    if system in {"frozen_C01_proposed", "matched_nsga2_tfidf"}:
        return summarize_one
    if system == "pacsum_tfidf_P08":
        return summarize_one_pacsum_tfidf
    if system == "pacsum_sbert_P03":
        return summarize_one_pacsum_sbert
    if system == "full_source_sbert_mmr_lambda_0.7":
        return summarize_one_sbert_mmr
    if system == "sbert_centroid":
        return summarize_one_sbert_centroid
    if system == "lead":
        return lambda doc, cfg: summarize_one_lead(
            doc, cfg, ordering="document_order", first_k=3
        )
    if system == "textrank":
        return summarize_one_textrank
    if system == "lexrank":
        return summarize_one_lexrank
    raise ValueError(f"unknown Multi-News E2 system {system!r}")


implementation._runner = _runner


if __name__ == "__main__":
    implementation.main()
