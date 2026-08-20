"""Run the preregistered Multi-News E3 route/provenance ablation on dev only."""

from scripts.audit import run_govreport_e3_ablation as implementation
from scripts.audit.run_greedy_sensitivity import REPO_ROOT


implementation.PARENT = REPO_ROOT / "configs/preregistrations/multinews_secondary_evidence_completion_v1.json"
implementation.ADDENDUM = REPO_ROOT / "configs/preregistrations/multinews_e3_execution_addendum_v1.json"
implementation.OUTPUT_ROOT = REPO_ROOT / "runs_v2/multinews_e3_route_provenance_v1/dev"
implementation.ANCHOR_CONFIG = REPO_ROOT / "runs_v2/d3b_cross_profile_combination_v1/multinews/dev/C01_combined_salience_route_weight/resolved_config.yaml"
implementation.ANCHOR_PREDICTIONS = REPO_ROOT / "runs_v2/d3b_cross_profile_combination_v1/multinews/dev/C01_combined_salience_route_weight/predictions.jsonl"
implementation.ANCHOR_SUMMARY = REPO_ROOT / "runs_v2/d3b_cross_profile_combination_v1/multinews/dev/C01_combined_salience_route_weight/candidate_summary.json"
implementation.DATASET_KEY = "multinews"
implementation.DATASET_LABEL = "Multi-News"
implementation.STUDY_ID = "multinews-e3-route-provenance-v1"
implementation.PARTITION_LABEL = "Multi-News frozen dev"
implementation.BASE_SEED = 20260920
implementation.CACHE_ROOT = REPO_ROOT / "runs_v2/gate2_baseline_matrix_v1/multinews/dev/plm/_embedding_cache"


if __name__ == "__main__":
    implementation.main()
