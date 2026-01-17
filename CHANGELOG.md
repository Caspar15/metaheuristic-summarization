# Changelog

All notable changes to the `metaheuristic-summarization` project will be documented in this file.

## [v0.4.0] - 2026-01-14 (Current Best Model)
### Added
- **3-Way Fusion Architecture**: Implemented a multi-view fusion pipeline combining Statistical (Base), Semantic (LLM), and Structural (Graph) scores.
- **Stage 1 Graph**: Added `src.features.graph` module implementing TextRank (PageRank) algorithm.
- **NSGA-II Integration**: Upgraded Stage 1 Base optimizer from `greedy` to `nsga2` for better candidate selection.
- **Multi-News Benchmark**: Achieved **ROUGE-1: 44.32** on Multi-News, surpassing the HiMAP benchmark (44.17).
- **Final Configs**: Standardized best-performing configurations in `configs/final/`.

### Changed
- **Pipeline Update**: `scripts/build_union_stage2.py` now supports 3 inputs: `--base_pred`, `--bert_pred`, and `--graph_pred`.
- **Optimization**: Tuned `max_tokens` to 245 and `lambda_coverage` to 2.5 for Multi-News dataset.

## [v0.3.0] - 2026-01-10
### Added
- **Multi-Objective Optimization**: Initial implementation of `nsga2` and `fast_nsga2` in Stage 2.
- **Extractive Pipeline**: Built core `select_sentences.py` with modular feature scoring.

## [v0.2.0] - 2025-12-10
### Added
- **Feature Correlation**: Scripts to analyze feature importance.

## [v0.1.0] - 2025-11-28
### Initial Release
- Basic Greedy optimizer.
- TF-ISF scoring.
