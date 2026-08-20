# Changelog

All notable changes to the `metaheuristic-summarization` project will be documented in this file.

> ⚠️ **Entries before v0.5.0 contain performance claims that a later audit
> invalidated.** They are kept for history, with corrections noted inline.
> See `docs/research/CODE_AUDIT_IEEE_Access.md`.

## [Unreleased] - GovReport-centered pre-test evidence freeze

- **Final frozen experiments completed (2026-08-20).** GovReport official test
  retains the scoped superiority claim (`+0.003700` macro vs SBERT+MMR,
  `p=0.000040`). Multi-News official test ranks Proposed first by macro but ties
  PacSum-TFIDF statistically (`+0.000091`, `p=0.907111`) with positive R-1/R-L
  and negative R-2 trade-offs.
- Multi-News E3 completed five preregistered route/provenance ablations; all
  macro endpoints pass Holm-20. A dependency-mismatched attempt is preserved
  and excluded; the authoritative rerun has exact frozen-anchor parity.
- Multi-News E2 completed 77 non-overlapping attempts across nine systems.
  Proposed cold/warm medians are `47.85/9.23s`; matched NSGA-II is
  `91.69/50.07s`. Scientific search is closed; remaining work is manuscript,
  artifact reproduction and submission compliance.

The older pre-test bullets immediately below are a historical checkpoint.

- **Documentation and execution status aligned (2026-08-15).** A1 through D3b,
  the two historical task profiles' Gate 2 matrices, six greedy-reference runs,
  and paired analyses are complete. Configuration search is closed.
- Frozen GovReport C01 scores `0.579106 / 0.249380 / 0.543725` (macro
  `0.457404`) under the internal evaluator, `+0.004636` over full-source
  SBERT+MMR; Multi-News remains a negative boundary (`−0.001323` macro and
  significantly worse ROUGE-2 versus PacSum).
- The next executable work is E1 official Stanza/Perl evaluator parity, E2
  controlled cold/warm runtime-memory-scaling, and E3 five preregistered
  route/provenance ablations on frozen GovReport dev. This is evidence
  completion, not test execution.
- GovReport official test remains locked pending E1-E3, a versioned test data
  policy, exact environment/commit freeze, and teacher/full-author signatures.
- Current regression evidence: 459 local tests passed; PR #17 clean-clone Linux
  CI reported 454 passed and 5 skipped.

### Historical Phase 1e / Phase 2 entries

The entries below preserve the chronological engineering record. Their
then-current pending statements are superseded by the checkpoint above.

- **First validation pilot measured (2026-08-03, diagnostic).** See
  `docs/research/CODE_AUDIT_IEEE_Access.md` F-17 and F-18. Headlines:

  > 🔴 **`greedy` could not complete a full validation run.** One document in
  > 5,621 (`validation_4066`) left it below `min_words`, `assert_feasible`
  > raised, and because the artifact is written atomically the whole run was
  > discarded. The root cause is shared with the Lead and Random baselines:
  > the lower bound is defined against `maximum_feasible_words`, an exact
  > arbitrary-subset optimum, while no actual selector is an optimal packer.
  > **Fixed in PR #12** — see the F-17 entry below.

  > 🟠 **No configuration beats Lead.** The best one so far,
  > `greedy + length_normalized`, appears to win ROUGE-1 (+0.0014) and
  > ROUGE-Lsum (+0.0019) — but a length bracket (Lead at 229.4 / 233.6 /
  > 258.8 words against the system's 244.0) shows both metrics rising
  > monotonically with word count, so the lead is explained entirely by
  > spending 10.4 more words. ROUGE-2 loses by 0.011–0.014 at every length.

  > 🟡 **The objective matters about 6× more than the optimizer.** Switching
  > `importance_aggregation` from `mean` to `length_normalized` is worth
  > +0.0232 ROUGE-Lsum for 10 minutes of compute; switching greedy to
  > NSGA-II is worth +0.0039 for 322 minutes.

  > 🔴 **Under `mean`, the system scores below the Random baseline on
  > ROUGE-Lsum** (greedy 0.3728, NSGA-II 0.3767, Random 0.3788).

  All of the above are single-seed, no paired bootstrap, MVP config only (no
  graph route, `position` weight 0), and were measured before F-17 was fixed,
  so they skipped the documents it made infeasible. They are **not** Gate 2
  results, and they have not been re-measured under the post-F-17 pipeline.

- **F-17 fixed (PR #12).** All four document-level infeasibility outcomes —
  `source_no_eligible_sentence`, `candidate_capacity_shortfall`,
  `selector_min_words_shortfall`, `optimizer_no_feasible_solution` — are now
  written as complete prediction rows with a reason code instead of aborting
  the batch. Upper-bound violations and config/schema/programming errors still
  fail loud. `evaluate` now scores **all rows by default** so methods share a
  denominator; `scripts/audit/paired_run_intersection.py` produces the
  common-feasible intersection for paired sensitivity. Verified on the full
  governed split: 5,621 rows completed, 5,620 feasible, 1 recorded.

- Added `scripts/audit/length_matched_lead.py` and
  `scripts/audit/selection_overlap.py` so the two measurements above are
  reproducible rather than asserted.
- First governed baseline artifact: `runs_v2/gate2_lead_document_order_validation/`
  (Multi-News validation, 5,621 rows, 0.433204 / 0.146768 / 0.394039).

- **PR #10 merged the first production baseline path**: a shared baseline
  contract, governed CLI, and Lead with `document_order`, `round_robin`, and
  diagnostic `fabbri_first_k` orderings. Lead shares the canonical data-policy
  preflight and upper-budget contract with the system pipeline. Its artifact
  explicitly records that the provisional `min_words` floor is not applied,
  together with the requested floor, source capacity, selected length, and
  reason; the full-split distribution is documented as F-16.
- The current master checkpoint passes **217 tests** (2026-08-02). This is a
  correctness checkpoint, not a baseline result: no governed Lead run has yet
  been completed on both GovReport and Multi-News, and Gate 2 remains open.

- **Sentence segmentation is now shared between the data layer and the
  evaluator** (PR #9). `src/eval/rouge.py` previously segmented with a
  hand-written regex, `(?<=[.!?。！？])\s+`, while `preprocess_multinews.py`
  used Punkt, so "a sentence" meant two different things on the two sides.
  The regex also split after every abbreviation period, turning
  `"Mr. Smith met U.S. officials on Tuesday."` into three sentences. Both
  sides now go through `src/data/sentence_split.py`.

  > ⛔ **Every previously recorded `rougeLsum` figure is stale.** Measured on
  > 800 real validation rows (Lead-style extract, 245-word budget):
  > `rougeLsum` 0.3893 → 0.3925 (**+0.0032**); `rouge1` and `rouge2` are
  > unchanged at +0.0000, since they do not use sentence boundaries.
  > Data-side behaviour did not change, so the frozen canonical fingerprint
  > and the data policy are unaffected.

- Added one shared evaluator for salience, facility coverage, redundancy, scalar
  utility, and non-empty/min-max length feasibility.
- Facility coverage now evaluates the full source-to-candidate matrix; its
  universe is no longer silently truncated to the candidate pool.
- Greedy, GRASP, and NSGA-II now search the same declared problem; infeasible
  results fail loudly instead of being emitted or silently repaired.
- NSGA-II predictions retain the feasible Pareto front, per-solution objective
  values, and selected row. The current weighted-sum policy remains provisional.
- Added hand-computed objective/constraint tests and cross-run seed checks.
- Added hand-computed TF-ISF/length/position and ROUGE protocol golden tests;
  invalid position methods, decay values, and length clips now fail loudly.
- Corrected TF-ISF v2 smoothing to `log((N+1)/(sf+1))`, preventing ubiquitous
  terms from becoming negative evidence; v1 remains available for legacy runs.

## [v0.6.0] - 2026-07-26 (Phase 1 research contracts)

Establishes the contracts the Phase 1 validation pilot depends on. Nothing here
is a research result: no configuration has been evaluated against a baseline yet.

### Added
- `src/data/schemas.py`, `preprocess_multinews.py`, `validate_dataset.py` —
  canonical DocumentExample with source-document boundaries, per-document
  sentence positions, deterministic NLTK Punkt segmentation, char-span mapping,
  pinned dataset revision, and a health/fingerprint report. The old flat
  Multi-News JSONL had lost `|||||` boundaries and cannot support
  cross-document objectives.
- `src/objectives/factory.py` — objectives are created only when the declared
  task profile makes them meaningful. Single-sentence tasks disable redundancy
  and subset search; profiled multi-sentence tasks reject raw-sum salience,
  which rewarded cardinality; declared-but-unimplemented document-group
  coverage raises rather than being reported as if it existed.
- `src/eval/protocol.py` — evaluation runs only under an explicitly named
  protocol. `scitldr_official` fails closed until a conformance-tested wrapper
  exists, so generic rouge-score output cannot be labelled official.
- `configs/phase1_mvp_multinews.yaml` — validation-only MVP isolating lexical +
  semantic candidate utility with a deterministic selector. Graph and NSGA-II
  are deliberately excluded from the first gate.
- GitHub Actions unit-test workflow with a light `requirements-ci.txt`.

### Changed
- Candidate generation: every enabled route now scores the complete input before
  any quota applies. `route_top_k` is proposal depth, `min_per_route` is a
  binding reservation, and RRF may fill the remaining cap only from the proposal
  union or explicit coverage guards. An infeasible cap raises; an unreachable
  cap is reported as `underfilled_by` rather than padded from the document.
- Selector salience is explicit and auditable (`base_score`, `membership_only`,
  `rrf_fusion`, `<route>_percentile`). The MVP uses `rrf_fusion`, so the
  semantic route influences ranking and not only pool membership; the other
  sources remain as ablation controls.
- The graph candidate route defaults to a bounded sparse TF-IDF kNN graph;
  dense `N x N` requires an explicit `dense_legacy` opt-in.
- Predictions no longer carry gold text; evaluation aligns by id via `--gold`.
- `length_control.unit: words` goes through the configured selector instead of a
  separate greedy path.

### Notes
- Route or feature failure fails the run. No zero-filling, no silent fallback.
- Tests at the v0.6.0 checkpoint: 108 passed. Still open: no baselines exist (Gate 2), so the central
  question — whether any configuration beats Lead — remains unanswered.

## [v0.5.0] - 2026-07-26 (Correctness refactor)
### Fixed
- **ROUGE protocol**: `src/eval/rouge.py` now uses `rougeLsum` for multi-sentence
  summaries (the old single-sequence `rougeL` under-scores extracts), applies the
  same segmentation to prediction and reference, and selects one reference by
  max ROUGE-1 for multi-reference data instead of concatenating references.
- **Similarity matrix corruption**: `src/features/graph.py` copied before
  thresholding; it previously mutated the caller's matrix in place, silently
  truncating the similarities NSGA-II used for coverage/redundancy.
- **Encoder reloading**: `src/models/extractive/encoder_rank.py` caches the
  tokenizer/model instead of calling `from_pretrained` once per document.
- **Unwired hyper-parameters**: `pop_size` / `n_gen` / `seed` are now read from
  config. Every previous NSGA-II run silently used the defaults (100/100)
  regardless of what the YAML declared.
- **Silent fallback removed**: `optimizer_dispatch.py` raises instead of quietly
  running greedy when NSGA-II is unavailable or the method is unknown.

### Added
- `src/eval/oracle.py` — greedy extractive oracle reference (not an exact upper bound).
- `scripts/audit/` — versioned diagnostics: local Lead comparison, selection-position
  analysis, per-dataset headroom, PLM load-vs-inference timing.
- `docs/research/` — audit findings, revision plan, target architecture, action plan.
- Guards on `scripts/quick_tune*.ps1` and `run_missing_experiments.ps1`, which tune
  on the test set.

## [v0.4.0] - 2026-01-14
### Added
- **3-Way Fusion Architecture**: Implemented a multi-view fusion pipeline combining Statistical (Base), Semantic (LLM), and Structural (Graph) scores.
- **Stage 1 Graph**: Added `src.features.graph` module implementing TextRank (PageRank) algorithm.
- **NSGA-II Integration**: Upgraded Stage 1 Base optimizer from `greedy` to `nsga2` for better candidate selection.
- ~~**Multi-News Benchmark**: Achieved **ROUGE-1: 44.32** on Multi-News, surpassing the HiMAP benchmark (44.17).~~
  > 🔴 **RETRACTED.** Three problems: (1) 44.32 does not correspond to any surviving
  > artifact — the runs in `runs/` give 43.52 (ExpB) and 43.37 (full benchmark);
  > (2) the configuration was selected using the **test set**, so the number is a
  > test-tuned artifact, not a valid result; (3) "surpassing HiMAP" compares against
  > a number from another paper computed with a different evaluator and preprocessing.
  > A local, ID-matched Lead baseline under the same evaluator scores
  > 0.4331 / 0.1453 / 0.3901 versus the system's 0.4352 / 0.1405 / 0.3880 —
  > i.e. the system does **not** beat Lead on ROUGE-2 or ROUGE-Lsum.
  > (2026-07-30, PR #9: the two ROUGE-Lsum values are stale — see the
  > segmentation note at the top of this file. The ROUGE-1 and ROUGE-2
  > figures, and therefore the conclusion itself, are unaffected.)
- ~~**Final Configs**: Standardized best-performing configurations in `configs/final/`.~~
  > That directory no longer exists; the current settings are in `configs/` (see its README).

### Changed
- **Pipeline Update**: `scripts/build_union_stage2.py` now supports 3 inputs: `--base_pred`, `--bert_pred`, and `--graph_pred`.
  > Note: that script now lives in `scripts/_archive/` and is excluded from version control.
- **Optimization**: Tuned `max_tokens` to 245 and `lambda_coverage` to 2.5 for Multi-News dataset.
  > Note: this tuning was performed on the test set (P0-01). `max_tokens` also counts
  > whitespace words, not model tokens.

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
