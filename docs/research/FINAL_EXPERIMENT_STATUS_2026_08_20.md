# Final experiment status — 2026-08-20

## Paper-level decision

The frozen experimental program is complete for both reported datasets. No further
quality tuning is permitted.

- **GovReport is the primary confirmatory benchmark.** Proposed ranks first and
  significantly beats the preregistered SBERT+MMR comparator in official macro ROUGE.
- **Multi-News is the secondary multi-document benchmark.** Proposed ranks first by
  official macro, but is statistically tied with the two strongest PacSum variants.
  It significantly improves R-1/R-L over PacSum-TFIDF while significantly losing R-2.
- The defensible contribution is a no-task-training, provenance-aware multi-route
  extractive framework with dataset-specific route/selector profiles and explicit
  quality-cost evidence. It is not a universal SOTA or an NSGA-II superiority paper.

## Final official test results

| Dataset | Proposed R-1 | R-2 | R-L | Macro | Primary comparison |
|---|---:|---:|---:|---:|---|
| GovReport, 973 rows | 0.58374 | 0.24711 | 0.54898 | **0.459943** | vs SBERT+MMR: `+0.003700`, 95% CI `[+0.002008,+0.005420]`, `p=0.000040` |
| Multi-News, 5,621 rows | **0.45011** | 0.14314 | **0.41351** | **0.335587** | vs PacSum-TFIDF: `+0.000091`, 95% CI `[-0.001542,+0.001730]`, `p=0.907111` |

Multi-News component differences versus PacSum-TFIDF are R-1 `+0.002389`
(Holm-3 `p=0.003520`), R-2 `-0.006758` (Holm-3 `p=0.000060`) and R-L
`+0.004641` (Holm-3 `p=0.000060`). Proposed also significantly beats Lead,
Random, TextRank, LexRank, SBERT centroid and SBERT+MMR in macro after Holm-32;
its macro difference versus PacSum-SBERT is not significant.

Authoritative result documents:

- `GOVREPORT_FINAL_TEST_RESULTS.md`
- `MULTINEWS_FINAL_TEST_RESULTS.md`

## Final Multi-News route/provenance ablation

The ablation uses all 3,935 frozen dev rows, 100,000 paired bootstrap resamples and
Holm correction over 20 endpoints. All five preregistered macro claims pass.

| Full minus ablation | Macro delta | 95% CI | Holm-20 p |
|---|---:|---:|---:|
| No semantic | +0.001917 | [+0.001149,+0.002685] | 0.000400 |
| No graph | +0.004385 | [+0.003466,+0.005311] | 0.000400 |
| Lexical only, capacity 80 | +0.010478 | [+0.009364,+0.011618] | 0.000400 |
| Exact pool, equal-weight RRF | +0.000742 | [+0.000135,+0.001353] | 0.049440 |
| Exact pool, lexical-only selector salience | +0.007139 | [+0.006188,+0.008099] | 0.000400 |

The equal-weight RRF result is positive but marginal after correction and must be
described conservatively. An initially completed run used dependencies that differed
from the frozen anchor; it is preserved and excluded. The final analysis uses exact
anchor dependency parity. Evidence: `runs_v2/multinews_e3_route_provenance_v1/`.

## Final Multi-News cost, memory and scaling evidence

Each value is the median of three fresh subprocess measurements on the frozen
reference-blind 30-document sample. Cold and warm must be reported separately.

| System | Cold wall (s) | Warm/control wall (s) | Cold RSS (MiB) | Warm/control RSS (MiB) |
|---|---:|---:|---:|---:|
| **Proposed** | **47.85** | **9.23** | **594.6** | **398.9** |
| PacSum-TFIDF P08 | 4.84 | 4.84 | 324.8 | 327.5 |
| PacSum-SBERT P03 | 46.05 | 7.27 | 584.3 | 396.0 |
| SBERT+MMR lambda=0.7 | 47.19 | 7.90 | 595.2 | 397.0 |
| SBERT centroid | 47.47 | 7.41 | 595.4 | 396.4 |
| Matched NSGA-II | 91.69 | 50.07 | 599.6 | 400.7 |
| Lead | 4.87 | 4.81 | 324.1 | 325.5 |
| TextRank | 6.39 | 6.26 | 327.9 | 328.7 |
| LexRank | 6.91 | 6.86 | 327.7 | 327.7 |

All 77 completed attempts are selected-index identical across repetitions and
cold/warm states. Proposed has approximately the same cold cost as the SBERT
baselines, a modestly slower warm path, and a far better cost profile than NSGA-II.
Evidence: `runs_v2/multinews_cost_scaling_v1/`.

## What remains before submission

The scientific runs required for the current paper are complete. Remaining work is
paper and artifact work, not more configuration search:

1. Write the IEEE Access manuscript and reviewer-response matrix from the frozen tables.
2. Add qualitative error cases and, if feasible, a separately preregistered human study;
   neither may alter the method or primary automatic results.
3. Complete clean-clone reproduction, lockfile/container, table-generation scripts and
   code/data availability instructions.
4. Verify ICACT citation/DOI, Outstanding Paper Award evidence, similarity report,
   biographies, ORCIDs and IEEE AI-use disclosure.
5. Run final equation-code-config-table consistency and English-language review.

CNN/DailyMail and SciTLDR are not part of this frozen revision matrix. Test scores may
not be used for any post-hoc method change.

## Repository verification

- Full unit suite: **486 passed + 5 subtests passed** on 2026-08-20.
- `python -m compileall -q -f src tests scripts`: passed.
- `git diff --check`: passed before the documentation closeout commit.
- The first sandboxed pytest attempt was excluded because Windows denied pytest temp
  and multiprocessing-pipe access; the same suite passed in the pinned `.venv` with
  an explicit repository-local basetemp.
