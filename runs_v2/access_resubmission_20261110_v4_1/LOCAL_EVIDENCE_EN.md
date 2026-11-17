# Reviewer-requested local evidence — execution v4.1

This document reports completed local work under the 11 November 2026 v4.1 protocol. It is an evidence report, not a complete resubmission. The original submitted manuscript, predictions, scores and final configurations have not been changed. New candidate-recall measurements are retrospective, reviewer-requested development diagnostics; their definitions and size-stratum membership were saved before joining greedy references. Original ROUGE outcomes were already known. No new significance tests or model tuning were performed.

## Candidate coverage (R2.3)

All 681 GovReport and 3,935 Multi-News development IDs were joined one-to-one across C01 and four controls. Candidate indices, source sentences, selected-set inclusion and summary reconstruction passed for all 23,080 prediction rows. C01/A06 differ only in the substantive reservation setting (20 versus 0); study metadata is excluded from this configuration comparison. A01/A02 change route availability, and A03 is the lexical-only top-80 setting. They are not pure quota interventions.

For each metric-specific greedy reference set G, candidate pool C and selected set S, pool recall is |G∩C|/|G|, selected recall is |G∩S|/|G|, and conditional retention is |G∩S|/|G∩C|. These measure matching greedy-selected sentence IDs, not the proportion of all important facts covered. Greedy references are neither exact optima nor official dataset oracles. Ratio denominators of zero are NA. The six empty Multi-News ROUGE-2 reference sets are retained in the population but excluded from the corresponding ratios.

The table shows all 30 prespecified condition/target combinations. Recall values are document-macro percentages. Exact unrounded values, micro values, numerators, denominators, size/word distributions and downstream descriptions are in `coverage/recall_conditions.tsv`.

| Dataset | Greedy target | Condition | Mean pool size | Pool recall (%) | Selected recall (%) | Valid reference N |
|---|---|---|---:|---:|---:|---:|
| govreport | rouge1 | C01 | 78.42 | 58.853 | 28.090 | 681 |
| govreport | rouge1 | A01_no_semantic | 71.37 | 53.006 | 25.693 | 681 |
| govreport | rouge1 | A02_no_graph | 70.65 | 52.734 | 23.059 | 681 |
| govreport | rouge1 | A03_lexical_only_capacity_80 | 79.52 | 50.576 | 13.986 | 681 |
| govreport | rouge1 | A06_no_route_reservation | 78.42 | 60.151 | 28.092 | 681 |
| govreport | rouge2 | C01 | 78.42 | 50.641 | 18.801 | 681 |
| govreport | rouge2 | A01_no_semantic | 71.37 | 42.089 | 15.158 | 681 |
| govreport | rouge2 | A02_no_graph | 70.65 | 42.095 | 16.471 | 681 |
| govreport | rouge2 | A03_lexical_only_capacity_80 | 79.52 | 34.842 | 5.749 | 681 |
| govreport | rouge2 | A06_no_route_reservation | 78.42 | 52.251 | 18.819 | 681 |
| govreport | rougeLsum | C01 | 78.42 | 56.755 | 26.206 | 681 |
| govreport | rougeLsum | A01_no_semantic | 71.37 | 50.766 | 23.687 | 681 |
| govreport | rougeLsum | A02_no_graph | 70.65 | 50.054 | 21.587 | 681 |
| govreport | rougeLsum | A03_lexical_only_capacity_80 | 79.52 | 47.270 | 12.312 | 681 |
| govreport | rougeLsum | A06_no_route_reservation | 78.42 | 58.095 | 26.223 | 681 |
| multinews | rouge1 | C01 | 51.52 | 87.224 | 30.792 | 3935 |
| multinews | rouge1 | A01_no_semantic | 47.36 | 84.102 | 29.694 | 3935 |
| multinews | rouge1 | A02_no_graph | 46.44 | 82.712 | 29.805 | 3935 |
| multinews | rouge1 | A03_lexical_only_capacity_80 | 56.39 | 90.804 | 31.194 | 3935 |
| multinews | rouge1 | A06_no_route_reservation | 51.52 | 87.248 | 30.837 | 3935 |
| multinews | rouge2 | C01 | 51.52 | 85.334 | 33.541 | 3929 |
| multinews | rouge2 | A01_no_semantic | 47.36 | 81.377 | 32.330 | 3929 |
| multinews | rouge2 | A02_no_graph | 46.44 | 79.545 | 31.741 | 3929 |
| multinews | rouge2 | A03_lexical_only_capacity_80 | 56.39 | 88.243 | 29.471 | 3929 |
| multinews | rouge2 | A06_no_route_reservation | 51.52 | 85.347 | 33.566 | 3929 |
| multinews | rougeLsum | C01 | 51.52 | 85.884 | 31.960 | 3935 |
| multinews | rougeLsum | A01_no_semantic | 47.36 | 82.602 | 30.666 | 3935 |
| multinews | rougeLsum | A02_no_graph | 46.44 | 80.897 | 30.513 | 3935 |
| multinews | rougeLsum | A03_lexical_only_capacity_80 | 56.39 | 89.640 | 31.043 | 3935 |
| multinews | rougeLsum | A06_no_route_reservation | 51.52 | 85.893 | 31.974 | 3935 |

### All 24 full-development contrasts

Direction is C01 minus control; units below are percentage points of pool recall, not ROUGE points. Values are paired mean differences. No p-values are assigned.

| Dataset | Greedy target | Control | Mean difference (pp) | Positive / zero / negative documents |
|---|---|---|---:|---|
| govreport | rouge1 | A01_no_semantic | +5.847 | 418 / 181 / 82 |
| govreport | rouge1 | A02_no_graph | +6.119 | 421 / 195 / 65 |
| govreport | rouge1 | A03_lexical_only_capacity_80 | +8.277 | 424 / 126 / 131 |
| govreport | rouge1 | A06_no_route_reservation | -1.297 | 59 / 445 / 177 |
| govreport | rouge2 | A01_no_semantic | +8.552 | 513 / 113 / 55 |
| govreport | rouge2 | A02_no_graph | +8.547 | 504 / 134 / 43 |
| govreport | rouge2 | A03_lexical_only_capacity_80 | +15.800 | 545 / 63 / 73 |
| govreport | rouge2 | A06_no_route_reservation | -1.609 | 53 / 408 / 220 |
| govreport | rougeLsum | A01_no_semantic | +5.989 | 428 / 169 / 84 |
| govreport | rougeLsum | A02_no_graph | +6.701 | 459 / 167 / 55 |
| govreport | rougeLsum | A03_lexical_only_capacity_80 | +9.485 | 458 / 112 / 111 |
| govreport | rougeLsum | A06_no_route_reservation | -1.341 | 56 / 437 / 188 |
| multinews | rouge1 | A01_no_semantic | +3.121 | 891 / 3001 / 43 |
| multinews | rouge1 | A02_no_graph | +4.511 | 1156 / 2742 / 37 |
| multinews | rouge1 | A03_lexical_only_capacity_80 | -3.581 | 393 / 2376 / 1166 |
| multinews | rouge1 | A06_no_route_reservation | -0.024 | 41 / 3840 / 54 |
| multinews | rouge2 | A01_no_semantic | +3.956 | 1043 / 2826 / 60 |
| multinews | rouge2 | A02_no_graph | +5.788 | 1318 / 2573 / 38 |
| multinews | rouge2 | A03_lexical_only_capacity_80 | -2.909 | 470 / 2386 / 1073 |
| multinews | rouge2 | A06_no_route_reservation | -0.014 | 61 / 3804 / 64 |
| multinews | rougeLsum | A01_no_semantic | +3.282 | 974 / 2913 / 48 |
| multinews | rougeLsum | A02_no_graph | +4.987 | 1279 / 2623 / 33 |
| multinews | rougeLsum | A03_lexical_only_capacity_80 | -3.755 | 435 / 2285 / 1215 |
| multinews | rougeLsum | A06_no_route_reservation | -0.008 | 51 / 3830 / 54 |

### Pool-size interpretation

All 144 prespecified size-subset/target/control rows, including empty groups, are in `coverage/recall_contrasts_by_size.tsv`; joint-valid and joint-NA counts for each paired metric are explicit in `coverage/paired_denominators.tsv`. Reference-blind membership was saved before scoring. Equal-size children (both 80, equal positive below 80, both empty) are nested, so their Ns must not be added to the parent N.

C01 has greater full-development pool recall than A01 and A02 on both datasets. Its pools are also larger on average, so those contrasts describe the operating configurations rather than isolating a pure route effect at fixed size. Against A03 on GovReport, C01 has greater recall across all three targets despite a slightly smaller mean pool. Against A03 on Multi-News, C01 has lower full-development recall and a smaller mean pool. The equal-size results do not replace these full-population observations.

For the ROUGE-Lsum greedy target, all prespecified A03 size groups are shown here for interpretation. All three targets and all controls remain available in the complete files.

| Dataset | Size subset | N | Mean C01 size | Mean A03 size | C01−A03 pool recall (pp) |
|---|---|---:|---:|---:|---:|
| govreport | C01_smaller | 104 | 70.35 | 77.54 | -0.904 |
| govreport | equal_size | 577 | 79.88 | 79.88 | +11.357 |
| govreport | C01_larger | 0 | NA | NA | NA |
| govreport | both_at_80 | 575 | 80.00 | 80.00 | +11.397 |
| govreport | equal_positive_below_80 | 2 | 44.00 | 44.00 | +0.000 |
| govreport | both_empty | 0 | NA | NA | NA |
| multinews | C01_smaller | 2168 | 60.27 | 69.11 | -7.700 |
| multinews | equal_size | 1767 | 40.79 | 40.79 | +1.085 |
| multinews | C01_larger | 0 | NA | NA | NA |
| multinews | both_at_80 | 420 | 80.00 | 80.00 | +4.565 |
| multinews | equal_positive_below_80 | 1347 | 28.56 | 28.56 | +0.000 |
| multinews | both_empty | 0 | NA | NA | NA |

Equal-size comparisons are conditional descriptions, not randomized or causal controls. In the 1,347 Multi-News A03 pairs with equal positive size below 80, pools are identical and differences are necessarily zero. Those observations do not demonstrate quality equivalence. C01/A06 have identical pool sizes in all development pairs; C01's pool recall is slightly lower across targets, especially on GovReport. This supports retaining reservation as a structural route-retention choice, not asserting an independent quality or greedy-reference-coverage improvement. Final selected recall is much closer between C01 and A06; full downstream values are retained.

Historical cap-120 comparisons yielded little change in development ROUGE under their own anchors. These results do not prove that entry coverage is irrelevant or that selection is the sole bottleneck. Candidate admission and subsequent selection both restrict the final matching sentence set.

## Historical design evidence and RRF (R2.3/R2.4)

`historical/development_conditions.tsv` retains all 24 D1 cheap-multiroute conditions and all nine D3a full-development finalist rows, including the unsuccessful historical GovReport G11 condition. Source hashes, configuration/evidence checks and 111 byte-preserved metadata copies are recorded in `historical/source_manifest.json` and `historical/source_snapshot.zip`. D1 and D3a have different stage anchors and selectors; neither is an exhaustive sensitivity study of the final C01 configuration. Development scores use the internal multisentence ROUGE-Lsum protocol, not the final test Perl ROUGE-L protocol.

D1 G02–G05 change top-K, reservation, cap or RRF K one at a time under the earlier two-route anchor. Their macro-ROUGE changes are within ±0.001. G06 is a separate salience experiment and falls by 0.002624 on GovReport and 0.004023 on Multi-News; it must not be included in the ±0.001 statement. G06 uses base lexical salience after membership, whereas A05 uses lexical percentile on an identical C01 pool. Neither is random selection.

For F(s)=Σ w_r/(K+rank_r(s)), weights are nonnegative and ranks cover all eligible sentences. Increasing a positive-weight route rank decreases its contribution; replacing route scores by strictly increasing transformations that preserve rankings leaves RRF unchanged. Multiplying all weights by the same positive scalar leaves the fused order and min-max normalized relevance unchanged. These are properties of the specified formulation, not new general RRF theory or a guarantee of summary quality.

The recorded case has 258 eligible sentences. Hypothetical equal-weight ranks (1,258,258) score 0.022683, whereas (20,20,20) score 0.037500 at K=60. Thus, broad consensus can outrank a route specialist, motivating explicit reservation when preserving route representation is desired. These are illustrative rank tuples, not asserted observations from that case. `historical/RRF_numeric_examples.json` also records the actual dataset weight settings. At K=60, the ratio of a route's rank-1 to rank-40 contribution is 100/61≈1.639; weights 0.5 and 2 directly rescale contributions. At K=30 the ratio is 70/31≈2.258. The D1 K=30 results provide empirical context for this rank-contrast change.

Global min-max normalization is over every eligible sentence, before pooling. It preserves fused order but changes the numerical relevance scale relative to the redundancy penalty in MMR. No claim is made that pool relevance universally lies in [0.8,1.0]: the original example's whole pool spans approximately [0.1886,1], while its selected sentences span approximately [0.7956,1].

For three routes reserving at most 20 sentences each and at most 20 guard sentences, 3×20+20=80 is a general sufficient worst-case bound for the mandatory union. It is not a minimum required for every document; overlap lowers the actual requirement, and a single-document GovReport guard contributes at most one, giving a corresponding bound of 61. Top-40 and reservation-20 were inherited fixed design choices, supported here by historical local perturbations, not mathematically optimized quality constants.

## Computational traceability and actual use (R2.6)

All six dataset/ablation groups passed: A04, A05 and A08 each preserve the C01 ordered candidate membership for all 681 GovReport and 3,935 Multi-News development examples (13,848 comparisons). The stored membership hashes, fixed-pool flags and inclusion reasons agree; route features and ranks remain unchanged, while configured weights and selector salience arithmetic match the intended intervention. This establishes actual reuse of provenance for controlled diagnostic comparisons. It does not establish that saving a log itself improves summary quality. A04/A05 provide the central illustration; the already-disclosed A08 result remains an additional diagnostic.

The reconstruction evidence has complementary scopes:

| Level | Population | Verified scope | Outcome |
|---|---|---|---|
| L0 | All 6,594 proposed test rows | Source mapping, selected⊆pool, source-order output, text concatenation, word counts and recorded feasibility | All checked rows pass |
| L1 | All 6,594 proposed test rows, with per-check denominators | Pool RRF arithmetic and exclusive-first reservation; GovReport stored-rank guard and 973 MMR selected-step traces | All applicable checks pass |
| L2 feasibility | 20 development examples per dataset | Frozen-code reproduction and independent downstream pool/selector verification | 40/40 pass |
| L2 formal | 100 test examples per dataset, fixed length-stratified hash sample | Full eligible rank/fusion/min-max/pool/guards; feasible alternatives, tie-breaking, selected IDs and output text | 200/200 pass |

Per-check pass/fail/not-checkable counts remain in `audit/L0_L1_summary.tsv`. Multi-News did not originally save Greedy steps; replayed steps are explicitly labeled reconstructed. Existing Multi-News length-shortfall records are valid audit observations, not omitted failures: L0 verifies consistency of the recorded status rather than requiring every summary to satisfy the minimum length. L1 uses recorded ranks and similarities for its arithmetic; global rank correctness, omitted nominees, normalization extrema and all-alternative argmax checks belong to L2.

L2 reuses exact archived source commits, canonical inputs, resolved configurations, matching installed dependency versions and read-only frozen embedding caches. It compares original selected IDs and summary text exactly, with numerical absolute tolerance 1e−8. A separate implementation verifies downstream rank/fusion/pool/selector decisions. Encoder, lexical and graph feature generation are shared inputs, so this is not a wholly independent reimplementation of the NLP system. Cache hashes and original source hashes are rechecked. No tolerance was relaxed to obtain discrete agreement.

The formal sample was fixed before replay and contains 100 unique GovReport clusters and 98 Multi-News clusters among 100 examples. Dependency components were formed over the full test populations using identical normalized full inputs or shared normalized source documents of at least 100 words, including transitive connections. These cluster annotations must carry into future LLM uncertainty estimates. No samples were replaced based on length, infeasibility, replay or quality outcomes.

### Table 10 order correction

The first five original table rows are the first five selected sentences in source/output order, not the first five MMR decisions. `audit/table10_order_verification.json` records both complete orders. Label these rows “Output 1–5” and explicitly describe source order. The selected set is unchanged. Human understanding speed or usability has not been measured; the verified claim is computational traceability/auditability with records plus frozen inputs and implementation.

## Integration status

This evidence can support R2.3, R2.4 and R2.6. R2.1/R2.2 still require the source-verified novelty comparison and literature integration. R2.5 must retain the GovReport-scoped modest improvement and measured cost comparison, alongside the nonsignificant Multi-News mean and its higher cost than PacSum. R2.7's four-dimensional LLM evaluation has not run. Model/access decisions, pilot/control validation, author QC, final manuscript changes, page-specific responses, clean/highlighted PDFs and submission packaging remain open. No acceptance probability or publication guarantee follows from passing these checks.
