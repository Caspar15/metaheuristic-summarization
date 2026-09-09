# Supplementary Material

Provenance-Aware Multi-Route Fusion for Long-Document Extractive Summarization Without Task-Specific Fine-Tuning

Shih-Wei Yang, Bo-Yu Chen, Shao-Chi Kuan, Sy-Yen Kuo, and Jiann-Liang Chen

PAMR-ES | IEEE Access submission companion | Version v1.0.0-ieee-access

## S1. Scope, protocols and final configuration

This supplement reports archived experimental evidence and details needed to inspect the final method. It adds no new model training, test prediction, evaluator run or configuration selection. Prespecified development ablations and post-test development diagnostics are reported as separate analysis families. Neither group replaces the frozen official test result.

Official quality tables use Stanza tokenize,mwt followed by Perl ROUGE-1.5.5. Development selector/ablation tables use the internal multi-sentence ROUGE-Lsum protocol. Scores are on the 0-1 scale. Mean denotes the arithmetic mean of the three stated ROUGE metrics. A corpus-score difference need not equal a mean paired difference because the archived official corpus outputs and per-document scores have different rounding/aggregation paths.

### Table S1. Final method settings

| Setting | GovReport | Multi-News |
| --- | --- | --- |
| Official test / frozen development rows | 973 / 681 | 5,621 / 3,935 |
| Routes | Lexical, semantic, sparse graph | Lexical, semantic, sparse graph |
| Per-route proposal limit / reservation | 40 / 20 | 40 / 20 |
| Candidate-pool cap / RRF constant | 80 / 60 | 80 / 60 |
| RRF weights: lexical, semantic, graph | 0.5, 1, 1 | 1, 1, 2 |
| Final selector | TF-IDF MMR, lambda=0.7 | Greedy-TFIDF |
| Requested summary words | 500-650 | 200-250 |
| Semantic encoder / input limit | all-MiniLM-L6-v2 / 256 tokens | Same |
| Task-specific fine-tuning | None | None |

The semantic encoder is pinned to revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf. Route weights and selectors were selected using development data and fixed before test execution. This is not a claim of zero configuration tuning. Full settings, policy identities and environment versions are in the two configs/final YAML files and their execution freezes.

## S2. Official test results and paired comparisons

### Table S2. GovReport official quality (973 rows)

| System | R-1 | R-2 | R-L | Mean |
| --- | --- | --- | --- | --- |
| PAMR-ES | 0.583740 | 0.247110 | 0.548980 | 0.459943 |
| SBERT+MMR | 0.580400 | 0.240920 | 0.547300 | 0.456207 |
| SBERT centroid | 0.578070 | 0.241060 | 0.545180 | 0.454770 |
| LexRank | 0.576220 | 0.238890 | 0.545100 | 0.453403 |
| PacSum-SBERT | 0.574240 | 0.241990 | 0.543790 | 0.453340 |
| TextRank | 0.557000 | 0.221630 | 0.522110 | 0.433580 |
| PacSum-TFIDF | 0.545620 | 0.212220 | 0.515490 | 0.424443 |
| Random (10 seeds) | 0.544267 | 0.182574 | 0.512904 | 0.413248 |
| Lead | 0.527460 | 0.203320 | 0.501780 | 0.410853 |

### Table S3. Multi-News official quality (5,621 rows)

| System | R-1 | R-2 | R-L | Mean |
| --- | --- | --- | --- | --- |
| PAMR-ES | 0.450110 | 0.143140 | 0.413510 | 0.335587 |
| PacSum-TFIDF | 0.447730 | 0.149900 | 0.408880 | 0.335503 |
| PacSum-SBERT | 0.445900 | 0.150020 | 0.406960 | 0.334293 |
| Lead | 0.439010 | 0.147290 | 0.400770 | 0.329023 |
| SBERT+MMR | 0.441470 | 0.139770 | 0.400780 | 0.327340 |
| LexRank | 0.438030 | 0.138840 | 0.398630 | 0.325167 |
| SBERT centroid | 0.432770 | 0.136140 | 0.392360 | 0.320423 |
| Random (10 seeds) | 0.423227 | 0.123884 | 0.386661 | 0.311257 |
| TextRank | 0.419470 | 0.129910 | 0.377920 | 0.309100 |

The Random row averages the ten fixed seeds; it is not the best seed. All system outputs are retained for the primary all-row analysis, including length shortfalls.

## S2.1 Prespecified paired comparisons

### Table S4. GovReport: PAMR-ES minus SBERT+MMR

| Metric | Difference | 95% paired CI | Raw p | Reported p |
| --- | --- | --- | --- | --- |
| R-1 | 0.003315 | [+0.001642, +0.004997] | 0.000140 | 0.000280 |
| R-2 | 0.006150 | [+0.004116, +0.008202] | 0.000020 | 0.000060 |
| R-L | 0.001636 | [-0.000072, +0.003366] | 0.060279 | 0.060279 |
| Mean | 0.003700 | [+0.002008, +0.005420] | 0.000040 | 0.000040 |

### Table S5. Multi-News: PAMR-ES minus PacSum-TFIDF

| Metric | Difference | 95% paired CI | Raw p | Reported p |
| --- | --- | --- | --- | --- |
| R-1 | 0.002389 | [+0.000776, +0.004000] | 0.003520 | 0.003520 |
| R-2 | -0.006758 | [-0.008667, -0.004863] | 0.000020 | 0.000060 |
| R-L | 0.004641 | [+0.003030, +0.006233] | 0.000020 | 0.000060 |
| Mean | 0.000091 | [-0.001542, +0.001730] | 0.907111 | 0.907111 |

Each comparison uses 100,000 paired bootstrap resamples. Component p-values use Holm-3; the prespecified Mean endpoint reports its two-sided p-value. GovReport supports the scoped Mean advantage; its R-L component is not significant. Multi-News does not show a significant overall Mean advantage, with higher R-1/R-L and lower R-2. Full exploratory baseline comparisons and their Holm-32 family remain in the linked final analyses.

## S3. Output length and feasibility

Values are means with population standard deviations. Random pools document-seed observations over ten fixed seeds; the remaining systems pool documents. Infeasible means the requested length contract was not satisfied, not that the output was omitted from evaluation.

### Table S6. GovReport output statistics

| System | Words: mean +/- SD | Sentences: mean +/- SD | Shortfalls |
| --- | --- | --- | --- |
| PAMR-ES | 647.08 +/- 8.61 | 16.73 +/- 3.66 | 0 |
| Lead | 633.77 +/- 16.61 | 25.82 +/- 4.45 | 0 |
| Random (10 seeds) | 649.30 +/- 8.21 | 28.37 +/- 5.09 | 0 |
| TextRank | 649.43 +/- 8.18 | 16.04 +/- 3.42 | 0 |
| LexRank | 649.36 +/- 8.19 | 21.80 +/- 4.40 | 0 |
| PacSum-TFIDF | 649.24 +/- 8.26 | 20.93 +/- 4.90 | 0 |
| PacSum-SBERT | 649.35 +/- 8.20 | 25.70 +/- 10.92 | 0 |
| SBERT centroid | 649.44 +/- 8.18 | 21.79 +/- 6.04 | 0 |
| SBERT+MMR | 649.43 +/- 8.17 | 21.62 +/- 5.41 | 0 |

### Table S7. Multi-News output statistics

| System | Words: mean +/- SD | Sentences: mean +/- SD | Shortfalls |
| --- | --- | --- | --- |
| PAMR-ES | 244.10 +/- 15.35 | 15.05 +/- 4.21 | 12 |
| Lead | 233.61 +/- 19.34 | 10.70 +/- 3.24 | 0 |
| Random (10 seeds) | 246.75 +/- 14.89 | 13.14 +/- 3.23 | 0 |
| TextRank | 247.25 +/- 14.74 | 8.02 +/- 2.60 | 0 |
| LexRank | 246.93 +/- 14.97 | 10.97 +/- 3.01 | 0 |
| PacSum-TFIDF | 245.92 +/- 15.12 | 12.04 +/- 3.94 | 0 |
| PacSum-SBERT | 246.90 +/- 14.93 | 11.36 +/- 3.15 | 0 |
| SBERT centroid | 247.16 +/- 14.92 | 9.94 +/- 3.89 | 0 |
| SBERT+MMR | 247.16 +/- 14.92 | 10.11 +/- 3.05 | 7 |

PAMR-ES does not produce systematically longer summaries than all strong baselines. Its comparative R-1/R-L performance cannot be attributed simply to filling more of the permitted word budget.

## S4. Prespecified ablations: GovReport

Frozen development only (681 rows); internal ROUGE-Lsum protocol. Each difference is full PAMR-ES minus the stated ablation. The five variants and four endpoints form a dataset-specific Holm-20 family with 100,000 paired bootstrap resamples.

### Table S8. Mean ROUGE by variant

| Variant | Mean |
| --- | --- |
| Full | 0.457404 |
| A01: no semantic route | 0.442022 |
| A02: no graph route | 0.448370 |
| A03: lexical only, capacity 80 | 0.359463 |
| A04: exact pool, equal-weight RRF | 0.449608 |
| A05: exact pool, lexical salience | 0.359569 |

### Table S9. Complete prespecified paired endpoints

| Variant | Metric | Difference | 95% CI | Raw p | Holm-20 p |
| --- | --- | --- | --- | --- | --- |
| A01 | R-1 | 0.014342 | [+0.012053, +0.016649] | 0.000020 | 0.000400 |
| A01 | R-2 | 0.016857 | [+0.014177, +0.019571] | 0.000020 | 0.000400 |
| A01 | R-Lsum | 0.014947 | [+0.012592, +0.017341] | 0.000020 | 0.000400 |
| A01 | Mean | 0.015382 | [+0.013057, +0.017719] | 0.000020 | 0.000400 |
| A02 | R-1 | 0.005850 | [+0.003858, +0.007848] | 0.000020 | 0.000400 |
| A02 | R-2 | 0.014724 | [+0.012203, +0.017270] | 0.000020 | 0.000400 |
| A02 | R-Lsum | 0.006527 | [+0.004439, +0.008610] | 0.000020 | 0.000400 |
| A02 | Mean | 0.009034 | [+0.006964, +0.011131] | 0.000020 | 0.000400 |
| A03 | R-1 | 0.093965 | [+0.089431, +0.098455] | 0.000020 | 0.000400 |
| A03 | R-2 | 0.102890 | [+0.098023, +0.107710] | 0.000020 | 0.000400 |
| A03 | R-Lsum | 0.096966 | [+0.092415, +0.101518] | 0.000020 | 0.000400 |
| A03 | Mean | 0.097940 | [+0.093473, +0.102404] | 0.000020 | 0.000400 |
| A04 | R-1 | 0.005853 | [+0.004263, +0.007451] | 0.000020 | 0.000400 |
| A04 | R-2 | 0.009981 | [+0.008102, +0.011840] | 0.000020 | 0.000400 |
| A04 | R-Lsum | 0.007553 | [+0.005892, +0.009220] | 0.000020 | 0.000400 |
| A04 | Mean | 0.007796 | [+0.006154, +0.009420] | 0.000020 | 0.000400 |
| A05 | R-1 | 0.092820 | [+0.088332, +0.097264] | 0.000020 | 0.000400 |
| A05 | R-2 | 0.105290 | [+0.100279, +0.110208] | 0.000020 | 0.000400 |
| A05 | R-Lsum | 0.095393 | [+0.090858, +0.099923] | 0.000020 | 0.000400 |
| A05 | Mean | 0.097834 | [+0.093306, +0.102323] | 0.000020 | 0.000400 |

A01/A02 remove one candidate-generation route. A03 retains lexical-only candidates at the matched capacity. A04 and A05 fix full-system candidate membership, changing only route weighting or selector salience. These contrasts support the studied semantic/graph and fusion-ranking effects within the frozen configuration; they do not establish that every route or reservation mechanism independently improves quality.

## S4. Prespecified ablations: Multi-News

Frozen development only (3,935 rows); internal ROUGE-Lsum protocol. Each difference is full PAMR-ES minus the stated ablation. The five variants and four endpoints form a dataset-specific Holm-20 family with 100,000 paired bootstrap resamples.

### Table S10. Mean ROUGE by variant

| Variant | Mean |
| --- | --- |
| Full | 0.330417 |
| A01: no semantic route | 0.328500 |
| A02: no graph route | 0.326033 |
| A03: lexical only, capacity 80 | 0.319939 |
| A04: exact pool, equal-weight RRF | 0.329675 |
| A05: exact pool, lexical salience | 0.323278 |

### Table S11. Complete prespecified paired endpoints

| Variant | Metric | Difference | 95% CI | Raw p | Holm-20 p |
| --- | --- | --- | --- | --- | --- |
| A01 | R-1 | 0.001833 | [+0.001043, +0.002622] | 0.000020 | 0.000400 |
| A01 | R-2 | 0.001321 | [+0.000421, +0.002226] | 0.004200 | 0.016800 |
| A01 | R-Lsum | 0.002598 | [+0.001812, +0.003382] | 0.000020 | 0.000400 |
| A01 | Mean | 0.001917 | [+0.001149, +0.002685] | 0.000020 | 0.000400 |
| A02 | R-1 | 0.004296 | [+0.003373, +0.005219] | 0.000020 | 0.000400 |
| A02 | R-2 | 0.003253 | [+0.002138, +0.004373] | 0.000020 | 0.000400 |
| A02 | R-Lsum | 0.005605 | [+0.004657, +0.006557] | 0.000020 | 0.000400 |
| A02 | Mean | 0.004385 | [+0.003466, +0.005311] | 0.000020 | 0.000400 |
| A03 | R-1 | 0.010988 | [+0.009831, +0.012142] | 0.000020 | 0.000400 |
| A03 | R-2 | 0.006590 | [+0.005307, +0.007882] | 0.000020 | 0.000400 |
| A03 | R-Lsum | 0.013857 | [+0.012684, +0.015038] | 0.000020 | 0.000400 |
| A03 | Mean | 0.010478 | [+0.009364, +0.011618] | 0.000020 | 0.000400 |
| A04 | R-1 | 0.000593 | [-0.000020, +0.001209] | 0.057559 | 0.115119 |
| A04 | R-2 | 0.000477 | [-0.000252, +0.001204] | 0.197698 | 0.197698 |
| A04 | R-Lsum | 0.001157 | [+0.000530, +0.001781] | 0.000240 | 0.001200 |
| A04 | Mean | 0.000742 | [+0.000135, +0.001353] | 0.016480 | 0.049440 |
| A05 | R-1 | 0.007293 | [+0.006334, +0.008261] | 0.000020 | 0.000400 |
| A05 | R-2 | 0.004986 | [+0.003859, +0.006109] | 0.000020 | 0.000400 |
| A05 | R-Lsum | 0.009138 | [+0.008153, +0.010131] | 0.000020 | 0.000400 |
| A05 | Mean | 0.007139 | [+0.006188, +0.008099] | 0.000020 | 0.000400 |

A01/A02 remove one candidate-generation route. A03 retains lexical-only candidates at the matched capacity. A04 and A05 fix full-system candidate membership, changing only route weighting or selector salience. These contrasts support the studied semantic/graph and fusion-ranking effects within the frozen configuration; they do not establish that every route or reservation mechanism independently improves quality.

## S5. Post-test development mechanism diagnostics

The following diagnostics were registered after final testing, before scoring their respective new variants. They use only frozen development membership and do not access dev-test/test for method selection. They are separate from the prespecified E3 family and did not alter the final system.

### Table S12. No route reservation: full minus variant

| Dataset | Metric | Difference | 95% CI | Raw p | Holm-4 p |
| --- | --- | --- | --- | --- | --- |
| GovReport | R-1 | -0.000023 | [-0.000059, +0.000011] | 0.196038 | 0.784152 |
| GovReport | R-2 | 0.000013 | [-0.000027, +0.000057] | 0.565694 | 1.000000 |
| GovReport | R-Lsum | 0.000009 | [-0.000027, +0.000043] | 0.617374 | 1.000000 |
| GovReport | Mean | -0.000001 | [-0.000029, +0.000028] | 0.968070 | 1.000000 |
| Multi-News | R-1 | -0.000286 | [-0.000503, -0.000065] | 0.011200 | 0.044800 |
| Multi-News | R-2 | -0.000158 | [-0.000390, +0.000076] | 0.184758 | 0.184758 |
| Multi-News | R-Lsum | -0.000234 | [-0.000453, -0.000015] | 0.036660 | 0.099539 |
| Multi-News | Mean | -0.000226 | [-0.000432, -0.000019] | 0.033180 | 0.099539 |

### Table S13. No lexical candidate route: full minus variant

| Dataset | Metric | Difference | 95% CI | Raw p | Holm-4 p |
| --- | --- | --- | --- | --- | --- |
| GovReport | R-1 | -0.002567 | [-0.004054, -0.001092] | 0.001040 | 0.001040 |
| GovReport | R-2 | -0.006138 | [-0.008038, -0.004231] | 0.000020 | 0.000080 |
| GovReport | R-Lsum | -0.004700 | [-0.006280, -0.003129] | 0.000020 | 0.000080 |
| GovReport | Mean | -0.004468 | [-0.006043, -0.002902] | 0.000020 | 0.000080 |
| Multi-News | R-1 | -0.004598 | [-0.005397, -0.003798] | 0.000020 | 0.000080 |
| Multi-News | R-2 | -0.003271 | [-0.004216, -0.002333] | 0.000020 | 0.000080 |
| Multi-News | R-Lsum | -0.005281 | [-0.006090, -0.004469] | 0.000020 | 0.000080 |
| Multi-News | Mean | -0.004383 | [-0.005171, -0.003595] | 0.000020 | 0.000080 |

Removing reservation changes min_per_route from 20 to zero with other settings held fixed. No multiplicity-corrected quality gain for retaining reservation is established. It is therefore described as a source-balancing and audit mechanism. Removing the lexical candidate route improves development Mean ROUGE in both datasets. TF-IDF still supplies selector features, so this is not removal of all lexical information.

## S5.1 Exact-pool zero lexical ranking weight

Candidate membership is fixed to the full system for every document. The lexical RRF weight alone is set to zero. Contrast A is full minus exact-pool zero-weight; contrast B is exact-pool zero-weight minus no lexical candidate route. The 16 endpoints across both datasets use global Holm-16 correction and 100,000 paired bootstrap resamples.

### Table S14. GovReport exact-pool contrasts

| Contrast | Metric | Difference | 95% CI | Raw p | Holm-16 p |
| --- | --- | --- | --- | --- | --- |
| A | R-1 | -0.002594 | [-0.004074, -0.001115] | 0.000480 | 0.002880 |
| A | R-2 | -0.006110 | [-0.008000, -0.004207] | 0.000020 | 0.000320 |
| A | R-Lsum | -0.004681 | [-0.006254, -0.003114] | 0.000020 | 0.000320 |
| A | Mean | -0.004462 | [-0.006049, -0.002903] | 0.000020 | 0.000320 |
| B | R-1 | 0.000027 | [-0.000087, +0.000150] | 0.667273 | 1.000000 |
| B | R-2 | -0.000027 | [-0.000110, +0.000050] | 0.500455 | 1.000000 |
| B | R-Lsum | -0.000019 | [-0.000141, +0.000109] | 0.748733 | 1.000000 |
| B | Mean | -0.000006 | [-0.000103, +0.000090] | 0.887111 | 1.000000 |

### Table S15. Multi-News exact-pool contrasts

| Contrast | Metric | Difference | 95% CI | Raw p | Holm-16 p |
| --- | --- | --- | --- | --- | --- |
| A | R-1 | -0.001498 | [-0.002166, -0.000826] | 0.000020 | 0.000320 |
| A | R-2 | -0.001134 | [-0.001962, -0.000309] | 0.007540 | 0.037700 |
| A | R-Lsum | -0.002456 | [-0.003138, -0.001779] | 0.000020 | 0.000320 |
| A | Mean | -0.001696 | [-0.002369, -0.001025] | 0.000020 | 0.000320 |
| B | R-1 | -0.003099 | [-0.003649, -0.002544] | 0.000020 | 0.000320 |
| B | R-2 | -0.002137 | [-0.002715, -0.001563] | 0.000020 | 0.000320 |
| B | R-Lsum | -0.002825 | [-0.003375, -0.002277] | 0.000020 | 0.000320 |
| B | Mean | -0.002687 | [-0.003211, -0.002164] | 0.000020 | 0.000320 |

In GovReport, the negative lexical effect is associated primarily with its ranking vote; retaining lexical-proposed pool members has no detectable additional effect in this contrast. Multi-News shows effects of both the ranking vote and candidate membership. These are configuration-specific mechanism diagnostics, not evidence for a newly selected final method.

## S6. Matched-row sensitivity and configuration budget

The 12 PAMR-ES shortfalls in Multi-News arise from whole-sentence packing under the 250-word ceiling. Their eligible source capacity exceeds 200 words; they are not short source documents. SBERT+MMR has seven shortfalls, six overlapping PAMR-ES. Actual outputs remain in the primary 5,621-row evaluation.

The sensitivity analysis removes the same 12 PAMR-ES-shortfall IDs from both PAMR-ES and PacSum-TFIDF, leaving 5,609 paired rows. Only existing per-document scores are reaggregated; summaries are not regenerated.

### Table S16. Matched-row PAMR-ES minus PacSum-TFIDF

| Metric | Difference | 95% CI | Raw p | Reported p |
| --- | --- | --- | --- | --- |
| R-1 | 0.002319 | [+0.000713, +0.003924] | 0.004980 | 0.004980 |
| R-2 | -0.006815 | [-0.008725, -0.004921] | 0.000020 | 0.000060 |
| R-L | 0.004593 | [+0.002999, +0.006194] | 0.000020 | 0.000060 |
| Mean | 0.000032 | [-0.001604, +0.001666] | 0.970830 | 0.970830 |

The component endpoints use Holm-3 and Mean reports its two-sided p-value. The conclusion is unchanged: no significant overall Mean difference, higher R-1/R-L, and lower R-2. Excluded IDs: test_62, test_1329, test_2382, test_2498, test_2546, test_2727, test_2732, test_3411, test_4331, test_4386, test_4514, test_5144.

### Table S17. Configuration comparisons

| Method / measure | GovReport | Multi-News |
| --- | --- | --- |
| Lead | 1 | 1 |
| Random | 10 fixed seeds; not selected by score | 10 fixed seeds; not selected by score |
| TextRank | 1 | 1 |
| LexRank | 1 | 1 |
| PacSum-TFIDF | 21 | 21 |
| PacSum-SBERT | 21 | 21 |
| SBERT centroid | 1 | 1 |
| SBERT+MMR | 5 | 5 |
| PAMR-ES: unique development config hashes | 69 | 68 |
| PAMR-ES: held-out dev-test score observations | 4 | 4 |

These counts describe configuration selection, not neural fine-tuning. The PAMR-ES program includes length, capacity, route, selector and combination studies that could inform its final profile; it excludes E3, cost, oracle and engineering checks. Methods did not receive identical search budgets. Search records and frozen data partitions make that scope auditable.

## S7. Fixed-rule decision-provenance example

The example is validation_crs_R44729, the first document in the frozen GovReport development prediction order, selected without consulting ROUGE or qualitative favorability. The summary has 647 words and 13 sentences. All selected sentences are provided below, followed by the first five reserved but unselected candidates in record order. Indices refer to the original canonical sentence order.

L/S/G denote lexical/semantic/graph route ranks. Retention reasons distinguish route nomination, reservation, guard or fusion fill from final selection. This example demonstrates a traceable selection record, not factuality or causal explanation. Source text is from the GovReport CRS document identified above; it is not original prose by the authors.

### Selected sentences

Sentence index 0 | L/S/G ranks 9/86/6 | fusion rank 11 | route agreement 2 | retained: route:lexical, route:graph, rrf_fill

On January 5, 2011, the House of Representatives adopted an amendment to House Rule XII to require that Members of the House state the constitutional basis for Congress's power to enact the proposed legislation when introducing a bill or joint resolution.

Sentence index 30 | L/S/G ranks 137/24/14 | fusion rank 16 | route agreement 2 | retained: route:semantic, route:graph, rrf_fill

The Supreme Court has interpreted the scope of Congress's power under the Necessary and Proper Clause as "broad," in that the clause leaves to "Congress a large discretion as to the means that may be employed in executing a given power."

Sentence index 36 | L/S/G ranks 7/40/24 | fusion rank 9 | route agreement 3 | retained: route:lexical, route:semantic, route:graph, rrf_fill

As the Supreme Court has noted, the clause is "not itself a grant of power, but a caveat that the Congress possesses all the means necessary to carry out the specifically granted 'foregoing' powers of § 8 'and all other Powers vested by this Constitution....'" Instead, in legislating, Congress "must rely upon its independent (though quite robust) Article I, § 8, powers" or in other powers implicitly or explicitly vested elsewhere in the Constitution to Congress.

Sentence index 67 | L/S/G ranks 3/85/9 | fusion rank 10 | route agreement 2 | retained: route:lexical, route:graph, rrf_fill

In its 1803 decision in Marbury v. Madison, the Supreme Court held that the logic of having a written Constitution that enumerates the legal limits imposed on the federal government, coupled with the tenure protections provided to the federal judiciary under the Constitution, confirmed the Supreme Court's role in interpreting the Constitution and invalidating acts of other branches of government that contravene this document in the context of a live case or controversy.

Sentence index 94 | L/S/G ranks 127/12/11 | fusion rank 6 | route agreement 2 | retained: route:semantic, route:graph, rrf_fill

While the rule, on its face, requires Members to provide as "specific[] as practicable" "a statement citing ... the power or powers to Congress in the Constitution to enact the bill or joint resolution," the CAS rule itself is silent on various issues.

Sentence index 98 | L/S/G ranks 148/30/3 | fusion rank 8 | route agreement 2 | retained: route:semantic, route:graph, rrf_fill

"The constitutional authority on which this bill rests is the power of Congress to make rules for the government and regulation of the land and naval forces, as enumerated in Article I, Section 8, Clause 14 of the United States Constitution."

Sentence index 109 | L/S/G ranks 85/10/5 | fusion rank 1 | route agreement 2 | retained: route:semantic, route:graph, guard:document

Nonetheless, the last example provided by the Rules Committee suggests that a citation to a provision of the Constitution that does not explicitly grant power to the Congress—such as the Tenth Amendment, which preserves the powers of the states —may suffice to comply with the rule.

Sentence index 208 | L/S/G ranks 24/13/54 | fusion rank 12 | route agreement 2 | retained: route:lexical, route:semantic, rrf_fill

Viewing this limitation on the use of a CAS as a shortcoming that prevents more robust constitutional debate, several proponents of the CAS rule have argued that the rule should apply during all stages of the legislative process, including during committee deliberations, so that the constitutionality of a bill or resolution is subject to broader consideration.

Sentence index 211 | L/S/G ranks 49/29/20 | fusion rank 15 | route agreement 2 | retained: route:semantic, route:graph, rrf_fill

In what may be the broadest means to allow more Members to weigh in on the constitutional implications of a bill, at least one commentator has suggested (but ultimately rejects) changing the House rule so that the CAS is part of the text of a bill, as opposed to a statement attached to the bill.

Sentence index 236 | L/S/G ranks 253/4/12 | fusion rank 4 | route agreement 2 | retained: route:semantic, route:graph, rrf_fill

Does the CAS cite to a specific clause of the Constitution?

Sentence index 243 | L/S/G ranks 71/3/30 | fusion rank 5 | route agreement 2 | retained: route:semantic, route:graph, rrf_fill

Citations in CASs to clauses in Article I, Section 9 of the Constitution, which contains a list of limitations on the powers of the federal government, or the Bill of Rights, which consists of a number of rights retained vis-á-vis the federal government, may suggest a broader interpretation of such clauses.

Sentence index 244 | L/S/G ranks 53/9/35 | fusion rank 7 | route agreement 2 | retained: route:semantic, route:graph, rrf_fill

To the extent a Member prefers to cite to a clause that is more generally recognized to grant an affirmative power to Congress, Article I, Section 8 contains the vast majority of commonly cited clauses that provide Congress the power to legislate with respect to various subjects.

Sentence index 252 | L/S/G ranks 13/47/2 | fusion rank 2 | route agreement 2 | retained: route:lexical, route:graph, rrf_fill

While the customary practice with regard to CASs, to date, has been to provide a short citation to the provision in the Constitution that affirmatively grants Congress the authority to enact the underlying legislation, it is not unprecedented for Members to cite sources beyond the text of the Constitution, such as Supreme Court case law, primary source materials on the Constitution, or a constitutional law treatise.

### Reserved candidates not selected

Sentence index 8 | L/S/G ranks 19/88/166 | fusion rank 85 | route agreement 1 | retained: route:lexical, reserve:lexical

The report contains two tables: Table 1 identifies the constitutional provisions most commonly cited in CASs during the last six months of the 114 th and 115 th Congresses, and Table 2 lists suggested constitutional authorities for various types of legislation.

Sentence index 9 | L/S/G ranks 183/99/19 | fusion rank 52 | route agreement 1 | retained: route:graph, reserve:graph

Understanding the purpose and logic of the CAS rule first requires an understanding of both the powers provided to the Congress under the Constitution and Congress's role in interpreting the Constitution.

Sentence index 13 | L/S/G ranks 210/75/34 | fusion rank 63 | route agreement 1 | retained: route:graph, reserve:graph

While only Congress may exercise the legislative power, this power, like those belonging to the other branches of the federal government, is cabined by the terms of the Constitution.

Sentence index 15 | L/S/G ranks 204/22/85 | fusion rank 53 | route agreement 1 | retained: route:semantic, reserve:semantic

As a result, the Supreme Court has interpreted Article I's Vesting Clause as creating a Congress of specified or "enumerated powers."

Sentence index 16 | L/S/G ranks 207/11/45 | fusion rank 24 | route agreement 1 | retained: route:semantic, reserve:semantic

As the Court noted in United States v. Morrison , "[e]very law enacted by Congress must be based on one or more of its powers enumerated in the Constitution."

## S8. Historical selector comparison and controlled cost

NSGA-II is a historical comparator, not a component of final PAMR-ES. The development selector comparison precedes final fusion configuration and holds candidate pools, TF-IDF similarities, length constraints and random seeds fixed. The shared comparison uses equal route weights and zero positional weight; its scores must not be substituted for the final development configuration.

### Table S18. Archived matched selector scores (candidate IDs preserve provenance)

| Dataset | Candidate | Mean |
| --- | --- | --- |
| Multi-News | S00_greedy_tfidf_anchor | 0.328077 |
| Multi-News | S03_mmr_tfidf_l03 | 0.321744 |
| Multi-News | S12_nsga2_tfidf_seed3407 | 0.322615 |
| GovReport | S00_greedy_tfidf_anchor | 0.417862 |
| GovReport | S05_mmr_tfidf_l07 | 0.446154 |
| GovReport | S12_nsga2_tfidf_seed3407 | 0.426844 |

The complete selector candidates and paired analyses remain in the D2 source JSON. NSGA-II was not the highest-quality selector on either dataset. The final selectors are MMR for GovReport and Greedy for Multi-News.

### Table S19. Controlled CPU cost: medians of three measured repetitions

| Dataset | System | Cold total s | Warm total s | Warm s/doc | Cold/warm RSS MiB |
| --- | --- | --- | --- | --- | --- |
| GovReport | Final PAMR-ES | 176.03 | 10.52 | 0.351 | 774.1 / 422.9 |
| GovReport | NSGA-II comparator | 222.61 | 56.74 | 1.891 | 777.8 / 421.9 |
| Multi-News | Final PAMR-ES | 47.85 | 9.23 | 0.308 | 594.6 / 398.9 |
| Multi-News | NSGA-II comparator | 91.69 | 50.07 | 1.669 | 599.6 / 400.7 |

Times are totals for the fixed 30-document sample, with a derived warm per-document average. Sampling is reference-blind across the 10th, 50th and 90th percentiles of log(1 + source sentence count), ten documents per stratum. Cold and warm-cache runs use fresh subprocesses, three measured repetitions, and process-tree RSS. Full system-level quartiles, CPU measurements, stratum measurements and descriptive scaling slopes are in the cost-analysis JSON files. Hardware timing is not a cross-machine speed claim.

## S9. Artifact access and source index

Code, configurations and compact evidence are versioned at https://github.com/Caspar15/metaheuristic-summarization/tree/v1.0.0-ieee-access. The matching release provides this supplement and the numerical-score/selected-index artifact. Raw input documents and reference summaries remain with their original providers; the artifact does not relicense or redistribute full source corpora.

The source package includes a standard-library snapshot verifier and result-table export. The numerical artifact includes a content manifest with per-file hashes and original-source hashes. Full output regeneration additionally requires the frozen canonical data, pinned semantic and Stanza model assets, and the specified Perl evaluator environment. Source-only verification is distinct from rerunning both full benchmarks.

Tables in this supplement are generated from archived JSON. Decimal presentation is rounded; machine-readable source values retain full precision. The following source paths and the companion source manifest identify the evidence used. Complete configuration paths are recorded by the individual execution and variant evidence files.

configs/final/govreport_final_v1.yaml

configs/final/multinews_final_v1.yaml

configs/preregistrations/govreport_final_execution_freeze_v1.json

configs/preregistrations/multinews_final_execution_freeze_v1.json

runs_v2/d2_selector_full_dev_v1/analysis/paired_summary.json

runs_v2/govreport_cost_scaling_v1/analysis.json

runs_v2/govreport_e3_route_provenance_v1/analysis.json

runs_v2/govreport_e3_route_provenance_v1/study_summary.json

runs_v2/govreport_final_test_v1/analysis.json

runs_v2/manuscript_supplemental_analysis_v1/analysis.json

runs_v2/manuscript_supplemental_analysis_v1/provenance_case.json

runs_v2/multinews_cost_scaling_v1/analysis.json

runs_v2/multinews_e3_route_provenance_v1/analysis.json

runs_v2/multinews_e3_route_provenance_v1/study_summary.json

runs_v2/multinews_final_test_v1/analysis.json

runs_v2/postfreeze_no_lexical_route_v1/analysis.json

runs_v2/postfreeze_no_reservation_v1/analysis.json

runs_v2/postfreeze_zero_lexical_weight_v1/analysis.json

AI assistance was used to organize and translate the supplementary text and implement its evidence-to-table formatting. No new experimental results were generated for this supplement; the authors retain responsibility for verification and the final content.
