# Numerical evidence for the November 2026 IEEE Access revision

The [v1.1.0 release](https://github.com/Caspar15/metaheuristic-summarization/releases/tag/v1.1.0-ieee-access) provides the revised supplement (Tables S1–S32), this evidence archive and SHA-256 checksums. The [original benchmark release](https://github.com/Caspar15/metaheuristic-summarization/releases/tag/v1.0.0-ieee-access) contains the September supplement, per-document benchmark scores and selected-sentence indices. The evidence used in Sections S9–S11 is also available at [its archived revision](https://github.com/Caspar15/metaheuristic-summarization/tree/d5772d6f96874552ac879acc290de55412ee16b0/runs_v2/access_resubmission_20261110_v4_1).

| Evidence file | Supplement | Contents |
| --- | --- | --- |
| `test_length_bounds.json` | S3, S6 | Counts below nominal targets and applied floors, and results for the subset in which both summaries meet the nominal range. |
| `R2_5_per_document_summary.json` | Table S28 | Paired Mean ROUGE differences, wins, ties and losses, overall and by source-length quartile. |
| `R2_7_control_rouge.json` | Table S29 | Numerical scores for 40 development controls and the direction of change by property. |
| `primary_results.json` | Tables S30–S31 | Sixteen judge-specific preference estimates, paired counts, intervals, missing-data bounds, order effects and flags for failed development controls. |
| `R2_7_repeat_diagnostics.json` | Table S32 | Agreement with the original judgment on repeated assessments and after changing source order, by dataset, judge and property. |

The primary judge study sampled 100 test examples per dataset. In the judge files, J1 denotes Claude and J2 denotes Gemini. A preference score of 1 favors PAMR-ES, 0.5 represents a tie, and 0 favors the comparator. The comparator is SBERT+MMR on GovReport and PacSum-TFIDF on Multi-News. Section S15 of the supplement gives the assessment protocol and interpretation.

`supplement_sources.json` identifies all 33 inputs by their repository paths and hashes. To check the extracted archive on a system with GNU Coreutils, run `sha256sum -c SHA256SUMS.txt` from the directory containing the JSON files.
