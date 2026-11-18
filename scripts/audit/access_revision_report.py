"""Render reviewer-facing evidence from completed, unchanged local outputs."""
from __future__ import annotations

from collections import Counter
from scripts.audit.access_revision_common import *


def table(path):
    with path.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f,delimiter='\t'))


def put(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f: f.write(text)


def run():
    for path,key in [('audit/dev40_gate.json','all_passed'),('audit/formal200_gate.json','all_passed'),('audit/L0_L1_complete.json','all_checkable_passed'),('audit/fixed_pool_complete.json','all_passed')]:
        if not readj(OUT/path)[key]: raise ValueError('required audit failed: '+path)
    conditions=table(OUT/'coverage/recall_conditions.tsv'); contrasts=table(OUT/'coverage/recall_contrasts_all.tsv'); subsets=table(OUT/'coverage/recall_contrasts_by_size.tsv')
    paired=[]
    for r in contrasts+subsets:
        for metric in ['pool_recall','selected_recall','conditional_retention']:
            n=int(r[metric+'_paired_delta_N'])
            paired.append({k:r[k] for k in ['dataset','target','control','subset']} | {'metric':metric,'subset_N':int(r['N']),'joint_valid_N':n,'joint_NA':int(r['N'])-n,'NA_rule':'empty greedy reference for recall; additionally empty G intersect pool on either side for conditional retention'})
    write_tsv(OUT/'coverage/paired_denominators.tsv',paired)
    source_case=next(x for x in rows(run_dir('govreport','C01')/'predictions.jsonl') if x['id']=='validation_crs_R44729')
    order=[source_case['candidate_records'][i]['original_index'] for i in source_case['optimizer_diagnostics']['selection_order']]
    case={'id':source_case['id'],'source_prediction_sha256':sha(run_dir('govreport','C01')/'predictions.jsonl'),'true_MMR_decision_order':order,'summary_source_order':source_case['selected_indices'],'table10_first_five_output_indices':source_case['selected_indices'][:5],'table10_correct_label':'Output 1–5 (source order), not MMR decision steps 1–5'}
    writej(OUT/'audit/table10_order_verification.json',case)
    headline='''# Reviewer-requested local evidence — execution v4.1

This document reports completed local work under the 11 November 2026 v4.1 protocol. It is an evidence report, not a complete resubmission. The original submitted manuscript, predictions, scores and final configurations have not been changed. New candidate-recall measurements are retrospective, reviewer-requested development diagnostics; their definitions and size-stratum membership were saved before joining greedy references. Original ROUGE outcomes were already known. No new significance tests or model tuning were performed.

## Candidate coverage (R2.3)

All 681 GovReport and 3,935 Multi-News development IDs were joined one-to-one across C01 and four controls. Candidate indices, source sentences, selected-set inclusion and summary reconstruction passed for all 23,080 prediction rows. C01/A06 differ only in the substantive reservation setting (20 versus 0); study metadata is excluded from this configuration comparison. A01/A02 change route availability, and A03 is the lexical-only top-80 setting. They are not pure quota interventions.

For each metric-specific greedy reference set G, candidate pool C and selected set S, pool recall is |G∩C|/|G|, selected recall is |G∩S|/|G|, and conditional retention is |G∩S|/|G∩C|. These measure matching greedy-selected sentence IDs, not the proportion of all important facts covered. Greedy references are neither exact optima nor official dataset oracles. Ratio denominators of zero are NA. The six empty Multi-News ROUGE-2 reference sets are retained in the population but excluded from the corresponding ratios.

The table shows all 30 prespecified condition/target combinations. Recall values are document-macro percentages. Exact unrounded values, micro values, numerators, denominators, size/word distributions and downstream descriptions are in `coverage/recall_conditions.tsv`.

| Dataset | Greedy target | Condition | Mean pool size | Pool recall (%) | Selected recall (%) | Valid reference N |
|---|---|---|---:|---:|---:|---:|
'''
    for r in conditions:
        headline+=f"| {r['dataset']} | {r['target']} | {r['condition']} | {float(r['pool_size_mean']):.2f} | {100*float(r['pool_recall_macro']):.3f} | {100*float(r['selected_recall_macro']):.3f} | {r['pool_recall_N']} |\n"
    headline+='''
### All 24 full-development contrasts

Direction is C01 minus control; units below are percentage points of pool recall, not ROUGE points. Values are paired mean differences. No p-values are assigned.

| Dataset | Greedy target | Control | Mean difference (pp) | Positive / zero / negative documents |
|---|---|---|---:|---|
'''
    for r in contrasts:
        headline+=f"| {r['dataset']} | {r['target']} | {r['control']} | {100*float(r['pool_recall_paired_delta_mean']):+.3f} | {r['pool_recall_positive_n']} / {r['pool_recall_zero_n']} / {r['pool_recall_negative_n']} |\n"
    headline+='''
### Pool-size interpretation

All 144 prespecified size-subset/target/control rows, including empty groups, are in `coverage/recall_contrasts_by_size.tsv`; joint-valid and joint-NA counts for each paired metric are explicit in `coverage/paired_denominators.tsv`. Reference-blind membership was saved before scoring. Equal-size children (both 80, equal positive below 80, both empty) are nested, so their Ns must not be added to the parent N.

C01 has greater full-development pool recall than A01 and A02 on both datasets. Its pools are also larger on average, so those contrasts describe the operating configurations rather than isolating a pure route effect at fixed size. Against A03 on GovReport, C01 has greater recall across all three targets despite a slightly smaller mean pool. Against A03 on Multi-News, C01 has lower full-development recall and a smaller mean pool. The equal-size results do not replace these full-population observations.

For the ROUGE-Lsum greedy target, all prespecified A03 size groups are shown here for interpretation. All three targets and all controls remain available in the complete files.

| Dataset | Size subset | N | Mean C01 size | Mean A03 size | C01−A03 pool recall (pp) |
|---|---|---:|---:|---:|---:|
'''
    for r in subsets:
        if r['target']=='rougeLsum' and r['control']=='A03_lexical_only_capacity_80':
            values=[f"{float(r[k]):.2f}" if r[k] else 'NA' for k in ['C01_pool_size_mean','control_pool_size_mean']]
            delta=f"{100*float(r['pool_recall_paired_delta_mean']):+.3f}" if r['pool_recall_paired_delta_mean'] else 'NA'
            headline+=f"| {r['dataset']} | {r['subset']} | {r['N']} | {values[0]} | {values[1]} | {delta} |\n"
    headline+='''
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
'''
    put(OUT/'LOCAL_EVIDENCE_EN.md',headline)
    writej(OUT/'execution_status.json',{'updated_at_utc':utc(),'protocol':'v4.1','completed':{'prediction_mapping_dev_rows':23080,'candidate_coverage_conditions':30,'candidate_coverage_full_contrasts':24,'candidate_coverage_size_rows':144,'fixed_pool_comparisons':13848,'L0_L1_test_rows':6594,'L2_dev_passed':40,'L2_formal_passed':200,'historical_conditions':33},'LLM_quality_calls':0,'paid_API_calls':0,'manuscript_modified':False,'submission_sent':False,'remaining':['source-verified novelty and full literature/bibliography integration','judge access and exact model/version/context/isolation preflight','four-dimension pilot, controls and author QC','formal four-dimension LLM judging and descriptive analysis','manuscript/supplement/response integration and PDF/source packaging'],'chronology_note':'Reference-blind groups and dev40 passed before recall. Full fixed-pool and historical rechecks completed after recall; no contrast definitions, membership or reporting scope changed.'})
    print({'stage':'report','conditions':len(conditions),'paired_denominator_rows':len(paired),'table10_true_first5':order[:5]},flush=True)


if __name__=='__main__': run()
