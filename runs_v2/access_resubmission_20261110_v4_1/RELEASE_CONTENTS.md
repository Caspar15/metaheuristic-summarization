# Contents of this evidence folder

This folder holds the compact, reviewer-requested evidence for the IEEE Access resubmission
(Access-2026-46834): candidate coverage, the capacity and RRF tables, decision-record verification,
and the walkthrough order check. It supports Supplementary Sections S9-S11 of the revised
supplement. `LOCAL_EVIDENCE_EN.md` explains definitions, scope and interpretation.

Nothing here changes the frozen final configurations, predictions or reported test results.
The scripts that produced these files are `scripts/audit/access_revision_*.py`; their offline tests
are in `tests/test_access_revision.py`.

## Included files

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `LOCAL_EVIDENCE_EN.md` | 16,920 | `7732c2efa90552255d599c11dca3359cb8b34ac7f548fc565ab1ab6486ee2b18` |
| `validation_checks.json` | 577 | `659dcce442c4e56950c5928a3cd489ab1515b4210b3e56d03a006c74ccbac5dc` |
| `review_pack_manifest.json` | 4,544 | `f267be0e4baf98b722e429e1f954b0daf66e9de4d5cf6dbc614a3c60fe040213` |
| `coverage/recall_conditions.tsv` | 22,630 | `a250feafa79525bdfff5387b25c887cae49dbaac65223bad505113e2ddb08275` |
| `coverage/recall_contrasts_all.tsv` | 27,644 | `547a8062bc62110921718664f1dce298b04052f93618910d144c738be084d055` |
| `coverage/recall_contrasts_by_size.tsv` | 99,884 | `31f4a89a46c738564d488d9fd37c81f3ebafcefa2de5feed06d26d0fa447258b` |
| `coverage/paired_denominators.tsv` | 95,700 | `72bde3ace6190fe73b89fb1a4584ef29d1e1b674116509de8d15e729fe52462f` |
| `coverage/pool_size_strata.tsv` | 2,567 | `c9a277889844e76178bb977fc0c993a191e5a156125822c5a6140049f3d75dfd` |
| `coverage/pool_set_relations.tsv` | 1,889 | `f97c69969a83280307092a4a270a92fe426913b013d5439268faa57b73bacca0` |
| `coverage/reference_validation.json` | 1,704 | `3d955e937bff35be2dd88480d1486bf4391c03ccfbac86cdbbdac6a3023b65e5` |
| `coverage/scoring_started.json` | 883 | `d6ba7b55adaa6f0ceb761691ce86382f0888989ab9d5e5fc70e1f198953596d2` |
| `coverage/scoring_complete.json` | 767 | `8b229d0d89107cf61a1b8c206672b42700373f9e16a51f6c12fcf44eb56b0e27` |
| `historical/development_conditions.tsv` | 8,435 | `2c777150df73dff6588fb1eb9fcd347fe00c91ec6c349c6a86b5b84e7fda76b4` |
| `historical/RRF_numeric_examples.json` | 2,207 | `76a8507d787ef1948776f3bed12e94cae3258a19adb547c9a9aea006f09b98d0` |
| `historical/source_manifest.json` | 32,247 | `daa5c63fe852e25fc6e949b2fee8ac13e059f70e42452e1ec4eae8c62882e895` |
| `historical/snapshot_packaging_note.json` | 376 | `2a7eda8d7c7660807ffbb423accb83ef2a0dc34dffab121ddd0c1333a40a5675` |
| `manifests/input_joins.tsv` | 484 | `a783796e39701051aff086a367680a2c2ed26ceddd3e726fe878f5154017a654` |
| `manifests/preparation_complete.json` | 1,404 | `5660845356c03e287020deeac5b91a2dbf391cdddea09a4867175f3cc0b9bb3f` |
| `manifests/formal_preparation_complete.json` | 2,724 | `8ec2c828733288b5734d1afab290042d1686a056cd387ea2f19b664f46bfe0b2` |
| `manifests/config_parity.json` | 2,855 | `0f88611c9833a5a3827bb24e3da3dfae81ea9c7020961769909087f52dfd1017` |
| `manifests/formal_sampling_summary.json` | 5,713 | `8686bffcc8cc783f3f2a257684bda232b430e123e04d97b62ff144534450c631` |
| `manifests/dev40.jsonl` | 9,096 | `1a6dff2358a3812ef34684144441dc67302430564372841121a232ec22bc2735` |
| `manifests/formal200_subsets.jsonl` | 25,925 | `60cf6815e33e0af59af5dcdcf40ce70925b6c0fd354d4dbc250e6d77ae36a283` |
| `manifests/formal200.jsonl` | 64,278 | `a27d7fc779d7e3eea17eda92000cafa02574dee6e0c64412240820de04d0b5f5` |
| `manifests/test_population_clusters.jsonl` | 661,222 | `41b8df3a765d7ab2e8dfd6850ef7a261507eb03e95928dd86006bd57a37b5eba` |
| `audit/L0_L1_summary.tsv` | 1,225 | `d431a0e6120e3dd521a4427f1fdbb2a33946aa8857f2aefb0b328a929f7d4462` |
| `audit/L0_L1_started.json` | 1,321 | `33b11fa96cd3579a027115276f2fbe8171a6e0fa1dafc41dce8965d729a516c7` |
| `audit/L0_L1_complete.json` | 230 | `c9606377a484ec04de7d15cb969dbe4c7114845cb34c166569abadc20e957c65` |
| `audit/fixed_pool_summary.tsv` | 325 | `7c69b9203e2bbb7fb1c601fe8f8d4c28dcc74ce3bf2751b2060b36531ee2a153` |
| `audit/fixed_pool_started.json` | 2,232 | `ac87561f1e5205ee349a47d667bc1ad2ac8ce6dd65c8480202a4934ffdd62a48` |
| `audit/fixed_pool_complete.json` | 230 | `afd1610b234270c55efd47318c03691ebee6dfb92ba628d526e354f1855308fd` |
| `audit/dev40_gate.json` | 251 | `d9b8ab9b50fbc3101a509235d66952ffe10aa108b603b2805cdc7da2cf643be0` |
| `audit/formal200_gate.json` | 256 | `8e9ab9d6961650097ee29626d5abc0d81e70d651c30cb323c04ec1fab9c78659` |
| `audit/table10_order_verification.json` | 593 | `248b6233a079a43e22ed8c82972f8c3afddf56189740f50a24da4035addaf91a` |
| `audit/dev40/govreport_summary.json` | 26,504 | `968517ffaa864ac3b38903709ca43e0910082da9491f33097b784d1edc863218` |
| `audit/dev40/multinews_summary.json` | 25,378 | `abc611912b7a64c8ca5898effde3010a1aa31fd03748c2e23a89a060df422076` |
| `audit/formal200/govreport_summary.json` | 126,466 | `49c63809fe482e9f09d71c2dc9a963c94c5d41b6ba091ca821df4b3124f9ed15` |
| `audit/formal200/multinews_summary.json` | 120,748 | `d85cb6fc45a0f5f3723c784629fd4272a79805171c255449fc4779b7c57ae911` |
| `protocol/execution_freeze.json` | 9,131 | `09100705f70eb54d00839257984d39e56c1c5a32a1dab6d2930f18a402219f96` |
| `protocol/implementation_correction_001.json` | 491 | `919a1d23d6bdb9ac7218ac622e1aa108a90b577e9379814903db968cf6a427bc` |

## Kept locally and not published

The following were produced in the same execution but are not part of this repository version.

**Large per-document tables.** They contain sentence identifiers and numeric values only, but are
too large for the repository. Their hashes are listed so that copies can be verified on request.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `coverage/recall_per_document.jsonl` | 61,170,261 | `9652c7a971c5fbce5b97bebdd83b14da1f58ea9fd8ece06a6993b927e612c448` |
| `coverage/pool_records.jsonl` | 15,363,791 | `eeee8724bcd6a8fde73c4f7d52717da08679061cda4e29cc5d5239ce51c46c87` |
| `coverage/pool_size_membership.jsonl` | 5,255,749 | `6aeb6d75b4f7e91d0627ed94f8fe09b31528f2443f77cd72c6dcbd30cb1eba84` |
| `audit/L0_L1_per_document.jsonl` | 3,165,765 | `1acd7624e80ad29169bff4297b57442ae3f19ba77b47efdac45b2083b1933e3b` |
| `audit/test_pool_descriptions.jsonl` | 6,107,499 | `9620084b3d01bd9389ea1627eb36f1e8848d54bb103ec090295b04ff63854e2b` |
| `audit/fixed_pool_per_document.jsonl` | 6,643,597 | `b86021400a3887befdde8d40fe8df10cb31698ee69285dc898b7eb08cb0fe19d` |

**Archives.** The repository does not track ZIP files. `Local_Evidence_Review_Pack_20261112.zip` bundles
earlier copies of files listed above (its contents and hashes are in `review_pack_manifest.json`).
`historical/source_snapshot.zip` holds byte-preserved copies of the historical metadata indexed in
`historical/source_manifest.json`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `Local_Evidence_Review_Pack_20261112.zip` | 63,542 | `f4c7f0fb58b93d9dbf04dbdfd49aa5f092e1d611e4583cb480bd97d5d3ec2930` |
| `historical/source_snapshot.zip` | 213,352 | `b650fb795dae36394f2ded231141425b319d059be65664cc05662f019a79054c` |

**Per-example replay outputs.** Each replayed example stores source sentences, reference text and
generated summary text, which the dataset terms and this repository's release policy do not
redistribute. Their pass/fail results are included in the `*_summary.json` and `*_gate.json` files.

| Folder | Examples | Files | Bytes |
| --- | ---: | ---: | ---: |
| `audit/dev40/govreport/` | 20 | 120 | 15,286,089 |
| `audit/dev40/multinews/` | 20 | 120 | 7,152,489 |
| `audit/formal200/govreport/` | 100 | 600 | 76,289,104 |
| `audit/formal200/multinews/` | 100 | 600 | 37,956,950 |

**Frozen source snapshots** (`protocol/frozen_src_*`). These are copies of commits 8d75aad, b0e0d28,
ebd6306 and faab8fb, which are available in this repository's history.

**Internal planning copies** (`protocol/plan_v4_1/`). These are the authors' internal planning and
correspondence documents and are not published.

**LLM-assisted evaluation in progress** (`execution_status*.json`, `JUDGE_ACCESS_READINESS_20261112.md`,
`cli_access_smoke/`, `antigravity_probe_v2/`, `judge_schema_smoke/`, `judge_pilot_v1/`). This work
addresses a separate reviewer point and is not complete. These files include account and access logs,
blinding keys and full source texts, and are not published.
