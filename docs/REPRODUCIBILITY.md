# Reproducing and inspecting PAMR-ES

This guide covers the frozen IEEE Access research implementation. Run commands
from the repository root. Python 3.11 is used in Linux unit-test CI; the recorded
experimental runtime is Windows 11 with Python 3.12.7. The optional web demo and
legacy metaheuristic experiments are separate from the final paper's method.

## 1. Inspect the archived results (Python standard library only)

```sh
python -m scripts.audit.verify_submission_snapshot
python -m scripts.audit.verify_submission_snapshot --export-dir review_tables
```

Use a new export directory each time. The second command creates JSON and CSV
with all nine systems on both official test sets (973 GovReport and 5,621
Multi-News documents). Scores are on the 0–1 scale. They are read from archived
analyses, not newly evaluated. Random is the archived ten-seed aggregate.
Paired effects, confidence intervals and multiplicity corrections remain in the
source analyses: subtracting rounded corpus scores does not reproduce those
paired estimates. This export covers the main quality results, not every paper
table or the full English Supplementary Material.

The dated [snapshot manifest](research/evidence/submission_snapshot_2026_09_09.json)
inventories research source, final configurations, historical freeze records,
final results, cost/ablation analyses and supplemental evidence. It records the
September review snapshot; it does not replace or backdate the original
pre-test freezes. Result evidence is checked byte for byte; Python source,
configuration text and requirements permit only CRLF-to-LF checkout normalization
(the existing policy hash function uses this same text normalization). A passing check
establishes consistency with the manifest, not independent authenticity or a
new reproduction of model outputs.

## 2. Install the unit-test environment

Create and activate a virtual environment:

```sh
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```sh
source .venv/bin/activate
```

Install dependencies and run the checks:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements-ci.txt
python -m compileall -q src tests scripts
python -m pytest -q
python -m scripts.audit.verify_govreport_freeze_package
python -m scripts.audit.verify_provenance --root runs_v2
```

These tests use fixtures and model stubs; they do not download datasets or model
weights. Use the virtual environment's Python if imports fail. On restricted
Windows hosts, `WinError 5` from pytest temporary directories or process pipes
requires a writable temporary location and permission to run local processes.
It should not be worked around by skipping failing tests.

The provenance validator currently reports 468 `legacy` pins and zero failures.
`legacy` means an exact match to the versioned CRLF-to-LF erratum; it is distinct
from an unexplained mismatch. The GovReport freeze validator explicitly defers
absent, untracked local evidence in a source-only download. Where the original
local artifacts are available, add `--require-local-evidence` for the strict check.

## 3. Reconstruct the scientific runtime

Install `requirements.txt` with the recorded core constraints:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements.txt -c environments/frozen-core-constraints.txt
```

The [core constraints](../environments/frozen-core-constraints.txt) come from both
original execution freezes. They are not a complete transitive lock. The separate
[September Windows environment snapshot](../environments/audit-windows-py312.freeze.txt)
records the environment used for the repository regression audit, including
optional packages. It is not claimed to be the complete historical August
environment or an environment installed and verified on another operating system.

For a fixed Windows/Python 3.12.7 research-and-test package set, the
[89-package lock](../environments/research-windows-py312.lock.txt) records the
transitive dependency closure of `requirements.txt` and `requirements-ci.txt`.
Its installed metadata constraints were checked with zero conflicts. After
installing CPU torch, use `python -m pip install -r environments/research-windows-py312.lock.txt`
in a new virtual environment, followed by `python -m pip check` and the tests.
Fresh installation of this lock has not been verified. It intentionally excludes
unrelated optional packages: the complete local audit environment has a
spaCy/thinc dependency conflict with NumPy 1.26.4. Those packages are not used by
the research/test dependency closure; do not upgrade the frozen NumPy to satisfy
the optional demo environment.

Full output reproduction additionally requires the following local resources:

| Resource | Exact identity / instructions |
|---|---|
| Canonical data and membership | `configs/data_policies/` and `configs/length_policies/`; retain source-document boundaries and all frozen exclusions |
| Final configurations | [GovReport](../configs/final/govreport_final_v1.yaml), [Multi-News](../configs/final/multinews_final_v1.yaml) |
| Semantic model | `sentence-transformers/all-MiniLM-L6-v2`, revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, CPU, max length 256 |
| Official evaluator | Stanza 1.10.1 `tokenize,mwt`, the pinned model files, pyrouge 0.1.3 and original Perl ROUGE-1.5.5; arguments `-c 95 -r 1000 -n 2 -m` |
| Runtime/model/data hashes | [GovReport execution freeze](../configs/preregistrations/govreport_final_execution_freeze_v1.json), [Multi-News execution freeze](../configs/preregistrations/multinews_final_execution_freeze_v1.json) |
| Windows evaluator setup | [Compute environment](research/COMPUTE_ENVIRONMENT.md) and the evaluator addenda referenced by the freezes |

Data, model weights, Perl runtime assets, full predictions and per-example scores
are not all in Git. Obtain them from their original providers or a separately
published author artifact, then verify against the recorded hashes. Do not use a
newly downloaded model revision or silently substitute the internal Python
ROUGE-Lsum evaluator for the manuscript's Stanza/Perl score columns.

The governed reproduction runners are
`python -m scripts.audit.run_govreport_final_test` and
`python -m scripts.audit.run_multinews_final_test`. Their CLI help and execution
freeze records specify dry-run/execute commands, expected freeze and activation
hashes, and the original output paths. These commands are not a source-only quick
start: prepare a separate checkout with the required resources and an empty run
destination before following the frozen protocol. Preserve the original evidence
archive; never overwrite it or tune the configuration after observing test scores.
Dry-run modes also write their evidence paths, so do not run them merely to inspect
an existing release. The read-only checks in sections 1–2 serve that purpose.

## 4. Locate the evidence

| Evidence | Location |
|---|---|
| Official quality and paired inference | `runs_v2/{govreport,multinews}_final_test_v1/analysis.json` |
| Cost, RSS and scaling | `runs_v2/{govreport,multinews}_cost_scaling_v1/analysis.json` |
| Prespecified component ablations | `runs_v2/{govreport,multinews}_e3_route_provenance_v1/analysis.json` |
| Length, matched-row sensitivity, configuration counts | `runs_v2/manuscript_supplemental_analysis_v1/analysis.json` |
| Fixed-rule provenance example | `runs_v2/manuscript_supplemental_analysis_v1/provenance_case.json` |
| Development-only mechanism diagnostics | `runs_v2/postfreeze_no_reservation_v1/`, `runs_v2/postfreeze_no_lexical_route_v1/`, `runs_v2/postfreeze_zero_lexical_weight_v1/` |

The final selectors are TF-IDF MMR (GovReport) and Greedy-TFIDF (Multi-News).
NSGA-II is a historical comparator. The primary claim is GovReport-scoped;
Multi-News has no significant overall Mean ROUGE advantage over PacSum-TFIDF.
Route reservation provides source balance and an audit trail; the diagnostics
do not establish an independent quality benefit for every candidate route.
No diagnostic changed the final system.

## 5. Distribution status

Source and compact evidence can be inspected in this repository. A fresh source
snapshot audit is distinct from installing an entirely new environment and
reproducing both complete benchmarks. The September audit did the former using
the existing tested environment; see the [audit report](research/REPOSITORY_AUDIT_2026_09_09.md).

Original software and software documentation use the [MIT License](../LICENSE);
dataset/model and manuscript publication terms remain separate as explained in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The submission version is
[`v1.0.0-ieee-access`](https://github.com/Caspar15/metaheuristic-summarization/releases/tag/v1.0.0-ieee-access).
The release provides the English supplement and the numerical selected-index /
per-document-score ZIP, with checksums and its own verification script. See
[submission/README.md](../submission/README.md) for exact contents and rebuilding.
The source ZIP does not contain all datasets, full predictions, or the main
English manuscript. The manuscript is maintained separately in
`ACCESS_latex_template_20260513`.
