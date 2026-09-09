# Repository audit — 2026-09-09

> This records the audit before release preparation. The subsequent MIT license,
> English supplement, numerical artifact and final manuscript checks are in
> [submission/README.md](../../submission/README.md) and
> [release_checks.json](../../submission/release_checks.json). Pending distribution
> items below describe the state at this audit checkpoint.

The local research repository has been reviewed and its reviewer entry points,
reproduction instructions and automated evidence checks updated. The final
summarization implementation, configurations, predictions and score artifacts
were not changed. This is a repository audit, not a new benchmark experiment or
a guarantee of journal acceptance.

## Scope and changes

- Inspected the 3,509-file tracked inventory, source/test organization, final
  configurations, execution freezes, result/evidence entry points, dependencies,
  CI and research documentation. Existing regression tests cover data contracts,
  candidate provenance, fusion/selection, evaluator behavior and governed runners.
- Added an English reproduction guide and a clear current-method README entry.
  Marked Phase 1 examples and legacy diagnostics as historical; corrected the
  misleading statement that every audit script uses internal Python ROUGE.
- Updated the research hub and action plan to reflect completion of the English
  manuscript/PDF/source package. Retained dated research history and the
  professor-specified biography decision.
- Added a standard-library snapshot verifier for 217 research source,
  configuration and compact evidence files, covering both datasets. It exports
  the 18 official corpus score rows to CSV/JSON without loading protected data,
  models, predictions, or rerunning evaluation. Paired statistics retain their
  own original analyses and are not inferred from rounded corpus scores.
- Added eight regression cases for inventory integrity, missing/tampered files,
  checkout line endings, unsafe paths and final populations. Added the snapshot,
  GovReport freeze and provenance checks to the existing unit-test workflow.
- Pinned final YAML checkout line endings. The snapshot normalizes CRLF only for
  source/configuration text and retains raw-byte hashes for result evidence.
- Recorded the historical core version constraints and a separate 89-package
  Windows/Python 3.12.7 research/test dependency closure. This closure satisfies
  installed metadata requirements. The full existing environment snapshot is
  retained for transparency, including unrelated optional packages.

## Verification evidence

| Check | Result |
|---|---|
| Original full regression | 496 tests + 5 subtests passed |
| Regression including new snapshot checks | 504 tests + 5 subtests passed |
| Python compilation | `src`, `scripts`, `tests`, `backend` passed |
| Strict local GovReport freeze verification | Passed; 11 local artifacts checked, none deferred |
| Versioned provenance records | 468 documented CRLF-era pins; zero unexplained hash failures |
| Snapshot verifier / main quality export | 217 files checked; 18 rows from the two final analyses |
| Optional frontend | TypeScript/Vite production build and ESLint passed |
| Tracked inventory hygiene | No file over 5 MiB; no matches for the checked private-key/GitHub-token/AWS-access-key patterns |
| Structured file scan | Research JSON parsed; the two frontend `tsconfig` files use valid JSON-with-comments syntax, not strict JSON |

The initial system-Python attempt lacked the project's dependencies. Tests in the
existing `.venv` then encountered Windows sandbox temporary-directory/process
restrictions; rerunning the same suite with the required process permissions
passed. These were environment failures, not skipped tests. The frontend's first
esbuild attempt similarly encountered `spawn EPERM`; the permitted build passed.

The entire local environment's `pip check` reports an optional `thinc`/NumPy
conflict. The 89-package research/test dependency closure excludes spaCy/thinc and
has zero metadata conflicts. No scientific package was upgraded to address an
unrelated optional dependency. The lock records installed versions, not wheel
hashes; fresh installation and cross-platform package resolution are not claimed.

## Clean source snapshot and limits

A separate directory was built from `git archive HEAD` plus the explicit local
review edits. It contains no `.git`, `.venv`, processed datasets or bulk
predictions. This tests the source-download case while reusing the existing Python
environment. It is not a network clone or a fresh dependency installation.

The source-only GovReport freeze check reports 11 deliberately absent untracked
artifacts as deferred; the full local strict check verifies them. Neither result
means full benchmark reproduction from public Git alone. The source-only
snapshot and regression outcomes are recorded in the completion addendum below.

### Completion addendum

The clean source snapshot passed all three evidence commands. The 217-file
inventory and 18-row export passed; the GovReport source-only validator passed
with the expected 11 deferred local artifacts; provenance again reported 468
documented legacy pins and zero failures. The complete source-snapshot test run
finished with **502 passed, 2 skipped and 5 subtests passed**. The two pre-existing
skips explicitly require the undistributed GovReport/Multi-News canonical test
files; both tests passed in the full local 504-test run. No new skip was added.

The first clean-snapshot test attempt used a custom pytest temporary path whose
parent did not exist. Correcting that invocation resolved the setup errors; it
did not require a source or test change. The initial source inventory also caught
Git checkout newline conversion in configuration text; explicitly using the
existing LF-normalized text identity resolved that portability issue without
changing any configuration value or frozen result.

## Remaining distribution work

- Publish an immutable public code version and an appropriate bulk artifact;
  verify the public branch/tag and manuscript availability statement agree.
- Choose software licensing with the rights holders. No license was invented and
  no dataset/model redistribution rights are implied.
- Fresh-install the pinned research environment on the intended reproduction
  host and provision the externally distributed data/model/Perl resources for
  full output reproduction, preserving the frozen scientific configuration.

The local base is `af51e1e15a41696f2b8776d32b63b3b7704a1cbc` on
`research/govreport-centered-freeze`, two commits ahead of the cached upstream
before these edits. No remote synchronization, branch rename, release tag, DOI,
software license or benchmark rerun is claimed by this audit. The manuscript
and its English Supplementary Material/publication form tasks remain distinct
from the repository's source checks.
