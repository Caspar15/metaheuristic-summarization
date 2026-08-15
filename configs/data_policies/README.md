# Frozen data policies

Current role decision (checked 2026-08-15): GovReport is the sole primary
quality domain, Multi-News is frozen boundary evidence, and no new dataset is
authorized. The validation policies below remain immutable historical data
contracts. E1-E3 are complete, but a GovReport official-test policy intentionally
does not yet exist. The frozen contracts disagree on whether policy materialization
or full-author signature comes first (F-76); no test membership or payload may be
read until the authors approve a non-circular, two-stage authorization order.

`multinews_validation_v1.json` is the pre-result data contract for the Phase 1
Multi-News validation pilot. It binds each allowed analysis to an exact row
count, canonical content fingerprint, file SHA-256, source revision, U+FFFD
count, and tracked row manifest.

- `main`: 5,621 structurally valid rows. The 72 U+FFFD rows are retained
  unchanged; text repair is forbidden.
- `clean_sensitivity`: 5,549 rows. It excludes exactly the 72 IDs in
  `multinews_validation_replacement_rows_v1.jsonl`; no other filtering or text
  repair is allowed.
- Canonical source row 4850 is excluded from both analyses because its source
  cluster is empty. The ignored local exclusion manifest is also SHA-checked.

Regenerate only the ignored clean artifact and verify it against the tracked
policy/manifest from the repository root:

```bash
python -m src.data.freeze_multinews_policy
```

The default command never rewrites the tracked policy or manifest. The
`--initialize_policy` switch is reserved for creating a new version before any
scores are observed; it must not be used as an ordinary setup command.

Do not regenerate or edit the policy after observing validation scores merely
to select a more favorable row set. A justified policy change requires a new
versioned policy ID, manifest, fingerprints, and an explicit research note.

Status checked on 2026-08-02: this same policy is enforced by both the proposed
pipeline and the PR #10 Lead baseline CLI. Adding a baseline does not authorize
a new subset or a rewritten policy; formal baseline outputs must carry the same
preflight identity as the system run they are compared with.

`govreport_validation_v1.json` is the corresponding pre-result contract for
the official author archive, not a flattened third-party mirror. The official
validation membership has 974 reports (362 CRS + 612 GAO). One pinned CRS row,
`98-228`, has an empty official reference and is excluded without fabricating a
target, leaving 973 canonical evaluation rows. The policy binds the official
archive SHA-256, validation-ID checksums, canonical artifact identity, empty
U+FFFD manifest, section/paragraph preservation rule, and CC-BY-4.0 license.

GovReport preprocessing reads validation membership only. Its streaming tar
pass can encounter other archive member names, but does not read test
membership or test payload bytes. Do not change the policy or inspect the test
split before the human freeze decision.

`govreport_centered_repositioning_v2.json` is an additive **role/claim-policy
addendum**, not a replacement canonical-data policy.  It records the
2026-08-10 author-side decision to use GovReport as the sole primary quality
domain and Multi-News as a frozen boundary condition.  It pins the v1
validation policies, manifests, length policy, D3b evidence, and final dev
candidate.  It does not change any row, split, or existing result, and it keeps
all protected splits locked until evidence completion and full human sign-off.
