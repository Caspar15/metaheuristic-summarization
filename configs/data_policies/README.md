# Frozen data policies

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
