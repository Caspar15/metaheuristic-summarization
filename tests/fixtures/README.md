# tests/fixtures/ provenance

`data/` is `.gitignore`d and not present in CI, so any test that wants a
"real data, not a synthetic toy case" sample must check a small, versioned
copy into `tests/fixtures/` instead of reading `data/processed/` directly.
This file records where each such fixture came from and how it was chosen,
so it can be verified or regenerated without guessing.

## `multi_news_validation_diagnostic_sample.jsonl`

- **Source file**: `data/processed/multi_news_validation_canonical.jsonl`
- **Source file SHA-256**:
  `24c23be3e61a9d76fe396ff4620474ff45cbd4c2dd46ee7efa610c7c70b6eef1`
- **Source `preprocessor_version`**: `multinews-canonical-v3`
  (`src.data.preprocess_multinews.PREPROCESSOR_VERSION`)
- **Generated**: 2026-08-01
- **Fixture file SHA-256**:
  `33018d8e1b9b6f4ab3b843f51f9d138a3ab52efef8da18fa5818d6a05d00aa04`
  (checked at test-collection time by `tests/test_baselines_random.py`;
  see `_assert_fixture_integrity` there — if you edit this fixture, update
  both the file and that constant together, or the check fails loud)
- **Rows are copied byte-for-byte** from the source file (no
  `json.loads`/`json.dumps` round-trip), so each row is bit-identical to
  its original canonical entry.

### Why this is not a blind "first N rows" sample

The original plan was to take the first 20 rows of the validation split.
Checked before committing to that (`src.objectives.evaluator.
resolve_effective_min_words` with `max_words=250, min_words=200`, plus a
scan for U+FFFD and for single-source-document rows): **all 20 of the first
rows are "clean"** -- multi-document, `min_words_relaxed=False`, no
replacement characters. A fixture built that way would claim to test
against "the real data distribution" while actually only ever exercising
the easy case, which defeats the reason this fixture exists. A full-file
scan (5,621 rows) found:

| property | count in full validation split |
|---|---|
| `min_words_relaxed=True` (source capacity < requested 200 words) | 72 |
| contains a U+FFFD replacement character | 72 |
| single source document (`len(doc["documents"]) == 1`) | 58 |

So the fixture is a **deliberately curated 15-row sample**, not a
random or head-N one: a handful of ordinary rows for baseline coverage,
plus at least a couple of rows from each of the three properties above.
Rows were picked as the smallest (by raw line byte size) among the first
few occurrences of each property in the file, purely to keep this fixture
small; that selection has no bearing on the Random baseline's behavior.

| id (== original row index) | original index | group | why selected |
|---|---|---|---|
| `validation_90`  | 90  | typical | ordinary 2-document row, `min_words_relaxed=False`, no U+FFFD -- baseline coverage |
| `validation_51`  | 51  | typical | same as above |
| `validation_64`  | 64  | typical | same as above |
| `validation_153` | 153 | typical | same as above |
| `validation_47`  | 47  | typical | same as above |
| `validation_26`  | 26  | typical | same as above |
| `validation_37`  | 37  | typical | same as above |
| `validation_161` | 161 | typical | 6-source-document row (the other 7 "typical" rows are all 2-document) -- covers a wider fan-in than the rest of the sample |
| `validation_538` | 538 | `min_words_relaxed` | `source_capacity_words=51`, smallest-capacity relaxed row found among the first 10 occurrences -- exercises F-16's 72-row class |
| `validation_308` | 308 | `min_words_relaxed` | `source_capacity_words=106`, second-smallest found -- a second, distinct relaxed row |
| `validation_701` | 701 | U+FFFD | smallest-byte row containing U+FFFD among the first 10 occurrences |
| `validation_423` | 423 | U+FFFD | second-smallest U+FFFD row found |
| `validation_1110`| 1110| single source document | 2 sentences -- smallest single-document row found |
| `validation_930` | 930 | single source document | 3 sentences |
| `validation_862` | 862 | single source document | 6 sentences |

**Overlap, stated honestly rather than glossed over**: the id happens to
equal the original 0-indexed row number in the source file for this
dataset (`build_document_example`'s `example_id` convention), which is why
the table above lists them as the same value. Also, all three
"single source document" rows independently turned out to *also* be
`min_words_relaxed=True` (`source_capacity_words` 31/70/133, all below the
200-word floor) -- not a selection mistake, but the expected structural
consequence of a single short news article having much less raw text than
a 2-8-document Multi-News cluster. So 5 of the 15 rows are
`min_words_relaxed=True` in total (the 2 chosen for that property plus
these 3), and none of the 15 rows is both `min_words_relaxed=True` and
contains U+FFFD at the same time (that combination was not searched for
separately; if a future test needs it, it is not yet covered here).

### Regenerating

There is no checked-in generator script (the exact row indices and order
below are the full specification); to reproduce: read
`data/processed/multi_news_validation_canonical.jsonl` line-by-line (0
indexed) and write out, in this exact order, with `\n` line endings and no
re-encoding:

```
90, 51, 64, 153, 47, 26, 37, 161, 538, 308, 701, 423, 1110, 930, 862
```

(the same order as the table above -- typical rows, then the two
`min_words_relaxed` rows, then the two U+FFFD rows, then the three
single-document rows). Verify the result's SHA-256 matches the value
recorded above before replacing the committed fixture.
