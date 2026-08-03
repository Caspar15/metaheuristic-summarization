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
- **Generated**: 2026-08-01; **de-lexicalized and extended**: 2026-08-03
  (PR #11 review, see "De-lexicalization" below)
- **Fixture file SHA-256**:
  `dca3bae4f93a1bb51fe0b66ed5c23367988fcc51a0a425a397a87d24c38bd848`
  (checked at test-collection time by `tests/test_baselines_random.py`;
  see `_assert_fixture_integrity` there — if you edit this fixture, update
  both the file and that constant together, or the check fails loud)
- **Sentence and reference text is placeholder text**, not the original
  Multi-News content — see "De-lexicalization" below for why and exactly
  how. Every other field (ids, structure, counts, `data_fingerprint`) is
  either copied from or recomputed against the source row; nothing here is
  a blind "first N rows" or byte-for-byte copy of copyrighted text.

### De-lexicalization — why the sentence text was replaced

This is a **public** repository with no `LICENSE` file, Multi-News' own
license is listed as `"other"` on its dataset card (not a permissive
license that clearly covers redistributing article/summary text), and
IEEE Access reviewers will have access to this repo during review. Keeping
15-16 verbatim news articles and their reference summaries checked into
version control indefinitely is an avoidable redistribution risk this
fixture does not need to take on, since none of the four tests that use
it (`test_per_row_independence_single_row_vs_full_file`,
`test_real_data_sample_is_feasible_with_min_words_not_applied`,
`test_validation_4576_succeeds_with_apply_min_words_false`,
`test_fixture_sha256_matches_recorded_value`, in
`tests/test_baselines_random.py`) reads sentence *content* at all — they
only depend on **word counts** (`src.utils.tokenizer.count_tokens` is a
plain `text.split()`), **sentence/document counts and structure**,
**U+FFFD presence/count**, and the derived `min_words_relaxed`/
`source_capacity_words` facts, none of which depend on which specific
words are in a sentence.

**⚠️ This fixture must not be used for any test that depends on actual
word content** — e.g. a future ROUGE-score test, a test asserting
something about specific vocabulary, or anything reading `references` as
real summary text. If a test genuinely needs real article/summary text,
it needs its own fixture (with its own licensing review), not this one.

**What changed, precisely** (every whitespace-delimited token, across all
sentence `text` fields and all `references` entries, in document order —
documents by `source_order`, then sections, then sentences, then
`references` — assigned a running per-row index `i` starting at 0):

- Every token is replaced by `f"w{i:06d}"`.
- If the *original* token contained one or more U+FFFD replacement
  characters, that many U+FFFD characters are appended to its replacement
  token — so U+FFFD **count is preserved exactly**, and **which token
  slot** (not necessarily which raw character offset) carries it is also
  preserved.
- Tokens are rejoined with a single space. Only token *count* is a tested
  property (via `.split()`), never the original inter-token whitespace, so
  this is not a loss of anything the tests check.
- **Unchanged as a result**: token count per sentence (hence per-sentence
  and per-document word counts), sentence count per document, document
  count per row, `min_words_relaxed`/`source_capacity_words` (both are
  pure functions of per-sentence word counts), and U+FFFD count per row.
  Verified for all 16 rows after the transform: `source_capacity_words`
  and `min_words_relaxed` are bit-identical to the pre-transform values.
- `data_fingerprint` **is recomputed** after the transform
  (`src.data.schemas.compute_data_fingerprint` over every field except
  itself) — `validate_document_example` checks this field against content,
  so a stale fingerprint would fail loud the moment any test touched a row.
- **Left stale, deliberately, not recomputed**: per-sentence/per-document
  `raw_sentence_sha256`/`raw_document_sha256`/`raw_cluster_sha256` and
  `document_char_start`/`document_char_end` in each sentence's `metadata`.
  These describe the *original* raw (pre-canonicalization) text for
  provenance/audit purposes; no code path validates them against the
  current `text` field, so they are left pointing at the original content's
  hashes/offsets rather than recomputed against placeholder text that would
  make them meaningless either way. Do not treat them as hashes of what is
  actually in this file.
- **Rows are no longer byte-for-byte copies** of the source file (unlike
  the original 2026-08-01 version of this fixture) — each row went through
  a `json.loads`/transform/`json.dumps` round-trip, so JSON key order may
  differ from the source even where content is unchanged.

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

So the fixture is a **deliberately curated 15-row sample** (16 as of
2026-08-03, see `validation_4576` at the end of the table below), not a
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
| `validation_4576`| 4576| `apply_min_words=False` regression | added 2026-08-03 (PR #11 review, "Blocking 1"): `source_capacity_words=244`, `min_words_relaxed=False`, 7 eligible sentences of word lengths `[45, 22, 72, 12, 22, 83, 7]`. Under the skip-tolerant `_select_random` selector this is the most fragile row found in a full-split rerun (`scripts/audit/random_baseline_min_words.py`) -- it fails to reach the 200-word floor under **all four** tested base seeds (0, 1, 42, 9999), because the 244-word optimum requires keeping *both* long sentences (72 and 83 words) while dropping the two shortest, a combination one random walk rarely lands on. Pinned by `tests/test_baselines_random.py`'s `test_validation_4576_succeeds_with_apply_min_words_false` to regression-guard the `apply_min_words=False` decision itself, not just document it. |

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

There is no checked-in generator script (the exact row indices/order and
the de-lexicalization transform above are the full specification); to
reproduce:

1. Read `data/processed/multi_news_validation_canonical.jsonl` line-by-line
   (0-indexed) and collect rows, in this exact order:

   ```
   90, 51, 64, 153, 47, 26, 37, 161, 538, 308, 701, 423, 1110, 930, 862, 4576
   ```

   (the same order as the table above -- typical rows, then the two
   `min_words_relaxed` rows, then the two U+FFFD rows, then the three
   single-document rows, then `validation_4576`).
2. Apply the de-lexicalization transform described under
   "De-lexicalization" above to each row independently (its own running
   token counter starting at 0), then recompute `data_fingerprint`.
3. Write the transformed rows out in the same order, one JSON object per
   line (`json.dumps(row, ensure_ascii=False)`), `\n` line endings.
4. Verify the result's SHA-256 matches the value recorded above before
   replacing the committed fixture.
