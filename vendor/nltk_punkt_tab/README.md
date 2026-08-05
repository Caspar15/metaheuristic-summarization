# Vendored NLTK `punkt_tab/english` data

## What this is

`tokenizers/punkt_tab/english/` (4 files, 244 KB) — the exact NLTK resource
`sumy`'s `Tokenizer("english")` needs. Vendored, not downloaded, so CI, this
machine, and an offline compute cluster all run against the same bytes
without any of the three needing network access.

```
vendor/nltk_punkt_tab/tokenizers/punkt_tab/english/
    sent_starters.txt
    abbrev_types.txt
    ortho_context.tab
    collocations.tab
```

## Why this exact directory shape

`nltk.data.find`/`nltk.data.load` resolve a resource name
(`tokenizers/punkt_tab/english/`) as a path *relative to* whatever root is on
`nltk.data.path` — the root's own name is never inspected, only the relative
subpath under it. So `tokenizers/punkt_tab/english/` must be reproduced
byte-for-byte as a directory structure under this vendor root; a missing or
extra directory level makes `find()` fail (it looks for a directory, since
the resource name ends in `/`). This is not cosmetic organization, it is the
literal lookup path NLTK computes.

The vendor root itself is deliberately **not** named `nltk_data` anywhere in
its path: `.gitignore` already has a bare `nltk_data/` rule (for local,
regenerable caches), which matches a directory of that name at any depth.
Naming this `vendor/nltk_punkt_tab/` avoids that collision entirely —
verified with `git check-ignore -v` on every file and every directory level
before committing.

## Why `punkt_tab`, not the classic `punkt` pickle

`sumy`'s own source (`sumy/nlp/tokenizers.py`) literally calls
`nltk.data.load("tokenizers/punkt/%s.pickle" % language)` — the old, pickle-
based path. But on this project's pinned `nltk==3.10.0`, NLTK 3.9+ ships a
`switch_punkt` compatibility shim in `nltk/data.py` that intercepts that
exact call and redirects it to `nltk.tokenize.punkt.PunktSentenceTokenizer
.load_lang`, which in turn calls `find("tokenizers/punkt_tab/%s/" % lang)` —
confirmed directly from a real CI traceback (GitHub Actions runner, clean
checkout, no local NLTK cache):

```
sumy/nlp/tokenizers.py:203   nltk.data.load("tokenizers/punkt/english.pickle")
nltk/data.py:1126            switch_punkt(fil)
nltk/tokenize/punkt.py:1769  find("tokenizers/punkt_tab/english/")
```

The classic `punkt` pickle is **never read** on this nltk version — only
`punkt_tab` is. This was previously verified experimentally (constructing a
`Tokenizer("english")` with only `punkt_tab` cached, no `punkt` pickle
present, succeeded); the CI traceback above confirms it from a real failure
on a genuinely clean machine, not just a local experiment.

## Coupling to the `nltk` version pin

`requirements.txt` pins `nltk==3.10.0` exactly, not a floor. This is not
just reproducibility — it is a precondition for this vendored data being
correct at all: `nltk` 3.8.x reads the classic `punkt` pickle path directly
(no `switch_punkt` shim), so a 3.8.x install would need the pickle instead
and would not even look for `punkt_tab`. If `nltk` is ever upgraded, re-run
the regeneration steps below and re-verify against a real clean-environment
run (see `docs/research/COMPUTE_ENVIRONMENT.md`) before assuming this
directory is still sufficient.

## How to regenerate (e.g. after an `nltk` version bump)

On a machine with network access:

```bash
python3 -c "import nltk; nltk.download('punkt_tab')"
# lands in ~/nltk_data/tokenizers/punkt_tab/ by default

mkdir -p vendor/nltk_punkt_tab/tokenizers/punkt_tab
cp -r ~/nltk_data/tokenizers/punkt_tab/english \
      vendor/nltk_punkt_tab/tokenizers/punkt_tab/
```

Only the `english/` subdirectory is vendored — `punkt_tab` ships ~19 other
languages and a `README`, none of which this project's TextRank/LexRank
baselines (English-only) need. Verified: constructing
`sumy.nlp.tokenizers.Tokenizer("english")` and running both `TextRankSummarizer`
and `LexRankSummarizer` succeeds with only `english/` present.

## Who actually needs this

Only `src/baselines/centrality.py` (the TextRank/LexRank baselines), via
`sumy.nlp.tokenizers.Tokenizer`. This project's own canonical sentence
splitter (`src/data/sentence_split.py`, used by Multi-News preprocessing and
ROUGE-Lsum evaluation) builds an **untrained** `PunktSentenceTokenizer`
directly from a manually maintained abbreviation list — it never calls
`nltk.data.load`/`nltk.data.find` at all, and needs neither `punkt` nor
`punkt_tab`. Confirmed by running it with `nltk.data.path` pointed at an
empty directory and sockets blocked (see
`docs/research/COMPUTE_ENVIRONMENT.md`). This vendored data is therefore
scoped to the sumy-based baselines, not a precondition for the rest of the
pipeline (Multi-News/GovReport preprocessing, evaluation).
