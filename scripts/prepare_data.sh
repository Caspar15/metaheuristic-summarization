#!/usr/bin/env bash
set -e

python - <<'PY'
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('NLTK resources downloaded.')
except Exception as e:
    print('Warning: NLTK download failed:', e)
PY

# Ensure sample exists (already added by repo); keep as no-op
if [ -f "data/samples/sample_en.jsonl" ]; then
  echo "Sample data present."
else
  echo "Missing data/samples/sample_en.jsonl; please re-pull repo." && exit 1
fi

