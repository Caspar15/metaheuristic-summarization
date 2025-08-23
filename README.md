# Summarization Lab (MVP)

A minimal, runnable extractive summarization pipeline (English) with:
- Preprocess and sentence splitting
- Features: TF, TF-ISF (sentence-level ISF), length, position
- Greedy sentence selection with simple redundancy removal
- ROUGE-1/2/L evaluation and CSV report

## Quickstart

- Create venv and install deps
  - Linux/macOS: `python -m venv .venv && . .venv/bin/activate`
  - Windows (PowerShell): `python -m venv .venv; . .venv\\Scripts\\Activate.ps1`
  - Install: `pip install -r requirements.txt`
- First-time NLTK data
  - `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`
- Optional spaCy English model: `python -m spacy download en_core_web_sm`

## One-command run
- Linux/macOS: `bash scripts/run_all.sh`
- Windows (PowerShell): `./scripts/run_all.ps1`

Outputs are written to `runs/<timestamp>/metrics.csv` and `runs/<timestamp>/predictions.jsonl`.

## Notes
- Default language is English. TODO: add Chinese (jieba/pkuseg).
- All CLI entries support `--config` and manage paths via YAML.
- The pipeline auto-creates required folders.
- Fails fast with clear messages if files/fields are missing.

