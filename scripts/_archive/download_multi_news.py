import os
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
import nltk

# Ensure nltk sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def preprocess_multi_news(split, out_path, limit=None):
    print(f"Loading multi_news split: {split}")
    try:
        # Use parquet version to avoid script loading issues on Windows
        dataset = load_dataset("Awesome075/multi_news_parquet", split=split)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    if limit:
        print(f"Limiting to first {limit} examples")
        dataset = dataset.select(range(limit))

    print(f"Processing {len(dataset)} examples...")
    rows = []
    
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        # Multi-news documents are separated by " ||||| " usually, or just newlines.
        doc_text = example['document']
        # Clean up the separator if present (some versions use |||||)
        doc_text = doc_text.replace("|||||", " ")
        
        sentences = nltk.sent_tokenize(doc_text)
        # Filter very short sentences (noise)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 3]
        
        summary = example['summary']
        # Summary sometimes starts with "- "
        if summary.startswith("- "):
            summary = summary[2:]
            
        rows.append({
            "id": f"{split}_{i}",
            "sentences": sentences,
            "highlights": summary
        })

    print(f"Writing to {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Done.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--limit", type=int, default=None, help="limit number of examples (None for full)")
    args = ap.parse_args()

    # Process ALL splits for a complete dataset
    splits = ["train", "validation", "test"]
    for split in splits:
        preprocess_multi_news(split, os.path.join(args.out_dir, f"multi_news_{split}.jsonl"), args.limit)

if __name__ == "__main__":
    main()
