from __future__ import annotations

from typing import List, Dict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


def _safe_stopwords(lang: str = "english") -> set[str]:
    try:
        return set(stopwords.words(lang))
    except LookupError:
        # Fallback: empty set if stopwords not available
        return set()


def split_sentences(text: str, lang: str = "en") -> List[str]:
    """Split a paragraph into sentences using NLTK (English default)."""
    # For English, NLTK's default works well
    return [s.strip() for s in sent_tokenize(text)]


def tokenize_words_en(text: str, lowercase: bool = True, remove_stop: bool = True) -> List[str]:
    """Tokenize English text into words using NLTK.

    Applies optional lowercasing and stopword removal.
    """
    tokens = word_tokenize(text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    # Basic cleaning: keep alphabetic/numeric tokens
    tokens = [t for t in tokens if any(ch.isalnum() for ch in t)]
    if remove_stop:
        sw = _safe_stopwords("english")
        if sw:
            tokens = [t for t in tokens if t not in sw]
    return tokens


def process_article(
    text: str,
    lang: str = "en",
    lowercase: bool = True,
    remove_stopwords: bool = True,
) -> List[Dict[str, object]]:
    """Process raw article text into sentence structures.

    Returns list of: {"text": raw_sentence, "tokens": [...], "token_count": int}
    """
    if lang != "en":
        # TODO: add Chinese tokenizers (jieba/pkuseg) if needed
        raise ValueError(f"Unsupported lang for MVP: {lang}")
    sentences = split_sentences(text, lang=lang)
    out: List[Dict[str, object]] = []
    for s in sentences:
        toks = tokenize_words_en(s, lowercase=lowercase, remove_stop=remove_stopwords)
        out.append({"text": s, "tokens": toks, "token_count": len(toks)})
    return out


# --- Compatibility wrappers for existing preprocess code ---
def clean_text(text: str) -> str:
    """Basic cleaning placeholder. MVP: strip only."""
    return text.strip()


def split_into_sentences(text: str, lang: str = "en", sent_tokenizer: str = "nltk") -> List[str]:
    """Wrapper matching older API; uses NLTK for English."""
    if lang != "en":
        raise ValueError(f"Unsupported lang for MVP: {lang}")
    return split_sentences(text, lang=lang)


def tokenize_sentence(
    text: str,
    lang: str = "en",
    word_tokenizer: str = "nltk",
    lowercase: bool = True,
    remove_stopwords: bool = True,
) -> List[str]:
    """Wrapper matching older API; uses NLTK word tokenizer for English."""
    if lang != "en":
        raise ValueError(f"Unsupported lang for MVP: {lang}")
    return tokenize_words_en(text, lowercase=lowercase, remove_stop=remove_stopwords)
