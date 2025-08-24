from typing import List


def length_scores(sentences: List[str], clip: int = 40) -> List[float]:
    lens = [len(s.split()) for s in sentences]
    if not lens:
        return []
    # cap overly long sentences and normalize
    lens = [min(x, clip) for x in lens]
    mx = max(lens) or 1
    return [x / mx for x in lens]

