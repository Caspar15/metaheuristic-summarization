from typing import List, Optional


def _ensure_imports():
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModel  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "需要安裝 transformers 與 torch 才能使用 BERT 排序器。"
        ) from e


def _sentence_embeddings(
    sentences: List[str],
    model_name: str = "bert-base-uncased",
    device: Optional[str] = None,
    use_auth_token: Optional[str] = None,
):
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
    model = AutoModel.from_pretrained(model_name, use_auth_token=use_auth_token)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    embs: List[torch.Tensor] = []
    batch_size = 16
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            last = out.last_hidden_state  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            masked = last * mask
            sum_vec = masked.sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            sent_emb = sum_vec / lengths  # mean pooling
            embs.append(sent_emb.detach().cpu())
    import torch as _torch

    return _torch.cat(embs, dim=0)  # [N, H]


def _cosine_scores_to_centroid(embs) -> List[float]:
    import torch

    if embs.size(0) == 0:
        return []
    centroid = embs.mean(dim=0, keepdim=True)  # [1, H]
    # cosine similarity
    a = embs / (embs.norm(dim=1, keepdim=True) + 1e-12)
    b = centroid / (centroid.norm(dim=1, keepdim=True) + 1e-12)
    sims = (a * b).sum(dim=1)
    return sims.tolist()


def _count_tokens(s: str) -> int:
    return len((s or "").split())


def bert_select(
    sentences: List[str],
    max_tokens: int,
    unit: str = "sentences",
    max_sentences: Optional[int] = 3,
    model_name: str = "bert-base-uncased",
    device: Optional[str] = None,
) -> List[int]:
    """使用預訓練 BERT 句向量與全文向量的 cosine 相似度做排名選句。

    不使用 greedy/GRASP/NSGA-II，直接依分數排序取前 N 句或在 token 預算內取高分句。
    """
    if not sentences:
        return []
    _ensure_imports()
    import torch

    import os
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    with torch.inference_mode():
        embs = _sentence_embeddings(
            sentences, model_name=model_name, device=device, use_auth_token=hf_token
        )  # [N, H]
    scores = _cosine_scores_to_centroid(embs)
    order = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)

    u = (unit or "sentences").lower()
    picked: List[int] = []
    if u == "sentences":
        limit = max_sentences if (max_sentences is not None and max_sentences > 0) else len(sentences)
        picked = order[: int(limit)]
    else:  # tokens budget
        budget = int(max_tokens)
        total = 0
        for i in order:
            t = _count_tokens(sentences[i])
            if total + t <= budget:
                picked.append(i)
                total += t
    picked.sort()
    return picked
