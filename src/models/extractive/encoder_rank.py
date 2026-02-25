from typing import List, Optional

from src.utils.tokenizer import count_tokens


def _ensure_imports():
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModel, AutoConfig  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "需要安裝 transformers 與 torch 才能使用編碼器排序"
        ) from e


def _sentence_embeddings(
    sentences: List[str],
    model_name: str = "bert-base-uncased",
    device: Optional[str] = None,
    token: Optional[str] = None,
):
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoConfig

    use_fast = True
    if isinstance(model_name, str) and ("xlnet" in model_name.lower()):
        use_fast = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=use_fast)

    if isinstance(model_name, str) and ("roberta" in model_name.lower()):
        try:
            cfg = AutoConfig.from_pretrained(model_name)
            try:
                cfg.add_pooling_layer = False  # type: ignore[attr-defined]
            except Exception:
                pass
            model = AutoModel.from_pretrained(model_name, token=token, config=cfg)
        except Exception:
            model = AutoModel.from_pretrained(model_name, token=token)
    else:
        model = AutoModel.from_pretrained(model_name, token=token)
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
    a = embs / (embs.norm(dim=1, keepdim=True) + 1e-12)
    b = centroid / (centroid.norm(dim=1, keepdim=True) + 1e-12)
    sims = (a * b).sum(dim=1)
    return sims.tolist()


def encoder_select(
    sentences: List[str],
    max_tokens: int,
    unit: str = "sentences",
    max_sentences: Optional[int] = 3,
    model_name: str = "bert-base-uncased",
    device: Optional[str] = None,
) -> List[int]:
    """通用編碼器排序：句向量（均值池化）→ 重心 cosine 排序 → 取前 N。
    不使用 greedy/GRASP/NSGA-II，直接依 `unit`（sentences/tokens）取上限。
    """
    if not sentences:
        return []
    _ensure_imports()
    import torch
    import os
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    with torch.inference_mode():
        embs = _sentence_embeddings(
            sentences, model_name=model_name, device=device, token=hf_token
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
            t = count_tokens(sentences[i])
            if total + t <= budget:
                picked.append(i)
                total += t
    picked.sort()
    return picked


# Backward-compatible alias
bert_select = encoder_select
