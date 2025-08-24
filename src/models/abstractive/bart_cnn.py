from typing import List


def bart_summarize(texts: List[str], max_length: int = 128, device: str | None = None) -> List[str]:
    try:
        from transformers import BartForConditionalGeneration, BartTokenizer
        import torch

        model_name = "facebook/bart-large-cnn"
        tok = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        outs = []
        for t in texts:
            inputs = tok([t], return_tensors="pt", truncation=True, max_length=1024).to(device)
            summary_ids = model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
            outs.append(tok.decode(summary_ids[0], skip_special_tokens=True))
        return outs
    except Exception as e:
        raise RuntimeError("需要安裝 transformers/torch 才能使用 BART") from e

