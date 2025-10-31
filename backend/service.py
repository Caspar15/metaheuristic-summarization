from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from src.data.preprocess import simple_sentence_split
from src.eval.rouge import rouge_scores
from src.pipeline.select_sentences import summarize_one
from src.utils.io import load_yaml

from .models import (
    LengthControl,
    StageSummary,
    SummaryRequest,
    SummaryResponse,
    Stage1BaseMethod,
    Stage1LLMMethod,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

BASE_STAGE1_CONFIG = ROOT_DIR / "configs/stage1/base/k10.yaml"
LLM_STAGE1_CONFIGS: Dict[Stage1LLMMethod, Path] = {
    "bert": ROOT_DIR / "configs/stage1/llm/bert/k10.yaml",
    "roberta": ROOT_DIR / "configs/stage1/llm/roberta/k10.yaml",
    "xlnet": ROOT_DIR / "configs/stage1/llm/xlnet/k10.yaml",
}
STAGE2_CONFIG = ROOT_DIR / "configs/stage2/fast/3sent.yaml"


class SummarizationService:
    """
    High-level orchestration of Stage1 (base + LLM) and Stage2 pipelines.
    """

    def __init__(self) -> None:
        if not BASE_STAGE1_CONFIG.exists():
            raise FileNotFoundError(f"Missing base config: {BASE_STAGE1_CONFIG}")
        if not STAGE2_CONFIG.exists():
            raise FileNotFoundError(f"Missing stage2 config: {STAGE2_CONFIG}")

        self.stage1_base_template = load_yaml(BASE_STAGE1_CONFIG)
        self.stage1_llm_templates = {
            name: load_yaml(path) for name, path in LLM_STAGE1_CONFIGS.items()
        }
        self.stage2_template = load_yaml(STAGE2_CONFIG)

    def run(self, request: SummaryRequest) -> SummaryResponse:
        sentences = self._collect_sentences(request.documents)
        if not sentences:
            raise ValueError("Input documents do not contain any sentences.")

        timing: Dict[str, float] = {}

        # Stage1 - Base algorithm (mandatory)
        base_method = request.stage1.algorithms[0]
        t0 = time.perf_counter()
        base_cfg = self._prepare_stage1_base_config(base_method, request.stage1.candidate_k)
        base_result = summarize_one(
            {"id": "doc", "sentences": sentences, "highlights": ""},
            copy.deepcopy(base_cfg),
        )
        timing["stage1_base_seconds"] = time.perf_counter() - t0

        base_indices = self._clamp_indices(
            base_result.get("selected_indices", []),
            len(sentences),
        )
        base_selected_sentences = [sentences[i] for i in base_indices]
        base_summary = StageSummary(
            method=base_method,
            summary=base_result.get("summary", " ".join(base_selected_sentences)),
            selected_indices=base_indices,
            sentences=base_selected_sentences,
        )

        # Stage1 - LLM encoders (optional)
        llm_summaries: List[StageSummary] = []
        llm_indices: List[int] = []
        if request.stage1.llms:
            for llm in request.stage1.llms:
                if llm not in self.stage1_llm_templates:
                    continue
                t_llm = time.perf_counter()
                llm_cfg = self._prepare_stage1_llm_config(llm, request.stage1.candidate_k)
                llm_result = summarize_one(
                    {"id": "doc", "sentences": sentences, "highlights": ""},
                    copy.deepcopy(llm_cfg),
                )
                timing[f"stage1_{llm}_seconds"] = time.perf_counter() - t_llm

                llm_sel = self._clamp_indices(
                    llm_result.get("selected_indices", []),
                    len(sentences),
                )
                llm_sentences = [sentences[i] for i in llm_sel]
                llm_indices.extend(llm_sel)
                llm_summaries.append(
                    StageSummary(
                        method=llm,
                        summary=llm_result.get("summary", " ".join(llm_sentences)),
                        selected_indices=llm_sel,
                        sentences=llm_sentences,
                    )
                )

        # Union candidates
        union_order = self._build_union(base_indices, llm_indices, request.stage2.union_cap)
        if not union_order:
            union_order = base_indices[: request.stage2.union_cap]
        candidate_sentences = [sentences[i] for i in union_order]

        # Stage2
        stage2_cfg = self._prepare_stage2_config(request.stage2.method, request.length_control)
        t2 = time.perf_counter()
        stage2_result = summarize_one(
            {"id": "doc", "sentences": candidate_sentences, "highlights": ""},
            copy.deepcopy(stage2_cfg),
        )
        timing["stage2_seconds"] = time.perf_counter() - t2

        selected_candidate_indices = stage2_result.get("selected_indices", [])
        final_indices = [
            union_order[i]
            for i in self._clamp_indices(selected_candidate_indices, len(candidate_sentences))
        ]
        final_sentences = [sentences[i] for i in final_indices]
        summary_text = " ".join(final_sentences) if final_sentences else stage2_result.get("summary", "")

        # Metrics (optional)
        metrics: Optional[Dict[str, float]] = None
        if request.reference:
            metrics = rouge_scores([summary_text], [request.reference])

        return SummaryResponse(
            summary=summary_text,
            sentences=final_sentences,
            selected_indices=final_indices,
            candidate_indices=union_order,
            stage1_base=base_summary,
            stage1_llms=llm_summaries,
            metrics=metrics,
            timing=timing,
        )

    @staticmethod
    def _collect_sentences(texts: Iterable[str]) -> List[str]:
        sentences: List[str] = []
        for text in texts:
            for sent in simple_sentence_split(text or ""):
                if sent:
                    sentences.append(sent)
        return sentences

    @staticmethod
    def _clamp_indices(indices: Iterable[int], max_len: int) -> List[int]:
        out: List[int] = []
        for idx in indices:
            if 0 <= idx < max_len:
                out.append(int(idx))
        return out

    @staticmethod
    def _build_union(
        base_indices: Iterable[int],
        llm_indices: Iterable[int],
        union_cap: int,
    ) -> List[int]:
        ordered: List[int] = []
        seen: Set[int] = set()
        for idx in base_indices:
            if idx not in seen:
                ordered.append(idx)
                seen.add(idx)
            if len(ordered) >= union_cap:
                return ordered[:union_cap]
        for idx in llm_indices:
            if idx not in seen:
                ordered.append(idx)
                seen.add(idx)
            if len(ordered) >= union_cap:
                break
        return ordered[:union_cap]

    def _prepare_stage1_base_config(
        self,
        method: Stage1BaseMethod,
        candidate_k: int,
    ) -> Dict:
        cfg = copy.deepcopy(self.stage1_base_template)
        cfg.setdefault("optimizer", {})["method"] = method
        length_cfg = cfg.setdefault("length_control", {})
        length_cfg["unit"] = "sentences"
        length_cfg["max_sentences"] = int(candidate_k)
        length_cfg.pop("max_tokens", None)
        cand_cfg = cfg.setdefault("candidates", {})
        cand_cfg["use"] = True
        cand_cfg["k"] = int(candidate_k)
        return cfg

    def _prepare_stage1_llm_config(
        self,
        method: Stage1LLMMethod,
        candidate_k: int,
    ) -> Dict:
        template = self.stage1_llm_templates.get(method)
        if template is None:
            raise ValueError(f"Unsupported LLM method: {method}")
        cfg = copy.deepcopy(template)
        cfg.setdefault("optimizer", {})["method"] = method
        length_cfg = cfg.setdefault("length_control", {})
        length_cfg["unit"] = "sentences"
        length_cfg["max_sentences"] = int(candidate_k)
        length_cfg.pop("max_tokens", None)
        cand_cfg = cfg.setdefault("candidates", {})
        cand_cfg["use"] = True
        cand_cfg["k"] = int(candidate_k)
        return cfg

    def _prepare_stage2_config(
        self,
        method: str,
        length: LengthControl,
    ) -> Dict:
        cfg = copy.deepcopy(self.stage2_template)
        cfg.setdefault("optimizer", {})["method"] = method
        length_cfg = cfg.setdefault("length_control", {})
        length_cfg["unit"] = length.unit
        if length.unit == "sentences":
            length_cfg["max_sentences"] = int(length.max_sentences or 3)
            length_cfg.pop("max_tokens", None)
        else:
            length_cfg["max_tokens"] = int(length.max_tokens or 400)
            length_cfg.pop("max_sentences", None)
        cand_cfg = cfg.setdefault("candidates", {})
        cand_cfg["use"] = False
        cand_cfg["k"] = max(1, cand_cfg.get("k", 0))
        return cfg
