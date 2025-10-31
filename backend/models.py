from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Stage1BaseMethod = Literal["greedy", "grasp", "nsga2"]
Stage1LLMMethod = Literal["bert", "roberta", "xlnet"]
Stage2Method = Literal["fast", "fast_grasp", "fast_nsga2", "greedy"]
LengthUnit = Literal["sentences", "tokens"]


class Stage1Config(BaseModel):
    algorithms: List[Stage1BaseMethod] = Field(..., min_length=1)
    llms: List[Stage1LLMMethod] = Field(default_factory=list)
    candidate_k: int = Field(default=25, ge=1)


class Stage2Config(BaseModel):
    method: Stage2Method = Field(default="fast")
    union_cap: int = Field(default=15, ge=1)


class LengthControl(BaseModel):
    unit: LengthUnit
    max_sentences: Optional[int] = Field(default=None, ge=1)
    max_tokens: Optional[int] = Field(default=None, ge=10)

    @model_validator(mode="after")
    def validate_length(self) -> "LengthControl":
        if self.unit == "sentences":
            if self.max_sentences is None:
                raise ValueError("max_sentences is required when unit='sentences'")
            self.max_tokens = None
        else:
            if self.max_tokens is None:
                raise ValueError("max_tokens is required when unit='tokens'")
            self.max_sentences = None
        return self


class SummaryRequest(BaseModel):
    documents: List[str] = Field(..., min_length=1)
    stage1: Stage1Config
    stage2: Stage2Config
    length_control: LengthControl
    reference: Optional[str] = None


class StageSummary(BaseModel):
    method: str
    summary: str
    selected_indices: List[int]
    sentences: List[str]


class SummaryResponse(BaseModel):
    summary: str
    sentences: List[str]
    selected_indices: List[int]
    candidate_indices: List[int]
    stage1_base: StageSummary
    stage1_llms: List[StageSummary] = Field(default_factory=list)
    metrics: Optional[Dict[str, float]] = None
    timing: Dict[str, float]

