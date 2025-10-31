from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import SummaryRequest, SummaryResponse
from .service import SummarizationService


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Metaheuristic Summarization API",
    version="0.1.0",
    description="REST API wrapper for the metaheuristic summarization pipeline",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = SummarizationService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: SummaryRequest) -> SummaryResponse:
    try:
        return service.run(request)
    except ValueError as exc:
        logger.warning("Bad request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failures
        logger.exception("Unexpected error during summarization")
        raise HTTPException(status_code=500, detail="Summarization failed") from exc


__all__ = ["app"]
