from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.inference import Analyzer, ModelOutputError, ModelUnavailableError
from app.schemas import AnalyzeRequest, AnalyzeResponse, ServiceStatus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

BASE_MODEL = os.getenv("BUG2JSON_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = Path(
    os.getenv("BUG2JSON_ADAPTER_PATH", str(PROJECT_ROOT / "models" / "bug2json-lora"))
)
if not ADAPTER_PATH.is_absolute():
    ADAPTER_PATH = PROJECT_ROOT / ADAPTER_PATH

analyzer = Analyzer(BASE_MODEL, ADAPTER_PATH)
app = FastAPI(
    title="Bug2JSON API",
    version="0.1.0",
    description="Extract a reviewable JSON summary from a software bug report.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/status", response_model=ServiceStatus)
def service_status() -> ServiceStatus:
    return ServiceStatus(
        status="ok",
        base_model=BASE_MODEL,
        adapter_available=analyzer.adapter_available,
        adapter_path=str(ADAPTER_PATH),
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    started = perf_counter()
    try:
        result, model_name, elapsed_ms = analyzer.analyze(request.report, request.mode)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelOutputError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AnalyzeResponse(
        analysis=result,
        mode=request.mode,
        model_name=model_name,
        elapsed_ms=elapsed_ms or round((perf_counter() - started) * 1000),
    )
