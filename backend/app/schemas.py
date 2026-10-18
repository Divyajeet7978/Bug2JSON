from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated

Category = Literal[
    "authentication",
    "ui",
    "api",
    "database",
    "performance",
    "validation",
    "integration",
    "security",
    "deployment",
    "compatibility",
    "other",
    "unknown",
]
Severity = Literal["critical", "high", "medium", "low", "unknown"]
Priority = Literal["P1", "P2", "P3", "P4", "unknown"]
Reproducible = bool | Literal["unknown"]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class BugAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Category
    severity: Severity
    priority: Priority
    environment: NonEmptyText
    reproducible: Reproducible
    probable_cause: NonEmptyText
    suggested_action: NonEmptyText


class AnalyzeRequest(BaseModel):
    report: Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=4000)]
    mode: Literal["base", "fine_tuned"] = "base"


class AnalyzeResponse(BaseModel):
    analysis: BugAnalysis
    mode: Literal["base", "fine_tuned"]
    model_name: str
    elapsed_ms: int = Field(ge=0)


class ServiceStatus(BaseModel):
    status: Literal["ok"]
    base_model: str
    adapter_available: bool
    adapter_path: str
