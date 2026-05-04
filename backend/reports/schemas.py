from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["High", "Medium", "Low"]


class RiskFactors(BaseModel):
    distance_factor: str = ""
    ttc_factor: str = ""
    obstacle_factor: str = ""
    repetition_factor: str = ""
    rule_agreement: str = ""


class ValidatedEvent(BaseModel):
    event_id: str
    event_time: str = "-"
    risk_level: RiskLevel
    risk_percent: int = Field(ge=0, le=100)
    judgement_reason: str
    risk_factors: RiskFactors
    session_id: str | None = None
    obstacle_name: str | None = None
    rep_distance_m: float | None = None
    ttc_s: float | None = None


class Chain1Output(BaseModel):
    validated_events: list[ValidatedEvent]
    risk_summary: dict[str, int]


class KeyCase(BaseModel):
    event_id: str | None = None
    event_time: str = "-"
    risk_level: RiskLevel
    risk_percent: int = Field(ge=0, le=100)
    obstacle_name: str | None = None
    rep_distance_m: float | None = None
    ttc_s: float | None = None
    snapshot_path: str | None = None
    snapshot_url: str | None = None
    why_key: str


class TopObstacle(BaseModel):
    name: str
    count: int = Field(ge=0)


class Chain2Output(BaseModel):
    summary: str
    risk_distribution: dict[str, int]
    total_events: int = Field(ge=0)
    key_cases: list[KeyCase]
    risk_patterns: list[str]
    improvements: list[str]
    top_obstacles: list[TopObstacle]


class Chain3Output(BaseModel):
    title: str = "Daily Collision Report"
    summary: str
    risk_distribution: dict[str, int]
    total_events: int = Field(ge=0)
    key_cases: list[KeyCase]
    risk_patterns: list[str]
    improvements: list[str]


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def validate_model(model_cls: type[BaseModel], payload: dict[str, Any]) -> dict[str, Any]:
    if hasattr(model_cls, "model_validate"):
        model = model_cls.model_validate(payload)
    else:
        model = model_cls.parse_obj(payload)
    return model_to_dict(model)
