from __future__ import annotations

import json
from typing import Any

from .chain1_event_validator import _rule_validate_events, validate_events
from .chain2_report_material import _rule_build_report_material, build_report_material
from .chain3_pdf_generator import prepare_final_report_material


sample_events = [
    {
        "event_id": "sample-001",
        "session_id": "session-100",
        "created_at": "2026-04-28T09:10:00+09:00",
        "risk": "DANGER",
        "risk_score": 0.86,
        "rep_distance_m": 0.38,
        "ttc_s": 1.2,
        "obstacle_name": "forklift",
    },
    {
        "event_id": "sample-002",
        "session_id": "session-100",
        "created_at": "2026-04-28T09:14:00+09:00",
        "risk": "WARNING",
        "risk_score": 0.58,
        "rep_distance_m": 0.82,
        "ttc_s": 2.4,
        "obstacle_name": "forklift",
    },
    {
        "event_id": "sample-003",
        "session_id": "session-100",
        "created_at": "2026-04-28T09:20:00+09:00",
        "risk": "WARNING",
        "risk_score": 0.42,
        "rep_distance_m": 1.35,
        "ttc_s": 3.6,
        "obstacle_name": "cart",
    },
]

sample_rule_output = _rule_validate_events(sample_events)
sample_rule_material = _rule_build_report_material(sample_rule_output)


def run_pipeline(
    events_json: str | list[dict[str, Any]],
    rule_output_json: str | dict[str, Any] | None = None,
    rule_material_json: str | dict[str, Any] | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    events = _coerce_json(events_json, expected_type=list)
    rule_output = _coerce_json(rule_output_json, expected_type=dict) if rule_output_json is not None else None
    rule_material = _coerce_json(rule_material_json, expected_type=dict) if rule_material_json is not None else None

    chain1 = validate_events(events, use_llm=use_llm, rule_output=rule_output)
    chain2 = build_report_material(chain1, use_llm=use_llm, rule_material=rule_material)
    chain3 = prepare_final_report_material(chain2, use_llm=use_llm)
    return {"chain1": chain1, "chain2": chain2, "chain3": chain3}


def _coerce_json(value: Any, expected_type: type) -> Any:
    if isinstance(value, expected_type):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, expected_type):
            return parsed
    raise TypeError(f"Expected {expected_type.__name__} JSON-compatible value")


if __name__ == "__main__":
    result = run_pipeline(sample_events, sample_rule_output, sample_rule_material, use_llm=False)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
