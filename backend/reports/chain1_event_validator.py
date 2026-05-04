from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .llm_utils import dumps_for_prompt, run_llm_json_chain
from .prompts import CHAIN1_HUMAN_TEMPLATE, CHAIN1_SYSTEM_PROMPT
from .schemas import Chain1Output, validate_model


def parse_event_datetime(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).astimezone()
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).astimezone()
    except ValueError:
        return None


def event_datetime(event: dict[str, Any]) -> datetime | None:
    return parse_event_datetime(event.get("created_at")) or parse_event_datetime(event.get("ts_epoch"))


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def risk_score(event: dict[str, Any]) -> float:
    score = _num(event.get("risk_score"))
    if score is not None:
        return max(0.0, min(1.0, score))
    risk = str(event.get("risk") or "").upper()
    if risk == "DANGER":
        return 0.85
    if risk == "WARNING":
        return 0.55
    return 0.2


def risk_percent(event: dict[str, Any]) -> int:
    return int(round(risk_score(event) * 100))


def classify_risk_level(event: dict[str, Any]) -> str:
    risk = str(event.get("risk") or "").upper()
    score = risk_score(event)
    distance = _num(event.get("rep_distance_m"))
    ttc = _num(event.get("ttc_s"))

    if risk == "DANGER" or score >= 0.75:
        return "High"
    if distance is not None and distance <= 0.5:
        return "High"
    if ttc is not None and ttc <= 1.5:
        return "High"
    if risk == "WARNING" or score >= 0.45:
        return "Medium"
    if distance is not None and distance <= 1.0:
        return "Medium"
    if ttc is not None and ttc <= 3.0:
        return "Medium"
    return "Low"


def judgement_reason(event: dict[str, Any], level: str) -> str:
    risk = str(event.get("risk") or "UNKNOWN").upper()
    parts = [f"dashboard risk={risk}", f"risk score={risk_percent(event)}%"]
    distance = _num(event.get("rep_distance_m"))
    ttc = _num(event.get("ttc_s"))
    if distance is not None:
        parts.append(f"distance={distance:.2f}m")
    if ttc is not None:
        parts.append(f"ttc={ttc:.2f}s")
    return f"{level} classification based on " + ", ".join(parts)


def risk_factors(event: dict[str, Any], level: str, repeated_count: int = 1) -> dict[str, str]:
    distance = _num(event.get("rep_distance_m"))
    ttc = _num(event.get("ttc_s"))
    obstacle = str(event.get("obstacle_name") or "unknown")
    rule_level = str(event.get("risk_level") or level)
    dashboard_risk = str(event.get("risk") or "UNKNOWN").upper()
    return {
        "distance_factor": (
            f"Representative distance is {distance:.2f}m."
            if distance is not None else "Representative distance is not available."
        ),
        "ttc_factor": f"TTC is {ttc:.2f}s." if ttc is not None else "TTC is not available.",
        "obstacle_factor": f"Related obstacle type is {obstacle}.",
        "repetition_factor": (
            f"Similar obstacle context appears {repeated_count} times."
            if repeated_count > 1 else "No repeated context is evident in the selected events."
        ),
        "rule_agreement": f"Rule helper={rule_level}, dashboard risk={dashboard_risk}.",
    }


def _rule_validate_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    validated_events = []
    risk_summary = {"high": 0, "medium": 0, "low": 0}
    repeated_counts = _obstacle_counts(events)

    for event in events:
        copied = dict(event)
        level = classify_risk_level(copied)
        event_dt = event_datetime(copied)
        copied["risk_level"] = level
        copied["risk_percent"] = risk_percent(copied)
        copied["judgement_reason"] = judgement_reason(copied, level)
        copied["risk_factors"] = risk_factors(
            copied,
            level,
            repeated_counts.get(str(copied.get("obstacle_name") or "unknown"), 1),
        )
        copied["event_time"] = event_dt.strftime("%Y-%m-%d %H:%M:%S") if event_dt else "-"
        validated_events.append(copied)
        risk_summary[level.lower()] += 1

    validated_events.sort(
        key=lambda row: (
            {"High": 2, "Medium": 1, "Low": 0}.get(str(row.get("risk_level")), 0),
            int(row.get("risk_percent") or 0),
            event_datetime(row) or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    return {"validated_events": validated_events, "risk_summary": risk_summary}


def validate_events(
    events: list[dict[str, Any]],
    use_llm: bool = True,
    rule_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rule_output = rule_output or _rule_validate_events(events)
    if not use_llm or not events:
        return rule_output

    repeated_counts = _obstacle_counts(events)
    compact_events = [
        {
            "event_id": event.get("event_id"),
            "session_id": event.get("session_id"),
            "event_time": event.get("event_time") or _format_event_time(event),
            "dashboard_risk": event.get("risk"),
            "risk_score": event.get("risk_score"),
            "rep_distance_m": event.get("rep_distance_m"),
            "ttc_s": event.get("ttc_s"),
            "obstacle_name": event.get("obstacle_name"),
            "repeated_context_count": repeated_counts.get(str(event.get("obstacle_name") or "unknown"), 1),
        }
        for event in events
    ]

    try:
        llm_output = run_llm_json_chain(
            CHAIN1_SYSTEM_PROMPT,
            CHAIN1_HUMAN_TEMPLATE,
            {
                "events_json": dumps_for_prompt(compact_events),
                "rule_output_json": dumps_for_prompt(rule_output),
            },
            output_model=Chain1Output,
        )
        return _normalize_llm_chain1_output(events, llm_output)
    except Exception as exc:
        fallback = dict(rule_output)
        fallback["llm_error"] = str(exc)
        return fallback


def _normalize_llm_chain1_output(
    source_events: list[dict[str, Any]],
    llm_output: dict[str, Any],
) -> dict[str, Any]:
    source_by_id = {str(event.get("event_id") or idx): event for idx, event in enumerate(source_events)}
    repeated_counts = _obstacle_counts(source_events)
    llm_events = llm_output.get("validated_events")
    if not isinstance(llm_events, list):
        return _rule_validate_events(source_events)

    validated_events = []
    risk_summary = {"high": 0, "medium": 0, "low": 0}
    for idx, item in enumerate(llm_events):
        if not isinstance(item, dict):
            continue
        event_id = str(item.get("event_id") or idx)
        original = dict(source_by_id.get(event_id) or (source_events[idx] if idx < len(source_events) else {}))
        level = str(item.get("risk_level") or classify_risk_level(original)).strip().title()
        if level not in {"High", "Medium", "Low"}:
            level = classify_risk_level(original)
        risk_pct_raw = item.get("risk_percent")
        risk_pct = _consistent_risk_percent(original, level, risk_pct_raw)
        event_dt = event_datetime(original)
        obstacle_name = str(original.get("obstacle_name") or "unknown")
        llm_factors = item.get("risk_factors") if isinstance(item.get("risk_factors"), dict) else {}
        original["risk_level"] = level
        original["risk_percent"] = risk_pct
        original["judgement_reason"] = str(item.get("judgement_reason") or judgement_reason(original, level))
        original["risk_factors"] = _normalize_risk_factors(
            llm_factors,
            risk_factors(original, level, repeated_counts.get(obstacle_name, 1)),
        )
        original["event_time"] = original.get("event_time") or (
            event_dt.strftime("%Y-%m-%d %H:%M:%S") if event_dt else "-"
        )
        validated_events.append(original)
        risk_summary[level.lower()] += 1

    if not validated_events:
        return _rule_validate_events(source_events)
    validated_events.sort(
        key=lambda row: (
            {"High": 2, "Medium": 1, "Low": 0}.get(str(row.get("risk_level")), 0),
            int(row.get("risk_percent") or 0),
            event_datetime(row) or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    return validate_model(Chain1Output, {"validated_events": validated_events, "risk_summary": risk_summary})


def _format_event_time(event: dict[str, Any]) -> str:
    event_dt = event_datetime(event)
    return event_dt.strftime("%Y-%m-%d %H:%M:%S") if event_dt else "-"


def _obstacle_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        key = str(event.get("obstacle_name") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _normalize_risk_factors(candidate: dict[str, Any], fallback: dict[str, str]) -> dict[str, str]:
    normalized = dict(fallback)
    for key in ("distance_factor", "ttc_factor", "obstacle_factor", "repetition_factor", "rule_agreement"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    return normalized


def _consistent_risk_percent(event: dict[str, Any], level: str, raw_percent: Any) -> int:
    base = risk_percent(event)
    if isinstance(raw_percent, (int, float)):
        base = int(round((base + float(raw_percent)) / 2.0))
    ranges = {
        "High": (70, 100),
        "Medium": (40, 79),
        "Low": (0, 49),
    }
    low, high = ranges[level]
    return max(low, min(high, max(0, min(100, base))))
