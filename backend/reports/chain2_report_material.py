from __future__ import annotations

from typing import Any

from .llm_utils import dumps_for_prompt, run_llm_json_chain
from .prompts import CHAIN2_HUMAN_TEMPLATE, CHAIN2_SYSTEM_PROMPT
from .schemas import Chain2Output, validate_model


def _rule_build_report_material(chain1_output: dict[str, Any]) -> dict[str, Any]:
    events = list(chain1_output.get("validated_events") or [])
    risk_summary = dict(chain1_output.get("risk_summary") or {"high": 0, "medium": 0, "low": 0})
    total = len(events)

    key_cases = _select_key_cases(events)
    obstacle_counts: dict[str, int] = {}
    for event in events:
        obstacle = str(event.get("obstacle_name") or "unknown")
        obstacle_counts[obstacle] = obstacle_counts.get(obstacle, 0) + 1

    top_obstacles = sorted(obstacle_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "summary": _build_summary_text(total, risk_summary, top_obstacles),
        "risk_distribution": risk_summary,
        "total_events": total,
        "key_cases": key_cases,
        "risk_patterns": _build_risk_patterns(events, top_obstacles),
        "improvements": _build_improvements(risk_summary, top_obstacles),
        "top_obstacles": [
            {"name": name, "count": count}
            for name, count in top_obstacles
        ],
    }


def build_report_material(
    chain1_output: dict[str, Any],
    use_llm: bool = True,
    rule_material: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rule_output = rule_material or _rule_build_report_material(chain1_output)
    if not use_llm:
        return rule_output

    try:
        llm_output = run_llm_json_chain(
            CHAIN2_SYSTEM_PROMPT,
            CHAIN2_HUMAN_TEMPLATE,
            {
                "chain1_json": dumps_for_prompt(chain1_output),
                "rule_material_json": dumps_for_prompt(rule_output),
            },
            output_model=Chain2Output,
        )
        return _normalize_llm_chain2_output(llm_output, rule_output, chain1_output)
    except Exception as exc:
        fallback = dict(rule_output)
        fallback["llm_error"] = str(exc)
        return fallback


def _normalize_llm_chain2_output(
    llm_output: dict[str, Any],
    rule_output: dict[str, Any],
    chain1_output: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(rule_output)
    required_distribution = dict(chain1_output.get("risk_summary") or rule_output.get("risk_distribution") or {})
    required_total = len(chain1_output.get("validated_events") or [])
    if isinstance(llm_output.get("summary"), str) and llm_output["summary"].strip():
        normalized["summary"] = llm_output["summary"].strip()
    if isinstance(llm_output.get("key_cases"), list):
        normalized["key_cases"] = _normalize_key_cases(llm_output["key_cases"], rule_output["key_cases"])
    if isinstance(llm_output.get("risk_patterns"), list):
        normalized["risk_patterns"] = [str(item) for item in llm_output["risk_patterns"] if str(item).strip()]
    if isinstance(llm_output.get("improvements"), list):
        normalized["improvements"] = [str(item) for item in llm_output["improvements"] if str(item).strip()]
    if isinstance(llm_output.get("top_obstacles"), list):
        normalized["top_obstacles"] = [
            item for item in llm_output["top_obstacles"]
            if isinstance(item, dict) and item.get("name") is not None
        ]
    normalized["risk_distribution"] = {
        "high": int(required_distribution.get("high") or 0),
        "medium": int(required_distribution.get("medium") or 0),
        "low": int(required_distribution.get("low") or 0),
    }
    normalized["total_events"] = required_total
    if not normalized.get("risk_patterns"):
        normalized["risk_patterns"] = rule_output.get("risk_patterns") or []
    return validate_model(Chain2Output, normalized)


def _build_summary_text(
    total: int,
    risk_summary: dict[str, int],
    top_obstacles: list[tuple[str, int]],
) -> str:
    if total == 0:
        return "No dashboard collision records were found for the selected report range."

    high = int(risk_summary.get("high") or 0)
    medium = int(risk_summary.get("medium") or 0)
    low = int(risk_summary.get("low") or 0)
    obstacle_text = ", ".join(f"{name}({count})" for name, count in top_obstacles) or "unknown"
    return (
        f"Total {total} dashboard collision records were analyzed. "
        f"Risk distribution is High {high}, Medium {medium}, Low {low}. "
        f"Main repeated obstacle groups are {obstacle_text}."
    )


def _build_improvements(
    risk_summary: dict[str, int],
    top_obstacles: list[tuple[str, int]],
) -> list[str]:
    improvements = [
        "Review the highest-risk key cases first and compare the saved snapshots with the actual work area.",
        "Adjust the movement path, obstacle placement, or detection zone around repeated collision points.",
    ]
    if int(risk_summary.get("high") or 0) > 0:
        improvements.append("Treat High cases as immediate safety review items before continuing the same workflow.")
    if top_obstacles:
        improvements.append(f"Install visible warning and separation controls around repeated {top_obstacles[0][0]} interaction zones.")
    return improvements


def _select_key_cases(events: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    ranked = sorted(events, key=_key_case_rank, reverse=True)
    key_cases = []
    for event in ranked[:limit]:
        copied = dict(event)
        copied["why_key"] = copied.get("judgement_reason") or "Selected by risk level, risk score, distance, TTC, and repeated context."
        key_cases.append(copied)
    return key_cases


def _key_case_rank(event: dict[str, Any]) -> tuple[int, int, float, float, int]:
    level_rank = {"High": 2, "Medium": 1, "Low": 0}.get(str(event.get("risk_level")), 0)
    risk_percent = int(event.get("risk_percent") or 0)
    ttc = event.get("ttc_s")
    distance = event.get("rep_distance_m")
    ttc_rank = -float(ttc) if isinstance(ttc, (int, float)) else -9999.0
    distance_rank = -float(distance) if isinstance(distance, (int, float)) else -9999.0
    factors = event.get("risk_factors") if isinstance(event.get("risk_factors"), dict) else {}
    repetition_text = str(factors.get("repetition_factor") or "")
    repeated_rank = 1 if any(char.isdigit() and char != "1" for char in repetition_text) else 0
    return level_rank, risk_percent, ttc_rank, distance_rank, repeated_rank


def _build_risk_patterns(
    events: list[dict[str, Any]],
    top_obstacles: list[tuple[str, int]],
) -> list[str]:
    patterns = []
    for obstacle, count in top_obstacles:
        if count > 1:
            high_count = sum(
                1 for event in events
                if str(event.get("obstacle_name") or "unknown") == obstacle
                and str(event.get("risk_level")) == "High"
            )
            patterns.append(f"{obstacle} appears repeatedly across {count} events, including {high_count} High cases.")
    if not patterns and events:
        patterns.append("No repeated obstacle pattern dominates, but key cases require individual review.")
    return patterns


def _normalize_key_cases(llm_cases: list[Any], fallback_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fallback_by_id = {str(case.get("event_id")): case for case in fallback_cases if case.get("event_id") is not None}
    normalized = []
    for item in llm_cases:
        if not isinstance(item, dict):
            continue
        event_id = str(item.get("event_id")) if item.get("event_id") is not None else ""
        merged = dict(fallback_by_id.get(event_id, {}))
        merged.update(item)
        if "why_key" not in merged or not str(merged["why_key"]).strip():
            merged["why_key"] = merged.get("judgement_reason") or "Selected as a key safety case."
        normalized.append(merged)
    return normalized or fallback_cases
