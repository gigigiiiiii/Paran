"""
collision_monitor/visualizer.py
================================
bbox 시각화 및 HUD 패널 그리기 함수.
FrameProcessor에서 분리된 순수 렌더링 레이어.
"""
from __future__ import annotations

import numpy as np
import cv2

from .config import CLASS_COLORS
from .output import draw_distance_graph


_PAIR_COLORS = {
    "SAFE":    (50,  220,  50),
    "WARNING": (0,   165, 255),
    "DANGER":  (0,    0,  220),
}


def draw_detections(
    canvas: np.ndarray,
    people: list[dict],
    obstacles: list[dict],
    all_risk_pairs: list[dict],
    color: tuple,
    has_depth: bool,
    line_max_dist: float = 0.0,
) -> np.ndarray:
    """사람·장애물 bbox + 모든 위험 쌍 연결선을 canvas에 그린다."""
    for p in people:
        x1, y1, x2, y2 = p["bbox"]
        tid    = p.get("track_id")
        id_tag = f" #{tid}" if tid is not None else ""
        label = f"person{id_tag}  {p['conf']:.0%}"
        box_color = CLASS_COLORS.get("person", (50, 220, 50))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(canvas, label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
        model_z = p.get("model_z")
        if model_z is not None:
            model_label = f"model:{model_z:.2f}m"
            if p["z"] is not None:
                err = model_z - p["z"]
                rel = err / p["z"] * 100.0
                model_label += f"  \u0394{err:+.2f}m/{rel:+.0f}%"
            cv2.putText(canvas, model_label,
                        (x1, max(36, y1 + 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 2)

    for o in obstacles:
        x1, y1, x2, y2 = o["bbox"]
        tid    = o.get("track_id")
        id_tag = f" #{tid}" if tid is not None else ""
        label = f"{o['name']}{id_tag}  {o['conf']:.0%}"
        box_color = CLASS_COLORS.get(o["name"], (220, 120, 60))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(canvas, label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
        model_z = o.get("model_z")
        if model_z is not None:
            model_label = f"model:{model_z:.2f}m"
            if o["z"] is not None:
                err = model_z - o["z"]
                rel = err / o["z"] * 100.0
                model_label += f"  \u0394{err:+.2f}m/{rel:+.0f}%"
            cv2.putText(canvas, model_label,
                        (x1, max(36, y1 + 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 2)

    for pair in all_risk_pairs:
        if line_max_dist > 0.0 and pair.get("dist") is not None:
            if float(pair["dist"]) > line_max_dist:
                continue
        person = pair["person"]
        obs    = pair["obs"]
        px1, py1, px2, py2 = person["bbox"]
        ox1, oy1, ox2, oy2 = obs["bbox"]
        ps = pair.get("ps") or {}
        os_ = pair.get("os_") or {}
        pc = ps.get("uv") or person.get("rep_uv") or ((px1 + px2) // 2, int(py2 * 0.9))
        oc = os_.get("uv") or obs.get("rep_uv") or ((ox1 + ox2) // 2, int(oy2 * 0.9))
        line_color = _PAIR_COLORS.get(pair["level"], color)
        thickness  = 4 if pair["level"] == "DANGER" else 2
        cv2.line(canvas, pc, oc, line_color, thickness)
        cv2.circle(canvas, pc, 4, line_color, -1)
        cv2.circle(canvas, oc, 4, line_color, -1)
        if pair["dist"] is not None:
            mid = ((pc[0] + oc[0]) // 2, (pc[1] + oc[1]) // 2)
            cv2.putText(canvas, f"{pair['dist']:.2f}m",
                        (mid[0] + 4, mid[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)

    return canvas


def draw_hud_panel(
    canvas: np.ndarray,
    level: str,
    color: tuple,
    rep_distance,
    min_distance,
    ttc,
    risk_score_smooth: float,
    risk_score_raw: float,
    history,
    depth_err_stats: dict | None = None,
) -> np.ndarray:
    """하단 HUD 패널(위험도·거리·TTC·그래프)을 canvas 아래에 붙인다."""
    panel_h = 174
    h, w    = canvas.shape[:2]
    full    = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
    full[:h] = canvas
    cv2.putText(full, f"RISK: {level}",
                (12, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(full,
                f"Rep distance: {rep_distance:.2f} m" if rep_distance else "Rep distance: N/A",
                (12, h + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
    cv2.putText(full,
                f"Min distance: {min_distance:.2f} m" if min_distance else "Min distance: N/A",
                (12, h + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
    cv2.putText(full,
                f"TTC: {ttc:.2f} s" if ttc else "TTC: N/A",
                (12, h + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)

    # depth 오차 통계 (RealSense vs 모델)
    if depth_err_stats and depth_err_stats.get("n", 0) > 0:
        n   = depth_err_stats["n"]
        mae = depth_err_stats["mae"]
        mre = depth_err_stats["mre"]
        err_text = f"Depth err(n={n}): MAE={mae:.2f}m  Rel={mre:+.0f}%"
        cv2.putText(full, err_text,
                    (12, h + 148), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 80, 255), 2)
    else:
        cv2.putText(full,
                    f"Risk score: {risk_score_smooth:.2f} (raw {risk_score_raw:.2f})",
                    (12, h + 148), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 2)

    graph_w = min(360, w // 2)
    graph_h = panel_h - 16
    draw_distance_graph(full, history, w - graph_w - 12, h + 8, graph_w, graph_h)
    return full
