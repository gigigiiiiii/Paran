import csv
import os

import cv2

try:
    import winsound
except ImportError:
    winsound = None


def draw_distance_graph(canvas, history, graph_x, graph_y, graph_w, graph_h):
    cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (120, 120, 120), 1)
    if len(history) < 2:
        cv2.putText(canvas, "Distance history", (graph_x + 8, graph_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        return

    distances = [d for d in history if d is not None]
    if not distances:
        return

    d_min = max(0.2, min(distances) - 0.2)
    d_max = max(d_min + 0.1, max(distances) + 0.2)

    points = []
    n = len(history)
    for i, d in enumerate(history):
        if d is None:
            continue
        px = graph_x + int(i / max(1, n - 1) * (graph_w - 1))
        py_norm = (d - d_min) / (d_max - d_min)
        py = graph_y + graph_h - int(py_norm * (graph_h - 1))
        points.append((px, py))

    for i in range(1, len(points)):
        cv2.line(canvas, points[i - 1], points[i], (0, 220, 255), 2)

    cv2.putText(canvas, f"Distance history ({d_min:.1f}m - {d_max:.1f}m)",
                (graph_x + 8, graph_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)


def maybe_open_log(log_path):
    if not log_path:
        return None, None
    need_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    log_f = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if need_header:
        writer.writerow([
            "ts_epoch", "risk", "min_distance_m", "ttc_s",
            "person_track_id", "obstacle_track_id", "obstacle_name", "angle_deg",
        ])
        log_f.flush()
    return log_f, writer


def maybe_beep(enabled, level, now_ts, last_beep_ts, cooldown):
    if not enabled or winsound is None:
        return last_beep_ts
    if now_ts - last_beep_ts < cooldown:
        return last_beep_ts
    if level == "DANGER":
        winsound.Beep(1700, 140)
        return now_ts
    if level == "WARNING":
        winsound.Beep(1200, 90)
        return now_ts
    return last_beep_ts

