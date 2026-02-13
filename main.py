import argparse
import csv
import math
import os
import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

try:
    import winsound
except ImportError:
    winsound = None


PERSON_CLASS_NAME = "person"
OBSTACLE_CLASSES_DEFAULT = {
    "chair", "couch", "bed", "dining table", "bench", "tv", "refrigerator",
    "oven", "microwave", "sink", "toilet", "car", "motorcycle", "bicycle",
    "truck", "bus",
}

FIXED_CLASSES_DEFAULT = {
    "chair", "couch", "bed", "dining table", "bench", "tv", "refrigerator",
    "oven", "microwave", "sink", "toilet",
}


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="RealSense + YOLO based collision risk detection and visualization."
    )
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="YOLO model path")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--rs-timeout-ms", type=int, default=10000, help="RealSense wait_for_frames timeout in ms")
    parser.add_argument("--rs-warmup", type=int, default=15, help="Number of initial frames to skip for sensor warmup")

    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--warn-dist", type=float, default=1.5, help="Warning distance in meters")
    parser.add_argument("--danger-dist", type=float, default=0.9, help="Danger distance in meters")
    parser.add_argument("--warn-ttc", type=float, default=2.0, help="Warning TTC in seconds")
    parser.add_argument("--danger-ttc", type=float, default=1.0, help="Danger TTC in seconds")
    parser.add_argument("--front-angle", type=float, default=70.0, help="Forward cone angle (deg)")
    parser.add_argument("--history-size", type=int, default=90, help="Distance history length")

    parser.add_argument("--obstacle-classes", type=str, default="", help="Comma separated obstacle class names")
    parser.add_argument("--fixed-classes", type=str, default="", help="Comma separated fixed obstacle class names")
    parser.add_argument("--all-non-person", action="store_true", help="Treat all non-person classes as obstacles.")

    parser.add_argument("--min-obstacle-area-ratio", type=float, default=0.01, help="Ignore small obstacle boxes (ratio)")
    parser.add_argument("--min-obstacle-size-m", type=float, default=0.35, help="Ignore obstacles smaller than this (m)")

    parser.add_argument("--lock-frames", type=int, default=10, help="Keep nearest pair lock for N frames")
    parser.add_argument("--lock-max-dist", type=float, default=0.6, help="Max per-object displacement (m) to keep lock")
    parser.add_argument("--match-dist", type=float, default=0.7, help="3D dist threshold (m) for ID matching")

    parser.add_argument("--beep", action="store_true", help="Enable warning sound")
    parser.add_argument("--beep-cooldown", type=float, default=0.8, help="Min seconds between beeps")

    parser.add_argument("--log-file", type=str, default="collision_log.csv", help="CSV log output path. Empty disables.")
    parser.add_argument("--use-yolo-track", action="store_true", help="Use YOLO built-in tracker (ByteTrack) for IDs.")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config (e.g., bytetrack.yaml).")

    # ---- NEW: stability knobs ----
    parser.add_argument("--vel-alpha", type=float, default=0.8, help="EMA alpha for velocity smoothing (0~1)")
    parser.add_argument("--depth-bottom-band", type=float, default=0.22, help="Bottom band ratio for depth median (0~1)")
    parser.add_argument("--depth-uv-win", type=int, default=7, help="Odd window size for (u,v) local depth median")
    parser.add_argument(
        "--ttc-mode",
        type=str,
        default="forward",
        choices=["forward", "los", "both_min"],
        help="TTC calculation mode: forward / los / both_min",
    )

    return parser.parse_args()


# -----------------------------
# Depth helpers (NEW)
# -----------------------------
def _median_depth_in_rect(depth_image, x1, y1, x2, y2, depth_scale, min_valid=30):
    h, w = depth_image.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    valid = patch[patch > 0]
    if valid.size < min_valid:
        return None
    return float(np.median(valid) * depth_scale)


def depth_median_bottom_band(depth_image, bbox, depth_scale, band_ratio=0.22):
    x1, y1, x2, y2 = bbox
    band_ratio = float(np.clip(band_ratio, 0.05, 0.7))
    bh = max(1, int((y2 - y1) * band_ratio))
    by1 = max(y1, y2 - bh)
    return _median_depth_in_rect(depth_image, x1, by1, x2, y2, depth_scale, min_valid=30)


def depth_median_around_uv(depth_image, u, v, depth_scale, win=7):
    win = int(win)
    if win < 3:
        win = 3
    if win % 2 == 0:
        win += 1
    r = win // 2
    return _median_depth_in_rect(depth_image, u - r, v - r, u + r + 1, v + r + 1, depth_scale, min_valid=10)


# (kept) fallback for whole bbox median
def median_depth_from_bbox(depth_image, bbox, depth_scale):
    x1, y1, x2, y2 = bbox
    return _median_depth_in_rect(depth_image, x1, y1, x2, y2, depth_scale, min_valid=20)


# -----------------------------
# Geometry
# -----------------------------
def pixel_to_3d(u, v, z, intrinsics):
    fx = intrinsics.fx
    fy = intrinsics.fy
    ppx = intrinsics.ppx
    ppy = intrinsics.ppy
    x = (u - ppx) / fx * z
    y = (v - ppy) / fy * z
    return np.array([x, y, z], dtype=np.float32)


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


def risk_level(min_distance, ttc, warn_dist, danger_dist, warn_ttc, danger_ttc):
    if min_distance is None:
        return "SAFE"
    if min_distance < danger_dist:
        return "DANGER"
    if min_distance < warn_dist:
        return "WARNING"
    if ttc is not None and ttc < danger_ttc:
        return "DANGER"
    if ttc is not None and ttc < warn_ttc:
        return "WARNING"
    return "SAFE"


def risk_color(level):
    if level == "DANGER":
        return (0, 0, 255)
    if level == "WARNING":
        return (0, 220, 255)
    return (0, 200, 0)


def angle_from_forward_vector(forward_vec, person_3d, target_3d):
    vec = target_3d - person_3d
    vec_norm = np.linalg.norm(vec)
    f_norm = np.linalg.norm(forward_vec)
    if vec_norm < 1e-6 or f_norm < 1e-6:
        # fallback: treat forward as camera Z
        forward_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        f_norm = 1.0
    cos_theta = float(np.dot(vec / vec_norm, forward_vec / f_norm))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


# -----------------------------
# Tracking
# -----------------------------
def assign_tracks(items, prev_points, next_track_id, max_dist):
    used_ids = set()
    for item in items:
        if item.get("track_id") is not None:
            used_ids.add(int(item["track_id"]))

    used_prev_ids = set()
    for item in items:
        if item.get("track_id") is not None:
            continue
        best_id = None
        best_dist = max_dist
        for track_id, prev_point in prev_points.items():
            if track_id in used_prev_ids or track_id in used_ids:
                continue
            dist = float(np.linalg.norm(item["point_3d"] - prev_point))
            if dist < best_dist:
                best_dist = dist
                best_id = track_id
        if best_id is None:
            best_id = next_track_id
            next_track_id += 1
        used_prev_ids.add(best_id)
        used_ids.add(best_id)
        item["track_id"] = best_id

    curr_points = {int(item["track_id"]): item["point_3d"]
                   for item in items if item.get("track_id") is not None}
    return curr_points, next_track_id


# -----------------------------
# Logging & Beep
# -----------------------------
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


# -----------------------------
# TTC helpers (NEW)
# -----------------------------
def _ttc_los(person, obs):
    rel_vec = obs["point_3d"] - person["point_3d"]
    rel_dist = float(np.linalg.norm(rel_vec))
    if rel_dist < 1e-6:
        return None
    rel_dir = rel_vec / rel_dist

    p_vel = person["velocity"] if person["velocity"] is not None else np.zeros(3, dtype=np.float32)
    o_vel = obs["velocity"] if obs["velocity"] is not None else np.zeros(3, dtype=np.float32)
    rel_vel = o_vel - p_vel

    closing_speed = -float(np.dot(rel_vel, rel_dir))  # >0이면 가까워지는 중
    if closing_speed > 1e-3:
        return rel_dist / closing_speed
    return None


def _ttc_forward(person, obs):
    # forward TTC: 사람 진행방향 기준으로 "앞쪽 거리/앞쪽 접근속도"
    f = person["velocity"]
    if f is None or float(np.linalg.norm(f)) < 1e-3:
        # fallback: camera forward
        f = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    f_norm = float(np.linalg.norm(f))
    if f_norm < 1e-6:
        return None
    f_hat = f / f_norm

    rel = obs["point_3d"] - person["point_3d"]
    d_forward = float(np.dot(rel, f_hat))  # 앞쪽(+) 투영 거리
    if d_forward <= 0.0:
        return None

    p_vel = person["velocity"] if person["velocity"] is not None else np.zeros(3, dtype=np.float32)
    o_vel = obs["velocity"] if obs["velocity"] is not None else np.zeros(3, dtype=np.float32)

    # 사람이 앞으로 가며 장애물에 접근하는 속도 성분(+)이 필요
    v_forward = float(np.dot(p_vel - o_vel, f_hat))
    if v_forward > 1e-3:
        return d_forward / v_forward
    return None


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    if args.all_non_person:
        obstacle_classes = set()
    else:
        obstacle_classes = (
            {x.strip() for x in args.obstacle_classes.split(",") if x.strip()}
            if args.obstacle_classes
            else OBSTACLE_CLASSES_DEFAULT
        )
    fixed_classes = (
        {x.strip() for x in args.fixed_classes.split(",") if x.strip()}
        if args.fixed_classes
        else FIXED_CLASSES_DEFAULT
    )

    if args.model.endswith(".pt") and not os.path.exists(args.model):
        print(f"Model file '{args.model}' not found locally. Ultralytics may try to download it.")
    model = YOLO(args.model)

    # names 안전 처리
    class_names = getattr(model, "names", None)
    if class_names is None:
        class_names = model.model.names

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()

    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    prev_time = time.time()
    history = deque(maxlen=args.history_size)

    prev_person_points = {}
    prev_obstacle_points = {}
    next_person_track_id = 1
    next_obstacle_track_id = 1

    # ---- NEW: velocity EMA per track ----
    vel_ema_person = {}   # track_id -> np.array(3)
    vel_ema_obstacle = {} # track_id -> np.array(3)

    last_beep_ts = 0.0
    log_file, log_writer = maybe_open_log(args.log_file)

    lock_pair = None
    lock_until = 0

    try:
        warmup_left = max(0, int(args.rs_warmup))
        while True:
            try:
                frames = pipeline.wait_for_frames(args.rs_timeout_ms)
            except RuntimeError:
                continue

            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            if warmup_left > 0:
                warmup_left -= 1
                continue

            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            if args.use_yolo_track:
                results = model.track(
                    color_image,
                    conf=args.conf,
                    tracker=args.tracker,
                    persist=True,
                    verbose=False,
                )[0]
            else:
                results = model.predict(color_image, conf=args.conf, verbose=False)[0]

            people = []
            obstacles = []

            frame_h, frame_w = color_image.shape[:2]
            frame_area = max(1.0, float(frame_w * frame_h))

            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else str(class_names[cls_id])

                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                box_w = max(0, x2 - x1)
                box_h = max(0, y2 - y1)
                area_ratio = (box_w * box_h) / frame_area

                # ---- NEW: bottom-band depth first, fallback to full bbox ----
                z = depth_median_bottom_band(depth_image, (x1, y1, x2, y2), depth_scale, band_ratio=args.depth_bottom_band)
                if z is None:
                    z = median_depth_from_bbox(depth_image, (x1, y1, x2, y2), depth_scale)
                if z is None:
                    continue

                # representative pixel: center-x, near-bottom-y
                u = (x1 + x2) // 2
                v = int(y2 * 0.9)

                # ---- NEW: local (u,v) depth refine to match pixel ----
                z_uv = depth_median_around_uv(depth_image, u, v, depth_scale, win=args.depth_uv_win)
                if z_uv is not None:
                    z = z_uv

                width_m = (box_w / intrinsics.fx) * z
                height_m = (box_h / intrinsics.fy) * z
                max_size_m = max(width_m, height_m)

                point_3d = pixel_to_3d(u, v, z, intrinsics)
                item = {
                    "bbox": (x1, y1, x2, y2),
                    "name": name,
                    "conf": conf,
                    "z": z,
                    "point_3d": point_3d,
                    "track_id": None,
                    "is_fixed": name in fixed_classes,
                    "velocity": None,
                }
                if args.use_yolo_track and box.id is not None:
                    item["track_id"] = int(box.id[0].item())

                if name == PERSON_CLASS_NAME:
                    people.append(item)
                elif args.all_non_person or name in obstacle_classes:
                    if area_ratio < args.min_obstacle_area_ratio:
                        continue
                    if max_size_m < args.min_obstacle_size_m:
                        continue
                    obstacles.append(item)

            now = time.time()
            dt = max(1e-3, now - prev_time)

            curr_person_points, next_person_track_id = assign_tracks(
                people, prev_person_points, next_person_track_id, args.match_dist
            )
            curr_obstacle_points, next_obstacle_track_id = assign_tracks(
                obstacles, prev_obstacle_points, next_obstacle_track_id, args.match_dist
            )

            # ---- Velocity + EMA smoothing (NEW) ----
            alpha = float(np.clip(args.vel_alpha, 0.0, 0.99))

            for p in people:
                tid = int(p["track_id"])
                prev_pt = prev_person_points.get(tid)
                if prev_pt is not None:
                    v_raw = (p["point_3d"] - prev_pt) / dt
                    v_prev = vel_ema_person.get(tid)
                    v_smooth = v_raw if v_prev is None else (alpha * v_prev + (1.0 - alpha) * v_raw)
                    vel_ema_person[tid] = v_smooth
                    p["velocity"] = v_smooth
                else:
                    # keep previous ema if exists, else None
                    p["velocity"] = vel_ema_person.get(tid, None)

            for o in obstacles:
                tid = int(o["track_id"])
                if o.get("is_fixed"):
                    vel_ema_obstacle[tid] = np.zeros(3, dtype=np.float32)
                    o["velocity"] = np.zeros(3, dtype=np.float32)
                    continue
                prev_pt = prev_obstacle_points.get(tid)
                if prev_pt is not None:
                    v_raw = (o["point_3d"] - prev_pt) / dt
                    v_prev = vel_ema_obstacle.get(tid)
                    v_smooth = v_raw if v_prev is None else (alpha * v_prev + (1.0 - alpha) * v_raw)
                    vel_ema_obstacle[tid] = v_smooth
                    o["velocity"] = v_smooth
                else:
                    o["velocity"] = vel_ema_obstacle.get(tid, None)

            min_distance = None
            ttc = None
            nearest_pair = None

            # ---- LOCK (FIXED): keep lock if each object doesn't "jump" too much + still in front cone ----
            if lock_pair is not None and lock_until > 0:
                lock_person_id, lock_obs_id = lock_pair
                lock_person = next((p for p in people if int(p.get("track_id", -1)) == lock_person_id), None)
                lock_obs = next((o for o in obstacles if int(o.get("track_id", -1)) == lock_obs_id), None)

                if lock_person is not None and lock_obs is not None:
                    prev_p = prev_person_points.get(lock_person_id)
                    prev_o = prev_obstacle_points.get(lock_obs_id)

                    p_disp = float(np.linalg.norm(lock_person["point_3d"] - prev_p)) if prev_p is not None else 0.0
                    o_disp = float(np.linalg.norm(lock_obs["point_3d"] - prev_o)) if prev_o is not None else 0.0

                    if p_disp <= args.lock_max_dist and o_disp <= args.lock_max_dist:
                        forward_vec = lock_person["velocity"]
                        if forward_vec is None or float(np.linalg.norm(forward_vec)) < 1e-3:
                            forward_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                        angle = angle_from_forward_vector(forward_vec, lock_person["point_3d"], lock_obs["point_3d"])
                        if angle <= args.front_angle:
                            lock_dist = float(np.linalg.norm(lock_person["point_3d"] - lock_obs["point_3d"]))
                            min_distance = lock_dist
                            nearest_pair = (lock_person, lock_obs, angle, forward_vec)
                            lock_until -= 1
                        else:
                            lock_pair = None
                            lock_until = 0
                    else:
                        lock_pair = None
                        lock_until = 0
                else:
                    lock_pair = None
                    lock_until = 0

            # ---- find nearest pair if no lock ----
            if nearest_pair is None:
                for person in people:
                    for obs in obstacles:
                        forward_vec = person["velocity"]
                        if forward_vec is None or float(np.linalg.norm(forward_vec)) < 1e-3:
                            forward_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                        angle = angle_from_forward_vector(forward_vec, person["point_3d"], obs["point_3d"])
                        if angle > args.front_angle:
                            continue

                        dist = float(np.linalg.norm(person["point_3d"] - obs["point_3d"]))
                        if min_distance is None or dist < min_distance:
                            min_distance = dist
                            nearest_pair = (person, obs, angle, forward_vec)

                if nearest_pair is not None:
                    lock_pair = (int(nearest_pair[0]["track_id"]), int(nearest_pair[1]["track_id"]))
                    lock_until = max(0, int(args.lock_frames))

            # ---- TTC (NEW): forward / los / both_min ----
            if nearest_pair is not None:
                person, obs, _, _ = nearest_pair

                ttc_f = _ttc_forward(person, obs)
                ttc_l = _ttc_los(person, obs)

                if args.ttc_mode == "forward":
                    ttc = ttc_f
                elif args.ttc_mode == "los":
                    ttc = ttc_l
                else:  # both_min
                    candidates = [x for x in [ttc_f, ttc_l] if x is not None and x > 0]
                    ttc = min(candidates) if candidates else None

            level = risk_level(
                min_distance=min_distance,
                ttc=ttc,
                warn_dist=args.warn_dist,
                danger_dist=args.danger_dist,
                warn_ttc=args.warn_ttc,
                danger_ttc=args.danger_ttc,
            )
            color = risk_color(level)

            # -----------------------------
            # Draw
            # -----------------------------
            for p in people:
                x1, y1, x2, y2 = p["bbox"]
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (50, 220, 50), 2)
                cv2.putText(
                    color_image, f"person {p['z']:.2f}m",
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 50), 2
                )

            for o in obstacles:
                x1, y1, x2, y2 = o["bbox"]
                box_color = (180, 120, 60) if o.get("is_fixed") else (220, 120, 60)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), box_color, 2)
                fixed_tag = " [fixed]" if o.get("is_fixed") else ""
                cv2.putText(
                    color_image, f"{o['name']}{fixed_tag} {o['z']:.2f}m",
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1
                )

            if nearest_pair is not None:
                person, obs, angle, forward_vec = nearest_pair
                px1, py1, px2, py2 = person["bbox"]
                ox1, oy1, ox2, oy2 = obs["bbox"]
                p_center = ((px1 + px2) // 2, int(py2 * 0.9))
                o_center = ((ox1 + ox2) // 2, int(oy2 * 0.9))

                cv2.line(color_image, p_center, o_center, color, 3)
                dist_txt = f"{min_distance:.2f}m" if min_distance is not None else "N/A"
                ttc_txt = f"{ttc:.2f}s" if ttc is not None else "N/A"

                cv2.putText(
                    color_image,
                    f"Nearest: {obs['name']}#{obs['track_id']}  Dist: {dist_txt}  TTC({args.ttc_mode}): {ttc_txt}  Angle: {angle:.0f}deg",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
                )

                # forward arrow
                fvec = person["velocity"]
                if fvec is None or float(np.linalg.norm(fvec)) < 1e-3:
                    fvec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                fvec_2d = np.array([fvec[0], fvec[2]], dtype=np.float32)
                if float(np.linalg.norm(fvec_2d)) > 1e-4:
                    fvec_2d = fvec_2d / float(np.linalg.norm(fvec_2d))
                    end_pt = (int(p_center[0] + fvec_2d[0] * 70), int(p_center[1] + fvec_2d[1] * 70))
                    cv2.arrowedLine(color_image, p_center, end_pt, (255, 255, 0), 2, tipLength=0.25)

            panel_h = 120
            h, w = color_image.shape[:2]
            canvas = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
            canvas[:h] = color_image

            cv2.putText(canvas, f"RISK: {level}", (12, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(canvas, f"Min distance: {min_distance:.2f} m" if min_distance is not None else "Min distance: N/A",
                        (12, h + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
            cv2.putText(canvas, f"TTC: {ttc:.2f} s" if ttc is not None else "TTC: N/A",
                        (12, h + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)

            history.append(min_distance)
            graph_w = min(360, w // 2)
            graph_h = panel_h - 16
            graph_x = w - graph_w - 12
            graph_y = h + 8
            draw_distance_graph(canvas, history, graph_x, graph_y, graph_w, graph_h)

            last_beep_ts = maybe_beep(args.beep, level, now, last_beep_ts, args.beep_cooldown)

            if log_writer is not None:
                if nearest_pair is None:
                    log_writer.writerow([f"{now:.3f}", level, "", "", "", "", "", ""])
                else:
                    person, obs, angle, _ = nearest_pair
                    log_writer.writerow([
                        f"{now:.3f}",
                        level,
                        f"{min_distance:.3f}" if min_distance is not None else "",
                        f"{ttc:.3f}" if ttc is not None else "",
                        person["track_id"],
                        obs["track_id"],
                        obs["name"],
                        f"{angle:.2f}",
                    ])
                log_file.flush()

            cv2.imshow("Collision Risk Monitor", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

            prev_time = now
            prev_person_points = curr_person_points
            prev_obstacle_points = curr_obstacle_points

    finally:
        if log_file is not None:
            log_file.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
