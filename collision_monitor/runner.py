import os
import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from .config import FIXED_CLASSES_DEFAULT
from .config import OBSTACLE_CLASSES_DEFAULT
from .config import PERSON_CLASS_NAME
from .depth import depth_median_around_uv
from .depth import depth_median_bottom_band
from .depth import median_depth_from_bbox
from .geometry import angle_from_forward_vector
from .geometry import build_collision_samples
from .geometry import min_distance_between_items
from .geometry import pixel_to_3d
from .output import draw_distance_graph
from .output import maybe_beep
from .output import maybe_open_log
from .risk import closing_speed_los
from .risk import compute_risk_score
from .risk import risk_color
from .risk import score_to_level
from .risk import ttc_forward
from .risk import ttc_los
from .tracking import assign_tracks


class PipelineRunner:
    def __init__(self, args, display=True, on_result=None):
        self.args = args
        self.display = bool(display)
        self.on_result = on_result
        self.person_grid_x = max(1, int(args.person_grid_x))
        self.person_grid_y = max(1, int(args.person_grid_y))
        self.obstacle_grid_x = max(1, int(args.obstacle_grid_x))
        self.obstacle_grid_y = max(1, int(args.obstacle_grid_y))
        self.near_weight = float(np.clip(args.depth_near_weight, 0.0, 1.0))
        self.sample_z_max_offset = max(0.0, float(args.sample_z_max_offset))
        self.pair_distance_percentile = float(np.clip(args.pair_distance_percentile, 1.0, 50.0))
        self.risk_up_frames = max(1, int(args.risk_up_frames))
        self.risk_down_frames = max(1, int(args.risk_down_frames))
        self.score_alpha = float(np.clip(args.score_alpha, 0.0, 0.99))
        self.score_dist_weight = max(0.0, float(args.score_dist_weight))
        self.score_ttc_weight = max(0.0, float(args.score_ttc_weight))
        self.score_close_weight = max(0.0, float(args.score_close_weight))
        self.score_close_ref = max(1e-3, float(args.score_close_ref))
        self.score_warn_on = float(np.clip(args.score_warn_on, 0.0, 1.0))
        self.score_danger_on = max(self.score_warn_on, float(np.clip(args.score_danger_on, 0.0, 1.0)))
        self.score_warn_off = min(self.score_warn_on, float(np.clip(args.score_warn_off, 0.0, 1.0)))
        self.score_danger_off = min(
            self.score_danger_on,
            max(self.score_warn_off, float(np.clip(args.score_danger_off, 0.0, 1.0))),
        )

        if args.all_non_person:
            self.obstacle_classes = set()
        else:
            self.obstacle_classes = (
                {x.strip() for x in args.obstacle_classes.split(",") if x.strip()}
                if args.obstacle_classes
                else OBSTACLE_CLASSES_DEFAULT
            )
        self.fixed_classes = (
            {x.strip() for x in args.fixed_classes.split(",") if x.strip()}
            if args.fixed_classes
            else FIXED_CLASSES_DEFAULT
        )

        if args.model.endswith(".pt") and not os.path.exists(args.model):
            print(f"Model file '{args.model}' not found locally. Ultralytics may try to download it.")
        self.model = YOLO(args.model)

        self.class_names = getattr(self.model, "names", None)
        if self.class_names is None:
            self.class_names = self.model.model.names

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        profile = self.pipeline.start(config)

        self.align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()

        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.prev_time = time.time()
        self.history = deque(maxlen=args.history_size)
        self.prev_person_points = {}
        self.prev_obstacle_points = {}
        self.next_person_track_id = 1
        self.next_obstacle_track_id = 1
        self.vel_ema_person = {}
        self.vel_ema_obstacle = {}
        self.last_beep_ts = 0.0
        self.log_file, self.log_writer = maybe_open_log(args.log_file)
        self.lock_pair = None
        self.lock_until = 0
        self.warmup_left = max(0, int(args.rs_warmup))
        self._closed = False
        self._stop_requested = False
        self.recording_enabled = True
        self.stable_level = "SAFE"
        self._pending_level = None
        self._pending_count = 0
        self.risk_score_raw = 0.0
        self.risk_score_smooth = 0.0
        self.last_result = None

    @staticmethod
    def _risk_rank(level):
        rank = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
        return rank.get(level, 0)

    def _stabilize_level(self, raw_level):
        if raw_level == self.stable_level:
            self._pending_level = None
            self._pending_count = 0
            return self.stable_level

        if raw_level == self._pending_level:
            self._pending_count += 1
        else:
            self._pending_level = raw_level
            self._pending_count = 1

        if self._risk_rank(raw_level) > self._risk_rank(self.stable_level):
            need = self.risk_up_frames
        else:
            need = self.risk_down_frames

        if self._pending_count >= need:
            self.stable_level = raw_level
            self._pending_level = None
            self._pending_count = 0

        return self.stable_level

    def run(self):
        try:
            while not self._stop_requested and self.process_frame():
                pass
        finally:
            self.close()

    def request_stop(self):
        self._stop_requested = True

    def set_recording_enabled(self, enabled):
        self.recording_enabled = bool(enabled)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.log_file is not None:
            self.log_file.close()
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def process_frame(self):
        if self._stop_requested:
            return False
        args = self.args
        try:
            frames = self.pipeline.wait_for_frames(args.rs_timeout_ms)
        except RuntimeError:
            return True

        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return True

        if self.warmup_left > 0:
            self.warmup_left -= 1
            return True

        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        if args.use_yolo_track:
            results = self.model.track(
                color_image,
                conf=args.conf,
                tracker=args.tracker,
                persist=True,
                verbose=False,
            )[0]
        else:
            results = self.model.predict(color_image, conf=args.conf, verbose=False)[0]

        people = []
        obstacles = []

        frame_h, frame_w = color_image.shape[:2]
        frame_area = max(1.0, float(frame_w * frame_h))

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = self.class_names.get(cls_id, str(cls_id)) if isinstance(self.class_names, dict) else str(self.class_names[cls_id])

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            area_ratio = (box_w * box_h) / frame_area

            z = depth_median_bottom_band(
                depth_image, (x1, y1, x2, y2), self.depth_scale, band_ratio=args.depth_bottom_band
            )
            if z is None:
                z = median_depth_from_bbox(depth_image, (x1, y1, x2, y2), self.depth_scale)
            if z is None:
                continue

            u = (x1 + x2) // 2
            v = int(y2 * 0.9)

            z_uv = depth_median_around_uv(depth_image, u, v, self.depth_scale, win=args.depth_uv_win)
            if z_uv is not None:
                z = z_uv

            width_m = (box_w / self.intrinsics.fx) * z
            height_m = (box_h / self.intrinsics.fy) * z
            max_size_m = max(width_m, height_m)

            point_3d = pixel_to_3d(u, v, z, self.intrinsics)
            item = {
                "bbox": (x1, y1, x2, y2),
                "name": name,
                "conf": conf,
                "z": z,
                "point_3d": point_3d,
                "rep_uv": (u, v),
                "track_id": None,
                "is_fixed": name in self.fixed_classes,
                "velocity": None,
                "collision_samples": [],
            }
            if args.use_yolo_track and box.id is not None:
                item["track_id"] = int(box.id[0].item())

            is_person = name == PERSON_CLASS_NAME
            is_obstacle = args.all_non_person or name in self.obstacle_classes

            if is_obstacle:
                if area_ratio < args.min_obstacle_area_ratio:
                    continue
                if max_size_m < args.min_obstacle_size_m:
                    continue

            if is_person or is_obstacle:
                gx = self.person_grid_x if is_person else self.obstacle_grid_x
                gy = self.person_grid_y if is_person else self.obstacle_grid_y
                item["collision_samples"] = build_collision_samples(
                    depth_image=depth_image,
                    bbox=item["bbox"],
                    intrinsics=self.intrinsics,
                    depth_scale=self.depth_scale,
                    grid_x=gx,
                    grid_y=gy,
                    win=args.depth_uv_win,
                    near_percentile=args.depth_near_percentile,
                    near_weight=self.near_weight,
                    base_z=z,
                    sample_z_max_offset=self.sample_z_max_offset,
                )
                if not item["collision_samples"]:
                    item["collision_samples"] = [{"uv": (u, v), "z": z, "point_3d": point_3d}]

            if is_person:
                people.append(item)
            elif is_obstacle:
                obstacles.append(item)

        now = time.time()
        dt = max(1e-3, now - self.prev_time)

        curr_person_points, self.next_person_track_id = assign_tracks(
            people, self.prev_person_points, self.next_person_track_id, args.match_dist
        )
        curr_obstacle_points, self.next_obstacle_track_id = assign_tracks(
            obstacles, self.prev_obstacle_points, self.next_obstacle_track_id, args.match_dist
        )

        alpha = float(np.clip(args.vel_alpha, 0.0, 0.99))
        for p in people:
            tid = int(p["track_id"])
            prev_pt = self.prev_person_points.get(tid)
            if prev_pt is not None:
                v_raw = (p["point_3d"] - prev_pt) / dt
                v_prev = self.vel_ema_person.get(tid)
                v_smooth = v_raw if v_prev is None else (alpha * v_prev + (1.0 - alpha) * v_raw)
                self.vel_ema_person[tid] = v_smooth
                p["velocity"] = v_smooth
            else:
                p["velocity"] = self.vel_ema_person.get(tid, None)

        for o in obstacles:
            tid = int(o["track_id"])
            if o.get("is_fixed"):
                self.vel_ema_obstacle[tid] = np.zeros(3, dtype=np.float32)
                o["velocity"] = np.zeros(3, dtype=np.float32)
                continue
            prev_pt = self.prev_obstacle_points.get(tid)
            if prev_pt is not None:
                v_raw = (o["point_3d"] - prev_pt) / dt
                v_prev = self.vel_ema_obstacle.get(tid)
                v_smooth = v_raw if v_prev is None else (alpha * v_prev + (1.0 - alpha) * v_raw)
                self.vel_ema_obstacle[tid] = v_smooth
                o["velocity"] = v_smooth
            else:
                o["velocity"] = self.vel_ema_obstacle.get(tid, None)

        min_distance = None
        ttc = None
        closing_speed = None
        nearest_pair = None

        if self.lock_pair is not None and self.lock_until > 0:
            lock_person_id, lock_obs_id = self.lock_pair
            lock_person = next((p for p in people if int(p.get("track_id", -1)) == lock_person_id), None)
            lock_obs = next((o for o in obstacles if int(o.get("track_id", -1)) == lock_obs_id), None)

            if lock_person is not None and lock_obs is not None:
                prev_p = self.prev_person_points.get(lock_person_id)
                prev_o = self.prev_obstacle_points.get(lock_obs_id)

                p_disp = float(np.linalg.norm(lock_person["point_3d"] - prev_p)) if prev_p is not None else 0.0
                o_disp = float(np.linalg.norm(lock_obs["point_3d"] - prev_o)) if prev_o is not None else 0.0

                if p_disp <= args.lock_max_dist and o_disp <= args.lock_max_dist:
                    forward_vec = lock_person["velocity"]
                    if forward_vec is None or float(np.linalg.norm(forward_vec)) < 1e-3:
                        forward_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                    lock_dist, lock_p_sample, lock_o_sample = min_distance_between_items(
                        lock_person, lock_obs, distance_percentile=self.pair_distance_percentile
                    )
                    if lock_dist is None:
                        self.lock_pair = None
                        self.lock_until = 0
                    else:
                        obs_point_for_angle = (
                            lock_o_sample["point_3d"] if lock_o_sample is not None else lock_obs["point_3d"]
                        )
                        angle = angle_from_forward_vector(forward_vec, lock_person["point_3d"], obs_point_for_angle)
                        if angle <= args.front_angle:
                            min_distance = lock_dist
                            nearest_pair = (lock_person, lock_obs, angle, forward_vec, lock_p_sample, lock_o_sample)
                            self.lock_until -= 1
                        else:
                            self.lock_pair = None
                            self.lock_until = 0
                else:
                    self.lock_pair = None
                    self.lock_until = 0
            else:
                self.lock_pair = None
                self.lock_until = 0

        if nearest_pair is None:
            for person in people:
                for obs in obstacles:
                    dist, p_sample, o_sample = min_distance_between_items(
                        person, obs, distance_percentile=self.pair_distance_percentile
                    )
                    if dist is None:
                        continue
                    forward_vec = person["velocity"]
                    if forward_vec is None or float(np.linalg.norm(forward_vec)) < 1e-3:
                        forward_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    obs_point_for_angle = o_sample["point_3d"] if o_sample is not None else obs["point_3d"]
                    angle = angle_from_forward_vector(forward_vec, person["point_3d"], obs_point_for_angle)
                    if angle > args.front_angle:
                        continue

                    if min_distance is None or dist < min_distance:
                        min_distance = dist
                        nearest_pair = (person, obs, angle, forward_vec, p_sample, o_sample)

            if nearest_pair is not None:
                self.lock_pair = (int(nearest_pair[0]["track_id"]), int(nearest_pair[1]["track_id"]))
                self.lock_until = max(0, int(args.lock_frames))

        if nearest_pair is not None:
            person, obs, _, _, p_sample, o_sample = nearest_pair
            person_point = p_sample["point_3d"] if p_sample is not None else person["point_3d"]
            obs_point = o_sample["point_3d"] if o_sample is not None else obs["point_3d"]
            person_for_ttc = {"point_3d": person_point, "velocity": person["velocity"]}
            obs_for_ttc = {"point_3d": obs_point, "velocity": obs["velocity"]}

            ttc_f = ttc_forward(person_for_ttc, obs_for_ttc)
            ttc_l = ttc_los(person_for_ttc, obs_for_ttc)
            closing_speed = closing_speed_los(person_for_ttc, obs_for_ttc)

            if args.ttc_mode == "forward":
                ttc = ttc_f
            elif args.ttc_mode == "los":
                ttc = ttc_l
            else:
                candidates = [x for x in [ttc_f, ttc_l] if x is not None and x > 0]
                ttc = min(candidates) if candidates else None

        self.risk_score_raw, _ = compute_risk_score(
            min_distance=min_distance,
            ttc=ttc,
            closing_speed=closing_speed,
            warn_dist=args.warn_dist,
            danger_dist=args.danger_dist,
            warn_ttc=args.warn_ttc,
            danger_ttc=args.danger_ttc,
            dist_weight=self.score_dist_weight,
            ttc_weight=self.score_ttc_weight,
            close_weight=self.score_close_weight,
            close_ref=self.score_close_ref,
        )
        self.risk_score_smooth = (
            self.score_alpha * self.risk_score_smooth + (1.0 - self.score_alpha) * self.risk_score_raw
        )
        score_level = score_to_level(
            score=self.risk_score_smooth,
            stable_level=self.stable_level,
            warn_on=self.score_warn_on,
            danger_on=self.score_danger_on,
            warn_off=self.score_warn_off,
            danger_off=self.score_danger_off,
        )
        level = self._stabilize_level(score_level)
        color = risk_color(level)
        draw_all_overlays = not bool(getattr(args, "no_overlay", False))
        hide_hud_panel = bool(getattr(args, "hide_hud_panel", False))
        draw_detection_overlay = draw_all_overlays
        draw_info_overlay = draw_all_overlays and not hide_hud_panel
        rep_distance = None
        if nearest_pair is not None:
            person, obs, _, _, _, _ = nearest_pair
            rep_distance = float(np.linalg.norm(person["point_3d"] - obs["point_3d"]))

        if draw_detection_overlay:
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

        if draw_detection_overlay and nearest_pair is not None:
            person, obs, _, _, _, _ = nearest_pair
            px1, py1, px2, py2 = person["bbox"]
            ox1, oy1, ox2, oy2 = obs["bbox"]
            p_center = person.get("rep_uv") if person.get("rep_uv") is not None else (
                (px1 + px2) // 2, int(py2 * 0.9)
            )
            o_center = obs.get("rep_uv") if obs.get("rep_uv") is not None else (
                (ox1 + ox2) // 2, int(oy2 * 0.9)
            )

            cv2.line(color_image, p_center, o_center, color, 3)
            cv2.circle(color_image, p_center, 4, (80, 255, 80), -1)
            cv2.circle(color_image, o_center, 4, (80, 180, 255), -1)
            cv2.putText(color_image, f"P({p_center[0]},{p_center[1]})",
                        (p_center[0] + 6, max(18, p_center[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 80), 1)
            cv2.putText(color_image, f"O({o_center[0]},{o_center[1]})",
                        (o_center[0] + 6, max(18, o_center[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 180, 255), 1)

        self.history.append(min_distance)
        if draw_info_overlay:
            panel_h = 174
            h, w = color_image.shape[:2]
            canvas = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
            canvas[:h] = color_image

            cv2.putText(canvas, f"RISK: {level}", (12, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(canvas, f"Rep distance: {rep_distance:.2f} m" if rep_distance is not None else "Rep distance: N/A",
                        (12, h + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
            cv2.putText(canvas, f"Min distance: {min_distance:.2f} m" if min_distance is not None else "Min distance: N/A",
                        (12, h + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
            cv2.putText(canvas, f"TTC: {ttc:.2f} s" if ttc is not None else "TTC: N/A",
                        (12, h + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
            cv2.putText(canvas, f"Risk score: {self.risk_score_smooth:.2f} (raw {self.risk_score_raw:.2f})",
                        (12, h + 148), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 2)

            graph_w = min(360, w // 2)
            graph_h = panel_h - 16
            graph_x = w - graph_w - 12
            graph_y = h + 8
            draw_distance_graph(canvas, self.history, graph_x, graph_y, graph_w, graph_h)
        else:
            canvas = color_image

        self.last_beep_ts = maybe_beep(args.beep, level, now, self.last_beep_ts, args.beep_cooldown)

        if self.log_writer is not None:
            if nearest_pair is None:
                self.log_writer.writerow([f"{now:.3f}", level, "", "", "", "", "", ""])
            else:
                person, obs, angle, _, _, _ = nearest_pair
                self.log_writer.writerow([
                    f"{now:.3f}",
                    level,
                    f"{min_distance:.3f}" if min_distance is not None else "",
                    f"{ttc:.3f}" if ttc is not None else "",
                    person["track_id"],
                    obs["track_id"],
                    obs["name"],
                    f"{angle:.2f}",
                ])
            self.log_file.flush()

        result = {
            "ts_epoch": float(now),
            "risk": level,
            "risk_score": float(self.risk_score_smooth),
            "risk_score_raw": float(self.risk_score_raw),
            "min_distance_m": float(min_distance) if min_distance is not None else None,
            "rep_distance_m": float(rep_distance) if rep_distance is not None else None,
            "ttc_s": float(ttc) if ttc is not None else None,
            "person_track_id": int(nearest_pair[0]["track_id"]) if nearest_pair is not None else None,
            "obstacle_track_id": int(nearest_pair[1]["track_id"]) if nearest_pair is not None else None,
            "obstacle_name": nearest_pair[1]["name"] if nearest_pair is not None else None,
            "angle_deg": float(nearest_pair[2]) if nearest_pair is not None else None,
        }
        self.last_result = result

        if self.on_result is not None:
            self.on_result(result, canvas)

        if self.display:
            cv2.imshow("Collision Risk Monitor", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                return False

        self.prev_time = now
        self.prev_person_points = curr_person_points
        self.prev_obstacle_points = curr_obstacle_points
        return not self._stop_requested
