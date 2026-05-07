"""
collision_monitor/frame_processor.py
=====================================
YOLO 탐지 + ByteTrack 트래킹 + 위험도 계산의 핵심 로직.

PipelineRunner(RealSense)와 VideoRunner(영상 파일) 양쪽에서 공유한다.
depth_image=None 으로 호출하면 depth 의존 계산(거리/TTC/위험도)을 건너뛴다.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from .byte_tracker import TrackHistory
from .config import FIXED_CLASSES_DEFAULT, OBSTACLE_CLASSES_DEFAULT, PERSON_CLASS_NAME, PERSON_CLASS_ALIASES
from .output import maybe_beep, maybe_open_log
from .visualizer import draw_detections, draw_hud_panel
from .risk import (
    closing_speed_los,
    signed_closing_speed_los,
    compute_risk_score,
    risk_color,
    score_to_level,
    ttc_forward,
    ttc_los,
)
from .tracking import assign_tracks


class FrameProcessor:
    """
    한 프레임에 대해 YOLO 탐지 → ByteTrack 트래킹 → 위험도 계산 → 시각화를 수행한다.

    process(color_image, depth_image, intrinsics, depth_scale) 를 호출한다.
    depth_image=None 이면 depth 의존 기능(거리·TTC·risk)은 건너뛴다.
    """

    def __init__(self, args):
        self.args = args

        # ── 위험도 파라미터 ───────────────────────────────────────────────────
        self.risk_up_frames    = max(1, int(args.risk_up_frames))
        self.risk_down_frames  = max(1, int(args.risk_down_frames))
        self.score_alpha       = float(np.clip(args.score_alpha, 0.0, 0.99))
        self.score_dist_weight = max(0.0, float(args.score_dist_weight))
        self.score_ttc_weight  = max(0.0, float(args.score_ttc_weight))
        self.score_close_weight= max(0.0, float(args.score_close_weight))
        self.score_close_ref   = max(1e-3, float(args.score_close_ref))
        self.score_warn_on     = float(np.clip(args.score_warn_on, 0.0, 1.0))
        self.score_danger_on   = max(self.score_warn_on,
                                     float(np.clip(args.score_danger_on, 0.0, 1.0)))
        self.score_warn_off    = min(self.score_warn_on,
                                     float(np.clip(args.score_warn_off, 0.0, 1.0)))
        self.score_danger_off  = min(
            self.score_danger_on,
            max(self.score_warn_off, float(np.clip(args.score_danger_off, 0.0, 1.0))),
        )
        self.near_weight           = float(np.clip(args.depth_near_weight, 0.0, 1.0))
        self.sample_z_max_offset   = max(0.0, float(args.sample_z_max_offset))
        self.depth_fusion          = str(getattr(args, "depth_fusion", "fallback")).lower()
        self.model_depth_weight    = float(np.clip(getattr(args, "model_depth_weight", 0.35), 0.0, 1.0))
        self.pair_distance_percentile = float(
            np.clip(args.pair_distance_percentile, 1.0, 50.0)
        )
        self.proximity_gate = max(0.0, float(getattr(args, "proximity_gate", 0.0)))
        self.line_max_dist = max(0.0, float(getattr(args, "line_max_dist", 0.0)))
        self.line_smooth_alpha = float(np.clip(getattr(args, "line_smooth_alpha", 0.75), 0.0, 0.99))
        self.display_distance_smooth_alpha = float(np.clip(getattr(args, "display_distance_smooth_alpha", 0.85), 0.0, 0.99))
        self.display_distance_step = max(0.0, float(getattr(args, "display_distance_step", 0.05)))
        self.distance_smooth_alpha = float(np.clip(getattr(args, "distance_smooth_alpha", 0.55), 0.0, 0.99))
        self.receding_speed_threshold = max(0.0, float(getattr(args, "receding_speed_threshold", 0.15)))
        self.receding_risk_scale = float(np.clip(getattr(args, "receding_risk_scale", 0.65), 0.0, 1.0))
        self.confidence_risk_floor = float(np.clip(getattr(args, "confidence_risk_floor", 0.65), 0.0, 1.0))
        self.person_grid_x   = max(1, int(args.person_grid_x))
        self.person_grid_y   = max(1, int(args.person_grid_y))
        self.obstacle_grid_x = max(1, int(args.obstacle_grid_x))
        self.obstacle_grid_y = max(1, int(args.obstacle_grid_y))
        self.vehicle_box_expand = max(1.0, float(getattr(args, "vehicle_box_expand", 1.0)))
        self.vehicle_box_expand_x = max(0.0, float(getattr(args, "vehicle_box_expand_x", 0.0)))
        self.vehicle_expand_classes = {"car", "truck", "bus", "train", "motorcycle"}

        # ── 클래스 분류 ───────────────────────────────────────────────────────
        if args.all_non_person:
            self.obstacle_classes: set[str] = set()
        else:
            self.obstacle_classes = (
                {x.strip() for x in args.obstacle_classes.split(",") if x.strip()}
                if args.obstacle_classes else OBSTACLE_CLASSES_DEFAULT
            )
        self.fixed_classes: set[str] = (
            {x.strip() for x in args.fixed_classes.split(",") if x.strip()}
            if args.fixed_classes else FIXED_CLASSES_DEFAULT
        )

        # ── YOLO 모델 ─────────────────────────────────────────────────────────
        self.model = YOLO(args.model)
        self.class_names = getattr(self.model, "names", None) or self.model.model.names

        import torch
        if torch.cuda.is_available():
            self._infer_device = 0
            self._use_half     = True
            print(f"[FrameProcessor] GPU: {torch.cuda.get_device_name(0)} | FP16 ON")
        else:
            self._infer_device = "cpu"
            self._use_half     = False
            print("[FrameProcessor] CPU 모드")


        # ── SAHI (소형 객체 탐지 강화) ────────────────────────────────────────
        self._use_sahi = bool(getattr(args, "use_sahi", False))
        self._sahi_model = None
        if self._use_sahi:
            try:
                from sahi import AutoDetectionModel
                device_str = f"cuda:{self._infer_device}" if isinstance(self._infer_device, int) else "cpu"
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=args.model,
                    confidence_threshold=args.conf,
                    device=device_str,
                )
                self._sahi_slice_size = int(getattr(args, "sahi_slice_size", 320))
                self._sahi_overlap    = float(getattr(args, "sahi_overlap", 0.2))
                print(f"[FrameProcessor] SAHI 활성화 | slice={self._sahi_slice_size} overlap={self._sahi_overlap}")
            except ImportError:
                print("[FrameProcessor][WARN] sahi 패키지 없음. pip install sahi 후 재시작 필요.")
                self._use_sahi = False

        # ── PPE 모델 (선택) ───────────────────────────────────────────────────
        from .ppe_processor import PPEProcessor
        ppe_model_path = getattr(args, "ppe_model", "")
        if ppe_model_path:
            self.ppe_processor: PPEProcessor | None = PPEProcessor(
                YOLO(ppe_model_path), frame_interval=3
            )
        else:
            self.ppe_processor = None

        # ── 트래킹 상태 ───────────────────────────────────────────────────────
        trail_len = int(getattr(args, "trail_len", 30))
        self.track_history = TrackHistory(
            trail_maxlen=trail_len, dead_track_ttl=45, bbox_alpha=0.7
        )
        self.prev_person_points    : dict = {}
        self.prev_obstacle_points  : dict = {}
        self.next_person_track_id    = 1
        self.next_obstacle_track_id  = 1
        self.vel_ema_person   : dict = {}
        self.vel_ema_obstacle : dict = {}

        # ── 위험도 상태 ───────────────────────────────────────────────────────
        self.stable_level      = "SAFE"
        self._pending_level    = None
        self._pending_count    = 0
        self.risk_score_raw    = 0.0
        self.risk_score_smooth = 0.0
        self.lock_pairs        : dict = {}  # {(person_id, obs_id): remaining_lock_frames}
        self.line_uv_smooth    : dict = {}  # {(person_id, obs_id): (person_uv, obstacle_uv)}
        self.pair_dist_smooth  : dict = {}  # {(person_id, obs_id): smoothed_distance_m}
        self.display_dist_smooth: dict = {}  # {(person_id, obs_id): visual_distance_m}

        # ── 기타 ─────────────────────────────────────────────────────────────
        self.history      = deque(maxlen=args.history_size)
        self.prev_time    = time.time()
        self.last_beep_ts = 0.0
        self.log_file, self.log_writer = maybe_open_log(getattr(args, "log_file", ""))
        self._log_flush_counter  = 0
        self._log_flush_interval = 30

    # ── 위험도 안정화 ─────────────────────────────────────────────────────────

    @staticmethod
    def _risk_rank(level: str) -> int:
        return {"SAFE": 0, "WARNING": 1, "DANGER": 2}.get(level, 0)

    def _stabilize_level(self, raw_level: str) -> str:
        if raw_level == self.stable_level:
            self._pending_level = None
            self._pending_count = 0
            return self.stable_level
        if raw_level == self._pending_level:
            self._pending_count += 1
        else:
            self._pending_level = raw_level
            self._pending_count = 1
        need = (self.risk_up_frames
                if self._risk_rank(raw_level) > self._risk_rank(self.stable_level)
                else self.risk_down_frames)
        if self._pending_count >= need:
            self.stable_level = raw_level
            self._pending_level = None
            self._pending_count = 0
        return self.stable_level

    def _expand_vehicle_bbox(self, bbox, frame_w, frame_h):
        if self.vehicle_box_expand <= 1.0:
            return bbox
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        extra_h = h * (self.vehicle_box_expand - 1.0)
        extra_x = w * self.vehicle_box_expand_x

        nx1 = int(round(x1 - extra_x))
        nx2 = int(round(x2 + extra_x))
        # Partial vehicle detections often sit on wheels, so grow mostly upward.
        ny1 = int(round(y1 - extra_h))
        ny2 = int(round(y2 + extra_h * 0.15))

        nx1 = max(0, min(frame_w - 1, nx1))
        nx2 = max(nx1 + 1, min(frame_w, nx2))
        ny1 = max(0, min(frame_h - 1, ny1))
        ny2 = max(ny1 + 1, min(frame_h, ny2))
        return (nx1, ny1, nx2, ny2)

    def _infer_sahi(self, color_image: np.ndarray, conf: float) -> list:
        """SAHI sliced inference → ultralytics box 호환 래퍼 리스트 반환"""
        import torch
        from sahi.predict import get_sliced_prediction

        rgb = color_image[:, :, ::-1]  # BGR → RGB
        result = get_sliced_prediction(
            rgb,
            self._sahi_model,
            slice_height=self._sahi_slice_size,
            slice_width=self._sahi_slice_size,
            overlap_height_ratio=self._sahi_overlap,
            overlap_width_ratio=self._sahi_overlap,
            verbose=0,
        )

        # class name → id 역매핑
        name_to_id = {v: k for k, v in self.class_names.items()}

        boxes = []
        for pred in result.object_prediction_list:
            if pred.score.value < conf:
                continue
            cls_name = pred.category.name
            cls_id   = name_to_id.get(cls_name, pred.category.id)
            x1, y1, x2, y2 = pred.bbox.to_xyxy()

            class _Box:
                pass

            b = _Box()
            b.cls  = torch.tensor([cls_id], dtype=torch.float32)
            b.conf = torch.tensor([pred.score.value], dtype=torch.float32)
            b.xyxy = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            b.id   = None  # SAHI는 tracking ID 없음 → assign_tracks로 대체
            boxes.append(b)

        return boxes

    def reset_tracking(self):
        """영상 루프 재시작 등 트래킹 상태를 초기화할 때 호출."""
        self.track_history.reset()
        self.prev_person_points   = {}
        self.prev_obstacle_points = {}
        self.next_person_track_id   = 1
        self.next_obstacle_track_id = 1
        self.vel_ema_person   = {}
        self.vel_ema_obstacle = {}
        self.lock_pairs.clear()
        self.line_uv_smooth.clear()
        self.pair_dist_smooth.clear()
        self.display_dist_smooth.clear()

    def close(self):
        if self.log_file is not None:
            self.log_file.close()

    # ── 핵심 처리 ─────────────────────────────────────────────────────────────

    def process(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray | None = None,
        intrinsics=None,
        depth_scale: float | None = None,
        model_depth_image: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        한 프레임을 처리하고 (canvas, result_dict) 를 반환한다.

        depth_image=None 이면 거리·TTC·위험도 계산을 건너뛴다.
        """
        args     = self.args
        has_depth = (depth_image is not None
                     and intrinsics is not None
                     and depth_scale is not None)
        now = time.time()
        dt  = min(max(1e-3, now - self.prev_time), 0.5)  # 최대 0.5s: 프레임 드롭 시 속도 추적 폭발 방지

        # ── YOLO + ByteTrack (또는 SAHI) ─────────────────────────────────────
        imgsz     = int(getattr(args, "imgsz", 1280))
        use_track = getattr(args, "use_yolo_track", True)

        if self._use_sahi and self._sahi_model is not None:
            raw_boxes = self._infer_sahi(color_image, args.conf)
        elif use_track:
            try:
                _r = self.model.track(
                    color_image,
                    conf=args.conf,
                    imgsz=imgsz,
                    iou=0.45,
                    agnostic_nms=True,
                    tracker=args.tracker,
                    persist=True,
                    verbose=False,
                    device=self._infer_device,
                    half=self._use_half,
                )[0]
            except Exception as e:
                print(f"[FrameProcessor][WARN] track() 실패, predict() fallback: {e}")
                _r = self.model.predict(
                    color_image, conf=args.conf, imgsz=imgsz, iou=0.45, verbose=False,
                    device=self._infer_device, half=self._use_half,
                )[0]
            raw_boxes = _r.boxes
        else:
            _r = self.model.predict(
                color_image, conf=args.conf, imgsz=imgsz, iou=0.45, verbose=False,
                device=self._infer_device, half=self._use_half,
            )[0]
            raw_boxes = _r.boxes

        # ── 탐지 결과 파싱 → people / obstacles 분리 ────────────────────────
        frame_h, frame_w = color_image.shape[:2]
        frame_area = max(1.0, float(frame_w * frame_h))
        people    : list[dict] = []
        obstacles : list[dict] = []

        for box in raw_boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            name   = (self.class_names.get(cls_id, str(cls_id))
                      if isinstance(self.class_names, dict)
                      else str(self.class_names[cls_id]))
            # VisDrone 클래스명 통일 (pedestrian/people → person)
            if name in PERSON_CLASS_ALIASES:
                name = PERSON_CLASS_NAME
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            det_bbox = (x1, y1, x2, y2)
            if name in self.vehicle_expand_classes:
                x1, y1, x2, y2 = self._expand_vehicle_bbox(
                    det_bbox, frame_w=frame_w, frame_h=frame_h
                )
            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            area_ratio = (box_w * box_h) / frame_area

            u = (x1 + x2) // 2
            v = int(y2 * 0.9)  # 발바닥 근처

            # depth 계산
            z = None
            model_z = None
            point_3d = None
            collision_samples = []

            if has_depth:
                from .depth import (depth_median_around_uv,
                                    depth_median_bottom_band,
                                    median_depth_from_bbox)
                from .geometry import (build_collision_samples,
                                       pixel_to_3d)

                z = depth_median_bottom_band(
                    depth_image, (x1, y1, x2, y2), depth_scale,
                    band_ratio=args.depth_bottom_band
                )
                if z is None:
                    z = median_depth_from_bbox(
                        depth_image, (x1, y1, x2, y2), depth_scale
                    )
                if model_depth_image is not None:
                    model_z = depth_median_bottom_band(
                        model_depth_image, (x1, y1, x2, y2), 1.0,
                        band_ratio=args.depth_bottom_band,
                    )
                    if model_z is None:
                        model_z = median_depth_from_bbox(
                            model_depth_image, (x1, y1, x2, y2), 1.0
                        )

                model_depth_scale_factor = 1.0
                if z is not None and model_z is not None and model_z > 0:
                    model_depth_scale_factor = float(z) / float(model_z)
                elif z is None and model_z is not None and self.depth_fusion != "realsense":
                    z = float(model_z)

                if z is None:
                    continue

                z_uv = depth_median_around_uv(
                    depth_image, u, v, depth_scale, win=args.depth_uv_win
                )
                if z_uv is not None:
                    z = z_uv
                    if model_z is not None and model_z > 0:
                        model_depth_scale_factor = float(z) / float(model_z)

                width_m  = (box_w / intrinsics.fx) * z
                height_m = (box_h / intrinsics.fy) * z
                max_size_m = max(width_m, height_m)

                point_3d = pixel_to_3d(u, v, z, intrinsics)

                is_person   = (name in PERSON_CLASS_ALIASES)
                is_obstacle = args.all_non_person or (name in self.obstacle_classes)
                if is_obstacle:
                    if area_ratio < args.min_obstacle_area_ratio:
                        continue
                    if max_size_m < args.min_obstacle_size_m:
                        continue

                if is_person or is_obstacle:
                    gx = self.person_grid_x if is_person else self.obstacle_grid_x
                    gy = self.person_grid_y if is_person else self.obstacle_grid_y
                    collision_samples = build_collision_samples(
                        depth_image=depth_image,
                        bbox=(x1, y1, x2, y2),
                        intrinsics=intrinsics,
                        depth_scale=depth_scale,
                        grid_x=gx, grid_y=gy,
                        win=args.depth_uv_win,
                        near_percentile=args.depth_near_percentile,
                        near_weight=self.near_weight,
                        base_z=z,
                        sample_z_max_offset=self.sample_z_max_offset,
                        model_depth_image=model_depth_image,
                        model_depth_scale=1.0,
                        model_depth_scale_factor=model_depth_scale_factor,
                        depth_fusion=self.depth_fusion,
                        model_depth_weight=self.model_depth_weight,
                    )
                    if not collision_samples:
                        collision_samples = [{"uv": (u, v), "z": z, "point_3d": point_3d}]
            else:
                # depth 없음: 픽셀 좌표를 point_3d 대신 사용 (assign_tracks 호환)
                point_3d = np.array([float(u), float(v), 0.0], dtype=np.float32)
                is_person   = (name in PERSON_CLASS_ALIASES)
                is_obstacle = args.all_non_person or (name in self.obstacle_classes)
                # depth 없을 때도 면적 비율로 너무 작은 장애물 제거
                if is_obstacle and area_ratio < args.min_obstacle_area_ratio:
                    continue

            if box.id is not None:
                track_id = int(box.id[0].item())
            else:
                track_id = None

            # 비교용 모델 depth (RealSense live 모드에서 depth 모델과 비교)
            if model_depth_image is not None and model_z is None:
                from .depth import depth_median_bottom_band, median_depth_from_bbox
                model_z = depth_median_bottom_band(
                    model_depth_image, (x1, y1, x2, y2), 1.0,
                    band_ratio=args.depth_bottom_band,
                )
                if model_z is None:
                    model_z = median_depth_from_bbox(
                        model_depth_image, (x1, y1, x2, y2), 1.0
                    )

            item = {
                "bbox"             : (x1, y1, x2, y2),
                "det_bbox"         : det_bbox,
                "name"             : name,
                "conf"             : conf,
                "z"                : z,
                "model_z"          : model_z,
                "point_3d"         : point_3d,
                "rep_uv"           : (u, v),
                "track_id"         : track_id,
                "is_fixed"         : name in self.fixed_classes,
                "velocity"         : None,
                "collision_samples": collision_samples,
            }

            if is_person:
                people.append(item)
            elif is_obstacle:
                obstacles.append(item)

        # ── assign_tracks ────────────────────────────────────────────────────
        curr_person_points, self.next_person_track_id = assign_tracks(
            people, self.prev_person_points,
            self.next_person_track_id, args.match_dist
        )
        curr_obstacle_points, self.next_obstacle_track_id = assign_tracks(
            obstacles, self.prev_obstacle_points,
            self.next_obstacle_track_id, args.match_dist
        )

        # ── TrackHistory 업데이트 ────────────────────────────────────────────
        active_ids: list[int] = []
        for item in people + obstacles:
            tid = item.get("track_id")
            if tid is None:
                continue
            track_data = self.track_history.update(
                track_id   = int(tid),
                class_name = item["name"],
                bbox       = item["bbox"],
                conf       = item["conf"],
            )
            item["bbox"]   = track_data.bbox
            item["rep_uv"] = track_data.center
            item["trail"]  = track_data.trail
            active_ids.append(int(tid))

        self.track_history.step(active_ids)

        # ── 속도 계산 (depth 있을 때만) ───────────────────────────────────────
        if has_depth:
            alpha = float(np.clip(args.vel_alpha, 0.0, 0.99))
            for p in people:
                tid = p.get("track_id")
                if tid is None:
                    continue
                prev_pt = self.prev_person_points.get(tid)
                if prev_pt is not None:
                    v_raw  = (p["point_3d"] - prev_pt) / dt
                    v_prev = self.vel_ema_person.get(tid)
                    v_s    = v_raw if v_prev is None else (alpha * v_prev + (1.0 - alpha) * v_raw)
                    self.vel_ema_person[tid] = v_s
                    p["velocity"] = v_s
                else:
                    p["velocity"] = self.vel_ema_person.get(tid)

            for o in obstacles:
                tid = o.get("track_id")
                if tid is None:
                    continue
                if o.get("is_fixed"):
                    self.vel_ema_obstacle[tid] = np.zeros(3, dtype=np.float32)
                    o["velocity"] = np.zeros(3, dtype=np.float32)
                    continue
                prev_pt = self.prev_obstacle_points.get(tid)
                if prev_pt is not None:
                    v_raw  = (o["point_3d"] - prev_pt) / dt
                    v_prev = self.vel_ema_obstacle.get(tid)
                    v_s    = v_raw if v_prev is None else (alpha * v_prev + (1.0 - alpha) * v_raw)
                    self.vel_ema_obstacle[tid] = v_s
                    o["velocity"] = v_s
                else:
                    o["velocity"] = self.vel_ema_obstacle.get(tid)

        # ── 거리·TTC·위험도 계산 (모든 사람-장비 쌍, 쌍별 lock 적용) ───────────
        min_distance  = None
        ttc           = None
        closing_speed = None
        nearest_pair  = None
        rep_distance  = None
        selected_pair_score = None
        all_risk_pairs: list[dict] = []

        if has_depth and people and obstacles:
            from .geometry import angle_from_forward_vector, min_distance_between_items

            active_keys: set = set()

            for person in people:
                for obs in obstacles:
                    p_tid = int(person.get("track_id", -1))
                    o_tid = int(obs.get("track_id", -1))
                    pair_key = (p_tid, o_tid)
                    lock_remaining = self.lock_pairs.get(pair_key, 0)

                    if self.proximity_gate > 0.0:
                        rep_delta = person["point_3d"] - obs["point_3d"]
                        rep_dist = float(np.sqrt(rep_delta[0]**2 + rep_delta[2]**2))
                        if rep_dist > self.proximity_gate:
                            self.lock_pairs.pop(pair_key, None)
                            self.pair_dist_smooth.pop(pair_key, None)
                            self.line_uv_smooth.pop(pair_key, None)
                            self.display_dist_smooth.pop(pair_key, None)
                            continue

                    dist, ps, os_ = min_distance_between_items(
                        person, obs,
                        distance_percentile=self.pair_distance_percentile
                    )
                    if dist is None:
                        self.lock_pairs.pop(pair_key, None)
                        self.pair_dist_smooth.pop(pair_key, None)
                        self.line_uv_smooth.pop(pair_key, None)
                        self.display_dist_smooth.pop(pair_key, None)
                        continue

                    raw_dist = float(dist)
                    prev_dist = self.pair_dist_smooth.get(pair_key)
                    if prev_dist is not None and self.distance_smooth_alpha > 0.0:
                        dist = (
                            self.distance_smooth_alpha * float(prev_dist)
                            + (1.0 - self.distance_smooth_alpha) * raw_dist
                        )
                    else:
                        dist = raw_dist
                    self.pair_dist_smooth[pair_key] = float(dist)

                    fwd = person["velocity"]
                    if fwd is None or float(np.linalg.norm(fwd)) < 1e-3:
                        fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    obs_pt = os_["point_3d"] if os_ else obs["point_3d"]
                    angle  = angle_from_forward_vector(fwd, person["point_3d"], obs_pt)

                    if angle > args.front_angle:
                        self.lock_pairs.pop(pair_key, None)
                        self.pair_dist_smooth.pop(pair_key, None)
                        self.line_uv_smooth.pop(pair_key, None)
                        self.display_dist_smooth.pop(pair_key, None)
                        continue

                    # 쌍별 lock 갱신
                    if lock_remaining > 0:
                        self.lock_pairs[pair_key] = lock_remaining - 1
                    else:
                        self.lock_pairs[pair_key] = max(0, int(args.lock_frames))
                    active_keys.add(pair_key)

                    # TTC / closing_speed / risk_score 계산
                    pp = ps["point_3d"] if ps else person["point_3d"]
                    op = os_["point_3d"] if os_ else obs["point_3d"]
                    pf = {"point_3d": pp, "velocity": person["velocity"]}
                    of = {"point_3d": op, "velocity": obs["velocity"]}
                    ttc_f    = ttc_forward(pf, of)
                    ttc_l    = ttc_los(pf, of)
                    pair_cs  = closing_speed_los(pf, of)
                    signed_pair_cs = signed_closing_speed_los(pf, of)
                    if args.ttc_mode == "forward":
                        pair_ttc = ttc_f
                    elif args.ttc_mode == "los":
                        pair_ttc = ttc_l
                    else:
                        cands    = [x for x in [ttc_f, ttc_l] if x is not None and x > 0]
                        pair_ttc = min(cands) if cands else None

                    pair_score, _ = compute_risk_score(
                        min_distance=dist, ttc=pair_ttc, closing_speed=pair_cs,
                        warn_dist=args.warn_dist, danger_dist=args.danger_dist,
                        warn_ttc=args.warn_ttc,   danger_ttc=args.danger_ttc,
                        dist_weight=self.score_dist_weight,
                        ttc_weight=self.score_ttc_weight,
                        close_weight=self.score_close_weight,
                        close_ref=self.score_close_ref,
                    )
                    is_receding = (
                        signed_pair_cs is not None
                        and signed_pair_cs < -self.receding_speed_threshold
                        and dist > args.danger_dist
                    )
                    p_sample_count = len(person.get("collision_samples") or [])
                    o_sample_count = len(obs.get("collision_samples") or [])
                    sample_conf = min(1.0, min(p_sample_count, o_sample_count) / 6.0)
                    det_conf = float(np.clip(min(person.get("conf", 1.0), obs.get("conf", 1.0)), 0.0, 1.0))
                    pair_conf = float(np.clip(0.65 * sample_conf + 0.35 * det_conf, 0.0, 1.0))
                    if is_receding:
                        pair_score *= self.receding_risk_scale
                        pair_ttc = None
                    pair_score *= self.confidence_risk_floor + (1.0 - self.confidence_risk_floor) * pair_conf
                    pair_score = float(np.clip(pair_score, 0.0, 1.0))
                    display_dist = float(dist)
                    prev_display_dist = self.display_dist_smooth.get(pair_key)
                    if prev_display_dist is not None and self.display_distance_smooth_alpha > 0.0:
                        a = self.display_distance_smooth_alpha
                        display_dist = a * float(prev_display_dist) + (1.0 - a) * display_dist
                    if self.display_distance_step > 0.0:
                        step = self.display_distance_step
                        display_dist = round(display_dist / step) * step
                    self.display_dist_smooth[pair_key] = float(display_dist)
                    pair_level = score_to_level(
                        score=pair_score,
                        stable_level="SAFE",
                        warn_on=self.score_warn_on,   danger_on=self.score_danger_on,
                        warn_off=self.score_warn_off, danger_off=self.score_danger_off,
                    )
                    display_ps_uv = ps.get("uv") if ps else None
                    display_os_uv = os_.get("uv") if os_ else None
                    if display_ps_uv is not None and display_os_uv is not None:
                        prev_uvs = self.line_uv_smooth.get(pair_key)
                        if prev_uvs is not None:
                            a = self.line_smooth_alpha
                            prev_p_uv, prev_o_uv = prev_uvs
                            display_ps_uv = (
                                int(round(a * prev_p_uv[0] + (1.0 - a) * display_ps_uv[0])),
                                int(round(a * prev_p_uv[1] + (1.0 - a) * display_ps_uv[1])),
                            )
                            display_os_uv = (
                                int(round(a * prev_o_uv[0] + (1.0 - a) * display_os_uv[0])),
                                int(round(a * prev_o_uv[1] + (1.0 - a) * display_os_uv[1])),
                            )
                        self.line_uv_smooth[pair_key] = (display_ps_uv, display_os_uv)
                    all_risk_pairs.append({
                        "person": person, "obs": obs,
                        "dist": dist, "raw_dist": raw_dist, "ttc": pair_ttc,
                        "display_dist": display_dist,
                        "closing_speed": pair_cs, "signed_closing_speed": signed_pair_cs,
                        "score": pair_score, "level": pair_level,
                        "pair_confidence": pair_conf, "is_receding": is_receding,
                        "angle": angle, "fwd": fwd, "ps": ps, "os_": os_,
                        "display_ps_uv": display_ps_uv, "display_os_uv": display_os_uv,
                    })

            # 이번 프레임에 활성화되지 않은 lock 쌍 TTL 감소 및 정리
            for k in list(self.lock_pairs.keys()):
                if k not in active_keys:
                    self.lock_pairs[k] -= 1
                    if self.lock_pairs[k] <= 0:
                        del self.lock_pairs[k]
                        self.line_uv_smooth.pop(k, None)
                        self.pair_dist_smooth.pop(k, None)
                        self.display_dist_smooth.pop(k, None)

            # 위험도 높은 순 정렬 → 최상위 쌍을 대표값으로 사용
            all_risk_pairs.sort(key=lambda x: x["score"], reverse=True)
            if all_risk_pairs:
                top          = all_risk_pairs[0]
                min_distance = top["dist"]
                ttc          = top["ttc"]
                closing_speed = top["closing_speed"]
                selected_pair_score = top["score"]
                rep_distance = float(top["dist"])
                nearest_pair = (
                    top["person"], top["obs"], top["angle"],
                    top["fwd"], top["ps"], top["os_"],
                )

        # ── Risk score ───────────────────────────────────────────────────────
        if has_depth:
            if selected_pair_score is not None:
                self.risk_score_raw = float(selected_pair_score)
            else:
                self.risk_score_raw, _ = compute_risk_score(
                min_distance=min_distance, ttc=ttc,
                closing_speed=closing_speed,
                warn_dist=args.warn_dist, danger_dist=args.danger_dist,
                warn_ttc=args.warn_ttc,  danger_ttc=args.danger_ttc,
                dist_weight=self.score_dist_weight,
                ttc_weight=self.score_ttc_weight,
                close_weight=self.score_close_weight,
                close_ref=self.score_close_ref,
                )
            self.risk_score_smooth = (
                self.score_alpha * self.risk_score_smooth
                + (1.0 - self.score_alpha) * self.risk_score_raw
            )
            score_level = score_to_level(
                score=self.risk_score_smooth,
                stable_level=self.stable_level,
                warn_on=self.score_warn_on,   danger_on=self.score_danger_on,
                warn_off=self.score_warn_off, danger_off=self.score_danger_off,
            )
            level = self._stabilize_level(score_level)
        else:
            self.risk_score_raw    = 0.0
            self.risk_score_smooth = 0.0
            level = "SAFE"

        color = risk_color(level)

        # ── 시각화 ───────────────────────────────────────────────────────────
        draw_all_overlays   = not bool(getattr(args, "no_overlay", False))
        hide_hud_panel      = bool(getattr(args, "hide_hud_panel", False))
        draw_detection      = draw_all_overlays
        draw_info_panel     = draw_all_overlays and not hide_hud_panel and has_depth

        canvas = color_image.copy()

        if draw_detection:
            canvas = draw_detections(
                canvas, people, obstacles, all_risk_pairs, color, has_depth,
                line_max_dist=self.line_max_dist,
            )

        if draw_info_panel:
            # depth 모델 vs RealSense 오차 통계 계산
            depth_err_stats = None
            if model_depth_image is not None:
                errs, rels = [], []
                for item in people + obstacles:
                    z_rs  = item.get("z")
                    z_mod = item.get("model_z")
                    if z_rs is not None and z_mod is not None and z_rs > 0:
                        errs.append(abs(z_mod - z_rs))
                        rels.append((z_mod - z_rs) / z_rs * 100.0)
                if errs:
                    depth_err_stats = {
                        "n":   len(errs),
                        "mae": float(np.mean(errs)),
                        "mre": float(np.mean(rels)),
                    }
            canvas = draw_hud_panel(
                canvas, level, color,
                rep_distance, min_distance, ttc,
                self.risk_score_smooth, self.risk_score_raw,
                self.history,
                depth_err_stats=depth_err_stats,
            )

        # ── PPE 추론 및 사람별 매핑 ───────────────────────────────────────────
        if self.ppe_processor is not None and draw_detection:
            try:
                canvas = self.ppe_processor.process(
                    canvas, color_image, people,
                    self._infer_device, self._use_half,
                    int(getattr(args, "imgsz", 1280)), args.conf,
                )
            except Exception as e:
                print(f"[FrameProcessor][WARN] PPE 추론 실패: {e}")

        # ── 로그·비프 ─────────────────────────────────────────────────────────
        self.history.append(min_distance)
        self.last_beep_ts = maybe_beep(
            args.beep, level, now, self.last_beep_ts, args.beep_cooldown
        )
        if self.log_writer is not None:
            if nearest_pair is None:
                self.log_writer.writerow([f"{now:.3f}", level, "", "", "", "", "", ""])
            else:
                person, obs, angle, _, _, _ = nearest_pair
                self.log_writer.writerow([
                    f"{now:.3f}", level,
                    f"{min_distance:.3f}" if min_distance else "",
                    f"{ttc:.3f}"          if ttc          else "",
                    person["track_id"], obs["track_id"], obs["name"],
                    f"{angle:.2f}",
                ])
            self._log_flush_counter += 1
            if self._log_flush_counter >= self._log_flush_interval:
                self.log_file.flush()
                self._log_flush_counter = 0

        # ── 상태 업데이트 ─────────────────────────────────────────────────────
        self.prev_time            = now
        self.prev_person_points   = curr_person_points
        self.prev_obstacle_points = curr_obstacle_points

        # ── result dict ───────────────────────────────────────────────────────
        result = {
            "ts_epoch"         : float(now),
            "risk"             : level,
            "risk_score"       : float(self.risk_score_smooth),
            "risk_score_raw"   : float(self.risk_score_raw),
            "min_distance_m"   : float(min_distance)  if min_distance  is not None else None,
            "rep_distance_m"   : float(rep_distance)  if rep_distance  is not None else None,
            "ttc_s"            : float(ttc)            if ttc            is not None else None,
            "person_track_id"  : int(nearest_pair[0]["track_id"]) if nearest_pair else None,
            "obstacle_track_id": int(nearest_pair[1]["track_id"]) if nearest_pair else None,
            "obstacle_name"    : nearest_pair[1]["name"]           if nearest_pair else None,
            "angle_deg"        : float(nearest_pair[2])            if nearest_pair else None,
            "track_count"      : len(people) + len(obstacles),
        }
        return canvas, result
