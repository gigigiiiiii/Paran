"""
backend/monitor.py
===================
MonitorService: 러너 스레드 관리, JPEG 프레임 버퍼, 이벤트 저장 조율.
Supabase HTTP 호출은 SupabaseClient에 위임한다.
"""
from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

# Support direct execution context (app.py sets sys.path before importing this).
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from collision_monitor.config import parse_args
from collision_monitor.runner import PipelineRunner
from collision_monitor.video_runner import VideoRunner

try:
    from .supabase_client import SupabaseClient  # 패키지로 임포트 시
except ImportError:
    from supabase_client import SupabaseClient   # 직접 실행 시


class MonitorService:
    def __init__(self):
        self._lock   = threading.Lock()
        self._runner: PipelineRunner | None = None
        self._thread: threading.Thread | None = None
        self._start_error: str | None = None
        self._args   = None
        self._current_mode: str = "live"

        self._latest_frame_jpeg: bytes | None = None
        self._latest_result: dict[str, Any]   = {}
        self._events = deque(maxlen=5000)
        self._recording_enabled    = False
        self._session_id: str | None = None
        self._session_start_ts: float | None = None
        self._recording_elapsed_sec = 0
        self._last_event_ts         = 0.0
        self._session_timeline: dict[str, dict[str, float | None]] = {}
        self._storage_error: str | None = None

        self._event_cooldown_sec = float(os.getenv("MONITOR_EVENT_COOLDOWN_SEC", "1.0"))
        try:
            self._jpeg_quality = int(os.getenv("MONITOR_JPEG_QUALITY", "92"))
        except ValueError:
            self._jpeg_quality = 92
        self._jpeg_quality = max(60, min(100, self._jpeg_quality))

        # Supabase 클라이언트 초기화
        try:
            retry_count = int(os.getenv("MONITOR_SNAPSHOT_RETRY_COUNT", "3"))
        except ValueError:
            retry_count = 3
        try:
            retry_delay = float(os.getenv("MONITOR_SNAPSHOT_RETRY_DELAY_SEC", "0.4"))
        except ValueError:
            retry_delay = 0.4

        snapshot_prefix = os.getenv("SUPABASE_SNAPSHOT_PREFIX", "events").strip().strip("/") or "events"
        self._db = SupabaseClient(
            url             = os.getenv("SUPABASE_URL", ""),
            key             = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", ""),
            table           = os.getenv("SUPABASE_EVENT_TABLE", "collision_events").strip() or "collision_events",
            snapshot_bucket = os.getenv("SUPABASE_SNAPSHOT_BUCKET", "collision-event-snaps").strip(),
            snapshot_prefix = snapshot_prefix,
            retry_count     = retry_count,
            retry_delay_sec = retry_delay,
        )
        if not self._db.enabled:
            self._storage_error = "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."

        self._persist_queue: queue.Queue[tuple[dict[str, Any], bytes | None] | None] = queue.Queue(maxsize=1000)
        self._persist_stop   = threading.Event()
        self._persist_thread: threading.Thread | None = None

    # ── 세션 유틸 ─────────────────────────────────────────────────────────────

    def _session_start_from_id(self, session_id: str | None) -> float:
        if not session_id:
            return 0.0
        match = re.match(r"^session-(\d+)$", str(session_id).strip())
        if match is None:
            return 0.0
        try:
            return float(int(match.group(1)))
        except ValueError:
            return 0.0

    def _touch_session_timeline(
        self,
        session_id: str | None,
        first_start_ts: float | None = None,
        last_stop_ts: float | None   = None,
    ):
        if not session_id:
            return
        existing = self._session_timeline.get(session_id)
        if existing is None:
            existing = {"first_start_ts": None, "last_stop_ts": None}
            self._session_timeline[session_id] = existing
        if first_start_ts is not None:
            current_start = existing.get("first_start_ts")
            if current_start is None or first_start_ts < current_start:
                existing["first_start_ts"] = float(first_start_ts)
        if last_stop_ts is not None:
            current_stop = existing.get("last_stop_ts")
            if current_stop is None or last_stop_ts > current_stop:
                existing["last_stop_ts"] = float(last_stop_ts)

    # ── 설정 빌드 ─────────────────────────────────────────────────────────────

    def _build_args(self):
        args         = parse_args([])
        args.log_file = ""

        _project_root = Path(__file__).resolve().parent.parent
        if os.getenv("MONITOR_MODEL"):
            _m = Path(os.getenv("MONITOR_MODEL"))
            args.model = str(_m if _m.is_absolute() else _project_root / _m)
        if os.getenv("MONITOR_PPE_MODEL"):
            _p = Path(os.getenv("MONITOR_PPE_MODEL"))
            args.ppe_model = str(_p if _p.is_absolute() else _project_root / _p)
        if os.getenv("MONITOR_CONF"):
            args.conf = float(os.getenv("MONITOR_CONF"))
        if os.getenv("MONITOR_IMGSZ"):
            args.imgsz = int(os.getenv("MONITOR_IMGSZ"))
        if os.getenv("MONITOR_VEHICLE_BOX_EXPAND"):
            args.vehicle_box_expand = float(os.getenv("MONITOR_VEHICLE_BOX_EXPAND"))
        if os.getenv("MONITOR_VEHICLE_BOX_EXPAND_X"):
            args.vehicle_box_expand_x = float(os.getenv("MONITOR_VEHICLE_BOX_EXPAND_X"))
        if os.getenv("MONITOR_WIDTH"):
            args.width = int(os.getenv("MONITOR_WIDTH"))
        if os.getenv("MONITOR_HEIGHT"):
            args.height = int(os.getenv("MONITOR_HEIGHT"))
        if os.getenv("MONITOR_FPS"):
            args.fps = int(os.getenv("MONITOR_FPS"))

        no_overlay = os.getenv("MONITOR_NO_OVERLAY", "0").strip().lower()
        args.no_overlay = no_overlay in {"1", "true", "yes", "on"}

        hide_hud_panel = os.getenv("MONITOR_HIDE_HUD_PANEL", "1").strip().lower()
        args.hide_hud_panel = hide_hud_panel in {"1", "true", "yes", "on"}

        use_track = os.getenv("MONITOR_USE_YOLO_TRACK", "1").strip().lower()
        args.use_yolo_track = use_track not in {"0", "false", "no", "off"}

        _default_tracker = str(Path(__file__).resolve().parent.parent / "bytetrack_config.yaml")
        args.tracker = os.getenv("MONITOR_TRACKER", _default_tracker)

        if os.getenv("MONITOR_TRAIL_LEN"):
            args.trail_len = int(os.getenv("MONITOR_TRAIL_LEN"))

        # 모노큘러 depth 모델 (RealSense 없이 depth 추정)
        if os.getenv("MONITOR_DEPTH_MODEL"):
            args.depth_model = os.getenv("MONITOR_DEPTH_MODEL").strip()
        if os.getenv("MONITOR_DEPTH_FOV"):
            args.depth_fov = float(os.getenv("MONITOR_DEPTH_FOV"))
        if os.getenv("MONITOR_DEPTH_COMPARE_INTERVAL"):
            args.depth_compare_interval = float(os.getenv("MONITOR_DEPTH_COMPARE_INTERVAL"))
        if os.getenv("MONITOR_MODEL_DEPTH_INTERVAL"):
            args.model_depth_interval = float(os.getenv("MONITOR_MODEL_DEPTH_INTERVAL"))
        if os.getenv("MONITOR_DISTANCE_SMOOTH_ALPHA"):
            args.distance_smooth_alpha = float(os.getenv("MONITOR_DISTANCE_SMOOTH_ALPHA"))
        if os.getenv("MONITOR_DISPLAY_DISTANCE_SMOOTH_ALPHA"):
            args.display_distance_smooth_alpha = float(os.getenv("MONITOR_DISPLAY_DISTANCE_SMOOTH_ALPHA"))
        if os.getenv("MONITOR_DISPLAY_DISTANCE_STEP"):
            args.display_distance_step = float(os.getenv("MONITOR_DISPLAY_DISTANCE_STEP"))
        if os.getenv("MONITOR_RECEDING_SPEED_THRESHOLD"):
            args.receding_speed_threshold = float(os.getenv("MONITOR_RECEDING_SPEED_THRESHOLD"))
        if os.getenv("MONITOR_RECEDING_RISK_SCALE"):
            args.receding_risk_scale = float(os.getenv("MONITOR_RECEDING_RISK_SCALE"))
        if os.getenv("MONITOR_CONFIDENCE_RISK_FLOOR"):
            args.confidence_risk_floor = float(os.getenv("MONITOR_CONFIDENCE_RISK_FLOOR"))
        if os.getenv("MONITOR_FRONT_ANGLE"):
            args.front_angle = float(os.getenv("MONITOR_FRONT_ANGLE"))
        if os.getenv("MONITOR_TTC_MODE"):
            args.ttc_mode = os.getenv("MONITOR_TTC_MODE").strip()

        use_sahi = os.getenv("MONITOR_USE_SAHI", "0").strip().lower()
        args.use_sahi = use_sahi in {"1", "true", "yes", "on"}
        if os.getenv("MONITOR_SAHI_SLICE_SIZE"):
            args.sahi_slice_size = int(os.getenv("MONITOR_SAHI_SLICE_SIZE"))
        if os.getenv("MONITOR_SAHI_OVERLAP"):
            args.sahi_overlap = float(os.getenv("MONITOR_SAHI_OVERLAP"))

        return args

    # ── 러너 생명주기 ─────────────────────────────────────────────────────────

    def start(self):
        self._start_persist_worker()

        args         = self._build_args()
        self._args   = args
        video_source = os.getenv("MONITOR_SOURCE", "").strip()
        with self._lock:
            self._current_mode = f"test:{Path(video_source).name}" if video_source else "live"
        try:
            if video_source:
                print(f"[INFO] VideoRunner 모드: {video_source}")
                args.video_source = video_source
                args.video_loop   = True
                self._runner = VideoRunner(args, display=False, on_result=self._on_runner_result)
            else:
                self._runner = PipelineRunner(args, display=False, on_result=self._on_runner_result)
        except Exception as exc:
            self._start_error = str(exc)
            return

        self._thread = threading.Thread(
            target=self._run_runner, daemon=True, name="collision-monitor-runner"
        )
        self._thread.start()

    def _run_runner(self):
        try:
            self._runner.run()
        except Exception as exc:
            err_msg = f"runner crashed: {exc}"
            print(f"[ERROR] {err_msg}")
            with self._lock:
                self._start_error = err_msg

    def stop(self):
        runner = self._runner
        thread = self._thread
        if runner is not None:
            runner.request_stop()
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        if runner is not None:
            runner.close()
        self._stop_persist_worker()

    def get_mode(self):
        test_videos_dir = Path(__file__).resolve().parent.parent / "test_videos"
        videos: list[str] = []
        if test_videos_dir.exists():
            for pat in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI"):
                for f in test_videos_dir.glob(pat):
                    if f.name not in videos:
                        videos.append(f.name)
        videos.sort()
        with self._lock:
            mode = self._current_mode
        return {"mode": mode, "test_videos": videos}

    def switch_source(self, source: str | None = None):
        with self._lock:
            runner = self._runner
            thread = self._thread
        if runner is not None:
            runner.request_stop()
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        if runner is not None:
            runner.close()

        with self._lock:
            self._runner             = None
            self._thread             = None
            self._start_error        = None
            self._latest_frame_jpeg  = None

        args        = self._build_args()
        self._args  = args
        source_name = Path(source).name if source else None
        with self._lock:
            self._current_mode = f"test:{source_name}" if source_name else "live"

        try:
            if source:
                print(f"[INFO] VideoRunner 전환: {source}")
                args.video_source = source
                args.video_loop   = True
                self._runner = VideoRunner(args, display=False, on_result=self._on_runner_result)
            else:
                print("[INFO] PipelineRunner(RealSense) 전환")
                self._runner = PipelineRunner(args, display=False, on_result=self._on_runner_result)
        except Exception as exc:
            with self._lock:
                self._start_error = str(exc)
            return {"ok": False, "error": str(exc)}

        self._thread = threading.Thread(
            target=self._run_runner, daemon=True, name="collision-monitor-runner"
        )
        self._thread.start()
        with self._lock:
            mode = self._current_mode
        return {"ok": True, "mode": mode}

    # ── 녹화 제어 ─────────────────────────────────────────────────────────────

    def start_recording(self):
        now_ts = time.time()
        with self._lock:
            if self._recording_enabled:
                if self._session_start_ts is not None:
                    self._recording_elapsed_sec = int(max(0.0, now_ts - self._session_start_ts))
                return {
                    "ok":                     True,
                    "recording_enabled":      True,
                    "session_id":             self._session_id,
                    "recording_elapsed_sec":  int(self._recording_elapsed_sec),
                }
            if self._session_id is None:
                self._session_id             = f"session-{int(now_ts)}"
                self._recording_elapsed_sec  = 0
                self._last_event_ts          = 0.0
                self._touch_session_timeline(self._session_id, first_start_ts=now_ts)
            else:
                parsed_start_ts = self._session_start_from_id(self._session_id)
                if parsed_start_ts > 0:
                    self._touch_session_timeline(self._session_id, first_start_ts=parsed_start_ts)
                else:
                    self._touch_session_timeline(self._session_id, first_start_ts=now_ts)
            self._recording_enabled  = True
            self._session_start_ts   = now_ts - float(self._recording_elapsed_sec)
            session_id               = self._session_id
        return {
            "ok":                    True,
            "recording_enabled":     True,
            "session_id":            session_id,
            "recording_elapsed_sec": int(self._recording_elapsed_sec),
        }

    def stop_recording(self):
        now_ts = time.time()
        with self._lock:
            if self._recording_enabled and self._session_start_ts is not None:
                elapsed = int(max(0.0, now_ts - self._session_start_ts))
                self._recording_elapsed_sec = max(self._recording_elapsed_sec, elapsed)
            self._recording_enabled = False
            self._session_start_ts  = None
            self._touch_session_timeline(self._session_id, last_stop_ts=now_ts)
        return {
            "ok":                    True,
            "recording_enabled":     False,
            "recording_elapsed_sec": int(self._recording_elapsed_sec),
        }

    def reset_recording(self):
        now_ts = time.time()
        with self._lock:
            if self._session_id is not None:
                parsed_start_ts = self._session_start_from_id(self._session_id)
                if parsed_start_ts > 0:
                    self._touch_session_timeline(self._session_id, first_start_ts=parsed_start_ts)
                if self._recording_enabled:
                    self._touch_session_timeline(self._session_id, last_stop_ts=now_ts)
            self._recording_enabled    = False
            self._session_id           = None
            self._session_start_ts     = None
            self._recording_elapsed_sec = 0
            self._last_event_ts        = 0.0
            self._events.clear()
        return {
            "ok":                    True,
            "recording_enabled":     False,
            "session_id":            None,
            "recording_elapsed_sec": 0,
            "event_count":           0,
        }

    # ── Supabase 비동기 저장 ─────────────────────────────────────────────────

    def _start_persist_worker(self):
        if not self._db.enabled:
            return
        if self._persist_thread is not None and self._persist_thread.is_alive():
            return
        self._persist_queue = queue.Queue(maxsize=1000)
        self._persist_stop.clear()
        self._persist_thread = threading.Thread(
            target=self._persist_worker,
            daemon=True,
            name="collision-monitor-supabase-writer",
        )
        self._persist_thread.start()

    def _stop_persist_worker(self):
        worker = self._persist_thread
        if worker is None or not worker.is_alive():
            return
        self._persist_stop.set()
        try:
            self._persist_queue.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=3.0)
        self._persist_thread = None

    def _persist_worker(self):
        while not self._persist_stop.is_set():
            try:
                item = self._persist_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if item is None:
                self._persist_queue.task_done()
                break
            event, snapshot_jpeg = item
            snapshot_error = None
            try:
                event_payload = dict(event)
                if snapshot_jpeg is not None and self._db.snapshot_enabled:
                    try:
                        snapshot_path = self._db.upload_snapshot_with_retry(event_payload, snapshot_jpeg)
                        event_payload["snapshot_path"] = snapshot_path
                    except Exception as exc:
                        snapshot_error = f"Supabase snapshot upload failed: {exc}"
                dropped_columns = self._db.insert_event_resilient(event_payload)
                with self._lock:
                    if dropped_columns:
                        dropped = ", ".join(dropped_columns)
                        if snapshot_error:
                            self._storage_error = f"{snapshot_error}; dropped columns: {dropped}"
                        else:
                            self._storage_error = f"Supabase schema mismatch; dropped columns: {dropped}"
                    else:
                        self._storage_error = snapshot_error
            except Exception as exc:
                with self._lock:
                    self._storage_error = f"Supabase insert failed: {exc}"
            finally:
                self._persist_queue.task_done()

    def _enqueue_supabase_event(self, event: dict[str, Any], snapshot_jpeg: bytes | None = None):
        if not self._db.enabled:
            return
        try:
            self._persist_queue.put_nowait((dict(event), snapshot_jpeg))
        except queue.Full:
            with self._lock:
                self._storage_error = "Supabase writer queue is full; dropping event"

    # ── 세션 삭제 ─────────────────────────────────────────────────────────────

    def _clear_session_from_memory(self, session_id: str) -> int:
        with self._lock:
            current_events = list(self._events)
            kept_events    = [
                row for row in current_events
                if str(row.get("session_id") or "").strip() != session_id
            ]
            removed_count  = len(current_events) - len(kept_events)
            self._events   = deque(kept_events, maxlen=self._events.maxlen)
            self._session_timeline.pop(session_id, None)
        return removed_count

    def delete_session(self, session_id: str):
        sid = str(session_id or "").strip()
        if not sid:
            raise ValueError("session_id is required")
        with self._lock:
            if self._session_id == sid:
                raise RuntimeError("active_session")

        snapshot_paths: set[str] = set()
        deleted_events      = 0
        deleted_images      = 0
        image_delete_errors: list[str] = []
        if self._db.enabled:
            try:
                rows = self._db.fetch_session_rows(sid)
            except Exception as exc:
                raise RuntimeError(f"Supabase fetch failed before delete: {exc}") from exc

            deleted_events = len(rows)
            session_part   = self._db._safe_path_part(sid)
            for row in rows:
                snapshot_path = str(row.get("snapshot_path") or "").strip().strip("/")
                if snapshot_path:
                    snapshot_paths.add(snapshot_path)
                event_id = str(row.get("event_id") or "").strip()
                if event_id:
                    fallback = self._db._safe_path_part(event_id)
                    snapshot_paths.add(f"{self._db._snapshot_prefix}/{session_part}/{fallback}.jpg")

            self._db.delete_session_rows(sid)
            for object_path in snapshot_paths:
                try:
                    if self._db.delete_snapshot_object(object_path):
                        deleted_images += 1
                except Exception as exc:
                    image_delete_errors.append(str(exc))

        removed_from_memory = self._clear_session_from_memory(sid)
        payload = {
            "ok":                   True,
            "session_id":           sid,
            "deleted_events":       int(deleted_events),
            "deleted_images":       int(deleted_images),
            "removed_from_memory":  int(removed_from_memory),
        }
        if image_delete_errors:
            payload["image_delete_errors"]      = image_delete_errors[:5]
            payload["image_delete_error_count"] = len(image_delete_errors)
        return payload

    # ── 이벤트 & 세션 조회 ────────────────────────────────────────────────────

    def _event_ts_ms(self, event: dict[str, Any]) -> int:
        ts_epoch = event.get("ts_epoch")
        if ts_epoch is not None:
            try:
                return int(float(ts_epoch) * 1000.0)
            except (TypeError, ValueError):
                pass
        created_at = event.get("created_at")
        if isinstance(created_at, str) and created_at.strip():
            try:
                return int(datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp() * 1000.0)
            except ValueError:
                return 0
        return 0

    def sessions(self, limit=40, scan_limit=5000, exclude_session_id: str | None = None):
        try:
            rows, total_count = self._db.fetch_events(limit=scan_limit, include_total=True)
            with self._lock:
                self._storage_error = None
        except Exception as exc:
            with self._lock:
                self._storage_error = f"Supabase fetch failed: {exc}"
            return {"sessions": [], "limit": limit, "source": "supabase", "error": "fetch_failed"}

        with self._lock:
            timeline_snapshot = {sid: dict(meta) for sid, meta in self._session_timeline.items()}

        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            sid = str(row.get("session_id") or "").strip()
            if not sid:
                continue
            if exclude_session_id and sid == exclude_session_id:
                continue
            ts_ms          = self._event_ts_ms(row)
            risk_score     = row.get("risk_score")
            rep_distance_m = row.get("rep_distance_m")

            bucket = grouped.get(sid)
            if bucket is None:
                bucket = {
                    "session_id":   sid,
                    "total":        0,
                    "warning":      0,
                    "danger":       0,
                    "risk_sum":     0.0,
                    "risk_count":   0,
                    "min_distance_m": None,
                    "start_ts_ms":  0,
                    "end_ts_ms":    0,
                }
                grouped[sid] = bucket

            bucket["total"] += 1
            risk = row.get("risk")
            if risk == "WARNING":
                bucket["warning"] += 1
            elif risk == "DANGER":
                bucket["danger"] += 1
            if isinstance(risk_score, (int, float)):
                bucket["risk_sum"]   += float(risk_score)
                bucket["risk_count"] += 1
            if isinstance(rep_distance_m, (int, float)):
                dist_val     = float(rep_distance_m)
                min_dist     = bucket["min_distance_m"]
                if min_dist is None or dist_val < min_dist:
                    bucket["min_distance_m"] = dist_val
            if ts_ms > 0:
                if bucket["start_ts_ms"] == 0 or ts_ms < bucket["start_ts_ms"]:
                    bucket["start_ts_ms"] = ts_ms
                if ts_ms > bucket["end_ts_ms"]:
                    bucket["end_ts_ms"] = ts_ms

        sessions = []
        for bucket in grouped.values():
            session_id = str(bucket["session_id"])
            timeline   = timeline_snapshot.get(session_id) or {}
            start_ts_ms = int(bucket["start_ts_ms"])
            end_ts_ms   = int(bucket["end_ts_ms"])

            meta_start_ts = timeline.get("first_start_ts")
            if isinstance(meta_start_ts, (int, float)) and float(meta_start_ts) > 0:
                start_ts_ms = int(float(meta_start_ts) * 1000.0)
            else:
                parsed_start_ts = self._session_start_from_id(session_id)
                if parsed_start_ts > 0:
                    start_ts_ms = int(parsed_start_ts * 1000.0)

            meta_stop_ts = timeline.get("last_stop_ts")
            if isinstance(meta_stop_ts, (int, float)) and float(meta_stop_ts) > 0:
                end_ts_ms = int(float(meta_stop_ts) * 1000.0)
            if end_ts_ms == 0 and start_ts_ms > 0:
                end_ts_ms = start_ts_ms

            risk_count   = int(bucket["risk_count"])
            avg_risk_pct = 0
            if risk_count > 0:
                avg_risk_pct = max(0, min(100, int(round((bucket["risk_sum"] / risk_count) * 100.0))))
            sessions.append({
                "session_id":     session_id,
                "total":          int(bucket["total"]),
                "warning":        int(bucket["warning"]),
                "danger":         int(bucket["danger"]),
                "avg_risk_pct":   avg_risk_pct,
                "min_distance_m": bucket["min_distance_m"],
                "start_ts_ms":    start_ts_ms,
                "end_ts_ms":      end_ts_ms,
            })

        sessions.sort(
            key=lambda row: (
                int(row.get("end_ts_ms", 0)),
                int(row.get("start_ts_ms", 0)),
                str(row.get("session_id", "")),
            ),
            reverse=True,
        )
        scanned_events = len(rows)
        return {
            "sessions":           sessions[:limit],
            "limit":              limit,
            "scan_limit":         scan_limit,
            "scanned_events":     scanned_events,
            "total_events":       total_count,
            "truncated":          bool(total_count is not None and total_count > scan_limit),
            "excluded_session_id": exclude_session_id,
            "source":             "supabase",
        }

    def events(self, limit=100, session_id: str | None = None, offset=0, include_total=False):
        try:
            rows, total_count = self._db.fetch_events(
                limit=limit, session_id=session_id, offset=offset, include_total=include_total
            )
            with self._lock:
                self._storage_error = None
        except Exception as exc:
            with self._lock:
                self._storage_error = f"Supabase fetch failed: {exc}"
            return {"events": [], "limit": limit, "source": "supabase", "error": "fetch_failed"}

        payload = {
            "events":     rows,
            "limit":      limit,
            "offset":     offset,
            "source":     "supabase",
            "session_id": session_id,
        }
        if include_total:
            payload["total"] = int(total_count) if isinstance(total_count, int) else len(rows)
        return payload

    # ── 프레임 콜백 & 스트리밍 ───────────────────────────────────────────────

    def _on_runner_result(self, result: dict[str, Any], canvas):
        ok, encoded = cv2.imencode(
            ".jpg",
            canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)],
        )
        now_ts        = float(result.get("ts_epoch", time.time()))
        encoded_bytes = encoded.tobytes() if ok else None

        event_to_persist = None
        with self._lock:
            self._latest_result = dict(result)
            self._latest_result["recording_enabled"] = self._recording_enabled
            if encoded_bytes is not None:
                self._latest_frame_jpeg = encoded_bytes

            should_save = (
                self._recording_enabled
                and result.get("risk") in {"WARNING", "DANGER"}
                and now_ts - self._last_event_ts >= self._event_cooldown_sec
            )
            if should_save:
                event_to_persist = dict(result)
                event_to_persist["event_id"]   = uuid.uuid4().hex
                event_to_persist["session_id"] = self._session_id
                event_to_persist["created_at"] = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()
                self._events.appendleft(event_to_persist)
                self._last_event_ts = now_ts

        if event_to_persist is not None:
            self._enqueue_supabase_event(event_to_persist, encoded_bytes)

    def frame_generator(self):
        while True:
            with self._lock:
                frame = self._latest_frame_jpeg
            if frame is None:
                time.sleep(0.03)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)

    # ── 상태 조회 ─────────────────────────────────────────────────────────────

    def _today_summary(self, events):
        today   = datetime.now().date()
        warning = 0
        danger  = 0
        for event in events:
            ts_epoch = event.get("ts_epoch")
            if ts_epoch is None:
                continue
            if datetime.fromtimestamp(float(ts_epoch)).date() != today:
                continue
            risk = event.get("risk")
            if risk == "WARNING":
                warning += 1
            elif risk == "DANGER":
                danger += 1
        return {
            "date":          today.isoformat(),
            "warning_count": warning,
            "danger_count":  danger,
            "total":         warning + danger,
        }

    def state(self):
        now_ts = time.time()
        with self._lock:
            latest               = dict(self._latest_result)
            events               = list(self._events)
            recording_enabled    = self._recording_enabled
            session_id           = self._session_id
            session_start_ts     = self._session_start_ts
            recording_elapsed_sec = int(self._recording_elapsed_sec)
            storage_error        = self._storage_error
        if recording_enabled and session_start_ts is not None:
            recording_elapsed_sec = max(recording_elapsed_sec, int(max(0.0, now_ts - session_start_ts)))
        stream_info = {
            "width":        int(self._args.width)        if self._args is not None else None,
            "height":       int(self._args.height)       if self._args is not None else None,
            "fps":          int(self._args.fps)          if self._args is not None else None,
            "jpeg_quality": int(self._jpeg_quality),
        }
        return {
            "recording_enabled":    recording_enabled,
            "session_id":           session_id,
            "recording_elapsed_sec": recording_elapsed_sec,
            "latest":               latest,
            "today_summary":        self._today_summary(events),
            "event_count":          len(events),
            "start_error":          self._start_error,
            "stream":               stream_info,
            "storage": {
                "backend":    "supabase" if self._db.enabled else "memory",
                "table":      self._db._table if self._db.enabled else None,
                "last_error": storage_error,
            },
        }
