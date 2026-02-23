from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

import cv2
from fastapi import FastAPI
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from collision_monitor.config import parse_args
from collision_monitor.runner import PipelineRunner


class MonitorService:
    def __init__(self):
        self._lock = threading.Lock()
        self._runner: PipelineRunner | None = None
        self._thread: threading.Thread | None = None
        self._start_error: str | None = None
        self._args = None

        self._latest_frame_jpeg: bytes | None = None
        self._latest_result: dict[str, Any] = {}
        self._events = deque(maxlen=5000)
        self._recording_enabled = False
        self._session_id: str | None = None
        self._session_start_ts: float | None = None
        self._recording_elapsed_sec = 0
        self._last_event_ts = 0.0

        self._event_cooldown_sec = float(os.getenv("MONITOR_EVENT_COOLDOWN_SEC", "1.0"))
        try:
            self._jpeg_quality = int(os.getenv("MONITOR_JPEG_QUALITY", "92"))
        except ValueError:
            self._jpeg_quality = 92
        self._jpeg_quality = max(60, min(100, self._jpeg_quality))

        self._supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self._supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "")
        self._supabase_table = os.getenv("SUPABASE_EVENT_TABLE", "collision_events").strip() or "collision_events"
        self._snapshot_bucket = os.getenv("SUPABASE_SNAPSHOT_BUCKET", "collision-event-snaps").strip()
        self._snapshot_prefix = os.getenv("SUPABASE_SNAPSHOT_PREFIX", "events").strip().strip("/")
        if not self._snapshot_prefix:
            self._snapshot_prefix = "events"
        self._snapshot_enabled = bool(self._snapshot_bucket)
        self._supabase_enabled = bool(self._supabase_url and self._supabase_key)
        self._storage_error: str | None = None
        if not self._supabase_enabled:
            self._storage_error = "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."

        self._persist_queue: queue.Queue[tuple[dict[str, Any], bytes | None] | None] = queue.Queue(maxsize=1000)
        self._persist_stop = threading.Event()
        self._persist_thread: threading.Thread | None = None
        # Ignore global HTTP(S)_PROXY for Supabase calls to avoid local proxy misconfiguration.
        self._supabase_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _build_args(self):
        args = parse_args([])
        args.log_file = ""

        if os.getenv("MONITOR_MODEL"):
            args.model = os.getenv("MONITOR_MODEL")
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
        return args

    def start(self):
        self._start_persist_worker()

        args = self._build_args()
        self._args = args
        try:
            self._runner = PipelineRunner(args, display=False, on_result=self._on_runner_result)
        except Exception as exc:
            self._start_error = str(exc)
            return

        self._thread = threading.Thread(target=self._runner.run, daemon=True, name="collision-monitor-runner")
        self._thread.start()

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

    def start_recording(self):
        now_ts = time.time()
        with self._lock:
            if self._recording_enabled:
                if self._session_start_ts is not None:
                    self._recording_elapsed_sec = int(max(0.0, now_ts - self._session_start_ts))
                return {
                    "ok": True,
                    "recording_enabled": True,
                    "session_id": self._session_id,
                    "recording_elapsed_sec": int(self._recording_elapsed_sec),
                }

            if self._session_id is None:
                self._session_id = f"session-{int(now_ts)}"
                self._recording_elapsed_sec = 0
                self._last_event_ts = 0.0

            # Resume from accumulated elapsed time for this session.
            self._recording_enabled = True
            self._session_start_ts = now_ts - float(self._recording_elapsed_sec)
            session_id = self._session_id
        return {
            "ok": True,
            "recording_enabled": True,
            "session_id": session_id,
            "recording_elapsed_sec": int(self._recording_elapsed_sec),
        }

    def stop_recording(self):
        now_ts = time.time()
        with self._lock:
            if self._recording_enabled and self._session_start_ts is not None:
                elapsed = int(max(0.0, now_ts - self._session_start_ts))
                self._recording_elapsed_sec = max(self._recording_elapsed_sec, elapsed)
            self._recording_enabled = False
            self._session_start_ts = None
        return {
            "ok": True,
            "recording_enabled": False,
            "recording_elapsed_sec": int(self._recording_elapsed_sec),
        }

    def reset_recording(self):
        with self._lock:
            self._recording_enabled = False
            self._session_id = None
            self._session_start_ts = None
            self._recording_elapsed_sec = 0
            self._last_event_ts = 0.0
            self._events.clear()
        return {
            "ok": True,
            "recording_enabled": False,
            "session_id": None,
            "recording_elapsed_sec": 0,
            "event_count": 0,
        }

    def _start_persist_worker(self):
        if not self._supabase_enabled:
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
                if snapshot_jpeg is not None and self._snapshot_enabled:
                    try:
                        snapshot_path = self._upload_snapshot_supabase(event_payload, snapshot_jpeg)
                        event_payload["snapshot_path"] = snapshot_path
                    except Exception as exc:
                        snapshot_error = f"Supabase snapshot upload failed: {exc}"
                dropped_columns = self._insert_event_supabase_resilient(event_payload)
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

    def _insert_event_supabase(self, event: dict[str, Any]):
        if not self._supabase_enabled:
            return
        table_name = urllib.parse.quote(self._supabase_table, safe="")
        url = f"{self._supabase_url}/rest/v1/{table_name}"
        payload = json.dumps(event, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=payload,
            method="POST",
            headers={
                "apikey": self._supabase_key,
                "Authorization": f"Bearer {self._supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
        )
        try:
            with self._supabase_opener.open(request, timeout=6) as response:
                status = int(getattr(response, "status", 0))
                if status not in {200, 201, 204}:
                    raise RuntimeError(f"HTTP {status}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    def _insert_event_supabase_resilient(self, event: dict[str, Any]):
        payload = dict(event)
        dropped_columns: list[str] = []
        # Handle schema drift gracefully if Supabase table is older than payload keys.
        for _ in range(12):
            try:
                self._insert_event_supabase(payload)
                return dropped_columns
            except Exception as exc:
                message = str(exc)
                match = re.search(r"Could not find the '([^']+)' column", message)
                if match is None:
                    raise
                missing_column = match.group(1)
                if missing_column not in payload:
                    raise
                payload.pop(missing_column, None)
                dropped_columns.append(missing_column)
        raise RuntimeError("Supabase insert failed after dropping missing columns")

    def _safe_path_part(self, value: str):
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
        safe = safe.strip("._")
        return safe or "unknown"

    def _build_snapshot_object_path(self, event: dict[str, Any]):
        created_at_raw = event.get("created_at")
        created_dt = datetime.now(timezone.utc)
        if isinstance(created_at_raw, str):
            try:
                created_dt = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
            except ValueError:
                created_dt = datetime.now(timezone.utc)

        date_part = created_dt.strftime("%Y/%m/%d")
        session_part = self._safe_path_part(str(event.get("session_id") or "no-session"))
        event_id = self._safe_path_part(str(event.get("event_id") or uuid.uuid4().hex))
        return f"{self._snapshot_prefix}/{date_part}/{session_part}/{event_id}.jpg"

    def _upload_snapshot_supabase(self, event: dict[str, Any], jpeg_bytes: bytes):
        if not self._supabase_enabled:
            raise RuntimeError("Supabase is not configured")
        if not self._snapshot_enabled:
            raise RuntimeError("SUPABASE_SNAPSHOT_BUCKET is not configured")

        object_path = self._build_snapshot_object_path(event)
        bucket_name = urllib.parse.quote(self._snapshot_bucket, safe="")
        object_path_encoded = urllib.parse.quote(object_path, safe="/")
        url = f"{self._supabase_url}/storage/v1/object/{bucket_name}/{object_path_encoded}"
        request = urllib.request.Request(
            url=url,
            data=jpeg_bytes,
            method="POST",
            headers={
                "apikey": self._supabase_key,
                "Authorization": f"Bearer {self._supabase_key}",
                "Content-Type": "image/jpeg",
                "x-upsert": "true",
            },
        )
        with self._supabase_opener.open(request, timeout=8) as response:
            status = int(getattr(response, "status", 0))
            if status not in {200, 201}:
                raise RuntimeError(f"HTTP {status}")
        return object_path

    def _enqueue_supabase_event(self, event: dict[str, Any], snapshot_jpeg: bytes | None = None):
        if not self._supabase_enabled:
            return
        try:
            self._persist_queue.put_nowait((dict(event), snapshot_jpeg))
        except queue.Full:
            with self._lock:
                self._storage_error = "Supabase writer queue is full; dropping event"

    def _fetch_events_supabase(self, limit: int):
        if not self._supabase_enabled:
            return None
        table_name = urllib.parse.quote(self._supabase_table, safe="")
        params = urllib.parse.urlencode(
            {
                "select": "*",
                "order": "created_at.desc",
                "limit": str(limit),
            }
        )
        url = f"{self._supabase_url}/rest/v1/{table_name}?{params}"
        request = urllib.request.Request(
            url=url,
            method="GET",
            headers={
                "apikey": self._supabase_key,
                "Authorization": f"Bearer {self._supabase_key}",
                "Accept": "application/json",
            },
        )
        try:
            with self._supabase_opener.open(request, timeout=6) as response:
                payload = response.read().decode("utf-8")
            rows = json.loads(payload) if payload else []
            if not isinstance(rows, list):
                raise RuntimeError("invalid payload type")
            with self._lock:
                self._storage_error = None
            return rows
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, RuntimeError) as exc:
            with self._lock:
                self._storage_error = f"Supabase fetch failed: {exc}"
            return None

    def _on_runner_result(self, result: dict[str, Any], canvas):
        ok, encoded = cv2.imencode(
            ".jpg",
            canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)],
        )
        now_ts = float(result.get("ts_epoch", time.time()))
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
                event_to_persist["event_id"] = uuid.uuid4().hex
                event_to_persist["session_id"] = self._session_id
                event_to_persist["created_at"] = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()
                self._events.appendleft(event_to_persist)
                self._last_event_ts = now_ts

        if event_to_persist is not None:
            self._enqueue_supabase_event(event_to_persist, encoded_bytes)

    def _today_summary(self, events):
        today = datetime.now().date()
        warning = 0
        danger = 0
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
            "date": today.isoformat(),
            "warning_count": warning,
            "danger_count": danger,
            "total": warning + danger,
        }

    def state(self):
        now_ts = time.time()
        with self._lock:
            latest = dict(self._latest_result)
            events = list(self._events)
            recording_enabled = self._recording_enabled
            session_id = self._session_id
            session_start_ts = self._session_start_ts
            recording_elapsed_sec = int(self._recording_elapsed_sec)
            storage_error = self._storage_error
        if recording_enabled and session_start_ts is not None:
            recording_elapsed_sec = max(recording_elapsed_sec, int(max(0.0, now_ts - session_start_ts)))
        stream_info = {
            "width": int(self._args.width) if self._args is not None else None,
            "height": int(self._args.height) if self._args is not None else None,
            "fps": int(self._args.fps) if self._args is not None else None,
            "jpeg_quality": int(self._jpeg_quality),
        }
        return {
            "recording_enabled": recording_enabled,
            "session_id": session_id,
            "recording_elapsed_sec": recording_elapsed_sec,
            "latest": latest,
            "today_summary": self._today_summary(events),
            "event_count": len(events),
            "start_error": self._start_error,
            "stream": stream_info,
            "storage": {
                "backend": "supabase" if self._supabase_enabled else "memory",
                "table": self._supabase_table if self._supabase_enabled else None,
                "last_error": storage_error,
            },
        }

    def events(self, limit=100):
        supabase_rows = self._fetch_events_supabase(limit=limit)
        if supabase_rows is None:
            return {"events": [], "limit": limit, "source": "supabase", "error": "fetch_failed"}
        return {"events": supabase_rows, "limit": limit, "source": "supabase"}

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


service = MonitorService()

app = FastAPI(title="Collision Monitor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    service.start()


@app.on_event("shutdown")
def on_shutdown():
    service.stop()


@app.get("/api/health")
def health():
    return {"ok": True, "service": "collision-monitor", "start_error": service.state()["start_error"]}


@app.get("/api/state")
def get_state():
    return service.state()


@app.get("/api/events")
def get_events(limit: int = Query(default=100, ge=1, le=1000)):
    return service.events(limit=limit)


@app.post("/api/control/start")
def start_recording():
    return service.start_recording()


@app.post("/api/control/stop")
def stop_recording():
    return service.stop_recording()


@app.post("/api/control/reset")
def reset_recording():
    return service.reset_recording()


@app.get("/api/stream")
def stream():
    return StreamingResponse(
        service.frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
