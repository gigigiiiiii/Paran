from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Support direct script execution (e.g. "Run Code" on backend/app.py).
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _load_env_file(path: Path):
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_env_file(Path(__file__).resolve().with_name(".env"))

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
        self._session_timeline: dict[str, dict[str, float | None]] = {}

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
        try:
            self._snapshot_retry_count = int(os.getenv("MONITOR_SNAPSHOT_RETRY_COUNT", "3"))
        except ValueError:
            self._snapshot_retry_count = 3
        self._snapshot_retry_count = max(1, min(10, self._snapshot_retry_count))
        try:
            self._snapshot_retry_delay_sec = float(os.getenv("MONITOR_SNAPSHOT_RETRY_DELAY_SEC", "0.4"))
        except ValueError:
            self._snapshot_retry_delay_sec = 0.4
        self._snapshot_retry_delay_sec = max(0.0, min(5.0, self._snapshot_retry_delay_sec))
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

    def _session_start_from_id(self, session_id: str | None):
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
        last_stop_ts: float | None = None,
    ):
        if not session_id:
            return
        existing = self._session_timeline.get(session_id)
        if existing is None:
            existing = {
                "first_start_ts": None,
                "last_stop_ts": None,
            }
            self._session_timeline[session_id] = existing
        if first_start_ts is not None:
            current_start = existing.get("first_start_ts")
            if current_start is None or first_start_ts < current_start:
                existing["first_start_ts"] = float(first_start_ts)
        if last_stop_ts is not None:
            current_stop = existing.get("last_stop_ts")
            if current_stop is None or last_stop_ts > current_stop:
                existing["last_stop_ts"] = float(last_stop_ts)

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
                self._touch_session_timeline(self._session_id, first_start_ts=now_ts)
            else:
                parsed_start_ts = self._session_start_from_id(self._session_id)
                if parsed_start_ts > 0:
                    self._touch_session_timeline(self._session_id, first_start_ts=parsed_start_ts)
                else:
                    self._touch_session_timeline(self._session_id, first_start_ts=now_ts)

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
            self._touch_session_timeline(self._session_id, last_stop_ts=now_ts)
        return {
            "ok": True,
            "recording_enabled": False,
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
                        snapshot_path = self._upload_snapshot_supabase_with_retry(event_payload, snapshot_jpeg)
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
        session_part = self._safe_path_part(str(event.get("session_id") or "no-session"))
        event_id = self._safe_path_part(str(event.get("event_id") or uuid.uuid4().hex))
        return f"{self._snapshot_prefix}/{session_part}/{event_id}.jpg"

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

    def _upload_snapshot_supabase_with_retry(self, event: dict[str, Any], jpeg_bytes: bytes):
        last_error = None
        for attempt in range(1, self._snapshot_retry_count + 1):
            try:
                return self._upload_snapshot_supabase(event, jpeg_bytes)
            except Exception as exc:
                last_error = exc
                if attempt < self._snapshot_retry_count and self._snapshot_retry_delay_sec > 0:
                    time.sleep(self._snapshot_retry_delay_sec * attempt)
        raise RuntimeError(f"after {self._snapshot_retry_count} attempts: {last_error}")

    def _enqueue_supabase_event(self, event: dict[str, Any], snapshot_jpeg: bytes | None = None):
        if not self._supabase_enabled:
            return
        try:
            self._persist_queue.put_nowait((dict(event), snapshot_jpeg))
        except queue.Full:
            with self._lock:
                self._storage_error = "Supabase writer queue is full; dropping event"

    def _fetch_events_supabase(
        self,
        limit: int,
        session_id: str | None = None,
        offset: int = 0,
        include_total: bool = False,
    ):
        if not self._supabase_enabled:
            return None
        table_name = urllib.parse.quote(self._supabase_table, safe="")
        query_params = {
            "select": "*",
            "order": "created_at.desc",
            "limit": str(limit),
            "offset": str(offset),
        }
        if session_id:
            query_params["session_id"] = f"eq.{session_id}"
        params = urllib.parse.urlencode(query_params)
        url = f"{self._supabase_url}/rest/v1/{table_name}?{params}"
        headers = {
            "apikey": self._supabase_key,
            "Authorization": f"Bearer {self._supabase_key}",
            "Accept": "application/json",
        }
        if include_total:
            headers["Prefer"] = "count=exact"
        request = urllib.request.Request(
            url=url,
            method="GET",
            headers=headers,
        )
        try:
            total_count = None
            with self._supabase_opener.open(request, timeout=6) as response:
                payload = response.read().decode("utf-8")
                if include_total:
                    content_range = response.headers.get("Content-Range", "")
                    if "/" in content_range:
                        count_part = content_range.rsplit("/", 1)[-1].strip()
                        if count_part.isdigit():
                            total_count = int(count_part)
            rows = json.loads(payload) if payload else []
            if not isinstance(rows, list):
                raise RuntimeError("invalid payload type")
            with self._lock:
                self._storage_error = None
            return rows, total_count
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, RuntimeError) as exc:
            with self._lock:
                self._storage_error = f"Supabase fetch failed: {exc}"
            return None

    def _fetch_session_rows_supabase(self, session_id: str, batch_size=1000):
        if not self._supabase_enabled:
            return []
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            fetched = self._fetch_events_supabase(
                limit=batch_size,
                session_id=session_id,
                offset=offset,
                include_total=False,
            )
            if fetched is None:
                return None
            batch, _ = fetched
            if not batch:
                break
            rows.extend(batch)
            batch_len = len(batch)
            if batch_len < batch_size:
                break
            offset += batch_len
            if offset > 200000:
                break
        return rows

    def _delete_session_rows_supabase(self, session_id: str):
        if not self._supabase_enabled:
            return
        table_name = urllib.parse.quote(self._supabase_table, safe="")
        params = urllib.parse.urlencode({"session_id": f"eq.{session_id}"})
        url = f"{self._supabase_url}/rest/v1/{table_name}?{params}"
        request = urllib.request.Request(
            url=url,
            method="DELETE",
            headers={
                "apikey": self._supabase_key,
                "Authorization": f"Bearer {self._supabase_key}",
                "Prefer": "return=minimal",
            },
        )
        try:
            with self._supabase_opener.open(request, timeout=8) as response:
                status = int(getattr(response, "status", 0))
                if status not in {200, 204}:
                    raise RuntimeError(f"HTTP {status}")
            with self._lock:
                self._storage_error = None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Supabase delete failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Supabase delete failed: {exc}") from exc

    def _delete_snapshot_object_supabase(self, object_path: str):
        if not self._supabase_enabled or not self._snapshot_enabled:
            return False
        clean_path = object_path.strip().strip("/")
        if not clean_path:
            return False
        bucket_name = urllib.parse.quote(self._snapshot_bucket, safe="")
        object_path_encoded = urllib.parse.quote(clean_path, safe="/")
        url = f"{self._supabase_url}/storage/v1/object/{bucket_name}/{object_path_encoded}"
        request = urllib.request.Request(
            url=url,
            method="DELETE",
            headers={
                "apikey": self._supabase_key,
                "Authorization": f"Bearer {self._supabase_key}",
            },
        )
        try:
            with self._supabase_opener.open(request, timeout=8) as response:
                status = int(getattr(response, "status", 0))
                if status not in {200, 204}:
                    raise RuntimeError(f"HTTP {status}")
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"snapshot delete failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"snapshot delete failed: {exc}") from exc

    def _clear_session_from_memory(self, session_id: str):
        with self._lock:
            current_events = list(self._events)
            kept_events = [
                row for row in current_events if str(row.get("session_id") or "").strip() != session_id
            ]
            removed_count = len(current_events) - len(kept_events)
            self._events = deque(kept_events, maxlen=self._events.maxlen)
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
        deleted_events = 0
        deleted_images = 0
        image_delete_errors: list[str] = []
        if self._supabase_enabled:
            rows = self._fetch_session_rows_supabase(sid)
            if rows is None:
                raise RuntimeError("Supabase fetch failed before delete")
            deleted_events = len(rows)
            session_part = self._safe_path_part(sid)
            for row in rows:
                snapshot_path = str(row.get("snapshot_path") or "").strip().strip("/")
                if snapshot_path:
                    snapshot_paths.add(snapshot_path)
                event_id = str(row.get("event_id") or "").strip()
                if event_id:
                    fallback_event_id = self._safe_path_part(event_id)
                    snapshot_paths.add(f"{self._snapshot_prefix}/{session_part}/{fallback_event_id}.jpg")

            self._delete_session_rows_supabase(sid)
            for object_path in snapshot_paths:
                try:
                    if self._delete_snapshot_object_supabase(object_path):
                        deleted_images += 1
                except Exception as exc:
                    image_delete_errors.append(str(exc))

        removed_from_memory = self._clear_session_from_memory(sid)
        payload = {
            "ok": True,
            "session_id": sid,
            "deleted_events": int(deleted_events),
            "deleted_images": int(deleted_images),
            "removed_from_memory": int(removed_from_memory),
        }
        if image_delete_errors:
            payload["image_delete_errors"] = image_delete_errors[:5]
            payload["image_delete_error_count"] = len(image_delete_errors)
        return payload

    def _event_ts_ms(self, event: dict[str, Any]):
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
        fetched = self._fetch_events_supabase(limit=scan_limit, include_total=True)
        if fetched is None:
            return {"sessions": [], "limit": limit, "source": "supabase", "error": "fetch_failed"}
        rows, total_count = fetched
        with self._lock:
            timeline_snapshot = {sid: dict(meta) for sid, meta in self._session_timeline.items()}
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            sid = str(row.get("session_id") or "").strip()
            if not sid:
                continue
            if exclude_session_id and sid == exclude_session_id:
                continue
            ts_ms = self._event_ts_ms(row)
            risk_score = row.get("risk_score")
            rep_distance_m = row.get("rep_distance_m")

            bucket = grouped.get(sid)
            if bucket is None:
                bucket = {
                    "session_id": sid,
                    "total": 0,
                    "warning": 0,
                    "danger": 0,
                    "risk_sum": 0.0,
                    "risk_count": 0,
                    "min_distance_m": None,
                    "start_ts_ms": 0,
                    "end_ts_ms": 0,
                }
                grouped[sid] = bucket

            bucket["total"] += 1
            risk = row.get("risk")
            if risk == "WARNING":
                bucket["warning"] += 1
            elif risk == "DANGER":
                bucket["danger"] += 1

            if isinstance(risk_score, (int, float)):
                bucket["risk_sum"] += float(risk_score)
                bucket["risk_count"] += 1

            if isinstance(rep_distance_m, (int, float)):
                distance_value = float(rep_distance_m)
                min_distance = bucket["min_distance_m"]
                if min_distance is None or distance_value < min_distance:
                    bucket["min_distance_m"] = distance_value

            if ts_ms > 0:
                if bucket["start_ts_ms"] == 0 or ts_ms < bucket["start_ts_ms"]:
                    bucket["start_ts_ms"] = ts_ms
                if ts_ms > bucket["end_ts_ms"]:
                    bucket["end_ts_ms"] = ts_ms

        sessions = []
        for bucket in grouped.values():
            session_id = str(bucket["session_id"])
            timeline = timeline_snapshot.get(session_id) or {}
            start_ts_ms = int(bucket["start_ts_ms"])
            end_ts_ms = int(bucket["end_ts_ms"])

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

            risk_count = int(bucket["risk_count"])
            avg_risk_pct = 0
            if risk_count > 0:
                avg_risk_pct = max(0, min(100, int(round((bucket["risk_sum"] / risk_count) * 100.0))))
            sessions.append(
                {
                    "session_id": session_id,
                    "total": int(bucket["total"]),
                    "warning": int(bucket["warning"]),
                    "danger": int(bucket["danger"]),
                    "avg_risk_pct": avg_risk_pct,
                    "min_distance_m": bucket["min_distance_m"],
                    "start_ts_ms": start_ts_ms,
                    "end_ts_ms": end_ts_ms,
                }
            )

        sessions.sort(
            key=lambda row: (
                int(row.get("end_ts_ms", 0)),
                int(row.get("start_ts_ms", 0)),
                str(row.get("session_id", "")),
            ),
            reverse=True,
        )
        limited_sessions = sessions[:limit]
        scanned_events = len(rows)
        return {
            "sessions": limited_sessions,
            "limit": limit,
            "scan_limit": scan_limit,
            "scanned_events": scanned_events,
            "total_events": total_count,
            "truncated": bool(total_count is not None and total_count > scan_limit),
            "excluded_session_id": exclude_session_id,
            "source": "supabase",
        }

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

    def events(self, limit=100, session_id: str | None = None, offset=0, include_total=False):
        fetched = self._fetch_events_supabase(
            limit=limit,
            session_id=session_id,
            offset=offset,
            include_total=include_total,
        )
        if fetched is None:
            return {"events": [], "limit": limit, "source": "supabase", "error": "fetch_failed"}
        rows, total_count = fetched
        payload = {
            "events": rows,
            "limit": limit,
            "offset": offset,
            "source": "supabase",
            "session_id": session_id,
        }
        if include_total:
            payload["total"] = int(total_count) if isinstance(total_count, int) else len(rows)
        return payload

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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service.start()
    try:
        yield
    finally:
        service.stop()


app = FastAPI(title="Collision Monitor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, "service": "collision-monitor", "start_error": service.state()["start_error"]}


@app.get("/api/state")
def get_state():
    return service.state()


@app.get("/api/events")
def get_events(
    limit: int = Query(default=100, ge=1, le=2000),
    offset: int = Query(default=0, ge=0, le=1000000),
    include_total: bool = Query(default=False),
    session_id: str | None = Query(default=None),
):
    normalized_session_id = session_id.strip() if session_id else None
    if normalized_session_id == "":
        normalized_session_id = None
    return service.events(
        limit=limit,
        session_id=normalized_session_id,
        offset=offset,
        include_total=include_total,
    )


@app.get("/api/sessions")
def get_sessions(
    limit: int = Query(default=40, ge=1, le=200),
    scan_limit: int = Query(default=5000, ge=200, le=20000),
    exclude_session_id: str | None = Query(default=None),
):
    normalized_exclude = exclude_session_id.strip() if exclude_session_id else None
    if normalized_exclude == "":
        normalized_exclude = None
    return service.sessions(limit=limit, scan_limit=scan_limit, exclude_session_id=normalized_exclude)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    try:
        return service.delete_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        if str(exc) == "active_session":
            raise HTTPException(status_code=409, detail="현재 사용 중인 세션은 삭제할 수 없습니다.") from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
