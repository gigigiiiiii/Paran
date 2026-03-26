"""
backend/supabase_client.py
===========================
Supabase REST API HTTP 클라이언트.
MonitorService에서 분리된 순수 HTTP 호출 레이어.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


class SupabaseClient:
    """Supabase REST API / Storage HTTP 호출을 담당하는 클래스."""

    def __init__(
        self,
        url: str,
        key: str,
        table: str,
        snapshot_bucket: str,
        snapshot_prefix: str,
        retry_count: int = 3,
        retry_delay_sec: float = 0.4,
    ):
        self._url    = url.rstrip("/")
        self._key    = key
        self._table  = table
        self._snapshot_bucket = snapshot_bucket
        self._snapshot_prefix = snapshot_prefix.strip().strip("/") or "events"
        self._retry_count     = max(1, min(10, retry_count))
        self._retry_delay_sec = max(0.0, min(5.0, retry_delay_sec))
        # 로컬 프록시 우회 (global HTTP_PROXY 무시)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def enabled(self) -> bool:
        return bool(self._url and self._key)

    @property
    def snapshot_enabled(self) -> bool:
        return bool(self._snapshot_bucket)

    # ── 경로 유틸 ─────────────────────────────────────────────────────────────

    def _safe_path_part(self, value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
        safe = safe.strip("._")
        return safe or "unknown"

    def _build_snapshot_object_path(self, event: dict[str, Any]) -> str:
        session_part = self._safe_path_part(str(event.get("session_id") or "no-session"))
        event_id     = self._safe_path_part(str(event.get("event_id") or uuid.uuid4().hex))
        return f"{self._snapshot_prefix}/{session_part}/{event_id}.jpg"

    # ── 이벤트 삽입 ───────────────────────────────────────────────────────────

    def insert_event(self, event: dict[str, Any]) -> None:
        table_name = urllib.parse.quote(self._table, safe="")
        url        = f"{self._url}/rest/v1/{table_name}"
        payload    = json.dumps(event, ensure_ascii=False).encode("utf-8")
        request    = urllib.request.Request(
            url=url,
            data=payload,
            method="POST",
            headers={
                "apikey":        self._key,
                "Authorization": f"Bearer {self._key}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
        )
        try:
            with self._opener.open(request, timeout=6) as response:
                status = int(getattr(response, "status", 0))
                if status not in {200, 201, 204}:
                    raise RuntimeError(f"HTTP {status}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    def insert_event_resilient(self, event: dict[str, Any]) -> list[str]:
        """컬럼 누락 오류 발생 시 해당 컬럼을 제거하고 재시도한다."""
        payload         = dict(event)
        dropped_columns: list[str] = []
        for _ in range(12):
            try:
                self.insert_event(payload)
                return dropped_columns
            except Exception as exc:
                message = str(exc)
                match   = re.search(r"Could not find the '([^']+)' column", message)
                if match is None:
                    raise
                missing_column = match.group(1)
                if missing_column not in payload:
                    raise
                payload.pop(missing_column, None)
                dropped_columns.append(missing_column)
        raise RuntimeError("Supabase insert failed after dropping missing columns")

    # ── 스냅샷 업로드 ─────────────────────────────────────────────────────────

    def upload_snapshot(self, event: dict[str, Any], jpeg_bytes: bytes) -> str:
        object_path          = self._build_snapshot_object_path(event)
        bucket_name          = urllib.parse.quote(self._snapshot_bucket, safe="")
        object_path_encoded  = urllib.parse.quote(object_path, safe="/")
        url                  = f"{self._url}/storage/v1/object/{bucket_name}/{object_path_encoded}"
        request              = urllib.request.Request(
            url=url,
            data=jpeg_bytes,
            method="POST",
            headers={
                "apikey":        self._key,
                "Authorization": f"Bearer {self._key}",
                "Content-Type":  "image/jpeg",
                "x-upsert":      "true",
            },
        )
        with self._opener.open(request, timeout=8) as response:
            status = int(getattr(response, "status", 0))
            if status not in {200, 201}:
                raise RuntimeError(f"HTTP {status}")
        return object_path

    def upload_snapshot_with_retry(self, event: dict[str, Any], jpeg_bytes: bytes) -> str:
        last_error = None
        for attempt in range(1, self._retry_count + 1):
            try:
                return self.upload_snapshot(event, jpeg_bytes)
            except Exception as exc:
                last_error = exc
                if attempt < self._retry_count and self._retry_delay_sec > 0:
                    time.sleep(self._retry_delay_sec * attempt)
        raise RuntimeError(f"after {self._retry_count} attempts: {last_error}")

    # ── 이벤트 조회 ───────────────────────────────────────────────────────────

    def fetch_events(
        self,
        limit: int,
        session_id: str | None = None,
        offset: int = 0,
        include_total: bool = False,
    ) -> tuple[list[dict[str, Any]], int | None]:
        table_name   = urllib.parse.quote(self._table, safe="")
        query_params = {
            "select": "*",
            "order":  "created_at.desc",
            "limit":  str(limit),
            "offset": str(offset),
        }
        if session_id:
            if not re.match(r"^session-\d+$", session_id):
                raise ValueError(f"Invalid session_id format: {session_id!r}")
            query_params["session_id"] = f"eq.{session_id}"
        params  = urllib.parse.urlencode(query_params)
        url     = f"{self._url}/rest/v1/{table_name}?{params}"
        headers = {
            "apikey":        self._key,
            "Authorization": f"Bearer {self._key}",
            "Accept":        "application/json",
        }
        if include_total:
            headers["Prefer"] = "count=exact"
        request = urllib.request.Request(url=url, method="GET", headers=headers)
        try:
            total_count = None
            with self._opener.open(request, timeout=6) as response:
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
            return rows, total_count
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Supabase fetch failed: {exc}") from exc

    def fetch_session_rows(self, session_id: str, batch_size: int = 1000) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            batch, _ = self.fetch_events(limit=batch_size, session_id=session_id, offset=offset)
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

    # ── 삭제 ──────────────────────────────────────────────────────────────────

    def delete_session_rows(self, session_id: str) -> None:
        table_name = urllib.parse.quote(self._table, safe="")
        params     = urllib.parse.urlencode({"session_id": f"eq.{session_id}"})
        url        = f"{self._url}/rest/v1/{table_name}?{params}"
        request    = urllib.request.Request(
            url=url,
            method="DELETE",
            headers={
                "apikey":        self._key,
                "Authorization": f"Bearer {self._key}",
                "Prefer":        "return=minimal",
            },
        )
        try:
            with self._opener.open(request, timeout=8) as response:
                status = int(getattr(response, "status", 0))
                if status not in {200, 204}:
                    raise RuntimeError(f"HTTP {status}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Supabase delete failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Supabase delete failed: {exc}") from exc

    def delete_snapshot_object(self, object_path: str) -> bool:
        clean_path = object_path.strip().strip("/")
        if not clean_path:
            return False
        bucket_name         = urllib.parse.quote(self._snapshot_bucket, safe="")
        object_path_encoded = urllib.parse.quote(clean_path, safe="/")
        url                 = f"{self._url}/storage/v1/object/{bucket_name}/{object_path_encoded}"
        request             = urllib.request.Request(
            url=url,
            method="DELETE",
            headers={
                "apikey":        self._key,
                "Authorization": f"Bearer {self._key}",
            },
        )
        try:
            with self._opener.open(request, timeout=8) as response:
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
