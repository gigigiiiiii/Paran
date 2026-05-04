from __future__ import annotations

import threading
from datetime import date, datetime
from typing import Any

from .chain3_pdf_generator import DashboardReportGenerator


class DailyReportScheduler:
    """Small background scheduler for dashboard-based daily reports."""

    def __init__(
        self,
        generator: DashboardReportGenerator,
        run_at: str = "23:59",
        output_format: str = "pdf",
    ):
        self._generator = generator
        self._run_at = run_at
        self._output_format = output_format
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_run_date: date | None = None
        self._last_result: dict[str, Any] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="daily-report-scheduler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None

    def state(self) -> dict[str, Any]:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "run_at": self._run_at,
            "output_format": self._output_format,
            "last_run_date": self._last_run_date.isoformat() if self._last_run_date else None,
            "last_result": self._last_result,
        }

    def run_once(self, target_date: date | None = None) -> dict[str, Any]:
        run_date = target_date or datetime.now().astimezone().date()
        result = self._generator.generate(target_date=run_date, output_format=self._output_format)
        self._last_run_date = run_date
        self._last_result = {
            "ok": result.get("ok"),
            "path": result.get("path"),
            "format": result.get("format"),
            "output": result.get("output"),
            "report_date": result.get("report_date"),
        }
        return result

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = datetime.now().astimezone()
            if now.strftime("%H:%M") >= self._run_at and self._last_run_date != now.date():
                try:
                    self.run_once(now.date())
                except Exception as exc:
                    self._last_result = {"ok": False, "error": str(exc)}
            self._stop.wait(30)
