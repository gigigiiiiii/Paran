from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
    
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
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_env_file(Path(__file__).resolve().with_name(".env"))

try:
    from .monitor import MonitorService  # 패키지로 임포트 시
    from .reports import DailyReportScheduler, DashboardReportGenerator
    from .reports.llm_utils import check_local_runtime, llm_runtime_info
except ImportError:
    from monitor import MonitorService   # uvicorn app:app 직접 실행 시
    from reports import DailyReportScheduler, DashboardReportGenerator
    from reports.llm_utils import check_local_runtime, llm_runtime_info


service = MonitorService()
report_output_dir = Path(os.getenv("REPORT_OUTPUT_DIR") or str(Path(__file__).resolve().parents[1] / "out_reports"))
report_generator = DashboardReportGenerator(
    event_source=service.events,
    output_dir=str(report_output_dir),
    supabase_url=os.getenv("SUPABASE_URL"),
    snapshot_bucket=os.getenv("SUPABASE_SNAPSHOT_BUCKET", "collision-event-snaps").strip(),
)
report_scheduler = DailyReportScheduler(
    report_generator,
    run_at=os.getenv("REPORT_DAILY_RUN_AT", "23:59"),
    output_format=os.getenv("REPORT_OUTPUT_FORMAT", "pdf"),
    llm_provider=os.getenv("REPORT_LLM_PROVIDER"),
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service.start()
    if os.getenv("REPORT_SCHEDULER_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}:
        report_scheduler.start()
    try:
        yield
    finally:
        report_scheduler.stop()
        service.stop()


app = FastAPI(title="Collision Monitor API", lifespan=lifespan)

# CORS: 환경변수 CORS_ORIGINS 로 허용 출처 지정 (콤마 구분)
# 기본값: localhost:3000 / 127.0.0.1:3000
_cors_origins_env = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).strip()
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] or [
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API 엔드포인트 ─────────────────────────────────────────────────────────────

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


@app.get("/api/mode")
def get_mode():
    return service.get_mode()


def _parse_report_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD") from exc


@app.get("/api/reports/daily")
def generate_daily_report(
    date_: str | None = Query(default=None, alias="date"),
    session_id: str | None = Query(default=None),
    format: str = Query(default="pdf", pattern="^pdf$"),
    llm_provider: str | None = Query(default=None, pattern="^(api|local|gemini|qwen)$"),
    download: bool = Query(default=False),
):
    normalized_session_id = session_id.strip() if session_id else None
    if normalized_session_id == "":
        normalized_session_id = None
    try:
        result = report_generator.generate(
            target_date=_parse_report_date(date_),
            session_id=normalized_session_id,
            output_format=format,
            llm_provider=llm_provider,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"보고서 생성 실패: {exc}") from exc
    if download:
        return FileResponse(result["path"], filename=Path(result["path"]).name)
    return {
        "ok": result["ok"],
        "path": result["path"],
        "format": result["format"],
        "output": result["output"],
        "llm_provider": result.get("llm_provider"),
        "download_url": f"/api/reports/files/{Path(result['path']).name}",
        "report_date": result["report_date"],
        "chain1": {
            "risk_summary": result["chain1"]["risk_summary"],
            "validated_count": len(result["chain1"]["validated_events"]),
        },
        "chain2": result["chain2"],
        "chain3": {
            "summary": result["chain3"].get("summary"),
            "improvements": result["chain3"].get("improvements"),
        },
    }


@app.get("/api/reports/scheduler")
def get_report_scheduler_state():
    return report_scheduler.state()


@app.get("/api/reports/llm/status")
def get_report_llm_status(
    llm_provider: str | None = Query(default=None, pattern="^(api|local|gemini|qwen)$"),
):
    info = llm_runtime_info(llm_provider)
    if info["provider"] == "local":
        return check_local_runtime(info["model"])
    return {
        **info,
        "api_key_configured": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    }


@app.get("/api/reports/files/{filename}")
def download_report_file(filename: str):
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="invalid report filename")
    full_path = (report_output_dir / safe_name).resolve()
    if not str(full_path).startswith(str(report_output_dir.resolve())):
        raise HTTPException(status_code=403, detail="invalid report path")
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="report file not found")
    return FileResponse(str(full_path), filename=safe_name, media_type="application/pdf")


@app.post("/api/reports/scheduler/run")
def run_report_scheduler_once(date_: str | None = Query(default=None, alias="date")):
    result = report_scheduler.run_once(_parse_report_date(date_))
    return {
        "ok": result["ok"],
        "path": result["path"],
        "format": result["format"],
        "output": result["output"],
        "llm_provider": result.get("llm_provider"),
        "report_date": result["report_date"],
        "chain2": result["chain2"],
        "chain3": {
            "summary": result["chain3"].get("summary"),
            "improvements": result["chain3"].get("improvements"),
        },
    }


@app.post("/api/mode/live")
def switch_to_live():
    return service.switch_source(None)


@app.post("/api/mode/test")
def switch_to_test(file: str = Query(..., description="test_videos 폴더 내 파일명")):
    test_videos_dir = Path(__file__).resolve().parent.parent / "test_videos"
    full_path       = (test_videos_dir / file).resolve()
    # 경로 순회 공격 방지
    if not str(full_path).startswith(str(test_videos_dir.resolve())):
        raise HTTPException(status_code=403, detail="접근이 허용되지 않는 경로입니다.")
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {file}")
    return service.switch_source(str(full_path))


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
