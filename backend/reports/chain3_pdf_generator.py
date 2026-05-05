from __future__ import annotations

import html
import os
import urllib.request
from io import BytesIO
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any, Callable

from .chain1_event_validator import event_datetime, validate_events
from .chain2_report_material import build_report_material
from .image_utils import attach_snapshot_urls
from .llm_utils import dumps_for_prompt, run_llm_json_chain
from .prompts import CHAIN3_HUMAN_TEMPLATE, CHAIN3_SYSTEM_PROMPT
from .schemas import Chain3Output, validate_model


EventSource = Callable[..., Any]


def _fmt(value: Any, suffix: str = "", empty: str = "-") -> str:
    if value is None:
        return empty
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _case_row(event: dict[str, Any]) -> list[str]:
    return [
        str(event.get("event_time") or "-"),
        str(event.get("risk_level") or "-"),
        f"{event.get('risk_percent') or 0}%",
        str(event.get("obstacle_name") or "-"),
        _fmt(event.get("rep_distance_m"), " m"),
        _fmt(event.get("ttc_s"), " s"),
    ]


def _table_style(font_size: int = 8, font_name: str = "Helvetica"):
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle

    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
    ])


_KOREAN_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\gulim.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
_KOREAN_FONT_NAME = "KoreanFont"


def _register_korean_font() -> bool:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for candidate in _KOREAN_FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                pdfmetrics.registerFont(TTFont(_KOREAN_FONT_NAME, candidate))
                return True
            except Exception:
                continue
    return False


def _apply_korean_font(styles: Any, font_available: bool) -> None:
    if not font_available:
        return
    for style_name in ("Title", "Heading2", "BodyText", "Normal"):
        if style_name in styles:
            styles[style_name].fontName = _KOREAN_FONT_NAME


def write_daily_report_pdf(
    chain2_output: dict[str, Any],
    output_dir: str | os.PathLike[str],
    report_date: date,
) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"daily_report_{report_date.isoformat()}.pdf"

    font_available = _register_korean_font()
    styles = getSampleStyleSheet()
    _apply_korean_font(styles, font_available)

    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=32, leftMargin=32, topMargin=32, bottomMargin=32)
    title = str(chain2_output.get("title") or "일일 충돌 위험 보고서")
    story = [
        Paragraph(f"{html.escape(title)} - {report_date.isoformat()}", styles["Title"]),
        Spacer(1, 12),
        Paragraph("1. 요약", styles["Heading2"]),
        Paragraph(html.escape(str(chain2_output.get("summary") or "-")), styles["BodyText"]),
        Spacer(1, 12),
    ]

    distribution = chain2_output.get("risk_distribution") or {}
    risk_table = Table([
        ["Total Events", "High", "Medium", "Low"],
        [
            str(chain2_output.get("total_events") or 0),
            str(distribution.get("high") or 0),
            str(distribution.get("medium") or 0),
            str(distribution.get("low") or 0),
        ],
    ], repeatRows=1)
    risk_table.setStyle(_table_style())
    story.extend([
        Paragraph("2. 위험 등급 분포", styles["Heading2"]),
        risk_table,
        Spacer(1, 12),
    ])

    case_rows = [["Time", "Level", "Risk", "Obstacle", "Distance", "TTC"]]
    for event in chain2_output.get("key_cases") or []:
        if isinstance(event, dict):
            case_rows.append(_case_row(event))
    if len(case_rows) == 1:
        case_rows.append(["-", "-", "-", "-", "-", "-"])

    case_table = Table(case_rows, repeatRows=1)
    case_table.setStyle(_table_style(font_size=7))
    story.extend([
        Paragraph("3. 핵심 케이스", styles["Heading2"]),
        case_table,
        Spacer(1, 12),
    ])

    snapshot_flowables = _snapshot_flowables(chain2_output.get("key_cases") or [], styles, Image)
    story.append(Paragraph("4. 증거 스냅샷", styles["Heading2"]))
    if snapshot_flowables:
        story.extend(snapshot_flowables)
    else:
        story.append(Paragraph("선택된 핵심 케이스에 대한 스냅샷 이미지를 불러올 수 없습니다.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    risk_patterns = chain2_output.get("risk_patterns") or []
    story.append(Paragraph("5. 위험 패턴", styles["Heading2"]))
    if risk_patterns:
        for item in risk_patterns:
            story.append(Paragraph(f"- {html.escape(str(item))}", styles["BodyText"]))
    else:
        story.append(Paragraph("-", styles["BodyText"]))
    story.append(Spacer(1, 12))

    improvements = chain2_output.get("improvements") or []
    story.append(Paragraph("6. 개선 조치", styles["Heading2"]))
    if improvements:
        for item in improvements:
            story.append(Paragraph(f"- {html.escape(str(item))}", styles["BodyText"]))
    else:
        story.append(Paragraph("-", styles["BodyText"]))

    doc.build(story)
    return path


def prepare_final_report_material(chain2_output: dict[str, Any], use_llm: bool = True) -> dict[str, Any]:
    if not use_llm:
        return chain2_output

    try:
        llm_output = run_llm_json_chain(
            CHAIN3_SYSTEM_PROMPT,
            CHAIN3_HUMAN_TEMPLATE,
            {"chain2_json": dumps_for_prompt(chain2_output)},
            output_model=Chain3Output,
        )
        final_output = dict(chain2_output)
        for key in ("title", "summary", "key_cases", "risk_patterns", "improvements"):
            if key in llm_output:
                final_output[key] = llm_output[key]
        if isinstance(final_output.get("key_cases"), list):
            final_output["key_cases"] = _preserve_case_evidence(
                chain2_output.get("key_cases") or [],
                final_output["key_cases"],
            )
        final_output["risk_distribution"] = dict(chain2_output.get("risk_distribution") or {})
        final_output["total_events"] = int(chain2_output.get("total_events") or 0)
        return validate_model(Chain3Output, final_output)
    except Exception as exc:
        fallback = dict(chain2_output)
        fallback["llm_error"] = str(exc)
        return fallback


class DashboardReportGenerator:
    """Runs chain1 -> chain2 -> chain3 for dashboard collision records."""

    def __init__(
        self,
        event_source: EventSource,
        output_dir: str | os.PathLike[str] | None = None,
        supabase_url: str | None = None,
        snapshot_bucket: str | None = None,
    ):
        self._event_source = event_source
        self._output_dir = Path(output_dir or Path(__file__).resolve().parents[2] / "out_reports")
        self._supabase_url = supabase_url
        self._snapshot_bucket = snapshot_bucket

    def fetch_dashboard_events(
        self,
        session_id: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        batch_size = min(1000, max(1, limit))
        while len(rows) < limit:
            payload = self._event_source(
                limit=batch_size,
                session_id=session_id,
                offset=offset,
                include_total=True,
            )
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(f"Supabase 이벤트 조회 실패: {payload['error']}")
            batch = payload.get("events", []) if isinstance(payload, dict) else payload
            if not isinstance(batch, list) or not batch:
                break
            rows.extend(dict(row) for row in batch if isinstance(row, dict))
            if len(batch) < batch_size:
                break
            offset += len(batch)
        return rows[:limit]

    def filter_events(
        self,
        events: list[dict[str, Any]],
        target_date: date | None = None,
    ) -> list[dict[str, Any]]:
        if target_date is None:
            return list(events)
        start = datetime.combine(target_date, dt_time.min).astimezone()
        end = datetime.combine(target_date, dt_time.max).astimezone()
        return [
            event for event in events
            if (event_dt := event_datetime(event)) is not None and start <= event_dt <= end
        ]

    def run_chains(
        self,
        target_date: date | None = None,
        session_id: str | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        report_date = target_date or datetime.now().astimezone().date()
        source_events = list(events) if events is not None else self.fetch_dashboard_events(session_id=session_id)
        filtered_events = self.filter_events(source_events, target_date=target_date)
        filtered_events = attach_snapshot_urls(filtered_events, self._supabase_url, self._snapshot_bucket)

        chain1_output = validate_events(filtered_events)
        chain2_output = build_report_material(chain1_output)
        chain3_output = prepare_final_report_material(chain2_output)
        pdf_path = write_daily_report_pdf(chain3_output, self._output_dir, report_date)
        return {
            "ok": True,
            "path": str(pdf_path),
            "format": "pdf",
            "report_date": report_date.isoformat(),
            "chain1": chain1_output,
            "chain2": chain2_output,
            "chain3": chain3_output,
            "output": pdf_path.name,
        }

    def generate(
        self,
        target_date: date | None = None,
        session_id: str | None = None,
        output_format: str = "pdf",
    ) -> dict[str, Any]:
        if output_format.lower() != "pdf":
            raise ValueError("daily report chain output is PDF only")
        return self.run_chains(target_date=target_date, session_id=session_id)


def _preserve_case_evidence(
    source_cases: list[dict[str, Any]],
    final_cases: list[Any],
) -> list[dict[str, Any]]:
    source_by_id = {str(case.get("event_id")): case for case in source_cases if case.get("event_id") is not None}
    merged_cases = []
    for idx, item in enumerate(final_cases):
        if not isinstance(item, dict):
            continue
        event_id = str(item.get("event_id")) if item.get("event_id") is not None else ""
        source = source_by_id.get(event_id) or (source_cases[idx] if idx < len(source_cases) else {})
        merged = dict(source)
        merged.update(item)
        if not merged.get("snapshot_url"):
            merged["snapshot_url"] = source.get("snapshot_url")
        if not merged.get("snapshot_path"):
            merged["snapshot_path"] = source.get("snapshot_path")
        merged_cases.append(merged)
    return merged_cases


def _snapshot_flowables(key_cases: list[Any], styles: Any, image_cls: Any) -> list[Any]:
    flowables = []
    added = 0
    for case in key_cases:
        if not isinstance(case, dict):
            continue
        snapshot_url = str(case.get("snapshot_url") or "").strip()
        if not snapshot_url:
            continue
        image_bytes = _download_image(snapshot_url)
        if image_bytes is None:
            continue
        caption = (
            f"{case.get('event_time', '-')} / {case.get('risk_level', '-')} / "
            f"{case.get('obstacle_name', '-')}"
        )
        image = image_cls(BytesIO(image_bytes), width=260, height=146)
        flowables.extend([
            image,
            Paragraph(html.escape(caption), styles["BodyText"]),
            Spacer(1, 8),
        ])
        added += 1
        if added >= 4:
            break
    return flowables


def _download_image(url: str) -> bytes | None:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "collision-report-generator/1.0"})
        with urllib.request.urlopen(request, timeout=8) as response:
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type.lower():
                return None
            return response.read(4 * 1024 * 1024)
    except Exception:
        return None
