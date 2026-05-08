"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_MONITOR_API ?? "http://127.0.0.1:8000";

type RiskSummary = {
  high?: number;
  medium?: number;
  low?: number;
};

type ReportResponse = {
  ok?: boolean;
  output?: string;
  download_url?: string;
  report_date?: string;
  llm_provider?: "api" | "local" | "gemini" | "qwen";
  chain1?: {
    validated_count?: number;
    risk_summary?: RiskSummary;
  };
  chain2?: {
    total_events?: number;
    risk_distribution?: RiskSummary;
    summary?: string;
    risk_patterns?: string[];
    improvements?: string[];
  };
  chain3?: {
    summary?: string;
    improvements?: string[];
  };
};

type LlmProvider = "api" | "local";

type SessionSummaryApi = {
  session_id?: string | null;
  total?: number;
  warning?: number;
  danger?: number;
  avg_risk_pct?: number;
  start_ts_ms?: number;
  end_ts_ms?: number;
};

type SessionsResponse = {
  sessions?: SessionSummaryApi[];
};

function formatSessionLabel(session: SessionSummaryApi): string {
  const sid = session.session_id ?? "";
  const total = session.total ?? 0;
  if (session.start_ts_ms && session.start_ts_ms > 0) {
    const dt = new Date(session.start_ts_ms).toLocaleString("ko-KR", { hour12: false });
    return `${sid} (${dt}, ${total}건)`;
  }
  return `${sid} (${total}건)`;
}

function buildReportQuery(sessionId: string, llmProvider: LlmProvider) {
  const query = new URLSearchParams({ format: "pdf" });
  query.set("llm_provider", llmProvider);
  if (sessionId) query.set("session_id", sessionId);
  return query.toString();
}

export default function ReportsPage() {
  const [sessionId, setSessionId] = useState<string>("");
  const [llmProvider, setLlmProvider] = useState<LlmProvider>("api");
  const [sessions, setSessions] = useState<SessionSummaryApi[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState(true);
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchSessions = async () => {
    setSessionsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/sessions?limit=40&scan_limit=5000`, { cache: "no-store" });
      if (!response.ok) throw new Error(`sessions status ${response.status}`);
      const data = (await response.json()) as SessionsResponse;
      setSessions(
        (data.sessions ?? [])
          .filter((s): s is SessionSummaryApi & { session_id: string } => Boolean(s.session_id?.trim()))
          .sort((a, b) => (b.end_ts_ms ?? 0) - (a.end_ts_ms ?? 0)),
      );
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    } finally {
      setSessionsLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  const downloadUrl = useMemo(() => {
    if (!report?.download_url) return null;
    return `${API_BASE}${report.download_url}`;
  }, [report?.download_url]);

  const generateReport = async () => {
    setLoading(true);
    setError(null);
    try {
      const query = buildReportQuery(sessionId, llmProvider);
      const response = await fetch(`${API_BASE}/api/reports/daily?${query}`, { cache: "no-store" });
      if (!response.ok) {
        let detail = `report status ${response.status}`;
        try {
          const payload = (await response.json()) as { detail?: string };
          if (payload.detail) detail = payload.detail;
        } catch {}
        throw new Error(detail);
      }
      const payload = (await response.json()) as ReportResponse;
      setReport(payload);
      setApiOnline(true);
    } catch (err) {
      setApiOnline(false);
      setError(err instanceof Error ? err.message : "보고서 생성에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboardRoot">
      <aside className="sideNav">
        <div className="brandRow">
          <img src="/brand-logo-mark.png" alt="안전모탐정단 로고" className="brandLogo" />
          <span className="brandTitle">안전모탐정단</span>
        </div>

        <nav className="navList">
          <Link href="/" className="navItem">
            <span className="material-symbols-outlined">dashboard</span>
            <span>Dashboard</span>
          </Link>
          <Link href="/event-logs" className="navItem">
            <span className="material-symbols-outlined">list_alt</span>
            <span>Event Logs</span>
          </Link>
          <Link href="/analytics" className="navItem">
            <span className="material-symbols-outlined">analytics</span>
            <span>Analytics</span>
          </Link>
          <Link href="/settings" className="navItem">
            <span className="material-symbols-outlined">settings</span>
            <span>Settings</span>
          </Link>
          <Link href="/reports" className="navItem navItemActive">
            <span className="material-symbols-outlined">smart_toy</span>
            <span>AI Report</span>
          </Link>
        </nav>
      </aside>

      <main className="mainArea">
        <header className="topBar">
          <div className="topLeft">
            <h2>AI Report</h2>
            <span className={`systemPill ${apiOnline ? "online" : "offline"}`}>
              <span className={`pulseDot ${apiOnline ? "online" : "offline"}`} /> SYSTEM {apiOnline ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        </header>

        <div className="contentScroll">
          <section className="reportGrid">
            <article className="reportPanel">
              <div className="cardHeader cardHeaderBorder">
                <h3>
                  <span className="material-symbols-outlined">description</span>
                  Report Generator
                </h3>
                <span className="tinyPill">Chain 1 → Chain 2 → Chain 3</span>
              </div>
              <div className="reportControls">
                <label className="logsField">
                  <span>Session</span>
                  <select
                    className="logsInput"
                    value={sessionId}
                    onChange={(event) => setSessionId(event.target.value)}
                    disabled={sessionsLoading}
                  >
                    <option value="">전체 세션</option>
                    {sessions.map((s) => (
                      <option key={s.session_id} value={s.session_id ?? ""}>
                        {formatSessionLabel(s)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="logsField">
                  <span>Output Mode</span>
                  <select
                    className="logsInput"
                    value={llmProvider}
                    onChange={(event) => setLlmProvider(event.target.value as LlmProvider)}
                    disabled={loading}
                  >
                    <option value="api">API</option>
                    <option value="local">Local</option>
                  </select>
                </label>
                <button type="button" className="reportPrimaryButton" disabled={loading} onClick={generateReport}>
                  <span className="material-symbols-outlined">{loading ? "hourglass_top" : "auto_awesome"}</span>
                  {loading ? "Generating..." : "Generate PDF"}
                </button>
              </div>
              <p className="reportHint">
                저장된 충돌 이벤트를 종합해 위험도 검증, 반복 패턴 분석, 개선 조치 제안을 포함한 PDF 보고서를 생성합니다.
              </p>
              {error && <div className="bannerError">{error}</div>}
            </article>

            <article className="reportPanel">
              <div className="cardHeader cardHeaderBorder">
                <h3>
                  <span className="material-symbols-outlined">task_alt</span>
                  Latest Output
                </h3>
                <span className="tinyPill">{report?.llm_provider ? report.llm_provider.toUpperCase() : (report?.output ?? "No report")}</span>
              </div>
              <div className="reportResult">
                <div className="reportMetricRow">
                  <div>
                    <span>Total Events</span>
                    <strong>{report?.chain2?.total_events ?? "-"}</strong>
                  </div>
                  <div>
                    <span>High</span>
                    <strong className="reportDanger">{report?.chain2?.risk_distribution?.high ?? "-"}</strong>
                  </div>
                  <div>
                    <span>Medium</span>
                    <strong className="reportWarning">{report?.chain2?.risk_distribution?.medium ?? "-"}</strong>
                  </div>
                  <div>
                    <span>Low</span>
                    <strong>{report?.chain2?.risk_distribution?.low ?? "-"}</strong>
                  </div>
                </div>
                <p className="reportSummary">{report?.chain3?.summary ?? report?.chain2?.summary ?? "생성된 보고서 요약이 여기에 표시됩니다."}</p>
                {downloadUrl && (
                  <a className="reportDownloadButton" href={downloadUrl} target="_blank" rel="noreferrer">
                    <span className="material-symbols-outlined">download</span>
                    Download PDF
                  </a>
                )}
              </div>
            </article>
          </section>

          <section className="reportDetailGrid">
            <article className="reportPanel">
              <div className="cardHeader cardHeaderBorder">
                <h3>Risk Patterns</h3>
              </div>
              <div className="analyticsList">
                {(report?.chain2?.risk_patterns ?? []).length === 0 && <p className="analyticsEmpty">생성 후 반복 위험 패턴이 표시됩니다.</p>}
                {(report?.chain2?.risk_patterns ?? []).map((item, index) => (
                  <div key={`${item}-${index}`} className="analyticsListRow">
                    <p className="reportListText">{item}</p>
                  </div>
                ))}
              </div>
            </article>

            <article className="reportPanel">
              <div className="cardHeader cardHeaderBorder">
                <h3>Improvements</h3>
              </div>
              <div className="analyticsList">
                {(report?.chain3?.improvements ?? report?.chain2?.improvements ?? []).length === 0 && (
                  <p className="analyticsEmpty">생성 후 개선 조치가 표시됩니다.</p>
                )}
                {(report?.chain3?.improvements ?? report?.chain2?.improvements ?? []).map((item, index) => (
                  <div key={`${item}-${index}`} className="analyticsListRow">
                    <p className="reportListText">{item}</p>
                  </div>
                ))}
              </div>
            </article>
          </section>
        </div>
      </main>
    </div>
  );
}
