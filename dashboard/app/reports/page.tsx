"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_MONITOR_API ?? "http://127.0.0.1:8000";

type RiskSummary = {
  high?: number;
  medium?: number;
  low?: number;
};

type StoredReport = {
  session_id?: string | null;
  generated_at?: string | null;
  report_date?: string | null;
  llm_provider?: "api" | "local" | "gemini" | "qwen" | null;
  summary?: string | null;
  total_events?: number;
  risk_distribution?: RiskSummary;
  pdf_filename?: string | null;
  download_url?: string | null;
};

type ReportResponse = {
  ok?: boolean;
  output?: string;
  download_url?: string;
  report_date?: string;
  llm_provider?: "api" | "local" | "gemini" | "qwen";
  report?: StoredReport;
  chain1?: {
    validated_count?: number;
    risk_summary?: RiskSummary;
  };
  chain2?: {
    total_events?: number;
    risk_distribution?: RiskSummary;
    summary?: string;
  };
  chain3?: {
    summary?: string;
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
  report?: StoredReport | null;
};

type SessionsResponse = {
  sessions?: SessionSummaryApi[];
};

function coerceRiskSummary(value: unknown): RiskSummary {
  if (typeof value === "string") {
    try {
      return coerceRiskSummary(JSON.parse(value) as unknown);
    } catch {
      return {};
    }
  }
  if (!value || typeof value !== "object") return {};
  const source = value as Record<string, unknown>;
  return {
    high: Number.isFinite(Number(source.high)) ? Number(source.high) : undefined,
    medium: Number.isFinite(Number(source.medium)) ? Number(source.medium) : undefined,
    low: Number.isFinite(Number(source.low)) ? Number(source.low) : undefined,
  };
}

function normalizeReport(payload: ReportResponse | StoredReport | null | undefined): StoredReport | null {
  if (!payload) return null;
  if ("report" in payload && payload.report) return normalizeReport(payload.report);
  if ("chain2" in payload || "chain3" in payload) {
    const response = payload as ReportResponse;
    return {
      report_date: response.report_date,
      llm_provider: response.llm_provider,
      summary: response.chain3?.summary ?? response.chain2?.summary,
      total_events: response.chain2?.total_events,
      risk_distribution: coerceRiskSummary(response.chain2?.risk_distribution),
      pdf_filename: response.output,
      download_url: response.download_url,
    };
  }
  const stored = payload as StoredReport;
  return {
    ...stored,
    total_events: Number.isFinite(Number(stored.total_events)) ? Number(stored.total_events) : undefined,
    risk_distribution: coerceRiskSummary(stored.risk_distribution),
  };
}

function formatSessionLabel(session: SessionSummaryApi): string {
  const sid = session.session_id ?? "";
  const total = session.total ?? 0;
  if (session.start_ts_ms && session.start_ts_ms > 0) {
    const dt = new Date(session.start_ts_ms).toLocaleString("ko-KR", { hour12: false });
    return `${sid} (${dt}, ${total} events)`;
  }
  return `${sid} (${total} events)`;
}

function formatDateTime(value: string | number | null | undefined) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString("ko-KR", { hour12: false });
}

function buildDashboardSummary(report: StoredReport | null, sessionId: string) {
  if (!report) {
    return sessionId ? "이 세션의 보고서가 아직 생성되지 않았습니다." : "세션을 선택하면 보고서 요약을 확인할 수 있습니다.";
  }
  const distribution = report.risk_distribution ?? {};
  const total = report.total_events ?? 0;
  return `총 ${total}건 중 고위험 ${distribution.high ?? 0}건, 중위험 ${distribution.medium ?? 0}건, 저위험 ${distribution.low ?? 0}건입니다. 상세 케이스, 스냅샷, 위험 패턴과 개선 조치는 PDF에서 확인하세요.`;
}

function buildReportQuery(sessionId: string, llmProvider: LlmProvider) {
  const query = new URLSearchParams({ format: "pdf", session_id: sessionId });
  query.set("llm_provider", llmProvider);
  return query.toString();
}

export default function ReportsPage() {
  const [sessionId, setSessionId] = useState<string>("");
  const [llmProvider, setLlmProvider] = useState<LlmProvider>("local");
  const [sessions, setSessions] = useState<SessionSummaryApi[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState(true);
  const [report, setReport] = useState<StoredReport | null>(null);
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

  const selectedSession = useMemo(
    () => sessions.find((session) => session.session_id === sessionId) ?? null,
    [sessionId, sessions],
  );

  useEffect(() => {
    setError(null);
    setReport(normalizeReport(selectedSession?.report));
  }, [selectedSession]);

  const activeReport = report ?? normalizeReport(selectedSession?.report);
  const selectedHasReport = Boolean(activeReport);

  const downloadUrl = useMemo(() => {
    if (!activeReport?.download_url) return null;
    if (activeReport.download_url.startsWith("http")) return activeReport.download_url;
    return `${API_BASE}${activeReport.download_url}`;
  }, [activeReport?.download_url]);

  const generateReport = async () => {
    if (!sessionId || selectedHasReport) return;
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
      setReport(normalizeReport(payload));
      await fetchSessions();
      setApiOnline(true);
    } catch (err) {
      setApiOnline(false);
      setError(err instanceof Error ? err.message : "Report generation failed.");
    } finally {
      setLoading(false);
    }
  };

  const riskDistribution = activeReport?.risk_distribution ?? {};
  const dangerRatio = activeReport?.total_events
    ? Math.round(((riskDistribution.high ?? 0) / activeReport.total_events) * 100)
    : null;

  return (
    <div className="dashboardRoot">
      <aside className="sideNav">
        <div className="brandRow">
          <img src="/brand-logo-mark.png" alt="Safety monitor logo" className="brandLogo" />
          <span className="brandTitle">Safety Monitor</span>
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
                  Session Report
                </h3>
                <span className="tinyPill">{selectedHasReport ? "Report ready" : "One report per session"}</span>
              </div>
              <div className="reportControls">
                <label className="logsField">
                  <span>Session</span>
                  <select
                    className="logsInput"
                    value={sessionId}
                    onChange={(event) => setSessionId(event.target.value)}
                    disabled={sessionsLoading || loading}
                  >
                    <option value="">Select a session</option>
                    {sessions.map((s) => (
                      <option key={s.session_id} value={s.session_id ?? ""}>
                        {formatSessionLabel(s)}{s.report ? " - Report ready" : ""}
                      </option>
                    ))}
                  </select>
                </label>
                {!selectedHasReport && (
                  <label className="logsField">
                    <span>Output Mode</span>
                    <select
                      className="logsInput"
                      value={llmProvider}
                      onChange={(event) => setLlmProvider(event.target.value as LlmProvider)}
                      disabled={loading || !sessionId}
                    >
                      <option value="local">Local</option>
                      <option value="api">API</option>
                    </select>
                  </label>
                )}
                {!selectedHasReport && (
                  <button type="button" className="reportPrimaryButton" disabled={loading || !sessionId} onClick={generateReport}>
                    <span className="material-symbols-outlined">{loading ? "hourglass_top" : "auto_awesome"}</span>
                    {loading ? "Generating..." : "Generate PDF"}
                  </button>
                )}
                {selectedHasReport && downloadUrl && (
                  <a className="reportDownloadButton" href={downloadUrl} target="_blank" rel="noreferrer">
                    <span className="material-symbols-outlined">download</span>
                    Download PDF
                  </a>
                )}
              </div>
              <p className="reportHint">
                Select a saved session. Sessions with an existing report show the saved summary and PDF download without regenerating.
              </p>
              {error && <div className="bannerError">{error}</div>}
            </article>

            <article className="reportPanel">
              <div className="cardHeader cardHeaderBorder">
                <h3>
                  <span className="material-symbols-outlined">task_alt</span>
                  Report Summary
                </h3>
                <span className="tinyPill">
                  {activeReport?.llm_provider ? activeReport.llm_provider.toUpperCase() : selectedSession ? "Not generated" : "No session"}
                </span>
              </div>
              <div className="reportResult">
                <div className="reportMetricRow">
                  <div>
                    <span>Total Events</span>
                    <strong>{activeReport?.total_events ?? "-"}</strong>
                  </div>
                  <div>
                    <span>High</span>
                    <strong className="reportDanger">{riskDistribution.high ?? "-"}</strong>
                  </div>
                  <div>
                    <span>Medium</span>
                    <strong className="reportWarning">{riskDistribution.medium ?? "-"}</strong>
                  </div>
                  <div>
                    <span>Low</span>
                    <strong>{riskDistribution.low ?? "-"}</strong>
                  </div>
                </div>
                <p className="reportSummary">
                  {buildDashboardSummary(activeReport, sessionId)}
                </p>
                {activeReport?.generated_at && (
                  <p className="reportHint">Generated at {formatDateTime(activeReport.generated_at)}</p>
                )}
                {dangerRatio !== null && <p className="reportHint">High risk ratio {dangerRatio}%</p>}
              </div>
            </article>
          </section>
        </div>
      </main>
    </div>
  );
}
