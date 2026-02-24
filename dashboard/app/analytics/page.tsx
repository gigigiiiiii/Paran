"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_MONITOR_API ?? "http://127.0.0.1:8000";
const SESSION_CARD_LIMIT = 40;
const SESSION_SCAN_LIMIT = 5000;
const ANALYTICS_EVENT_LIMIT = 2000;
const SESSION_TABLE_PAGE_SIZE = 30;

type RiskLevel = "SAFE" | "WARNING" | "DANGER";

type MonitorEvent = {
  event_id?: string;
  session_id?: string | null;
  created_at?: string;
  ts_epoch?: number;
  risk?: RiskLevel;
  obstacle_name?: string | null;
  rep_distance_m?: number | null;
  min_distance_m?: number | null;
  ttc_s?: number | null;
  risk_score?: number | null;
};

type MonitorState = {
  start_error?: string | null;
  session_id?: string | null;
};

type SessionSummary = {
  key: string;
  label: string;
  total: number;
  warning: number;
  danger: number;
  avgRisk: number;
  minDistance: number | null;
  startTs: number;
  endTs: number;
};

type SessionSummaryApi = {
  session_id?: string | null;
  total?: number;
  warning?: number;
  danger?: number;
  avg_risk_pct?: number;
  min_distance_m?: number | null;
  start_ts_ms?: number;
  end_ts_ms?: number;
};

type SessionsResponse = {
  sessions?: SessionSummaryApi[];
  truncated?: boolean;
};

type EventsResponse = {
  events?: MonitorEvent[];
  total?: number;
};

function clampPercent(value: number) {
  return Math.max(0, Math.min(100, Math.round(value)));
}

function formatNum(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}

function parseEventTs(row: MonitorEvent) {
  if (row.created_at) {
    const parsed = new Date(row.created_at).getTime();
    if (!Number.isNaN(parsed)) return parsed;
  }
  if (row.ts_epoch !== undefined && row.ts_epoch !== null) {
    return Number(row.ts_epoch) * 1000;
  }
  return 0;
}

function formatTime(value: number | null) {
  if (!value) return "-";
  return new Date(value).toLocaleTimeString("en-US", { hour12: false });
}

function formatDateTime(value: number | null) {
  if (!value) return "-";
  return new Date(value).toLocaleString("en-US", { hour12: false });
}

function formatDurationMs(startMs: number | null, endMs: number | null) {
  if (!startMs || !endMs || endMs < startMs) return "-";
  const totalSec = Math.floor((endMs - startMs) / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function riskPillClass(risk: RiskLevel | undefined) {
  if (risk === "DANGER") return "riskPillDanger";
  if (risk === "WARNING") return "riskPillWarning";
  return "riskPillSafe";
}

export default function AnalyticsPage() {
  const [sessionSummaries, setSessionSummaries] = useState<SessionSummary[]>([]);
  const [sessionEvents, setSessionEvents] = useState<MonitorEvent[]>([]);
  const [sessionEventsTotal, setSessionEventsTotal] = useState(0);
  const [sessionListTruncated, setSessionListTruncated] = useState(false);
  const [sessionPage, setSessionPage] = useState(0);
  const [sessionEventsLoading, setSessionEventsLoading] = useState(false);
  const [state, setState] = useState<MonitorState | null>(null);
  const [apiOnline, setApiOnline] = useState(true);
  const [loading, setLoading] = useState(true);
  const [lastSyncedAt, setLastSyncedAt] = useState<number | null>(null);
  const [activeSessionKey, setActiveSessionKey] = useState<string | null>(null);
  const [deletingSessionKey, setDeletingSessionKey] = useState<string | null>(null);
  const [deleteConfirmSessionKey, setDeleteConfirmSessionKey] = useState<string | null>(null);

  const fetchAnalytics = async () => {
    try {
      const stateResponse = await fetch(`${API_BASE}/api/state`, { cache: "no-store" });
      if (!stateResponse.ok) throw new Error(`state status ${stateResponse.status}`);
      const stateData = (await stateResponse.json()) as MonitorState;
      const runtimeSessionId = stateData.session_id?.trim() || null;

      const sessionQuery = new URLSearchParams({
        limit: String(SESSION_CARD_LIMIT),
        scan_limit: String(SESSION_SCAN_LIMIT),
      });
      if (runtimeSessionId) {
        sessionQuery.set("exclude_session_id", runtimeSessionId);
      }

      const sessionsResponse = await fetch(`${API_BASE}/api/sessions?${sessionQuery.toString()}`, { cache: "no-store" });
      if (!sessionsResponse.ok) throw new Error(`sessions status ${sessionsResponse.status}`);
      const sessionsData = (await sessionsResponse.json()) as SessionsResponse;

      const baseSummaries = (sessionsData.sessions ?? [])
        .map((row) => {
          const sid = row.session_id?.trim();
          if (!sid) return null;
          return {
            key: sid,
            label: sid,
            total: Math.max(0, Number(row.total ?? 0)),
            warning: Math.max(0, Number(row.warning ?? 0)),
            danger: Math.max(0, Number(row.danger ?? 0)),
            avgRisk: clampPercent(Number(row.avg_risk_pct ?? 0)),
            minDistance: row.min_distance_m ?? null,
            startTs: Math.max(0, Number(row.start_ts_ms ?? 0)),
            endTs: Math.max(0, Number(row.end_ts_ms ?? 0)),
          } satisfies SessionSummary;
        })
        .filter((row): row is SessionSummary => row !== null);

      setSessionSummaries(baseSummaries.sort((a, b) => b.endTs - a.endTs));
      setSessionListTruncated(Boolean(sessionsData.truncated));
      setState(stateData);
      setApiOnline(true);
      setLastSyncedAt(Date.now());
    } catch {
      setApiOnline(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
    const timer = window.setInterval(fetchAnalytics, 3000);
    return () => window.clearInterval(timer);
  }, []);

  const activeRuntimeSessionKey = useMemo(() => {
    const sid = state?.session_id?.trim();
    return sid ? sid : null;
  }, [state?.session_id]);

  useEffect(() => {
    if (sessionSummaries.length === 0) {
      if (activeSessionKey !== null) setActiveSessionKey(null);
      return;
    }
    if (activeSessionKey && sessionSummaries.some((row) => row.key === activeSessionKey)) return;
    setActiveSessionKey(sessionSummaries[0].key);
  }, [activeSessionKey, sessionSummaries]);

  const fetchSessionEvents = async (sessionKey: string | null) => {
    if (!sessionKey) {
      setSessionEvents([]);
      setSessionEventsTotal(0);
      return;
    }
    setSessionEventsLoading(true);
    try {
      const query = new URLSearchParams({
        session_id: sessionKey,
        limit: String(ANALYTICS_EVENT_LIMIT),
        offset: "0",
        include_total: "true",
      });
      const response = await fetch(`${API_BASE}/api/events?${query.toString()}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`events status ${response.status}`);
      const data = (await response.json()) as EventsResponse;
      setSessionEvents(data.events ?? []);
      setSessionEventsTotal(typeof data.total === "number" ? data.total : data.events?.length ?? 0);
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    } finally {
      setSessionEventsLoading(false);
    }
  };

  const deleteSession = async (sessionKey: string) => {
    if (deletingSessionKey) return;
    setDeleteConfirmSessionKey(null);
    setDeletingSessionKey(sessionKey);
    try {
      const response = await fetch(`${API_BASE}/api/sessions/${encodeURIComponent(sessionKey)}`, {
        method: "DELETE",
        cache: "no-store",
      });
      if (!response.ok) {
        let detail = `delete status ${response.status}`;
        try {
          const payload = (await response.json()) as { detail?: string };
          if (payload?.detail) detail = payload.detail;
        } catch {
          // ignore parse errors
        }
        throw new Error(detail);
      }
      if (activeSessionKey === sessionKey) {
        setActiveSessionKey(null);
        setSessionEvents([]);
        setSessionEventsTotal(0);
        setSessionPage(0);
      }
      await fetchAnalytics();
    } catch (error) {
      const message = error instanceof Error ? error.message : "세션 삭제에 실패했습니다.";
      window.alert(message);
    } finally {
      setDeletingSessionKey(null);
    }
  };

  useEffect(() => {
    setSessionPage(0);
    fetchSessionEvents(activeSessionKey);
  }, [activeSessionKey]);

  useEffect(() => {
    if (!activeSessionKey) return;
    const timer = window.setInterval(() => {
      fetchSessionEvents(activeSessionKey);
    }, 2500);
    return () => window.clearInterval(timer);
  }, [activeSessionKey]);

  const filteredEvents = useMemo(() => {
    if (!activeSessionKey) return [];
    return sessionEvents;
  }, [activeSessionKey, sessionEvents]);

  const selectedSession = useMemo(
    () => sessionSummaries.find((row) => row.key === activeSessionKey) ?? null,
    [activeSessionKey, sessionSummaries],
  );

  const sessionRowsSorted = useMemo(
    () => [...filteredEvents].sort((a, b) => parseEventTs(b) - parseEventTs(a)),
    [filteredEvents],
  );
  const sessionPageCount = Math.max(1, Math.ceil(sessionRowsSorted.length / SESSION_TABLE_PAGE_SIZE));
  useEffect(() => {
    if (sessionPage >= sessionPageCount) {
      setSessionPage(Math.max(0, sessionPageCount - 1));
    }
  }, [sessionPage, sessionPageCount]);

  const sessionRows = useMemo(() => {
    const start = sessionPage * SESSION_TABLE_PAGE_SIZE;
    return sessionRowsSorted.slice(start, start + SESSION_TABLE_PAGE_SIZE);
  }, [sessionPage, sessionRowsSorted]);

  const warningCountFallback = useMemo(
    () => filteredEvents.filter((row) => row.risk === "WARNING").length,
    [filteredEvents],
  );
  const dangerCountFallback = useMemo(
    () => filteredEvents.filter((row) => row.risk === "DANGER").length,
    [filteredEvents],
  );
  const totalEvents = selectedSession?.total ?? filteredEvents.length;
  const warningCount = selectedSession?.warning ?? warningCountFallback;
  const dangerCount = selectedSession?.danger ?? dangerCountFallback;

  const avgRiskFallback = useMemo(() => {
    if (filteredEvents.length === 0) return 0;
    const sum = filteredEvents.reduce((acc, row) => acc + (row.risk_score ?? 0), 0);
    return clampPercent((sum / filteredEvents.length) * 100);
  }, [filteredEvents]);
  const avgRiskScore = selectedSession?.avgRisk ?? avgRiskFallback;

  const peakRiskScore = useMemo(() => {
    if (filteredEvents.length === 0) return 0;
    return clampPercent(Math.max(...filteredEvents.map((row) => (row.risk_score ?? 0) * 100)));
  }, [filteredEvents]);

  const closestRepDistanceFallback = useMemo(() => {
    const values = filteredEvents
      .map((row) => row.rep_distance_m)
      .filter((value): value is number => value !== null && value !== undefined && Number.isFinite(value));
    if (values.length === 0) return null;
    return Math.min(...values);
  }, [filteredEvents]);
  const closestRepDistance = selectedSession?.minDistance ?? closestRepDistanceFallback;

  const shortestTtc = useMemo(() => {
    const values = filteredEvents
      .map((row) => row.ttc_s)
      .filter((value): value is number => value !== null && value !== undefined && value > 0 && Number.isFinite(value));
    if (values.length === 0) return null;
    return Math.min(...values);
  }, [filteredEvents]);

  const latestEventTs = useMemo(() => {
    if (filteredEvents.length === 0) return null;
    return filteredEvents.reduce((maxTs, row) => {
      const ts = parseEventTs(row);
      return ts > maxTs ? ts : maxTs;
    }, 0);
  }, [filteredEvents]);
  const latestEventDisplayTs = selectedSession?.endTs ?? latestEventTs;

  const riskMix = useMemo(
    () => [
      {
        label: "Danger",
        count: dangerCount,
        percent: totalEvents > 0 ? clampPercent((dangerCount / totalEvents) * 100) : 0,
        tone: "danger",
      },
      {
        label: "Warning",
        count: warningCount,
        percent: totalEvents > 0 ? clampPercent((warningCount / totalEvents) * 100) : 0,
        tone: "warning",
      },
    ],
    [dangerCount, totalEvents, warningCount],
  );

  const obstacleTop = useMemo(() => {
    const map = new Map<string, { count: number; riskSum: number }>();
    filteredEvents.forEach((row) => {
      const key = row.obstacle_name?.trim() || "unknown";
      const prev = map.get(key) ?? { count: 0, riskSum: 0 };
      prev.count += 1;
      prev.riskSum += row.risk_score ?? 0;
      map.set(key, prev);
    });
    return [...map.entries()]
      .map(([name, value]) => ({
        name,
        count: value.count,
        avgRisk: value.count > 0 ? clampPercent((value.riskSum / value.count) * 100) : 0,
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 6);
  }, [filteredEvents]);

  const maxObstacleCount = useMemo(() => Math.max(1, ...obstacleTop.map((row) => row.count)), [obstacleTop]);
  const sessionDataTruncated = sessionEventsTotal > filteredEvents.length;

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
          <Link href="/analytics" className="navItem navItemActive">
            <span className="material-symbols-outlined">analytics</span>
            <span>Analytics</span>
          </Link>
          <Link href="/settings" className="navItem">
            <span className="material-symbols-outlined">settings</span>
            <span>Settings</span>
          </Link>
          <button type="button" className="navItem">
            <span className="material-symbols-outlined">smart_toy</span>
            <span>AI Report</span>
          </button>
        </nav>
      </aside>

      <main className="mainArea">
        <header className="topBar">
          <div className="topLeft">
            <h2>Analytics</h2>
            <span className={`systemPill ${apiOnline ? "online" : "offline"}`}>
              <span className={`pulseDot ${apiOnline ? "online" : "offline"}`} /> SYSTEM {apiOnline ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        </header>

        <div className="contentScroll">
          <section className="analyticsSessionSection">
            <div className="cardHeader cardHeaderBorder">
              <h3>
                <span className="material-symbols-outlined">folder_managed</span>
                Session Filter
              </h3>
              <span className="tinyPill">{selectedSession?.label ?? "No session selected"}</span>
            </div>
            <div className="analyticsSessionGrid">
              {sessionSummaries.length === 0 && <p className="analyticsEmpty">No saved sessions.</p>}
              {activeRuntimeSessionKey && (
                <p className="analyticsEmpty">현재 진행 중인 세션은 RESET 이후 레코드 카드에 표시됩니다.</p>
              )}
              {sessionListTruncated && (
                <p className="analyticsEmpty">세션 목록은 최신 범위만 스캔했습니다. 오래된 세션은 제외될 수 있습니다.</p>
              )}
              {sessionSummaries.map((session) => (
                <div key={session.key} className="analyticsSessionCardWrap">
                  <button
                    type="button"
                    className={`analyticsSessionCard ${activeSessionKey === session.key ? "active" : ""}`}
                    onClick={() => setActiveSessionKey(session.key)}
                    disabled={deletingSessionKey === session.key}
                  >
                    <p className="analyticsSessionTitle">{session.label}</p>
                    <p className="analyticsSessionStat">{session.total} events</p>
                    <p className="analyticsSessionSub">Danger {session.danger} | Warning {session.warning}</p>
                    <p className="analyticsSessionSub">Avg risk {session.avgRisk}/100</p>
                    <p className="analyticsSessionRange">{formatDateTime(session.startTs)} - {formatTime(session.endTs)}</p>
                    <p className="analyticsSessionSub">Duration {formatDurationMs(session.startTs, session.endTs)}</p>
                  </button>
                  <button
                    type="button"
                    className="analyticsSessionDeleteButton"
                    disabled={deletingSessionKey !== null}
                    onClick={() => {
                      setDeleteConfirmSessionKey(session.key);
                    }}
                  >
                    {deletingSessionKey === session.key ? "Deleting..." : "Delete Session"}
                  </button>
                </div>
              ))}
            </div>
          </section>

          <section className="analyticsKpiGrid">
            <article className="analyticsKpiCard">
              <div className="analyticsKpiHead">
                <span className="analyticsKpiLabel">Total Events</span>
                <span className="material-symbols-outlined">dataset</span>
              </div>
              <strong className="analyticsKpiValue">{totalEvents}</strong>
              <p className="analyticsKpiHint">Latest at {formatTime(latestEventDisplayTs)}</p>
            </article>

            <article className="analyticsKpiCard analyticsKpiCardDanger">
              <div className="analyticsKpiHead">
                <span className="analyticsKpiLabel">Danger Ratio</span>
                <span className="material-symbols-outlined">crisis_alert</span>
              </div>
              <strong className="analyticsKpiValue">{totalEvents > 0 ? clampPercent((dangerCount / totalEvents) * 100) : 0}%</strong>
              <p className="analyticsKpiHint">{dangerCount} of {totalEvents} events</p>
            </article>

            <article className="analyticsKpiCard">
              <div className="analyticsKpiHead">
                <span className="analyticsKpiLabel">Avg / Peak Risk</span>
                <span className="material-symbols-outlined">monitoring</span>
              </div>
              <strong className="analyticsKpiValue">{avgRiskScore} / {peakRiskScore}</strong>
              <p className="analyticsKpiHint">risk score out of 100</p>
            </article>

            <article className="analyticsKpiCard">
              <div className="analyticsKpiHead">
                <span className="analyticsKpiLabel">Closest Dist / TTC</span>
                <span className="material-symbols-outlined">straighten</span>
              </div>
              <strong className="analyticsKpiValue">{formatNum(closestRepDistance)}m</strong>
              <p className="analyticsKpiHint">shortest TTC {formatNum(shortestTtc)}s</p>
            </article>
          </section>

          <section className="analyticsGrid">
            <article className="analyticsCard">
              <div className="cardHeader cardHeaderBorder">
                <h3>
                  <span className="material-symbols-outlined">pie_chart</span>
                  Risk Mix
                </h3>
                <span className="tinyPill">Filtered</span>
              </div>
              <div className="analyticsList">
                {riskMix.map((row) => (
                  <div key={row.label} className="analyticsListRow">
                    <div className="analyticsListTop">
                      <span>{row.label}</span>
                      <span>{row.count} ({row.percent}%)</span>
                    </div>
                    <div className="analyticsProgress">
                      <div
                        className={`analyticsProgressFill ${row.tone}`}
                        style={{ width: `${Math.max(row.percent, row.count > 0 ? 4 : 0)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </article>

            <article className="analyticsCard">
              <div className="cardHeader cardHeaderBorder">
                <h3>
                  <span className="material-symbols-outlined">target</span>
                  Top Obstacles
                </h3>
                <span className="tinyPill">By frequency</span>
              </div>
              <div className="analyticsList">
                {obstacleTop.length === 0 && <p className="analyticsEmpty">No obstacle data yet.</p>}
                {obstacleTop.map((row) => {
                  const width = clampPercent((row.count / maxObstacleCount) * 100);
                  return (
                    <div key={row.name} className="analyticsListRow">
                      <div className="analyticsListTop">
                        <span>{row.name}</span>
                        <span>{row.count} events</span>
                      </div>
                      <div className="analyticsProgress">
                        <div className="analyticsProgressFill info" style={{ width: `${Math.max(width, 6)}%` }} />
                      </div>
                      <p className="analyticsRowMeta">avg risk {row.avgRisk}/100</p>
                    </div>
                  );
                })}
              </div>
            </article>

            <article className="analyticsCard analyticsCardWide analyticsCardSpanFull">
              <div className="cardHeader cardHeaderBorder">
                <h3>
                  <span className="material-symbols-outlined">table_rows</span>
                  Session Events
                </h3>
                <span className="tinyPill">Page {sessionPage + 1} / {sessionPageCount}</span>
              </div>
              <div className="eventsTableWrap">
                <table className="eventsTable">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Risk</th>
                      <th>Object</th>
                      <th>Rep Dist (m)</th>
                      <th>TTC (s)</th>
                      <th>Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sessionEventsLoading && sessionRows.length === 0 && (
                      <tr>
                        <td colSpan={6} className="eventsTableEmpty">Loading session events...</td>
                      </tr>
                    )}
                    {!sessionEventsLoading && sessionRows.length === 0 && (
                      <tr>
                        <td colSpan={6} className="eventsTableEmpty">No events for selected session.</td>
                      </tr>
                    )}
                    {sessionRows.map((row, index) => {
                      const score = clampPercent((row.risk_score ?? 0) * 100);
                      const rowKey = row.event_id ?? `${row.created_at ?? "event"}-${index}`;
                      return (
                        <tr key={rowKey}>
                          <td>{formatDateTime(parseEventTs(row))}</td>
                          <td>
                            <span className={`riskPill ${riskPillClass(row.risk)}`}>{row.risk ?? "SAFE"}</span>
                          </td>
                          <td>{row.obstacle_name ?? "-"}</td>
                          <td>{formatNum(row.rep_distance_m)}</td>
                          <td>{formatNum(row.ttc_s)}</td>
                          <td>{score}/100</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="tablePager">
                <button
                  type="button"
                  className="pagerButton"
                  disabled={sessionPage <= 0}
                  onClick={() => setSessionPage((value) => Math.max(0, value - 1))}
                >
                  Prev
                </button>
                <p className="pagerStatus">
                  Showing {sessionRows.length} of {sessionRowsSorted.length} loaded rows
                </p>
                <button
                  type="button"
                  className="pagerButton"
                  disabled={sessionPage + 1 >= sessionPageCount}
                  onClick={() => setSessionPage((value) => value + 1)}
                >
                  Next
                </button>
              </div>
              {sessionDataTruncated && (
                <p className="analyticsEmpty">
                  This session has {sessionEventsTotal} events. Showing latest {sessionRowsSorted.length} rows.
                </p>
              )}
            </article>
          </section>

          {loading && <p className="analyticsEmpty">Loading analytics...</p>}
          {state?.start_error && <p className="bannerError">Camera start error: {state.start_error}</p>}
        </div>
      </main>

      {deleteConfirmSessionKey && (
        <div className="analyticsConfirmOverlay" role="presentation">
          <div className="analyticsConfirmDialog" role="dialog" aria-modal="true" aria-labelledby="delete-session-title">
            <h3 id="delete-session-title" className="analyticsConfirmTitle">
              Delete Session?
            </h3>
            <p className="analyticsConfirmMessage">
              This will permanently remove all event rows and snapshot images for the selected session.
            </p>
            <p className="analyticsConfirmSession">{deleteConfirmSessionKey}</p>
            <div className="analyticsConfirmActions">
              <button
                type="button"
                className="analyticsConfirmButton analyticsConfirmCancel"
                disabled={deletingSessionKey !== null}
                onClick={() => setDeleteConfirmSessionKey(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="analyticsConfirmButton analyticsConfirmDelete"
                disabled={deletingSessionKey !== null}
                onClick={() => {
                  void deleteSession(deleteConfirmSessionKey);
                }}
              >
                {deletingSessionKey === deleteConfirmSessionKey ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
