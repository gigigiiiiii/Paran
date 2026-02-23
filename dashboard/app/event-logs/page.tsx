"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_MONITOR_API ?? "http://127.0.0.1:8000";

type RiskLevel = "SAFE" | "WARNING" | "DANGER";
type RiskFilter = "ALL" | RiskLevel;
type SortKey = "latest" | "oldest" | "risk_high" | "distance_low";

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

function formatNum(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}

function clampPercent(value: number) {
  return Math.max(0, Math.min(100, Math.round(value)));
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

function formatDateTime(row: MonitorEvent) {
  const ts = parseEventTs(row);
  if (!ts) return "-";
  return new Date(ts).toLocaleString("ko-KR", { hour12: false });
}

function riskPillClass(risk: RiskLevel | undefined) {
  if (risk === "DANGER") return "riskPillDanger";
  if (risk === "WARNING") return "riskPillWarning";
  return "riskPillSafe";
}

export default function EventLogsPage() {
  const [events, setEvents] = useState<MonitorEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [apiOnline, setApiOnline] = useState(true);
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("ALL");
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("latest");

  const fetchEvents = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/events?limit=1000`, { cache: "no-store" });
      if (!response.ok) throw new Error(`events status ${response.status}`);
      const data = (await response.json()) as { events?: MonitorEvent[] };
      setEvents(data.events ?? []);
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();
    const timer = window.setInterval(fetchEvents, 2000);
    return () => window.clearInterval(timer);
  }, []);

  const warningCount = useMemo(
    () => events.filter((row) => row.risk === "WARNING").length,
    [events],
  );
  const dangerCount = useMemo(
    () => events.filter((row) => row.risk === "DANGER").length,
    [events],
  );

  const filteredRows = useMemo(() => {
    const search = query.trim().toLowerCase();
    let rows = events.filter((row) => {
      if (riskFilter !== "ALL" && (row.risk ?? "SAFE") !== riskFilter) return false;
      if (!search) return true;

      const obstacle = (row.obstacle_name ?? "").toLowerCase();
      const eventId = (row.event_id ?? "").toLowerCase();
      const sessionId = (row.session_id ?? "").toLowerCase();
      return obstacle.includes(search) || eventId.includes(search) || sessionId.includes(search);
    });

    rows = [...rows].sort((a, b) => {
      if (sortKey === "oldest") return parseEventTs(a) - parseEventTs(b);
      if (sortKey === "risk_high") {
        return (b.risk_score ?? 0) - (a.risk_score ?? 0);
      }
      if (sortKey === "distance_low") {
        return (a.rep_distance_m ?? Number.POSITIVE_INFINITY) - (b.rep_distance_m ?? Number.POSITIVE_INFINITY);
      }
      return parseEventTs(b) - parseEventTs(a);
    });

    return rows;
  }, [events, query, riskFilter, sortKey]);

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
          <button type="button" className="navItem">
            <span className="material-symbols-outlined">summarize</span>
            <span>Today&apos;s Summary</span>
          </button>
          <Link href="/event-logs" className="navItem navItemActive">
            <span className="material-symbols-outlined">list_alt</span>
            <span>Event Logs</span>
          </Link>
          <button type="button" className="navItem">
            <span className="material-symbols-outlined">analytics</span>
            <span>Analytics</span>
          </button>
          <button type="button" className="navItem">
            <span className="material-symbols-outlined">smart_toy</span>
            <span>AI Report</span>
          </button>
        </nav>
      </aside>

      <main className="mainArea">
        <header className="topBar">
          <div className="topLeft">
            <h2>Event Logs</h2>
            <span className={`systemPill ${apiOnline ? "online" : "offline"}`}>
              <span className={`pulseDot ${apiOnline ? "online" : "offline"}`} /> SYSTEM {apiOnline ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        </header>

        <div className="contentScroll">
          <section className="logsSummaryGrid">
            <article className="logsSummaryCard">
              <span className="logsSummaryLabel">Total Events</span>
              <strong className="logsSummaryValue">{events.length}</strong>
            </article>
            <article className="logsSummaryCard">
              <span className="logsSummaryLabel">Warning</span>
              <strong className="logsSummaryValue logsSummaryValueWarn">{warningCount}</strong>
            </article>
            <article className="logsSummaryCard">
              <span className="logsSummaryLabel">Danger</span>
              <strong className="logsSummaryValue logsSummaryValueDanger">{dangerCount}</strong>
            </article>
          </section>

          <section className="logsFilterBar">
            <label className="logsField">
              <span>Risk</span>
              <select className="logsInput" value={riskFilter} onChange={(event) => setRiskFilter(event.target.value as RiskFilter)}>
                <option value="ALL">All</option>
                <option value="WARNING">Warning</option>
                <option value="DANGER">Danger</option>
                <option value="SAFE">Safe</option>
              </select>
            </label>

            <label className="logsField logsFieldGrow">
              <span>Search</span>
              <input
                className="logsInput"
                type="text"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="object, event id, session id"
              />
            </label>

            <label className="logsField">
              <span>Sort</span>
              <select className="logsInput" value={sortKey} onChange={(event) => setSortKey(event.target.value as SortKey)}>
                <option value="latest">Latest</option>
                <option value="oldest">Oldest</option>
                <option value="risk_high">Risk Score (High)</option>
                <option value="distance_low">Distance (Near)</option>
              </select>
            </label>
          </section>

          <article className="eventsTableCard">
            <div className="cardHeader cardHeaderBorder">
              <h3>Event Records</h3>
              <span className="tinyPill">Showing {filteredRows.length} / {events.length}</span>
            </div>

            <div className="eventsTableWrap">
              <table className="eventsTable">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Risk</th>
                    <th>Object</th>
                    <th>Rep Dist (m)</th>
                    <th>Min Dist (m)</th>
                    <th>TTC (s)</th>
                    <th>Score</th>
                    <th>Session</th>
                  </tr>
                </thead>
                <tbody>
                  {loading && events.length === 0 && (
                    <tr>
                      <td colSpan={8} className="eventsTableEmpty">Loading events...</td>
                    </tr>
                  )}

                  {!loading && filteredRows.length === 0 && (
                    <tr>
                      <td colSpan={8} className="eventsTableEmpty">No events found for current filters.</td>
                    </tr>
                  )}

                  {filteredRows.map((row, index) => {
                    const score = clampPercent((row.risk_score ?? 0) * 100);
                    const rowKey = row.event_id ?? `${row.created_at ?? "event"}-${index}`;
                    return (
                      <tr key={rowKey}>
                        <td>{formatDateTime(row)}</td>
                        <td>
                          <span className={`riskPill ${riskPillClass(row.risk)}`}>{row.risk ?? "SAFE"}</span>
                        </td>
                        <td>{row.obstacle_name ?? "-"}</td>
                        <td>{formatNum(row.rep_distance_m)}</td>
                        <td>{formatNum(row.min_distance_m)}</td>
                        <td>{formatNum(row.ttc_s)}</td>
                        <td>{score}/100</td>
                        <td className="sessionMono">{row.session_id ?? "-"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </article>
        </div>
      </main>
    </div>
  );
}
