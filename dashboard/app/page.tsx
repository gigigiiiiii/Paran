"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_MONITOR_API ?? "http://127.0.0.1:8000";
const DASHBOARD_EVENT_LIMIT = 200;

type RiskLevel = "SAFE" | "WARNING" | "DANGER";

type MonitorEvent = {
  event_id?: string;
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
  recording_enabled: boolean;
  session_id?: string | null;
  recording_elapsed_sec?: number;
  event_count?: number;
  start_error?: string | null;
  latest?: {
    risk?: RiskLevel;
    risk_score?: number;
    risk_score_raw?: number;
    rep_distance_m?: number | null;
    min_distance_m?: number | null;
    ttc_s?: number | null;
    obstacle_name?: string | null;
    ts_epoch?: number;
  };
  today_summary?: {
    date?: string;
    warning_count?: number;
    danger_count?: number;
    total?: number;
  };
  stream?: {
    width?: number | null;
    height?: number | null;
    fps?: number | null;
    jpeg_quality?: number | null;
  };
};

function formatNum(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}

function clampPercent(value: number) {
  return Math.max(0, Math.min(100, Math.round(value)));
}

function riskTone(risk: RiskLevel | undefined) {
  if (risk === "DANGER") return "tone-danger";
  if (risk === "WARNING") return "tone-warning";
  return "tone-safe";
}

function formatEventTime(iso: string | undefined) {
  if (!iso) return "--:--:--";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "--:--:--";
  return date.toLocaleTimeString("ko-KR", { hour12: false });
}

function formatElapsed(totalSeconds: number) {
  const safe = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(safe / 3600);
  const m = Math.floor((safe % 3600) / 60);
  const s = safe % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export default function DashboardPage() {
  const [state, setState] = useState<MonitorState | null>(null);
  const [events, setEvents] = useState<MonitorEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [apiOnline, setApiOnline] = useState(true);
  const [streamError, setStreamError] = useState(false);
  const [saveNoticeVisible, setSaveNoticeVisible] = useState(false);
  const saveNoticeTimerRef = useRef<number | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  const latest = state?.latest;
  const streamMeta = state?.stream;
  const recordingEnabled = Boolean(state?.recording_enabled);

  const streamUrl = useMemo(() => `${API_BASE}/api/stream`, []);

  const risk = latest?.risk ?? "SAFE";
  const riskScorePct = clampPercent((latest?.risk_score ?? 0) * 100);
  const distanceRiskPct = clampPercent(
    latest?.rep_distance_m !== null && latest?.rep_distance_m !== undefined
      ? (1 - latest.rep_distance_m / 3.0) * 100
      : 0,
  );
  const ttcRiskPct = clampPercent(
    latest?.ttc_s !== null && latest?.ttc_s !== undefined ? (1 - latest.ttc_s / 3.0) * 100 : 0,
  );
  const avgRiskPct = clampPercent(
    events.length > 0
      ? (events.reduce((sum, row) => sum + (row.risk_score ?? 0), 0) / events.length) * 100
      : 0,
  );
  const totalEvents = events.length;
  const criticalEvents = useMemo(
    () => events.filter((row) => row.risk === "DANGER").length,
    [events],
  );
  const recentEvents = useMemo(() => events.slice(0, 5), [events]);

  const fetchState = async (): Promise<MonitorState | null> => {
    try {
      const response = await fetch(`${API_BASE}/api/state`, { cache: "no-store" });
      if (!response.ok) throw new Error(`state status ${response.status}`);
      const data = (await response.json()) as MonitorState;
      setState(data);
      sessionIdRef.current = data.session_id ?? null;
      setApiOnline(true);
      setStreamError(false);
      return data;
    } catch {
      setApiOnline(false);
      return null;
    }
  };

  const fetchEvents = async (sessionIdOverride?: string | null) => {
    try {
      const sessionId = sessionIdOverride !== undefined ? sessionIdOverride : sessionIdRef.current;
      if (!sessionId) {
        setEvents([]);
        return;
      }
      const query = new URLSearchParams({ limit: String(DASHBOARD_EVENT_LIMIT), session_id: sessionId });
      const response = await fetch(`${API_BASE}/api/events?${query.toString()}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`events status ${response.status}`);
      const data = (await response.json()) as { events: MonitorEvent[] };
      setEvents(data.events ?? []);
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    }
  };

  const callControl = async (mode: "start" | "stop" | "reset") => {
    setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/api/control/${mode}`, { method: "POST" });
      if (!response.ok) throw new Error(`control status ${response.status}`);
      const latestState = await fetchState();
      await fetchEvents(latestState?.session_id ?? null);
      if (mode === "reset") {
        if (saveNoticeTimerRef.current !== null) {
          window.clearTimeout(saveNoticeTimerRef.current);
        }
        setSaveNoticeVisible(true);
        saveNoticeTimerRef.current = window.setTimeout(() => {
          setSaveNoticeVisible(false);
          saveNoticeTimerRef.current = null;
        }, 2600);
      }
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    const loadInitial = async () => {
      const latestState = await fetchState();
      await fetchEvents(latestState?.session_id ?? null);
    };
    loadInitial();
    const stateTimer = window.setInterval(fetchState, 1000);
    const eventTimer = window.setInterval(fetchEvents, 1500);
    return () => {
      window.clearInterval(stateTimer);
      window.clearInterval(eventTimer);
      if (saveNoticeTimerRef.current !== null) {
        window.clearTimeout(saveNoticeTimerRef.current);
      }
    };
  }, []);

  const streamAvailable = apiOnline && !streamError && !state?.start_error && state !== null;
  const elapsedSec = Math.max(0, Math.floor(state?.recording_elapsed_sec ?? 0));
  const streamInfoText =
    streamMeta?.width && streamMeta?.height && streamMeta?.fps
      ? `${streamMeta.width}x${streamMeta.height} @ ${streamMeta.fps}fps`
      : "N/A";

  return (
    <div className="dashboardRoot">
      <aside className="sideNav">
        <div className="brandRow">
          <img src="/brand-logo-mark.png" alt="안전모탐정단 로고" className="brandLogo" />
          <span className="brandTitle">안전모탐정단</span>
        </div>

        <nav className="navList">
          <Link href="/" className="navItem navItemActive">
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
          <button type="button" className="navItem">
            <span className="material-symbols-outlined">smart_toy</span>
            <span>AI Report</span>
          </button>
        </nav>
      </aside>

      <main className="mainArea">
        <header className="topBar">
          <div className="topLeft">
            <h2>Live Operations Center</h2>
            <span className={`systemPill ${apiOnline ? "online" : "offline"}`}>
              <span className={`pulseDot ${apiOnline ? "online" : "offline"}`} /> SYSTEM {apiOnline ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        </header>

        <div className="contentScroll">
          <div className="dashboardGrid">
            <section className="leftColumn">
              <article className="feedCard">
                <div className="feedViewport">
                  {streamAvailable ? (
                    <img
                      src={streamUrl}
                      alt="Live monitor"
                      className="feedImage"
                      onError={() => setStreamError(true)}
                    />
                  ) : (
                    <div className="offlinePanel" role="status" aria-live="polite">
                      <span className="material-symbols-outlined offlinePanelIcon">videocam_off</span>
                    </div>
                  )}

                  {streamAvailable && (
                    <div className="feedTopOverlay">
                      <div className={`riskBadge ${riskTone(risk)}`}>
                        <span className="material-symbols-outlined">warning</span>
                        <div>
                          <p>Risk Level</p>
                          <strong>{risk}</strong>
                        </div>
                      </div>
                    </div>
                  )}

                  {streamAvailable && (
                    <div className="feedBottomOverlay">
                      <div className="cameraInfo">{streamInfoText}</div>
                    </div>
                  )}
                </div>
              </article>

              <section className="metricGrid">
                <article className="metricCard">
                  <div className="metricHead">
                    <span className="metricLabel">Rep Distance</span>
                    <span className="material-symbols-outlined">straighten</span>
                  </div>
                  <div className="metricValue">
                    <strong>{formatNum(latest?.rep_distance_m)}</strong>
                    <span>m</span>
                  </div>
                  <div className="meter">
                    <div className="meterBar meterDanger" style={{ width: `${distanceRiskPct}%` }} />
                  </div>
                </article>

                <article className="metricCard">
                  <div className="metricHead">
                    <span className="metricLabel">Time to Collision</span>
                    <span className="material-symbols-outlined">timer</span>
                  </div>
                  <div className="metricValue metricDangerText">
                    <strong>{formatNum(latest?.ttc_s)}</strong>
                    <span>s</span>
                  </div>
                  <div className="meter">
                    <div className="meterBar meterDanger" style={{ width: `${ttcRiskPct}%` }} />
                  </div>
                </article>

                <article className="metricCard">
                  <div className="metricHead">
                    <span className="metricLabel">Min Distance</span>
                    <span className="material-symbols-outlined">speed</span>
                  </div>
                  <div className="metricValue">
                    <strong>{formatNum(latest?.min_distance_m)}</strong>
                    <span>m</span>
                  </div>
                  <div className="meter">
                    <div className="meterBar meterWarn" style={{ width: `${distanceRiskPct}%` }} />
                  </div>
                </article>

                <article className="metricCard metricCardDanger">
                  <div className="metricHead">
                    <span className="metricLabel">Risk Score</span>
                    <span className="material-symbols-outlined">crisis_alert</span>
                  </div>
                  <div className="metricValue metricDangerText">
                    <strong>{riskScorePct}</strong>
                    <span>/100</span>
                  </div>
                  <div className="meter">
                    <div className="meterBar meterGradient" style={{ width: `${riskScorePct}%` }} />
                  </div>
                </article>
              </section>
            </section>

            <section className="rightColumn">
              <article className="overviewCard">
                <div className="cardHeader">
                  <h3>
                    <span className="material-symbols-outlined">calendar_today</span>
                    Current Session Overview
                  </h3>
                </div>
                <div className="overviewGrid">
                  <div className="overviewStat">
                    <strong>{totalEvents}</strong>
                    <span>Total Events</span>
                  </div>
                  <div className="overviewStat overviewDanger">
                    <strong>{criticalEvents}</strong>
                    <span>Critical</span>
                  </div>
                  <div className="overviewStat">
                    <strong>{avgRiskPct}</strong>
                    <span>Avg Risk</span>
                  </div>
                </div>
              </article>

              <article className="eventsCard">
                <div className="cardHeader cardHeaderBorder">
                  <h3>Recent Events</h3>
                  <span className="tinyPill">Latest 5</span>
                </div>
                <div className="eventsList">
                  {recentEvents.length === 0 && <p className="emptyState">No recorded events in this session.</p>}
                  {recentEvents.map((event, index) => {
                    const tone = riskTone(event.risk);
                    const icon = event.risk === "DANGER" ? "priority_high" : event.risk === "WARNING" ? "warning" : "check_circle";
                    const title = event.obstacle_name ? `${event.obstacle_name} proximity` : "Proximity Alert";
                    const rowKey = event.event_id ?? `${event.created_at ?? "event"}-${index}`;
                    return (
                      <div key={rowKey} className={`eventRow ${tone}`}>
                        <div className="eventIcon">
                          <span className="material-symbols-outlined">{icon}</span>
                        </div>
                        <div className="eventMeta">
                          <div className="eventMetaTop">
                            <p>{title}</p>
                            <span>{formatEventTime(event.created_at)}</span>
                          </div>
                          <div className="eventMetaBottom">
                            <span>{formatNum(event.rep_distance_m)}m</span>
                            <span>{formatNum(event.ttc_s)}s</span>
                            <span>{clampPercent((event.risk_score ?? 0) * 100)}/100</span>
                          </div>
                        </div>
                        <span className="eventDot" />
                      </div>
                    );
                  })}
                </div>
              </article>

              <section className="actionSection">
                <div className="recordControls">
                  <button
                    type="button"
                    disabled={busy}
                    className={recordingEnabled ? "recordButton stop" : "recordButton start"}
                    onClick={() => callControl(recordingEnabled ? "stop" : "start")}
                  >
                    {recordingEnabled ? "STOP" : "START"}
                  </button>
                  <button
                    type="button"
                    disabled={busy || recordingEnabled}
                    className="recordButton reset"
                    onClick={() => callControl("reset")}
                  >
                    SAVE
                  </button>
                </div>
                <p className="recordHint">START creates a session, STOP pauses it, SAVE closes it and clears the view.</p>

                <article className="recordTimeCard">
                  <div className="recordTimeHeader">
                    <h4>
                      <span className="material-symbols-outlined">schedule</span>
                      Recording Time
                    </h4>
                    <span className={recordingEnabled ? "timePill live" : "timePill paused"}>
                      {recordingEnabled ? "RUNNING" : "PAUSED"}
                    </span>
                  </div>
                  <p className="recordTimeValue">{formatElapsed(elapsedSec)}</p>
                </article>
              </section>
            </section>
          </div>

          {state?.start_error && <p className="bannerError">Camera start error: {state.start_error}</p>}
        </div>
      </main>

      {saveNoticeVisible && (
        <div className="saveToast" role="status" aria-live="polite">
          <span className="material-symbols-outlined">check_circle</span>
          <div className="saveToastText">
            <strong>Saved</strong>
            <span>Recording has been saved.</span>
          </div>
        </div>
      )}
    </div>
  );
}

