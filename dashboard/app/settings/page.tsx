"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_MONITOR_API ?? "http://127.0.0.1:8000";

type MonitorState = {
  start_error?: string | null;
  stream?: {
    width?: number | null;
    height?: number | null;
    fps?: number | null;
    jpeg_quality?: number | null;
  };
  storage?: {
    backend?: string | null;
    table?: string | null;
    last_error?: string | null;
  };
};

export default function SettingsPage() {
  const [apiOnline, setApiOnline] = useState(true);
  const [state, setState] = useState<MonitorState | null>(null);

  useEffect(() => {
    let active = true;
    const fetchState = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/state`, { cache: "no-store" });
        if (!response.ok) throw new Error("state fetch failed");
        const data = (await response.json()) as MonitorState;
        if (!active) return;
        setState(data);
        setApiOnline(true);
      } catch {
        if (!active) return;
        setApiOnline(false);
      }
    };
    fetchState();
    const timer = window.setInterval(fetchState, 2000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const streamText =
    state?.stream?.width && state?.stream?.height && state?.stream?.fps
      ? `${state.stream.width}x${state.stream.height} @ ${state.stream.fps}fps`
      : "N/A";

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
          <Link href="/settings" className="navItem navItemActive">
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
            <h2>Settings</h2>
            <span className={`systemPill ${apiOnline ? "online" : "offline"}`}>
              <span className={`pulseDot ${apiOnline ? "online" : "offline"}`} /> SYSTEM {apiOnline ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        </header>

        <div className="contentScroll">
          <section className="logsSummaryGrid">
            <article className="logsSummaryCard">
              <span className="logsSummaryLabel">API Base</span>
              <strong className="logsSummaryValue" style={{ fontSize: 18 }}>{API_BASE}</strong>
            </article>
            <article className="logsSummaryCard">
              <span className="logsSummaryLabel">Stream</span>
              <strong className="logsSummaryValue" style={{ fontSize: 20 }}>{streamText}</strong>
            </article>
            <article className="logsSummaryCard">
              <span className="logsSummaryLabel">Storage</span>
              <strong className="logsSummaryValue" style={{ fontSize: 20 }}>{state?.storage?.backend ?? "N/A"}</strong>
            </article>
          </section>

          <article className="eventsTableCard">
            <div className="cardHeader cardHeaderBorder">
              <h3>Runtime Status</h3>
              <span className="tinyPill">Read-only</span>
            </div>
            <div className="analyticsList">
              <div className="analyticsListRow">
                <div className="analyticsListTop">
                  <span>Backend API</span>
                  <span>{apiOnline ? "Connected" : "Disconnected"}</span>
                </div>
              </div>
              <div className="analyticsListRow">
                <div className="analyticsListTop">
                  <span>Storage Table</span>
                  <span>{state?.storage?.table ?? "-"}</span>
                </div>
              </div>
              <div className="analyticsListRow">
                <div className="analyticsListTop">
                  <span>JPEG Quality</span>
                  <span>{state?.stream?.jpeg_quality ?? "-"}</span>
                </div>
              </div>
              <div className="analyticsListRow">
                <div className="analyticsListTop">
                  <span>Camera Start Error</span>
                  <span>{state?.start_error ?? "-"}</span>
                </div>
              </div>
              <div className="analyticsListRow">
                <div className="analyticsListTop">
                  <span>Storage Last Error</span>
                  <span>{state?.storage?.last_error ?? "-"}</span>
                </div>
              </div>
            </div>
          </article>
        </div>
      </main>
    </div>
  );
}
