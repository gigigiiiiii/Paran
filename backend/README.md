# Collision Monitor Backend

## Run

```powershell
cd c:\Users\Administrator\Desktop\Paran
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## API

- `GET /api/health`
- `GET /api/state`
- `GET /api/events`
- `POST /api/control/start`
- `POST /api/control/stop`
- `POST /api/control/reset`
- `GET /api/stream` (MJPEG)

`/api/state` includes `recording_elapsed_sec`, `stream.width`, `stream.height`, `stream.fps`, and `stream.jpeg_quality`.
`start -> stop -> start` resumes the same recording session; `reset` clears session/time/recent events.
Event persistence is written to Supabase (no local JSONL file write).
`/api/events` reads from Supabase only (no memory merge fallback).

## Environment Variables

- `MONITOR_MODEL` (default: `yolo26n.pt`)
- `MONITOR_WIDTH` (default: `640`)
- `MONITOR_HEIGHT` (default: `480`)
- `MONITOR_FPS` (default: `30`)
- `MONITOR_JPEG_QUALITY` (default: `92`, range: `60~100`)
- `MONITOR_NO_OVERLAY` (default: `0`; `1/true` = raw camera stream, no boxes/HUD)
- `MONITOR_HIDE_HUD_PANEL` (default: `1`; `1/true` = keep YOLO boxes, hide bottom risk panel)
- `MONITOR_EVENT_COOLDOWN_SEC` (default: `1.0`)
- `MONITOR_DEPTH_MODEL` (optional; enables monocular metric depth for test/video mode)
- `MONITOR_DEPTH_FOV` (default: `79.0`; horizontal FOV used for model-depth intrinsics)
- `MONITOR_MODEL_DEPTH_INTERVAL` (default: `0.2`; used only when async video depth is enabled)
- `MONITOR_VIDEO_DEPTH_SYNC` (default: `1`; test/video mode uses the same RGB frame for depth inference and distance calculation)
- `SUPABASE_URL` (required for remote persistence)
- `SUPABASE_SERVICE_ROLE_KEY` (recommended for backend insert/select)
- `SUPABASE_ANON_KEY` (optional fallback if service role key not set)
- `SUPABASE_EVENT_TABLE` (default: `collision_events`)
- `SUPABASE_SNAPSHOT_BUCKET` (default: `collision-event-snaps`)
- `SUPABASE_SNAPSHOT_PREFIX` (default: `events`)

### LLM Configuration (Reports)

- `REPORT_LLM_PROVIDER` (default: `api`; supported: `api`, `local`; legacy aliases: `gemini`, `qwen`)
- `REPORT_LLM_USE_OLLAMA` (legacy fallback; `1/true` selects `local` when `REPORT_LLM_PROVIDER` is not set)
- `REPORT_LLM_MODEL` (shared fallback model override)
- `REPORT_GEMINI_MODEL` (default: `gemini-2.5-flash`)
- `REPORT_OLLAMA_MODEL` (default: `qwen2.5:7b`; current local test model: `qwen2.5vl:3b`)
- `REPORT_LLM_TEMPERATURE` (default: `0.2`)
- `REPORT_OLLAMA_NUM_CTX` (default: `4096`)
- `REPORT_OLLAMA_NUM_GPU` (default: `-1`; ask Ollama to offload as many layers as possible to GPU)
- `REPORT_OLLAMA_TIMEOUT_SEC` (default: `120`)
- `REPORT_OLLAMA_KEEP_ALIVE` (default: `5m`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`; used when `llm_provider=local`)

**Local Setup (WiFi-independent):**
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull qwen2.5vl:3b` (or set `REPORT_OLLAMA_MODEL` to another installed Qwen model)
3. Start Ollama server: `ollama serve` (runs on `http://localhost:11434`)
4. Choose `Local` in the report UI, or set `REPORT_LLM_PROVIDER=local` in `.env`
5. Check readiness: `GET /api/reports/llm/status?llm_provider=local`

## Supabase Table

Create table `collision_events` (or set `SUPABASE_EVENT_TABLE`):

```sql
create table if not exists public.collision_events (
  event_id text primary key,
  session_id text,
  created_at timestamptz not null,
  ts_epoch double precision,
  risk text,
  risk_score double precision,
  risk_score_raw double precision,
  min_distance_m double precision,
  rep_distance_m double precision,
  ttc_s double precision,
  person_track_id bigint,
  obstacle_track_id bigint,
  obstacle_name text,
  angle_deg double precision,
  snapshot_path text
);

alter table public.collision_events add column if not exists snapshot_path text;
alter table public.collision_events drop column if exists snapshot_bucket;

create index if not exists collision_events_created_at_idx
  on public.collision_events (created_at desc);
```
