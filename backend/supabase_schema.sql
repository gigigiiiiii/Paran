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

create table if not exists public.session_reports (
  session_id text primary key,
  generated_at timestamptz not null default now(),
  report_date date not null,
  llm_provider text,
  summary text,
  total_events integer not null default 0,
  risk_distribution jsonb not null default '{}'::jsonb,
  risk_patterns jsonb not null default '[]'::jsonb,
  improvements jsonb not null default '[]'::jsonb,
  key_cases jsonb not null default '[]'::jsonb,
  top_obstacles jsonb not null default '[]'::jsonb,
  pdf_bucket text not null default 'collision-report-pdfs',
  pdf_path text not null,
  pdf_filename text not null
);

alter table public.session_reports add column if not exists key_cases jsonb not null default '[]'::jsonb;
alter table public.session_reports add column if not exists top_obstacles jsonb not null default '[]'::jsonb;

create index if not exists session_reports_generated_at_idx
  on public.session_reports (generated_at desc);

alter table public.session_reports enable row level security;

insert into storage.buckets (id, name, public)
values ('collision-report-pdfs', 'collision-report-pdfs', false)
on conflict (id) do nothing;
