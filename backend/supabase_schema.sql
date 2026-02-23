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
