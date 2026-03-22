# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

공장 고정형 CCTV 영상에서 사람·이동장비를 실시간 탐지하고, 충돌 위험도를 계산해 대시보드로 보여주는 시스템.

- **카메라**: Intel RealSense (RGB + Depth) / 또는 테스트 영상(MP4 등)
- **탐지**: YOLO (`yolo26m.pt`, COCO 학습)
- **트래킹**: ByteTrack (Ultralytics 내장, `bytetrack_config.yaml`)
- **위험도**: 거리(m) + TTC + 접근속도 → 가중합 스코어 → SAFE / WARNING / DANGER
- **백엔드**: FastAPI + MJPEG 스트리밍 (`backend/app.py`)
- **프론트엔드**: Next.js 대시보드 (`dashboard/`)
- **이벤트 저장**: Supabase (PostgreSQL + Storage)

---

## 실행 방법

### 백엔드 (FastAPI)
```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 대시보드 (Next.js)
```bash
cd dashboard
npm run dev
```

### 직접 실행 (RealSense)
```bash
python -m collision_monitor.runner --model yolo26m.pt --conf 0.35
```

### 직접 실행 (영상 파일)
```bash
python -m collision_monitor.video_runner --video test_videos/xxx.mp4 --model yolo26m.pt
```

---

## 아키텍처 — 데이터 흐름

```
[카메라/영상]
    ↓
PipelineRunner (runner.py) 또는 VideoRunner (video_runner.py)
    ↓  color_image, depth_image, intrinsics, depth_scale
FrameProcessor.process() (frame_processor.py)  ← 핵심 처리
    ↓  YOLO.track() → ByteTrack 트래킹 ID
    ↓  depth 기반 3D 좌표 + 속도 계산
    ↓  risk.py 위험도 스코어 계산
    ↓  (canvas: np.ndarray, result: dict)
MonitorService (backend/app.py)
    ├── JPEG 인코딩 → _latest_frame_jpeg
    ├── result dict → _latest_result
    └── 이벤트 조건 충족 시 → _persist_queue → Supabase
```

`MonitorService`는 두 개의 백그라운드 스레드를 관리한다:
- **runner thread**: `PipelineRunner` 또는 `VideoRunner` 루프
- **persist thread**: Supabase DB/Storage에 비동기 이벤트 저장

프론트엔드는 1초마다 `/api/state`, 1.5초마다 `/api/events`를 폴링한다.

---

## 핵심 파일 역할

### `collision_monitor/frame_processor.py`
- `FrameProcessor.process(color_image, depth_image, intrinsics, depth_scale)` 가 메인 진입점
- `depth_image=None` 으로 호출하면 depth 의존 계산(거리·TTC·risk)을 건너뜀 → 테스트 영상 모드
- 반환값: `(canvas: np.ndarray, result: dict)` — result dict에 위험도, 거리, TTC 포함
- 시각화: bbox, 트래킹 ID, 위험도 텍스트만 표시 (trail 선 없음)

### `collision_monitor/byte_tracker.py`
- `TrackHistory.update(track_id, class_name, bbox, conf) → TrackData`: bbox EMA 스무딩(`bbox_alpha=0.5`) + trail 누적
- `TrackHistory.step(active_ids)`: 프레임 종료 시 반드시 호출, TTL 감소 및 dead track 정리
- `TrackData.center`: bbox 중앙점 `((sx1+sx2)/2, (sy1+sy2)/2)` — bbox EMA 적용 후 계산

### `collision_monitor/config.py`
- 모든 CLI argparse 기본값이 여기에 정의됨
- 백엔드(`app.py`)는 `parse_args([])`로 기본값을 로드한 후 env 변수로 덮어씀
- 주요 기본값: `--conf 0.35`, `--imgsz 1280`, `--warn-dist 1.5`, `--danger-dist 0.9`, `--warn-ttc 2.0`, `--danger-ttc 1.0`

### `collision_monitor/risk.py`
- `compute_risk_score()`: dist·ttc·closing_speed 각각 0~1 점수화 후 가중합
- `score_to_level()`: 히스테리시스 적용 (warn_on/off, danger_on/off 별도)
- `FrameProcessor._stabilize_level()`: N프레임 연속 조건으로 레벨 전환 안정화

### `backend/app.py`
- `MonitorService`: 런너 스레드 관리, JPEG 프레임 버퍼, Supabase 이벤트 저장
- `MONITOR_SOURCE` 환경변수가 있으면 `VideoRunner`, 없으면 `PipelineRunner(RealSense)`
- Supabase HTTP 호출은 `ProxyHandler({})` opener 사용 (로컬 프록시 우회)
- API 엔드포인트:
  - `GET /api/stream` — MJPEG 스트림
  - `GET /api/state` — 현재 상태 (위험도, 이벤트 수, 스트림 메타 등)
  - `GET /api/events?session_id=&limit=` — 이벤트 목록
  - `GET /api/mode` — 현재 모드 + test_videos 목록
  - `POST /api/mode/test?file=xxx.mp4` — 테스트 영상 전환
  - `POST /api/mode/live` — RealSense 전환
  - `POST /api/control/start|stop|reset` — 녹화 제어

---

## 환경변수 (`backend/.env`)

```
# 소스
MONITOR_MODEL=yolo26m.pt          # YOLO 가중치 파일명 (프로젝트 루트 기준)
MONITOR_SOURCE=                   # 비어있으면 RealSense, 경로 지정시 영상 파일
MONITOR_WIDTH=640
MONITOR_HEIGHT=480
MONITOR_FPS=30

# 처리
MONITOR_USE_YOLO_TRACK=1          # 1=ByteTrack 활성, 0=공간매칭
MONITOR_TRACKER=bytetrack_config.yaml
MONITOR_TRAIL_LEN=30
MONITOR_NO_OVERLAY=0              # 1=시각화 오버레이 전체 끔
MONITOR_HIDE_HUD_PANEL=1          # 1=하단 그래프 숨김
MONITOR_JPEG_QUALITY=92           # MJPEG 품질 (60~100)
MONITOR_EVENT_COOLDOWN_SEC=1.0    # 이벤트 저장 쿨다운 (초)

# Supabase
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...     # 절대 커밋 금지
SUPABASE_EVENT_TABLE=collision_events
SUPABASE_SNAPSHOT_BUCKET=collision-event-snaps
SUPABASE_SNAPSHOT_PREFIX=events
MONITOR_SNAPSHOT_RETRY_COUNT=3
MONITOR_SNAPSHOT_RETRY_DELAY_SEC=0.4
```

프론트엔드는 `NEXT_PUBLIC_MONITOR_API` (기본 `http://127.0.0.1:8000`)로 백엔드 주소를 설정한다.

---

## 위험도 계산 로직

| 변수 | 설명 |
|------|------|
| `dist_m` | 사람-장애물 샘플 포인트 간 최솟값 (RealSense depth 기반) |
| `TTC` | Time-To-Collision (`ttc_forward` / `ttc_los` / `both_min`) |
| `closing_speed` | LOS 방향 접근속도 |
| `risk_score` | `dist_weight×dist_score + ttc_weight×ttc_score + close_weight×close_score` (기본 가중치: 0.5 / 0.35 / 0.15) |
| SAFE→WARNING | score ≥ 0.45 (2프레임 연속) |
| WARNING→DANGER | score ≥ 0.75 (2프레임 연속) |
| DANGER→WARNING | score < 0.65 (4프레임 연속) |

> `depth_image=None` 이면 위 계산 전체 건너뜀. 테스트 영상에서는 위험도 표시 안 됨.

---

## ByteTrack 튜닝 (`bytetrack_config.yaml`)

공장 저FPS(~15fps), 원거리 소형 객체 환경에 맞게 튜닝됨:
- `track_high_thresh: 0.35` — 원거리 소형 객체 confidence가 낮아 낮게 설정
- `track_buffer: 60` — 15fps 기준 약 4초 동안 가려져도 같은 ID 유지
- `match_thresh: 0.7` — 저FPS 큰 이동량 허용

---

## 대시보드 페이지 구성

- `/` — Live Operations Center: MJPEG 스트림, 실시간 지표, 세션 녹화 제어
- `/event-logs` — 이벤트 로그 목록
- `/analytics` — 분석 차트
- `/settings` — 설정

---

## 현재 모델 현황

- **사용 모델**: `yolo26m.pt` (COCO 80클래스 학습)
- **탐지 클래스**: person, car, motorcycle, truck, bus 등 (`config.py`의 `OBSTACLE_CLASSES_DEFAULT` 참조)
- **VisDrone 학습 시도**: 드론 수직(90°) 앵글 데이터 → 공장 사선(45~60°) 환경에 도메인 불일치로 성능 저하. 폐기.
- **향후 계획**: 실제 공장 CCTV 사선 앵글 데이터 수집 및 커스텀 파인튜닝 필요

---

## 주의 사항

- `best.pt` / `last.pt` 등 대용량 가중치 파일은 git에 포함하지 않음
- `backend/.env` 는 절대 커밋 금지 (Supabase 키 포함)
- RealSense SDK (`pyrealsense2`) 는 Windows/Linux 전용, Mac 미지원
- 테스트 영상은 `test_videos/` 폴더에 배치하면 대시보드에서 자동 인식
- `backend/requirements.txt`는 `fastapi`, `uvicorn`만 포함 — `ultralytics`, `opencv-python`, `numpy`, `pyrealsense2` 등은 별도 설치 필요

---

## 향후 개발 로드맵

1. **Depth Anything V2 Metric** — RealSense depth 없이 RGB만으로 절대거리(m) 계산
2. **공장 CCTV 커스텀 학습** — 사선 앵글 데이터 수집 → CVAT 라벨링 → YOLO 파인튜닝
3. **Supabase MCP 연결** — Claude Code에서 직접 DB 조회/관리
