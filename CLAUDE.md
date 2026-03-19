# Paran — 공장 충돌 위험 감지 시스템 CLAUDE.md

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

## 디렉토리 구조

```
Paran/
├── collision_monitor/        # 탐지·트래킹·위험도 핵심 로직
│   ├── frame_processor.py    # YOLO 탐지 + ByteTrack + 위험도 + 시각화 (핵심)
│   ├── runner.py             # RealSense 카메라 소스 루프
│   ├── video_runner.py       # 영상 파일 소스 루프
│   ├── byte_tracker.py       # TrackHistory / TrackData (EMA 스무딩, trail 관리)
│   ├── config.py             # argparse 기본값 모음
│   ├── risk.py               # TTC / 위험도 스코어 계산
│   ├── tracking.py           # assign_tracks (공간 매칭)
│   ├── depth.py              # RealSense depth 유틸
│   ├── geometry.py           # 3D 좌표 변환
│   └── output.py             # 그래프 그리기 / 로그 / 비프음
├── backend/
│   ├── app.py                # FastAPI 서버 + MonitorService
│   ├── .env                  # 환경변수 (gitignore — 커밋 금지)
│   ├── requirements.txt
│   └── supabase_schema.sql   # Supabase 테이블 스키마
├── dashboard/                # Next.js 프론트엔드
├── test_videos/              # 테스트용 영상 파일 (MP4 등)
├── bytetrack_config.yaml     # ByteTrack 튜닝 파라미터
└── yolo26m.pt                # 현재 사용 YOLO 가중치
```

---

## 핵심 파일 역할

### `collision_monitor/frame_processor.py`
- `FrameProcessor.process(color_image, depth_image, intrinsics, depth_scale)` 가 메인 진입점
- `depth_image=None` 으로 호출하면 depth 의존 계산(거리·TTC·risk)을 건너뜀 → 테스트 영상 모드
- PipelineRunner(RealSense)와 VideoRunner(영상 파일) 양쪽이 이 클래스를 공유
- 시각화: bbox, 트래킹 ID, 위험도 텍스트만 표시 (circle dot, trail 선 없음)

### `collision_monitor/byte_tracker.py`
- `TrackHistory`: 트랙별 bbox EMA 스무딩(`bbox_alpha=0.5`), trail 좌표 보관
- `TrackData.smooth_center`: bbox 중앙 `((sx1+sx2)/2, (sy1+sy2)/2)` — 발이 가려지는 공장 환경 대응

### `collision_monitor/config.py`
- 주요 기본값:
  - `--model`: `yolo26m.pt`
  - `--conf`: `0.35`
  - `--tracker`: `bytetrack_config.yaml`
  - `--imgsz`: `1280`
  - `--warn-dist`: `1.5m` / `--danger-dist`: `0.9m`

### `backend/app.py`
- `MonitorService`: 런너 스레드 관리, JPEG 프레임 버퍼, Supabase 이벤트 저장
- `MONITOR_SOURCE` 환경변수가 있으면 `VideoRunner`, 없으면 `PipelineRunner(RealSense)`
- API 엔드포인트:
  - `GET /api/stream` — MJPEG 스트림
  - `GET /api/state` — 현재 상태 (위험도, 이벤트 수 등)
  - `GET /api/mode` — 현재 모드 + test_videos 목록
  - `POST /api/mode/test?file=xxx.mp4` — 테스트 영상 전환
  - `POST /api/mode/live` — RealSense 전환
  - `POST /api/control/start|stop|reset` — 녹화 제어

---

## 환경변수 (`backend/.env`)

```
MONITOR_MODEL=yolo26m.pt          # YOLO 가중치 파일명 (프로젝트 루트 기준)
MONITOR_SOURCE=                   # 비어있으면 RealSense, 경로 지정시 영상 파일
MONITOR_USE_YOLO_TRACK=1          # 1=ByteTrack 활성, 0=공간매칭
MONITOR_HIDE_HUD_PANEL=1          # 1=하단 그래프 숨김
MONITOR_JPEG_QUALITY=92           # MJPEG 품질 (60~100)
SUPABASE_URL=...                  # Supabase 프로젝트 URL
SUPABASE_SERVICE_ROLE_KEY=...     # Service Role Key (절대 커밋 금지)
SUPABASE_EVENT_TABLE=collision_events
SUPABASE_SNAPSHOT_BUCKET=collision-event-snaps
```

> `.env`는 `.gitignore`에 포함되어 있음. 키를 코드에 하드코딩하거나 커밋하지 말 것.

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

## ByteTrack 튜닝 (`bytetrack_config.yaml`)

공장 저FPS(~15fps), 원거리 소형 객체 환경에 맞게 튜닝됨:
- `track_high_thresh: 0.35` — 원거리 소형 객체 confidence가 낮아 낮게 설정
- `track_buffer: 60` — 15fps 기준 약 4초 동안 가려져도 같은 ID 유지
- `match_thresh: 0.7` — 저FPS 큰 이동량 허용

---

## 위험도 계산 로직

| 변수 | 설명 |
|------|------|
| `dist_m` | 사람-장애물 3D 거리 (RealSense depth 기반) |
| `TTC` | Time-To-Collision (거리 / 접근속도) |
| `closing_speed` | LOS 방향 접근속도 |
| `risk_score` | `dist_weight×dist_score + ttc_weight×ttc_score + close_weight×close_score` |
| SAFE→WARNING | score ≥ 0.45 (2프레임 연속) |
| WARNING→DANGER | score ≥ 0.75 (2프레임 연속) |

> depth_image=None 이면 위 계산 전체 건너뜀. 테스트 영상에서는 위험도 표시 안 됨.

---

## 현재 모델 현황

- **사용 모델**: `yolo26m.pt` (COCO 80클래스 학습)
- **탐지 클래스**: person, car, motorcycle, truck, bus 등
- **VisDrone 학습 시도**: 드론 수직(90°) 앵글 데이터 → 공장 사선(45~60°) 환경에 도메인 불일치로 성능 저하. 폐기.
- **향후 계획**: 실제 공장 CCTV 사선 앵글 데이터 수집 및 커스텀 파인튜닝 필요

---

## 주의 사항

- `best.pt` / `last.pt` 등 대용량 가중치 파일은 git에 포함하지 않음
- `backend/.env` 는 절대 커밋 금지 (Supabase 키 포함)
- RealSense SDK (`pyrealsense2`) 는 Windows/Linux 전용, Mac 미지원
- 테스트 영상은 `test_videos/` 폴더에 배치하면 대시보드에서 자동 인식

---

## 향후 개발 로드맵

1. **Depth Anything V2 Metric** — RealSense depth 없이 RGB만으로 절대거리(m) 계산
2. **공장 CCTV 커스텀 학습** — 사선 앵글 데이터 수집 → CVAT 라벨링 → YOLO 파인튜닝
3. **Supabase MCP 연결** — Claude Code에서 직접 DB 조회/관리
