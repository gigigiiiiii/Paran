"""
track_demo.py
=============
YOLO + ByteTrack 기반 객체 추적 데모 스크립트.

공장 고정형 CCTV 영상에서 사람(person)과 이동장비(vehicle 등)를 검출하고
ByteTrack으로 프레임 간 동일 ID를 유지하면서 이동 궤적(trail)을 시각화한다.

RealSense 깊이 카메라 없이도 단독 실행 가능하다 (일반 영상/웹캠).

실행 예시
---------
  # 웹캠
  python track_demo.py --source 0

  # 비디오 파일
  python track_demo.py --source path/to/video.mp4

  # RTSP 스트림
  python track_demo.py --source "rtsp://192.168.0.1/stream"

  # detect 전용 (track_id, trail 없음) — 비교 기준
  python track_demo.py --source video.mp4 --mode detect

  # detect / ByteTrack 나란히 비교
  python track_demo.py --source video.mp4 --mode compare

  # 결과 영상 저장
  python track_demo.py --source video.mp4 --save --output out.mp4

  # 신뢰도 낮추기 + 모든 클래스 추적
  python track_demo.py --source video.mp4 --conf 0.25 --target-classes ""

키보드 단축키
-------------
  q / ESC : 종료
  t       : detect ↔ track 모드 토글
  s       : 현재 프레임 스크린샷 저장

설계 확장 포인트
----------------
  - 속도/방향 계산 : TrackHistory.estimate_velocity_px() 구현 후 TrackData에 추가
  - TTC 계산       : 두 TrackData의 trail로 closig speed 추정 후 거리/속도로 산출
  - SAHI 통합      : _run_inference() 함수만 교체하면 됨
  - 커스텀 학습    : --model 인자에 커스텀 .pt 경로만 지정하면 됨
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# 이 프로젝트의 TrackHistory / TrackData 모듈
from collision_monitor.byte_tracker import TrackData, TrackHistory


# ── 클래스별 bbox 색상 (BGR) ─────────────────────────────────────────────────
# 새 클래스를 추적하려면 여기에 추가하면 된다.
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "person":     ( 50, 220,  50),   # 녹색
    "car":        (220, 120,  50),   # 주황
    "truck":      (200,  80,  80),   # 빨강
    "bus":        (200,  80, 200),   # 보라
    "motorcycle": (180, 200,  50),   # 연두
    "bicycle":    (100, 200, 220),   # 하늘
    "forklift":   (255, 180,   0),   # 금색
    "cart":       (255, 220,  50),   # 노랑
    "vehicle":    (180, 150, 255),   # 라벤더
}

# 미정의 클래스 기본 색상
_DEFAULT_COLOR: Tuple[int, int, int] = (180, 180, 180)

# 기본 추적 대상 클래스 (비워두면 전체 클래스 추적)
TARGET_CLASSES_DEFAULT: Set[str] = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "forklift", "cart", "vehicle",
}

# ByteTrack 설정 파일 경로
_SCRIPT_DIR = Path(__file__).parent
_LOCAL_TRACKER_CFG = _SCRIPT_DIR / "bytetrack_config.yaml"
# 로컬 설정 파일 있으면 사용, 없으면 Ultralytics 내장 bytetrack.yaml 사용
DEFAULT_TRACKER_CFG = str(_LOCAL_TRACKER_CFG) if _LOCAL_TRACKER_CFG.exists() else "bytetrack.yaml"


# ── 색상 헬퍼 ────────────────────────────────────────────────────────────────

def _color(class_name: str) -> Tuple[int, int, int]:
    """클래스명에 해당하는 BGR 색상 반환."""
    return CLASS_COLORS.get(class_name, _DEFAULT_COLOR)


# ── 시각화 함수 ──────────────────────────────────────────────────────────────

def draw_track(frame: np.ndarray, track: TrackData, show_trail: bool = True) -> None:
    """
    단일 TrackData를 프레임에 그린다.

    그리는 요소
    -----------
    1. bounding box (클래스별 색상)
    2. 라벨: "<class> #<ID> <conf%>"
    3. 중심점 (채워진 원)
    4. trail — 최근 N 프레임 이동 궤적 (오래될수록 얇아짐)
    """
    color = _color(track.class_name)
    x1, y1, x2, y2 = track.bbox
    cx, cy = track.center

    # 1. bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 2. 라벨 배경 + 텍스트
    label = f"{track.class_name} #{track.track_id}  {track.confidence:.0%}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    lx = x1
    ly = max(y1 - 4, th + 4)  # 프레임 위로 넘어가지 않도록
    # 라벨 배경 (불투명 채우기)
    cv2.rectangle(frame, (lx, ly - th - baseline), (lx + tw + 4, ly + baseline), color, -1)
    # 텍스트 (검은색)
    cv2.putText(frame, label, (lx + 2, ly), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # 3. 중심점
    cv2.circle(frame, (cx, cy), 5, color, -1)
    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)  # 흰 테두리

    # 4. trail (궤적)
    if show_trail and len(track.trail) >= 2:
        pts = np.array(track.trail, dtype=np.int32)
        n = len(pts)
        for i in range(1, n):
            # 오래된 점일수록 얇고 투명하게 (굵기 1~3 선형 보간)
            alpha = i / n                      # 0→1 (오래된→최신)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), color, thickness)


def draw_detect_only(
    frame      : np.ndarray,
    results,
    class_names: dict,
    target_cls : Set[str],
) -> int:
    """
    YOLO predict() 결과를 track_id / trail 없이 그린다 (비교용 detect 모드).

    Returns
    -------
    int : 이번 프레임 감지 객체 수
    """
    count = 0
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        name = _get_class_name(class_names, cls_id)
        if target_cls and name not in target_cls:
            continue

        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
        color = _color(name)

        # bbox만 그림 — track_id, trail 없음
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} {conf:.0%}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
        count += 1
    return count


def draw_hud(frame: np.ndarray, mode: str, fps: float, count: int) -> None:
    """
    우상단에 모드 / FPS / 감지 수 오버레이를 그린다.
    배경 없이 검은 외곽선 + 흰 텍스트 방식.
    """
    _, w = frame.shape[:2]
    lines = [
        f"Mode : {mode}",
        f"FPS  : {fps:.1f}",
        f"Count: {count}",
    ]
    for i, txt in enumerate(lines):
        y = 24 + i * 24
        x = w - 210
        # 검은 외곽선
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        # 흰 텍스트
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)


# ── 추론 헬퍼 ────────────────────────────────────────────────────────────────

def _get_class_name(class_names, cls_id: int) -> str:
    """YOLO 모델의 names (dict 또는 list 모두 대응)."""
    if isinstance(class_names, dict):
        return class_names.get(cls_id, str(cls_id))
    return str(class_names[cls_id])


def _run_inference(model, frame: np.ndarray, conf: float):
    """
    detect 전용 추론.
    추후 SAHI 통합 시 이 함수만 교체하면 된다.
    """
    return model.predict(frame, conf=conf, verbose=False)[0]


def _run_track(model, frame: np.ndarray, conf: float, tracker_cfg: str):
    """
    ByteTrack 추론.
    persist=True 가 핵심: Ultralytics가 내부 tracker 상태를 유지해준다.
    추후 SAHI 통합 시 이 함수의 반환값 형식에 맞게 래핑하면 된다.
    """
    return model.track(
        frame,
        conf=conf,
        tracker=tracker_cfg,
        persist=True,   # 프레임 간 tracker 상태 유지 (ID 연속성의 핵심)
        verbose=False,
    )[0]


# ── 핵심 처리 루프 ────────────────────────────────────────────────────────────

def _process_track_frame(
    frame        : np.ndarray,
    results,
    class_names  : dict,
    target_cls   : Set[str],
    track_history: TrackHistory,
) -> Tuple[np.ndarray, List[TrackData]]:
    """
    model.track() 결과를 파싱 → TrackHistory 업데이트 → 시각화.

    반환값
    ------
    (시각화된 프레임, 이번 프레임의 TrackData 리스트)
    """
    vis = frame.copy()
    tracks: List[TrackData] = []
    active_ids: List[int] = []

    for box in results.boxes:
        # box.id가 None이면 ByteTrack이 아직 track_id를 확정 못 한 것 → 스킵
        if box.id is None:
            continue

        cls_id = int(box.cls[0].item())
        name   = _get_class_name(class_names, cls_id)

        # 추적 대상 클래스 필터 (target_cls가 비어 있으면 전체 허용)
        if target_cls and name not in target_cls:
            continue

        track_id = int(box.id[0].item())
        conf     = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # TrackHistory 업데이트 → 이번 프레임 TrackData 반환
        track = track_history.update(
            track_id   = track_id,
            class_name = name,
            bbox       = (x1, y1, x2, y2),
            conf       = conf,
        )
        tracks.append(track)
        active_ids.append(track_id)

    # 이번 프레임에 보이지 않은 track의 TTL 감소 및 정리
    track_history.step(active_ids)

    # 시각화
    for track in tracks:
        draw_track(vis, track, show_trail=True)

    return vis, tracks


# ── 직렬화 / 후처리용 헬퍼 ───────────────────────────────────────────────────

def _track_to_dict(track: TrackData) -> dict:
    """
    TrackData → JSON 직렬화 가능한 dict.

    추후 속도·방향·TTC 계산 결과를 여기에 추가하면 된다:
      "velocity_px"   : [vx, vy],
      "direction_deg" : 45.0,
      "speed_pxps"    : 12.3,
    """
    return {
        "track_id"   : track.track_id,
        "class_name" : track.class_name,
        "bbox"       : list(track.bbox),
        "confidence" : round(track.confidence, 4),
        "center"     : list(track.center),
        # 메모리 절약: 최근 5개 중심점만 직렬화 (전체 trail은 TrackHistory에 있음)
        "trail_last5": [list(p) for p in track.trail[-5:]],
    }


# ── 메인 추적 루프 ────────────────────────────────────────────────────────────

def run_tracker(
    source      : str,
    model_path  : str,
    tracker_cfg : str,
    conf_thres  : float,
    target_cls  : Set[str],
    trail_len   : int,
    mode        : str,        # "detect" | "track" | "compare"
    save_output : bool,
    output_path : str,
) -> Dict[int, List[dict]]:
    """
    메인 추적 루프.

    Parameters
    ----------
    source      : 입력 소스 (숫자=웹캠, 경로=파일, URL=스트림)
    model_path  : YOLO 모델 경로
    tracker_cfg : ByteTrack 설정 파일 경로
    conf_thres  : 탐지 신뢰도 임계값
    target_cls  : 추적 대상 클래스 집합
    trail_len   : 궤적 유지 프레임 수
    mode        : 실행 모드 ("detect" / "track" / "compare")
    save_output : 결과 영상 저장 여부
    output_path : 저장 경로

    Returns
    -------
    Dict[frame_idx, List[dict]]
        프레임별 track 결과 (후처리, 분석용)
    """
    print(f"[INFO] 모델 로딩: {model_path}")
    model = YOLO(model_path)
    class_names = getattr(model, "names", {})
    if not class_names:
        class_names = model.model.names  # fallback

    # 프레임 간 궤적 관리
    track_history = TrackHistory(trail_maxlen=trail_len, dead_track_ttl=30)

    # 프레임별 track 결과 저장 (추후 속도·방향·TTC 계산에 활용)
    frame_results: Dict[int, List[dict]] = {}

    # 입력 소스 열기 (숫자 → 웹캠 인덱스, 문자열 → 파일/URL)
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"소스를 열 수 없습니다: {source}")

    frame_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 결과 영상 저장 설정
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w  = frame_w * 2 if mode == "compare" else frame_w
        writer = cv2.VideoWriter(output_path, fourcc, fps_src, (out_w, frame_h))
        print(f"[INFO] 결과 영상 저장: {output_path}")

    frame_idx = 0
    prev_t    = time.time()

    print(f"[INFO] 소스: {source}  |  모드: {mode}  |  conf: {conf_thres}")
    print(f"[INFO] 추적 대상 클래스: {sorted(target_cls) if target_cls else '(전체)'}")
    print("[INFO] 단축키: q=종료  t=모드전환  s=스크린샷")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        now       = time.time()
        fps_disp  = 1.0 / max(now - prev_t, 1e-4)
        prev_t    = now

        # ── detect 전용 모드 ──────────────────────────────────────────────
        if mode == "detect":
            det_res = _run_inference(model, frame, conf_thres)
            vis     = frame.copy()
            n_det   = draw_detect_only(vis, det_res, class_names, target_cls)
            draw_hud(vis, "DETECT ONLY", fps_disp, n_det)
            display = vis

        # ── track 모드 ────────────────────────────────────────────────────
        elif mode == "track":
            trk_res           = _run_track(model, frame, conf_thres, tracker_cfg)
            vis, tracks       = _process_track_frame(
                frame, trk_res, class_names, target_cls, track_history
            )
            frame_results[frame_idx] = [_track_to_dict(t) for t in tracks]
            draw_hud(vis, "BYTETRACK", fps_disp, len(tracks))
            display = vis

        # ── compare 모드 (좌: detect 표시, 우: ByteTrack) ───────────────────
        else:
            # track()을 한 번만 실행한다.
            # predict()와 track()을 동일 모델 객체에 교대로 호출하면
            # Ultralytics 내부 tracker 상태가 초기화될 수 있으므로,
            # track 결과를 두 가지 방식으로 시각화해서 비교한다.
            #   왼쪽: track 결과를 trail/ID 없이 표시 → detect-only 효과
            #   오른쪽: track 결과를 trail/ID 포함 표시 → ByteTrack 효과
            trk_res       = _run_track(model, frame, conf_thres, tracker_cfg)
            right, tracks = _process_track_frame(
                frame, trk_res, class_names, target_cls, track_history
            )
            frame_results[frame_idx] = [_track_to_dict(t) for t in tracks]

            # 왼쪽: 동일 results를 trail/ID 없이 그리기 (detect 시뮬레이션)
            left  = frame.copy()
            n_det = draw_detect_only(left, trk_res, class_names, target_cls)
            draw_hud(left, "DETECT ONLY (no ID/trail)", fps_disp, n_det)
            draw_hud(right, "BYTETRACK (ID + trail)", fps_disp, len(tracks))

            # 좌우 나란히 합치기
            display = np.hstack([left, right])

        # 저장
        if writer is not None:
            writer.write(display)

        cv2.imshow("Paran Tracker", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):           # q 또는 ESC: 종료
            break
        elif key == ord("t"):               # t: detect ↔ track 토글
            if mode == "track":
                mode = "detect"
            elif mode == "detect":
                mode = "track"
                track_history.reset()       # 모드 전환 시 궤적 초기화
            # compare 모드에서는 토글 무시
            print(f"[INFO] 모드 전환 → {mode}")
        elif key == ord("s"):               # s: 스크린샷
            ss_path = f"screenshot_{frame_idx:06d}.jpg"
            cv2.imwrite(ss_path, display)
            print(f"[INFO] 스크린샷 저장: {ss_path}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n[INFO] 완료. 총 {frame_idx} 프레임 처리.")
    return frame_results


# ── 결과 요약 출력 ────────────────────────────────────────────────────────────

def print_summary(frame_results: Dict[int, List[dict]]) -> None:
    """
    추적 결과 요약을 출력한다.

    각 고유 track_id가 몇 프레임에 걸쳐 등장했는지 보여준다.
    추후 속도·방향 분석의 입력 데이터로 활용 가능하다.
    """
    if not frame_results:
        return

    # track_id → {class_name, frame 등장 횟수}
    track_summary: Dict[int, dict] = {}
    for tracks in frame_results.values():
        for t in tracks:
            tid = t["track_id"]
            if tid not in track_summary:
                track_summary[tid] = {"class_name": t["class_name"], "frames": 0}
            track_summary[tid]["frames"] += 1

    print("\n" + "=" * 50)
    print("[추적 결과 요약]")
    print(f"  총 처리 프레임 : {len(frame_results)}")
    print(f"  고유 track 수  : {len(track_summary)}")
    print("-" * 50)
    print(f"  {'ID':>4}  {'클래스':<12}  {'등장 프레임':>10}")
    print("-" * 50)
    for tid, info in sorted(track_summary.items()):
        print(f"  {tid:>4}  {info['class_name']:<12}  {info['frames']:>10}")
    print("=" * 50)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YOLO + ByteTrack 공장 CCTV 추적 데모",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source", type=str, default="0",
        help="입력 소스: 숫자(웹캠 인덱스), 파일 경로, RTSP URL",
    )
    p.add_argument(
        "--model", type=str, default="yolo26n.pt",
        help="YOLO 모델 파일 경로 (.pt)",
    )
    p.add_argument(
        "--tracker", type=str, default=DEFAULT_TRACKER_CFG,
        help="ByteTrack 설정 파일 경로 (yaml)",
    )
    p.add_argument(
        "--conf", type=float, default=0.35,
        help="탐지 신뢰도 임계값 (0.0 ~ 1.0)",
    )
    p.add_argument(
        "--mode", type=str, default="track",
        choices=["detect", "track", "compare"],
        help="실행 모드: detect(탐지만) / track(ByteTrack) / compare(나란히 비교)",
    )
    p.add_argument(
        "--target-classes", type=str, default="",
        help="추적 대상 클래스 (쉼표 구분). 비어 있으면 기본값 사용.",
    )
    p.add_argument(
        "--trail-len", type=int, default=30,
        help="궤적(trail)으로 유지할 최대 프레임 수",
    )
    p.add_argument(
        "--save", action="store_true",
        help="결과 영상을 파일로 저장",
    )
    p.add_argument(
        "--output", type=str, default="track_output.mp4",
        help="저장 파일 경로",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # 추적 대상 클래스 파싱
    if args.target_classes.strip():
        target_cls = {c.strip() for c in args.target_classes.split(",") if c.strip()}
    else:
        target_cls = TARGET_CLASSES_DEFAULT

    # 메인 루프 실행
    frame_results = run_tracker(
        source      = args.source,
        model_path  = args.model,
        tracker_cfg = args.tracker,
        conf_thres  = args.conf,
        target_cls  = target_cls,
        trail_len   = args.trail_len,
        mode        = args.mode,
        save_output = args.save,
        output_path = args.output,
    )

    # 결과 요약 출력
    print_summary(frame_results)


if __name__ == "__main__":
    main()
