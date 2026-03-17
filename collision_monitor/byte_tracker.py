"""
collision_monitor/byte_tracker.py
==================================
ByteTrack 기반 트래킹 히스토리 관리 모듈.

역할
----
- 각 track의 중심점 궤적(trail)을 프레임 간에 유지한다.
- Ultralytics YOLO model.track()이 반환하는 box.id를 그대로 사용한다.
- 추후 속도(velocity), 진행방향(direction), TTC 계산을 위해
  프레임별 center 시퀀스를 TrackHistory에 누적한다.

설계 원칙
---------
- TrackData  : 단일 프레임에서 track 1개의 스냅샷 (불변값)
- TrackHistory : 여러 프레임에 걸친 trail 상태를 관리 (mutable)
- 속도/방향 계산은 이 파일이 아니라 별도 모듈에서 trail 데이터를 받아 수행한다.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── 타입 별칭 ────────────────────────────────────────────────────────────────
BBox   = Tuple[int, int, int, int]   # (x1, y1, x2, y2)
Center = Tuple[int, int]             # (cx, cy) 픽셀 좌표
Trail  = List[Center]                # center 점의 시퀀스


@dataclass
class TrackData:
    """
    단일 track의 현재 프레임 스냅샷.

    Attributes
    ----------
    track_id   : ByteTrack이 부여한 고유 ID (프레임 간 유지)
    class_name : YOLO 클래스명 (예: 'person', 'car', 'forklift')
    bbox       : (x1, y1, x2, y2) 픽셀 좌표
    confidence : 탐지 신뢰도 (0.0 ~ 1.0)
    center     : (cx, cy) bbox 중심점
    trail      : 최근 N 프레임의 center 목록 (가장 오래된 것이 [0])

    추후 확장 예시 (speed/direction/TTC 붙일 때 주석 해제)
    -----------------------------------------------------------
    velocity_px   : Optional[Tuple[float, float]]  # 픽셀/프레임 속도벡터
    direction_deg : Optional[float]                 # 이동 방향 (북=0, 시계방향)
    speed_pxps    : Optional[float]                 # 픽셀/초 속도 크기
    """

    track_id   : int
    class_name : str
    bbox       : BBox
    confidence : float
    center     : Center
    trail      : Trail

    # ── 추후 확장용 필드 (현재 미사용) ─────────────────────────────────────
    # velocity_px   : Optional[Tuple[float, float]] = field(default=None, compare=False)
    # direction_deg : Optional[float]               = field(default=None, compare=False)
    # speed_pxps    : Optional[float]               = field(default=None, compare=False)


class TrackHistory:
    """
    프레임 간 track 상태(궤적)를 관리하는 클래스.

    주요 기능
    ---------
    - track_id 별 center 궤적(trail) 누적 (deque 방식, 최대 길이 제한)
    - 사라진 track에 TTL(Time-To-Live)을 부여해 일정 프레임 후 자동 정리
      → 잠깐 가려진 객체가 사라지지 않도록 버퍼를 둠

    확장 포인트
    -----------
    - velocity/direction 계산 메서드를 이 클래스에 추가할 수 있다.
      (trail 데이터가 이미 누적되어 있으므로 간단히 구현 가능)

    Parameters
    ----------
    trail_maxlen   : 궤적 최대 길이 (프레임 단위). 기본 30.
    dead_track_ttl : track이 사라진 후 유지할 프레임 수. 기본 30.
                     (30fps 기준 약 1초 동안 trail 유지 후 정리)
    """

    def __init__(
        self,
        trail_maxlen   : int   = 30,
        dead_track_ttl : int   = 30,
        bbox_alpha     : float = 0.7,
    ) -> None:
        """
        Parameters
        ----------
        trail_maxlen   : 궤적 최대 길이 (프레임 단위). 기본 30.
        dead_track_ttl : track이 사라진 후 유지할 프레임 수. 기본 30.
        bbox_alpha     : bbox EMA 계수 (0.0 ~ 1.0).
                         높을수록 이전 값을 더 유지 → 흔들림 감소, 반응 느림.
                         낮을수록 현재 값에 빠르게 반응 → 흔들림 증가, 반응 빠름.
                         기본 0.7 (정지 객체 흔들림 억제에 적합).
        """
        self.trail_maxlen   = trail_maxlen
        self.dead_track_ttl = dead_track_ttl
        self.bbox_alpha     = float(max(0.0, min(0.99, bbox_alpha)))

        # track_id → deque((cx, cy)) — 스무딩된 중심점 시퀀스
        self._trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.trail_maxlen)
        )

        # track_id → 스무딩된 bbox [x1, y1, x2, y2] (float)
        # 첫 프레임은 raw 값으로 초기화, 이후 EMA 적용
        self._smooth_bboxes: Dict[int, List[float]] = {}

        # track_id → 남은 TTL (0이 되면 정리)
        self._ttl: Dict[int, int] = {}

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def update(
        self,
        track_id   : int,
        class_name : str,
        bbox       : BBox,
        conf       : float,
    ) -> TrackData:
        """
        한 track의 현재 프레임 정보를 업데이트하고 TrackData를 반환한다.

        이 메서드를 프레임마다 감지된 각 box에 대해 호출한다.
        호출 후 반드시 step(active_ids)을 한 번 호출해야 TTL이 관리된다.

        center는 스무딩된 bbox에서 자동 계산되므로 별도로 전달하지 않는다.

        Parameters
        ----------
        track_id   : ByteTrack이 부여한 ID (box.id)
        class_name : YOLO 클래스명
        bbox       : (x1, y1, x2, y2) raw 탐지 좌표
        conf       : 신뢰도

        Returns
        -------
        TrackData : 스무딩된 bbox/center/trail이 포함된 스냅샷
        """
        # ── bbox EMA 스무딩 ──────────────────────────────────────────────
        # 첫 등장: raw 값으로 초기화 (이전 값 없음)
        # 이후  : smoothed = alpha * prev + (1 - alpha) * raw
        raw = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        prev = self._smooth_bboxes.get(track_id)
        if prev is None:
            smoothed = raw
        else:
            a = self.bbox_alpha
            smoothed = [a * prev[i] + (1.0 - a) * raw[i] for i in range(4)]
        self._smooth_bboxes[track_id] = smoothed

        # 스무딩된 bbox에서 center 재계산 (trail도 스무딩 좌표 기준)
        # center는 발바닥 기준: x=bbox 중앙, y=bbox 하단
        sx1, sy1, sx2, sy2 = smoothed
        smooth_center: Center = (int((sx1 + sx2) / 2), int(sy2))
        smooth_bbox: BBox = (int(sx1), int(sy1), int(sx2), int(sy2))

        # 궤적에 스무딩된 중심점 추가
        self._trails[track_id].append(smooth_center)

        # TTL 갱신 (다시 감지됐으므로 최대값으로 리셋)
        self._ttl[track_id] = self.dead_track_ttl

        return TrackData(
            track_id   = track_id,
            class_name = class_name,
            bbox       = smooth_bbox,       # 스무딩된 bbox
            confidence = float(conf),
            center     = smooth_center,     # 스무딩된 중심점
            trail      = list(self._trails[track_id]),
        )

    def step(self, active_ids: List[int]) -> None:
        """
        프레임 종료 시 한 번 호출한다.
        이번 프레임에 보이지 않은 track의 TTL을 감소시키고,
        TTL이 0 이하가 된 track을 정리한다.

        Parameters
        ----------
        active_ids : 이번 프레임에 update()가 호출된 track_id 목록
        """
        active_set = set(active_ids)

        # 감지되지 않은 track → TTL 감소
        dead_candidates = set(self._ttl.keys()) - active_set
        for tid in dead_candidates:
            self._ttl[tid] -= 1
            if self._ttl[tid] <= 0:
                self._trails.pop(tid, None)
                self._ttl.pop(tid, None)

    def get_trail(self, track_id: int) -> Trail:
        """특정 track의 궤적 반환 (읽기 전용 복사본)."""
        return list(self._trails.get(track_id, []))

    def reset(self) -> None:
        """모든 히스토리와 TTL을 초기화한다. 모드 전환 시 호출."""
        self._trails.clear()
        self._ttl.clear()

    # ── 확장 포인트: 추후 속도/방향 계산 메서드 추가 위치 ───────────────────
    #
    # def estimate_velocity_px(self, track_id: int, fps: float) -> Optional[Tuple[float, float]]:
    #     """trail의 최근 N 점으로 픽셀/초 속도 벡터 추정."""
    #     trail = self.get_trail(track_id)
    #     if len(trail) < 2:
    #         return None
    #     dx = trail[-1][0] - trail[-2][0]
    #     dy = trail[-1][1] - trail[-2][1]
    #     return (dx * fps, dy * fps)
    #
    # def estimate_direction_deg(self, track_id: int) -> Optional[float]:
    #     """trail의 최근 이동 벡터로 진행 방향 추정 (북=0, 시계방향 양수)."""
    #     trail = self.get_trail(track_id)
    #     if len(trail) < 2:
    #         return None
    #     import math
    #     dx = trail[-1][0] - trail[-2][0]
    #     dy = trail[-1][1] - trail[-2][1]
    #     return (math.degrees(math.atan2(dx, -dy))) % 360
