"""
collision_monitor/video_runner.py
==================================
영상 파일 / 웹캠에서 프레임을 읽어 FrameProcessor에 전달한다.
탐지·트래킹 로직은 PipelineRunner(RealSense)와 완전히 동일하다.
depth 없이 실행하므로 거리·TTC·위험도는 계산하지 않는다.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .frame_processor import FrameProcessor


class VideoRunner:
    def __init__(self, args, display: bool = False, on_result=None):
        self.args      = args
        self.display   = display
        self.on_result = on_result
        self._stop     = False
        self._closed   = False

        # 입력 소스 (숫자=웹캠, 문자열=파일경로)
        source       = getattr(args, "video_source", "0")
        self._source = int(source) if str(source).isdigit() else source
        self._loop   = bool(getattr(args, "video_loop", True))

        # ── FrameProcessor (PipelineRunner와 동일한 탐지·트래킹 로직) ─────────
        print(f"[VideoRunner] 모델 로딩: {args.model}")
        self.processor = FrameProcessor(args)

    # ── 공개 인터페이스 ───────────────────────────────────────────────────────

    def run(self):
        try:
            self._loop_video()
        finally:
            self.close()

    def request_stop(self):
        self._stop = True

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.processor.close()
        cv2.destroyAllWindows()

    # ── 내부 루프 ────────────────────────────────────────────────────────────

    def _loop_video(self):
        while not self._stop:
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                print(f"[VideoRunner] 소스를 열 수 없습니다: {self._source}")
                return

            fps_src  = cap.get(cv2.CAP_PROP_FPS) or 30.0
            interval = 1.0 / fps_src
            print(f"[VideoRunner] 소스: {self._source} | {fps_src:.0f}fps")

            while not self._stop:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                # depth=None → 거리/TTC/위험도 없이 탐지+트래킹만 수행
                canvas, result = self.processor.process(
                    color_image=frame,
                    depth_image=None,
                )

                if self.on_result is not None:
                    self.on_result(result, canvas)

                if self.display:
                    cv2.imshow("VideoRunner", canvas)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        self._stop = True
                        break

                elapsed = time.time() - t0
                sleep_t = max(0.0, interval - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)

            cap.release()

            if not self._loop or self._stop:
                break

            print("[VideoRunner] 영상 끝 → 처음부터 재시작")
            self.processor.reset_tracking()
