"""
collision_monitor/video_runner.py
==================================
영상 파일 / 웹캠에서 프레임을 읽어 FrameProcessor에 전달한다.
탐지·트래킹 로직은 PipelineRunner(RealSense)와 완전히 동일하다.

--depth-model 옵션을 지정하면 Metric3D V2 모델로 metric depth를 추정한다.
depth 추론은 백그라운드 스레드에서 실행되며, 메인 루프는 블로킹 없이
이전 결과를 재사용한다 (라이브 모드와 동일한 구조).
"""

from __future__ import annotations

import threading
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

        # ── Depth 모델 백그라운드 스레드 (라이브 모드와 동일 구조) ────────────
        self._depth_model       = None
        self._depth_scale       = None
        self._pseudo_intrinsics = None

        self._dm_lock           = threading.Lock()
        self._dm_input_frame    = None   # 메인 → 워커
        self._dm_latest_result  = None   # 워커 → 메인
        self._dm_stop           = threading.Event()
        self._dm_thread         = None
        self._dm_last_submit_ts = 0.0
        self._dm_interval_sec   = max(0.0, float(getattr(args, "model_depth_interval", 0.2)))
        self._sync_depth        = bool(getattr(args, "video_depth_sync", False))

        depth_model_id = getattr(args, "depth_model", "none")
        if depth_model_id and depth_model_id.lower() != "none":
            from .depth_model import DepthAnythingV2Wrapper, make_intrinsics_from_fov
            self._depth_fov = float(getattr(args, "depth_fov", 79.0))
            self._depth_model = DepthAnythingV2Wrapper(
                model_id=depth_model_id, hfov_deg=self._depth_fov
            )
            self._depth_scale     = 1.0
            self._make_intrinsics = make_intrinsics_from_fov

            if not self._sync_depth:
                self._dm_thread = threading.Thread(
                    target=self._depth_worker, daemon=True, name="video-depth-worker"
                )
                self._dm_thread.start()
            if self._sync_depth:
                print(
                    f"[VideoRunner] Depth 모델 활성화 (동기, RGB-depth 프레임 일치): "
                    f"{depth_model_id}"
                )
            else:
                print(
                    f"[VideoRunner] Depth 모델 활성화 (백그라운드, "
                    f"간격={self._dm_interval_sec}s): {depth_model_id}"
                )
        else:
            print("[VideoRunner] Depth 모델 없음 — 탐지+트래킹만 수행")

        # ── FrameProcessor ───────────────────────────────────────────────────
        print(f"[VideoRunner] 모델 로딩: {args.model}")
        self.processor = FrameProcessor(args)

    # ── Depth 백그라운드 워커 (runner.py와 동일) ──────────────────────────────

    def _depth_worker(self):
        while not self._dm_stop.is_set():
            with self._dm_lock:
                frame = self._dm_input_frame
                self._dm_input_frame = None

            if frame is None:
                time.sleep(0.01)
                continue

            try:
                result = self._depth_model.infer(frame)
                with self._dm_lock:
                    self._dm_latest_result = result
            except Exception as exc:
                print(f"[VideoDepthWorker] 추론 오류: {exc}")

            # 추론 후 대기 → GPU를 YOLO에 양보
            self._dm_stop.wait(timeout=self._dm_interval_sec)

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
        self._dm_stop.set()
        if self._dm_thread is not None:
            self._dm_thread.join(timeout=2.0)
        self.processor.close()
        cv2.destroyAllWindows()

    # ── 내부 루프 ────────────────────────────────────────────────────────────

    def _loop_video(self):
        while not self._stop:
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                print(f"[VideoRunner] 소스를 열 수 없습니다: {self._source}")
                return

            fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
            print(f"[VideoRunner] 소스: {self._source} | {fps_src:.0f}fps")

            # ── 최신 프레임 버퍼 (Reader → Infer) ──────────────────────────
            _buf_lock   = threading.Lock()
            _latest_buf = [None]   # [frame | None]
            _eof        = [False]

            def _reader_thread():
                interval = 1.0 / fps_src
                while not self._stop:
                    t0 = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        with _buf_lock:
                            _eof[0] = True
                        break
                    with _buf_lock:
                        _latest_buf[0] = frame   # 이전 미처리 프레임 덮어씀
                    elapsed = time.time() - t0
                    sleep_t = max(0.0, interval - elapsed)
                    if sleep_t > 0:
                        time.sleep(sleep_t)

            reader = threading.Thread(target=_reader_thread, daemon=True, name="video-reader")
            reader.start()

            # ── 추론 루프 ───────────────────────────────────────────────────
            while not self._stop:
                with _buf_lock:
                    frame = _latest_buf[0]
                    _latest_buf[0] = None
                    eof = _eof[0]

                if frame is None:
                    if eof:
                        break
                    time.sleep(0.005)
                    continue

                depth_image = None
                intrinsics  = None
                depth_scale = None

                if self._depth_model is not None:
                    if self._pseudo_intrinsics is None:
                        h, w = frame.shape[:2]
                        self._pseudo_intrinsics = self._make_intrinsics(
                            w, h, self._depth_fov
                        )
                        print(f"[VideoRunner] Intrinsics: {self._pseudo_intrinsics}")

                    if self._sync_depth:
                        try:
                            depth_image = self._depth_model.infer(frame)
                        except Exception as exc:
                            print(f"[VideoRunner] Sync depth inference error: {exc}")
                            depth_image = None
                    else:
                        now_ts = time.time()
                        with self._dm_lock:
                            if now_ts - self._dm_last_submit_ts >= self._dm_interval_sec:
                                self._dm_input_frame    = frame.copy()
                                self._dm_last_submit_ts = now_ts
                            depth_image = self._dm_latest_result

                    intrinsics  = self._pseudo_intrinsics
                    depth_scale = self._depth_scale

                canvas, result = self.processor.process(
                    color_image=frame,
                    depth_image=depth_image,
                    intrinsics=intrinsics,
                    depth_scale=depth_scale,
                )

                if self.on_result is not None:
                    self.on_result(result, canvas)

                if self.display:
                    cv2.imshow("VideoRunner", canvas)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        self._stop = True
                        break

            reader.join(timeout=2.0)
            cap.release()

            if not self._loop or self._stop:
                break

            print("[VideoRunner] 영상 끝 → 처음부터 재시작")
            self.processor.reset_tracking()
