"""
collision_monitor/runner.py
============================
RealSense 카메라에서 RGB + Depth 프레임을 읽어 FrameProcessor에 전달한다.
탐지·트래킹·위험도 계산은 모두 FrameProcessor 에서 수행한다.
"""

import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from .frame_processor import FrameProcessor


class PipelineRunner:
    def __init__(self, args, display=True, on_result=None):
        self.args      = args
        self.display   = bool(display)
        self.on_result = on_result
        self._closed   = False
        self._stop_requested = False

        # ── FrameProcessor (탐지·트래킹·위험도) ──────────────────────────────
        self.processor = FrameProcessor(args)

        # ── Depth 모델 (백그라운드 스레드, RealSense depth와 비교용) ──────────
        self._depth_model       = None
        self._dm_lock           = threading.Lock()
        self._dm_input_frame    = None   # 메인 → 워커로 전달할 최신 프레임
        self._dm_latest_result  = None   # 워커 → 메인으로 전달할 최신 depth
        self._dm_stop           = threading.Event()
        self._dm_thread         = None
        self._dm_last_submit_ts = 0.0
        # depth 추론 간격 (초): GPU 경쟁 방지, YOLO fps 보호
        # 기본 2초 → depth 0.5fps, YOLO는 비경쟁 구간에서 30fps 유지
        self._dm_interval_sec   = float(getattr(args, "depth_compare_interval", 2.0))

        depth_model_id = getattr(args, "depth_model", "none")
        if depth_model_id and depth_model_id.lower() != "none":
            from .depth_model import DepthAnythingV2Wrapper
            _depth_fov = float(getattr(args, "depth_fov", 79.0))
            self._depth_model = DepthAnythingV2Wrapper(
                model_id=depth_model_id, hfov_deg=_depth_fov
            )
            print(
                f"[PipelineRunner] Depth 비교 모델 활성화 (비동기, "
                f"간격={self._dm_interval_sec}s): {depth_model_id}"
            )
            self._dm_thread = threading.Thread(
                target=self._depth_worker, daemon=True, name="depth-model-worker"
            )
            self._dm_thread.start()

        # ── RealSense 초기화 ──────────────────────────────────────────────────
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        profile = self.pipeline.start(config)

        self.align       = rs.align(rs.stream.color)
        depth_sensor     = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.spatial  = rs.spatial_filter()
        self.temporal = rs.temporal_filter()

        color_stream    = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.warmup_left = max(0, int(args.rs_warmup))

    # ── Depth 모델 백그라운드 워커 ────────────────────────────────────────────

    def _depth_worker(self):
        """별도 스레드에서 depth 모델 추론.
        GPU 경쟁을 막기 위해 추론 후 _dm_interval_sec 동안 대기한다.
        YOLO(26ms)가 depth(107ms)와 GPU를 동시에 쓰면 YOLO가 4배 느려지므로
        depth를 간헐적으로만 실행해 메인 루프 fps를 보호한다.
        """
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
                print(f"[DepthWorker] 추론 오류: {exc}")

            # 추론 완료 후 대기 → GPU를 YOLO에 양보
            self._dm_stop.wait(timeout=self._dm_interval_sec)

    # ── 공개 인터페이스 ───────────────────────────────────────────────────────

    def run(self):
        try:
            while not self._stop_requested and self.process_frame():
                pass
        finally:
            self.close()

    def request_stop(self):
        self._stop_requested = True

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._dm_stop.set()
        if self._dm_thread is not None:
            self._dm_thread.join(timeout=2.0)
        self.processor.close()
        self.pipeline.stop()
        cv2.destroyAllWindows()

    # ── 프레임 처리 ───────────────────────────────────────────────────────────

    def process_frame(self) -> bool:
        if self._stop_requested:
            return False
        args = self.args
        try:
            frames = self.pipeline.wait_for_frames(args.rs_timeout_ms)
        except RuntimeError:
            return True

        aligned     = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return True

        if self.warmup_left > 0:
            self.warmup_left -= 1
            return True

        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 워커에 새 프레임 전달 (쓰로틀링), 직전 결과 읽기
        model_depth_image = None
        if self._depth_model is not None:
            now_ts = time.time()
            with self._dm_lock:
                # _dm_interval_sec마다 한 번만 새 프레임 제출 → GPU 경쟁 최소화
                if now_ts - self._dm_last_submit_ts >= self._dm_interval_sec:
                    self._dm_input_frame      = color_image.copy()
                    self._dm_last_submit_ts   = now_ts
                model_depth_image = self._dm_latest_result

        canvas, result = self.processor.process(
            color_image       = color_image,
            depth_image       = depth_image,
            intrinsics        = self.intrinsics,
            depth_scale       = self.depth_scale,
            model_depth_image = model_depth_image,
        )

        if self.on_result is not None:
            self.on_result(result, canvas)

        if self.display:
            cv2.imshow("Collision Risk Monitor", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                return False

        return not self._stop_requested
