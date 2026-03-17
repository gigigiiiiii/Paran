"""
collision_monitor/runner.py
============================
RealSense 카메라에서 RGB + Depth 프레임을 읽어 FrameProcessor에 전달한다.
탐지·트래킹·위험도 계산은 모두 FrameProcessor 에서 수행한다.
"""

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

        color_stream   = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.warmup_left = max(0, int(args.rs_warmup))

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

        depth_frame  = self.spatial.process(depth_frame)
        depth_frame  = self.temporal.process(depth_frame)
        color_image  = np.asanyarray(color_frame.get_data())
        depth_image  = np.asanyarray(depth_frame.get_data())

        canvas, result = self.processor.process(
            color_image  = color_image,
            depth_image  = depth_image,
            intrinsics   = self.intrinsics,
            depth_scale  = self.depth_scale,
        )

        if self.on_result is not None:
            self.on_result(result, canvas)

        if self.display:
            cv2.imshow("Collision Risk Monitor", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                return False

        return not self._stop_requested
