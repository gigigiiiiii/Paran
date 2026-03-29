"""
collision_monitor/depth_model.py
================================
모노큘러 metric depth 추정 래퍼.

사용 모델: Depth Anything V2 Metric Indoor (DepthAnythingV2Wrapper)
    depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf  ← 현재 사용
    depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
    depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf

설치:
    pip install transformers torch torchvision pillow

사용법:
    model = DepthAnythingV2Wrapper()
    depth_image = model.infer(color_image)   # float32 (H, W), 단위 미터
    intrinsics = make_intrinsics_from_fov(W, H, hfov_deg=79.0)
    depth_scale = 1.0  # 이미 미터 단위
"""

from __future__ import annotations

import math

import cv2
import numpy as np


# ── Intrinsics 대체 객체 ──────────────────────────────────────────────────────

class FakeIntrinsics:
    """RealSense intrinsics 객체와 동일한 인터페이스를 제공하는 단순 대체 클래스."""

    def __init__(self, fx: float, fy: float, ppx: float, ppy: float):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy

    def __repr__(self) -> str:
        return (
            f"FakeIntrinsics(fx={self.fx:.1f}, fy={self.fy:.1f}, "
            f"ppx={self.ppx:.1f}, ppy={self.ppy:.1f})"
        )


def make_intrinsics_from_fov(
    width: int, height: int, hfov_deg: float = 79.0
) -> FakeIntrinsics:
    """
    영상 해상도와 수평 FOV(도)로 카메라 intrinsics를 근사 계산한다.

    일반 CCTV/웹캠의 전형적 수평 FOV:
      - 광각 (대부분 IP 카메라):  90~110°
      - 표준 (RealSense D435):    ~79°
      - 협각 (망원형):            40~60°

    Args:
        width:    영상 가로 픽셀
        height:   영상 세로 픽셀
        hfov_deg: 수평 화각(°). 기본값 79도 = RealSense D435 기준.

    Returns:
        FakeIntrinsics: pixel_to_3d() 함수와 호환되는 intrinsics 객체
    """
    hfov_rad = math.radians(float(hfov_deg))
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx  # 정사각 픽셀(square pixel) 가정
    ppx = width / 2.0
    ppy = height / 2.0
    return FakeIntrinsics(fx=fx, fy=fy, ppx=ppx, ppy=ppy)


# ── Depth Anything V2 래퍼 ────────────────────────────────────────────────────

class DepthAnythingV2Wrapper:
    """
    Depth Anything V2 Metric Indoor 모노큘러 depth 추정 래퍼.
    HuggingFace pipeline 사용.

    Args:
        model_id: HuggingFace 모델 ID
        hfov_deg: 미사용 (API 통일용)
    """

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        hfov_deg: float = 79.0,
    ):
        print(f"[DepthAnythingV2] 모델 로딩: {model_id}")
        try:
            from transformers import pipeline as hf_pipeline  # noqa: F401
        except ImportError:
            raise ImportError(
                "Depth Anything V2 사용을 위해 다음을 설치하세요:\n"
                "  pip install transformers torch torchvision pillow"
            )

        device = 0 if _has_cuda() else -1
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=model_id,
            device=device,
        )
        self._device_label = "cuda" if device == 0 else "cpu"
        print(f"[DepthAnythingV2] 로딩 완료 (device={self._device_label})")

    def infer(self, color_image: np.ndarray) -> np.ndarray:
        """
        단일 BGR 프레임에서 metric depth를 추정한다.

        Args:
            color_image: BGR uint8 numpy array (H, W, 3)

        Returns:
            depth: float32 numpy array (H, W), 단위 미터
        """
        from PIL import Image

        h, w = color_image.shape[:2]
        rgb = color_image[:, :, ::-1]
        pil_img = Image.fromarray(rgb.astype(np.uint8))

        result = self._pipe(pil_img)
        pred = result["predicted_depth"]  # torch.Tensor, 미터 단위
        pred = pred.squeeze()
        depth = pred.detach().cpu().float().numpy()

        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth.astype(np.float32)


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
