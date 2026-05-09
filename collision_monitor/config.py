import argparse


PERSON_CLASS_NAME = "person"
# VisDrone 모델 클래스명 (pedestrian, people → person으로 통합)
PERSON_CLASS_ALIASES = {"person", "Person", "pedestrian", "people"}

# yolo26m.pt (COCO 80클래스) 기준 장애물 클래스
OBSTACLE_CLASSES_DEFAULT = {
    "car", "motorcycle", "bicycle", "bus", "truck", "train",
}

FIXED_CLASSES_DEFAULT: set = set()

# 클래스별 바운딩 박스 색상 (BGR) — yolo26m.pt (COCO) 기준
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    # 사람
    "person":     ( 50, 220,  50),   # 초록
    # 이동 장비 (COCO)
    "car":        (  0, 165, 255),   # 주황
    "truck":      (  0, 100, 255),   # 주황-빨강
    "bus":        (  0,  60, 255),   # 빨강
    "motorcycle": ( 50, 200, 255),   # 노랑-주황
    "bicycle":    (255, 200,  50),   # 하늘-노랑
    "train":      (200,  50, 200),   # 보라
    # 기타
    "none":       (160, 160, 160),   # 회색
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="RealSense + YOLO based collision risk detection and visualization."
    )
    parser.add_argument("--model", type=str, default="yolo26m.pt", help="YOLO model path")
    parser.add_argument("--ppe-model", type=str, default="", dest="ppe_model", help="PPE detection model path (optional, runs in parallel)")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--rs-timeout-ms", type=int, default=10000, help="RealSense wait_for_frames timeout in ms")
    parser.add_argument("--rs-warmup", type=int, default=15, help="Number of initial frames to skip for sensor warmup")

    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--warn-dist", type=float, default=1.5, help="Warning distance in meters")
    parser.add_argument("--danger-dist", type=float, default=0.9, help="Danger distance in meters")
    parser.add_argument("--warn-ttc", type=float, default=2.0, help="Warning TTC in seconds")
    parser.add_argument("--danger-ttc", type=float, default=1.0, help="Danger TTC in seconds")
    parser.add_argument("--front-angle", type=float, default=70.0, help="Forward cone angle (deg)")
    parser.add_argument("--history-size", type=int, default=90, help="Distance history length")

    parser.add_argument("--obstacle-classes", type=str, default="", help="Comma separated obstacle class names")
    parser.add_argument("--fixed-classes", type=str, default="", help="Comma separated fixed obstacle class names")
    parser.add_argument("--all-non-person", action="store_true", help="Treat all non-person classes as obstacles.")

    parser.add_argument("--use-sahi", action="store_true", dest="use_sahi",
                        help="Use SAHI sliced inference for small object detection")
    parser.add_argument("--sahi-slice-size", type=int, default=320, dest="sahi_slice_size",
                        help="Slice size (px) for SAHI inference")
    parser.add_argument("--sahi-overlap", type=float, default=0.2, dest="sahi_overlap",
                        help="Overlap ratio between SAHI slices")

    parser.add_argument("--min-obstacle-area-ratio", type=float, default=0.01, help="Ignore small obstacle boxes (ratio)")
    parser.add_argument("--min-obstacle-size-m", type=float, default=0.35, help="Ignore obstacles smaller than this (m)")
    parser.add_argument("--vehicle-box-expand", type=float, default=1.0, dest="vehicle_box_expand",
                        help="Expand vehicle-class boxes for collision geometry when detections cover only parts like wheels. 1.0 disables.")
    parser.add_argument("--vehicle-box-expand-x", type=float, default=0.25, dest="vehicle_box_expand_x",
                        help="Horizontal vehicle bbox expansion ratio per side when vehicle-box-expand > 1.0")

    parser.add_argument("--lock-frames", type=int, default=10, help="Keep nearest pair lock for N frames")
    parser.add_argument("--lock-max-dist", type=float, default=0.6, help="Max per-object displacement (m) to keep lock")
    parser.add_argument("--match-dist", type=float, default=0.7, help="3D dist threshold (m) for ID matching")

    parser.add_argument("--beep", action="store_true", help="Enable warning sound")
    parser.add_argument("--beep-cooldown", type=float, default=0.8, help="Min seconds between beeps")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Disable all drawing overlays on output frame (raw camera only)")
    parser.add_argument("--hide-hud-panel", action="store_true",
                        help="Hide bottom HUD panel/graph while keeping detection boxes")

    parser.add_argument("--log-file", type=str, default="collision_log.csv", help="CSV log output path. Empty disables.")
    parser.add_argument("--use-yolo-track", action="store_true", help="Use YOLO built-in tracker (ByteTrack) for IDs.")
    parser.add_argument("--tracker", type=str, default="botsort_reid_config.yaml", help="Tracker config (e.g., botsort_reid_config.yaml).")
    parser.add_argument("--trail-len", type=int, default=30, help="Trail (중심점 궤적) 유지 최대 프레임 수")

    parser.add_argument("--vel-alpha", type=float, default=0.8, help="EMA alpha for velocity smoothing (0~1)")
    parser.add_argument("--depth-bottom-band", type=float, default=0.22, help="Bottom band ratio for depth median (0~1)")
    parser.add_argument("--depth-uv-win", type=int, default=7, help="Odd window size for (u,v) local depth median")
    parser.add_argument("--person-grid-x", type=int, default=3, help="Collision sampling grid X for person bbox")
    parser.add_argument("--person-grid-y", type=int, default=4, help="Collision sampling grid Y for person bbox")
    parser.add_argument("--obstacle-grid-x", type=int, default=3, help="Collision sampling grid X for obstacle bbox")
    parser.add_argument("--obstacle-grid-y", type=int, default=3, help="Collision sampling grid Y for obstacle bbox")
    parser.add_argument("--depth-near-percentile", type=float, default=30.0,
                        help="Near depth percentile (1~50) used for collision samples")
    parser.add_argument("--depth-near-weight", type=float, default=0.25,
                        help="Blend weight of near depth percentile against median depth (0~1)")
    parser.add_argument("--sample-z-max-offset", type=float, default=0.8,
                        help="Reject collision samples if depth differs from object base depth by more than this (m)")
    parser.add_argument("--depth-fusion", type=str, default="fallback",
                        choices=["realsense", "fallback", "fused"],
                        help="Collision depth source: realsense / fallback(model only when RealSense is invalid) / fused")
    parser.add_argument("--model-depth-weight", type=float, default=0.35, dest="model_depth_weight",
                        help="Model depth blend weight when --depth-fusion=fused (0~1)")
    parser.add_argument("--pair-distance-percentile", type=float, default=20.0,
                        help="Robust percentile (1~50) for person-obstacle sample distance")
    parser.add_argument("--proximity-gate", type=float, default=4.0, dest="proximity_gate",
                        help="센터-투-센터 3D 거리가 이 값(m) 초과인 쌍은 거리 계산·연결선 생략. 0=비활성")
    parser.add_argument("--line-max-dist", type=float, default=0.0, dest="line_max_dist",
                        help="이 거리(m) 이하인 쌍만 연결선을 표시. 0=계산된 모든 쌍 표시")
    parser.add_argument("--line-smooth-alpha", type=float, default=0.75, dest="line_smooth_alpha",
                        help="연결선 끝점 스무딩 EMA 계수(0~0.99). 높을수록 덜 흔들리고 반응은 느림")
    parser.add_argument("--display-distance-smooth-alpha", type=float, default=0.85, dest="display_distance_smooth_alpha",
                        help="Visual-only EMA alpha for distance labels on pair lines")
    parser.add_argument("--display-distance-step", type=float, default=0.05, dest="display_distance_step",
                        help="Visual-only distance label rounding step in meters. 0 disables")
    parser.add_argument("--distance-smooth-alpha", type=float, default=0.55, dest="distance_smooth_alpha",
                        help="Pair distance EMA alpha used for risk calculation (0 disables)")
    parser.add_argument("--receding-speed-threshold", type=float, default=0.15, dest="receding_speed_threshold",
                        help="Treat pairs separating faster than this m/s as receding")
    parser.add_argument("--receding-risk-scale", type=float, default=0.65, dest="receding_risk_scale",
                        help="Risk score multiplier for receding pairs outside danger distance")
    parser.add_argument("--confidence-risk-floor", type=float, default=0.65, dest="confidence_risk_floor",
                        help="Lowest multiplier applied when pair confidence is weak (0~1)")
    parser.add_argument("--risk-up-frames", type=int, default=2,
                        help="Consecutive frames required to raise risk level")
    parser.add_argument("--risk-down-frames", type=int, default=4,
                        help="Consecutive frames required to lower risk level")
    parser.add_argument("--score-alpha", type=float, default=0.85,
                        help="EMA alpha for risk score smoothing (0~1)")
    parser.add_argument("--score-dist-weight", type=float, default=0.35,
                        help="Risk score weight for distance component")
    parser.add_argument("--score-ttc-weight", type=float, default=0.20,
                        help="Risk score weight for TTC component")
    parser.add_argument("--score-close-weight", type=float, default=0.10,
                        help="Risk score weight for relative closing speed component")
    parser.add_argument("--score-cpa-weight", type=float, default=0.35,
                        help="Risk score weight for closest-point-of-approach prediction")
    parser.add_argument("--score-close-ref", type=float, default=1.2,
                        help="Closing speed (m/s) that maps close-score to 1.0")
    parser.add_argument("--score-warn-on", type=float, default=0.40,
                        help="Score threshold to enter WARNING from SAFE")
    parser.add_argument("--score-danger-on", type=float, default=0.58,
                        help="Score threshold to enter DANGER")
    parser.add_argument("--score-warn-off", type=float, default=0.30,
                        help="Score threshold to leave WARNING")
    parser.add_argument("--score-danger-off", type=float, default=0.48,
                        help="Score threshold to leave DANGER")
    parser.add_argument(
        "--ttc-mode",
        type=str,
        default="forward",
        choices=["forward", "los", "both_min"],
        help="TTC calculation mode: forward / los / both_min",
    )
    parser.add_argument("--prediction-horizon", type=float, default=3.0,
                        help="CPA prediction horizon in seconds")
    parser.add_argument("--person-radius", type=float, default=0.35,
                        help="Person collision radius in meters for CPA")
    parser.add_argument("--obstacle-radius-min", type=float, default=0.35,
                        help="Minimum obstacle collision radius in meters for CPA")
    parser.add_argument("--safety-margin", type=float, default=0.30,
                        help="Additional safety margin in meters for CPA")
    parser.add_argument("--track-velocity-window", type=int, default=8,
                        help="Recent per-track positions used for regression velocity")

    # ── 모노큘러 Depth 모델 (RealSense 없이 depth 추정) ─────────────────────
    parser.add_argument(
        "--depth-compare-interval",
        type=float,
        default=2.0,
        dest="depth_compare_interval",
        help="RealSense 비교용 depth 모델 추론 간격(초). 기본 2.0초.",
    )
    parser.add_argument(
        "--model-depth-interval",
        type=float,
        default=0.2,
        dest="model_depth_interval",
        help="RealSense 없이 모델 depth를 실제 거리로 쓸 때 추론 간격(초). 0=가능한 매 프레임",
    )
    parser.add_argument(
        "--async-video-depth",
        action="store_false",
        dest="video_depth_sync",
        help=(
            "영상/웹캠 테스트 모드에서 depth 모델을 백그라운드 비동기로 실행한다. "
            "기본값은 정확도를 위해 현재 RGB 프레임과 같은 프레임의 depth를 동기 추론한다."
        ),
    )
    parser.add_argument(
        "--sync-video-depth",
        action="store_true",
        dest="video_depth_sync",
        help="Run video/webcam depth inference synchronously on every processed frame for RGB-depth alignment.",
    )
    parser.set_defaults(video_depth_sync=False)
    parser.add_argument(
        "--depth-model",
        type=str,
        default="none",
        dest="depth_model",
        help=(
            "모노큘러 metric depth 모델 ID. 'none'이면 비활성화. "
            "예: depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        ),
    )
    parser.add_argument(
        "--depth-fov",
        type=float,
        default=79.0,
        dest="depth_fov",
        help="영상 수평 화각(°). depth_model 사용 시 intrinsics 계산에 사용. 기본값 79.0(RealSense D435 기준)",
    )

    return parser.parse_args(argv)
