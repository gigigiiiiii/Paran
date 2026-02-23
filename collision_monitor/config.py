import argparse


PERSON_CLASS_NAME = "person"
OBSTACLE_CLASSES_DEFAULT = {
    "chair", "couch", "bed", "dining table", "bench", "tv", "refrigerator",
    "oven", "microwave", "sink", "toilet", "car", "motorcycle", "bicycle",
    "truck", "bus",
}

FIXED_CLASSES_DEFAULT = {
    "chair", "couch", "bed", "dining table", "bench", "tv", "refrigerator",
    "oven", "microwave", "sink", "toilet",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="RealSense + YOLO based collision risk detection and visualization."
    )
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="YOLO model path")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--rs-timeout-ms", type=int, default=10000, help="RealSense wait_for_frames timeout in ms")
    parser.add_argument("--rs-warmup", type=int, default=15, help="Number of initial frames to skip for sensor warmup")

    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--warn-dist", type=float, default=1.5, help="Warning distance in meters")
    parser.add_argument("--danger-dist", type=float, default=0.9, help="Danger distance in meters")
    parser.add_argument("--warn-ttc", type=float, default=2.0, help="Warning TTC in seconds")
    parser.add_argument("--danger-ttc", type=float, default=1.0, help="Danger TTC in seconds")
    parser.add_argument("--front-angle", type=float, default=70.0, help="Forward cone angle (deg)")
    parser.add_argument("--history-size", type=int, default=90, help="Distance history length")

    parser.add_argument("--obstacle-classes", type=str, default="", help="Comma separated obstacle class names")
    parser.add_argument("--fixed-classes", type=str, default="", help="Comma separated fixed obstacle class names")
    parser.add_argument("--all-non-person", action="store_true", help="Treat all non-person classes as obstacles.")

    parser.add_argument("--min-obstacle-area-ratio", type=float, default=0.01, help="Ignore small obstacle boxes (ratio)")
    parser.add_argument("--min-obstacle-size-m", type=float, default=0.35, help="Ignore obstacles smaller than this (m)")

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
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config (e.g., bytetrack.yaml).")

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
    parser.add_argument("--pair-distance-percentile", type=float, default=20.0,
                        help="Robust percentile (1~50) for person-obstacle sample distance")
    parser.add_argument("--risk-up-frames", type=int, default=2,
                        help="Consecutive frames required to raise risk level")
    parser.add_argument("--risk-down-frames", type=int, default=4,
                        help="Consecutive frames required to lower risk level")
    parser.add_argument("--score-alpha", type=float, default=0.85,
                        help="EMA alpha for risk score smoothing (0~1)")
    parser.add_argument("--score-dist-weight", type=float, default=0.5,
                        help="Risk score weight for distance component")
    parser.add_argument("--score-ttc-weight", type=float, default=0.35,
                        help="Risk score weight for TTC component")
    parser.add_argument("--score-close-weight", type=float, default=0.15,
                        help="Risk score weight for relative closing speed component")
    parser.add_argument("--score-close-ref", type=float, default=1.2,
                        help="Closing speed (m/s) that maps close-score to 1.0")
    parser.add_argument("--score-warn-on", type=float, default=0.45,
                        help="Score threshold to enter WARNING from SAFE")
    parser.add_argument("--score-danger-on", type=float, default=0.75,
                        help="Score threshold to enter DANGER")
    parser.add_argument("--score-warn-off", type=float, default=0.35,
                        help="Score threshold to leave WARNING")
    parser.add_argument("--score-danger-off", type=float, default=0.65,
                        help="Score threshold to leave DANGER")
    parser.add_argument(
        "--ttc-mode",
        type=str,
        default="forward",
        choices=["forward", "los", "both_min"],
        help="TTC calculation mode: forward / los / both_min",
    )

    return parser.parse_args(argv)
