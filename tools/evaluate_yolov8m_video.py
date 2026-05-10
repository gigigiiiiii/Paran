from __future__ import annotations

import argparse
import csv
import json
import math
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


GT_CLASSES = ("person", "bicycle", "motorbike", "car", "truck")
OBSTACLE_CLASSES = {"bicycle", "motorbike", "car", "truck"}
YOLO_TO_GT = {
    "person": "person",
    "pedestrian": "person",
    "people": "person",
    "bicycle": "bicycle",
    "motorcycle": "motorbike",
    "motor": "motorbike",
    "tricycle": "motorbike",
    "awning-tricycle": "motorbike",
    "car": "car",
    "van": "car",
    "truck": "truck",
    "bus": "truck",
}


def parse_cvat_tracks(xml_path: Path) -> tuple[dict[int, list[dict]], dict[int, dict[int, dict]]]:
    root = ET.parse(xml_path).getroot()
    per_frame: dict[int, list[dict]] = defaultdict(list)
    tracks: dict[int, dict[int, dict]] = defaultdict(dict)
    for track in root.findall("track"):
        label = track.attrib.get("label", "")
        if label not in GT_CLASSES:
            continue
        track_id = int(track.attrib["id"])
        for box in track.findall("box"):
            frame = int(box.attrib["frame"])
            if box.attrib.get("outside") == "1":
                continue
            x1 = float(box.attrib["xtl"])
            y1 = float(box.attrib["ytl"])
            x2 = float(box.attrib["xbr"])
            y2 = float(box.attrib["ybr"])
            if x2 <= x1 or y2 <= y1:
                continue
            item = {
                "frame": frame,
                "track_id": track_id,
                "cls": label,
                "box": [x1, y1, x2, y2],
            }
            per_frame[frame].append(item)
            tracks[track_id][frame] = item
    return dict(per_frame), dict(tracks)


def iou_matrix(a: list[list[float]], b: list[list[float]]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    ix1 = np.maximum(aa[:, None, 0], bb[None, :, 0])
    iy1 = np.maximum(aa[:, None, 1], bb[None, :, 1])
    ix2 = np.minimum(aa[:, None, 2], bb[None, :, 2])
    iy2 = np.minimum(aa[:, None, 3], bb[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = np.maximum(0.0, aa[:, 2] - aa[:, 0]) * np.maximum(0.0, aa[:, 3] - aa[:, 1])
    area_b = np.maximum(0.0, bb[:, 2] - bb[:, 0]) * np.maximum(0.0, bb[:, 3] - bb[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def compute_detection_metrics(
    gt_by_frame: dict[int, list[dict]],
    preds_by_frame: dict[int, list[dict]],
    iou_thr: float,
    eval_conf: float,
):
    classes = list(GT_CLASSES)
    gt_count = {cls: 0 for cls in classes}
    pred_count = {cls: 0 for cls in classes}
    tp_fixed = {cls: 0 for cls in classes}
    fp_fixed = {cls: 0 for cls in classes}
    fn_fixed = {cls: 0 for cls in classes}
    ap_by_class = {}

    all_frames = sorted(set(gt_by_frame) | set(preds_by_frame))
    for cls in classes:
        records = []
        total_gt = 0
        for frame in all_frames:
            gt_boxes = [g["box"] for g in gt_by_frame.get(frame, []) if g["cls"] == cls]
            pred_items_all = [p for p in preds_by_frame.get(frame, []) if p["cls"] == cls]
            pred_items = [p for p in pred_items_all if p["conf"] >= eval_conf]
            total_gt += len(gt_boxes)
            pred_items_sorted = sorted(pred_items, key=lambda p: p["conf"], reverse=True)
            pred_count[cls] += len(pred_items_sorted)

            used_fixed: set[int] = set()
            mat = iou_matrix([p["box"] for p in pred_items_sorted], gt_boxes)
            for pi, pred in enumerate(pred_items_sorted):
                best_gi = -1
                best_iou = 0.0
                if gt_boxes:
                    for gi in range(len(gt_boxes)):
                        if gi in used_fixed:
                            continue
                        val = float(mat[pi, gi])
                        if val > best_iou:
                            best_iou = val
                            best_gi = gi
                is_tp = best_gi >= 0 and best_iou >= iou_thr
                if is_tp:
                    used_fixed.add(best_gi)
                    tp_fixed[cls] += 1
                else:
                    fp_fixed[cls] += 1
            fn_fixed[cls] += max(0, len(gt_boxes) - len(used_fixed))

            for pred in pred_items_all:
                records.append((float(pred["conf"]), frame, pred["box"]))

        gt_count[cls] = total_gt
        if total_gt == 0:
            ap_by_class[cls] = None
            continue

        # AP is computed from all retained predictions sorted globally by confidence.
        records.sort(key=lambda r: r[0], reverse=True)
        matched_by_frame: dict[int, set[int]] = defaultdict(set)
        tp_curve = []
        fp_curve = []
        for conf, frame, pred_box in records:
            gt_boxes = [g["box"] for g in gt_by_frame.get(frame, []) if g["cls"] == cls]
            mat_one = iou_matrix([pred_box], gt_boxes)
            best_gi = -1
            best_iou = 0.0
            for gi in range(len(gt_boxes)):
                if gi in matched_by_frame[frame]:
                    continue
                val = float(mat_one[0, gi])
                if val > best_iou:
                    best_iou = val
                    best_gi = gi
            if best_gi >= 0 and best_iou >= iou_thr:
                matched_by_frame[frame].add(best_gi)
                tp_curve.append(1.0)
                fp_curve.append(0.0)
            else:
                tp_curve.append(0.0)
                fp_curve.append(1.0)

        if not records:
            ap_by_class[cls] = 0.0
            continue
        tp_cum = np.cumsum(tp_curve)
        fp_cum = np.cumsum(fp_curve)
        recall = tp_cum / max(total_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap_by_class[cls] = voc_ap(recall, precision)

    total_tp = sum(tp_fixed.values())
    total_fp = sum(fp_fixed.values())
    total_fn = sum(fn_fixed.values())
    valid_aps = [ap for cls, ap in ap_by_class.items() if ap is not None and gt_count[cls] > 0]
    return {
        "iou_threshold": iou_thr,
        "precision_recall_conf_threshold": eval_conf,
        "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
        "recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
        "mAP@0.5": float(np.mean(valid_aps)) if valid_aps else 0.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_class": {
            cls: {
                "gt": gt_count[cls],
                "pred": pred_count[cls],
                "tp": tp_fixed[cls],
                "fp": fp_fixed[cls],
                "fn": fn_fixed[cls],
                "precision": tp_fixed[cls] / (tp_fixed[cls] + fp_fixed[cls]) if tp_fixed[cls] + fp_fixed[cls] else 0.0,
                "recall": tp_fixed[cls] / (tp_fixed[cls] + fn_fixed[cls]) if tp_fixed[cls] + fn_fixed[cls] else 0.0,
                "AP@0.5": ap_by_class[cls],
            }
            for cls in classes
        },
    }


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def run_yolo(video_path: Path, model_path: Path, imgsz: int, conf: float, device: str | None):
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    preds_by_frame: dict[int, list[dict]] = defaultdict(list)
    frame_count = 0
    inference_s = 0.0
    class_ids = [idx for idx, name in model.names.items() if name in YOLO_TO_GT]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        result = model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            iou=0.45,
            classes=class_ids,
            device=device,
            verbose=False,
        )[0]
        inference_s += time.perf_counter() - t0
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
            confs = boxes.conf.detach().cpu().numpy()
            for box, cls_id, score in zip(xyxy, cls_ids, confs):
                yolo_name = model.names[int(cls_id)]
                gt_cls = YOLO_TO_GT.get(yolo_name)
                if gt_cls is None:
                    continue
                preds_by_frame[frame_count].append({
                    "frame": frame_count,
                    "cls": gt_cls,
                    "conf": float(score),
                    "box": [float(v) for v in box.tolist()],
                })
        frame_count += 1
        if frame_count % 250 == 0:
            print(f"processed {frame_count} frames")
    cap.release()
    return dict(preds_by_frame), frame_count, inference_s


def box_center(box: list[float]) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)


def box_height(box: list[float]) -> float:
    return max(1.0, float(box[3] - box[1]))


def boxes_overlap(a: list[float], b: list[float]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def point_box_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def detect_risk_start_from_tracks(
    tracks: dict[int, dict[int, dict]],
    fps: float,
    horizon_s: float = 2.0,
    sustain_frames: int = 3,
    dist_factor: float = 0.5,
) -> dict:
    person_ids = [tid for tid, items in tracks.items() if next(iter(items.values()))["cls"] == "person"]
    obs_ids = [tid for tid, items in tracks.items() if next(iter(items.values()))["cls"] in OBSTACLE_CLASSES]
    max_frame = max((f for items in tracks.values() for f in items), default=-1)
    consecutive = 0
    first_candidate = None
    best = None
    for frame in range(max_frame + 1):
        risky_pairs = []
        for pid in person_ids:
            p_now = tracks[pid].get(frame)
            p_prev = tracks[pid].get(frame - 1)
            if p_now is None or p_prev is None:
                continue
            pc = box_center(p_now["box"])
            pv = (pc - box_center(p_prev["box"])) * fps
            for oid in obs_ids:
                o_now = tracks[oid].get(frame)
                o_prev = tracks[oid].get(frame - 1)
                if o_now is None or o_prev is None:
                    continue
                oc = box_center(o_now["box"])
                ov = (oc - box_center(o_prev["box"])) * fps
                rel_p = pc - oc
                rel_v = pv - ov
                denom = float(np.dot(rel_v, rel_v))
                if denom <= 1e-9:
                    t_cpa = 0.0
                else:
                    t_cpa = float(np.clip(-np.dot(rel_p, rel_v) / denom, 0.0, horizon_s))
                pred_p = pc + pv * t_cpa
                pred_o = oc + ov * t_cpa
                d_cpa = point_box_distance(pred_p, pred_o)
                threshold = box_height(p_now["box"]) * dist_factor
                overlap_now = boxes_overlap(p_now["box"], o_now["box"])
                is_risky = overlap_now or (t_cpa <= horizon_s and d_cpa <= threshold)
                if is_risky:
                    risky_pairs.append({
                        "person_track_id": pid,
                        "obstacle_track_id": oid,
                        "obstacle_class": o_now["cls"],
                        "t_cpa_s": t_cpa,
                        "d_cpa_px": d_cpa,
                        "threshold_px": threshold,
                        "overlap_now": overlap_now,
                    })
        if risky_pairs:
            if consecutive == 0:
                first_candidate = frame
            consecutive += 1
            if consecutive >= sustain_frames:
                best = min(risky_pairs, key=lambda r: r["d_cpa_px"])
                return {"frame": first_candidate, "time_s": first_candidate / fps, "pair": best}
        else:
            consecutive = 0
            first_candidate = None
    return {"frame": None, "time_s": None, "pair": None}


def build_pred_tracks_by_iou(preds_by_frame: dict[int, list[dict]], iou_thr: float = 0.3) -> dict[int, dict[int, dict]]:
    tracks: dict[int, dict[int, dict]] = {}
    active: dict[int, dict] = {}
    next_id = 1
    for frame in sorted(preds_by_frame):
        current = sorted(preds_by_frame.get(frame, []), key=lambda p: p["conf"], reverse=True)
        assigned_tracks = set()
        for pred in current:
            best_tid = None
            best_iou = 0.0
            for tid, prev in active.items():
                if tid in assigned_tracks or prev["cls"] != pred["cls"] or frame - prev["frame"] > 5:
                    continue
                val = float(iou_matrix([pred["box"]], [prev["box"]])[0, 0])
                if val > best_iou:
                    best_iou = val
                    best_tid = tid
            if best_tid is None or best_iou < iou_thr:
                best_tid = next_id
                next_id += 1
                tracks[best_tid] = {}
            item = dict(pred)
            item["track_id"] = best_tid
            tracks[best_tid][frame] = item
            active[best_tid] = item
            assigned_tracks.add(best_tid)
        for tid in list(active):
            if frame - active[tid]["frame"] > 5:
                active.pop(tid, None)
    return tracks


def save_predictions(path: Path, preds_by_frame: dict[int, list[dict]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "class", "confidence", "x1", "y1", "x2", "y2"])
        for frame in sorted(preds_by_frame):
            for p in preds_by_frame[frame]:
                writer.writerow([frame, p["cls"], f"{p['conf']:.8f}", *[f"{v:.3f}" for v in p["box"]]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=Path("annotations.xml"))
    parser.add_argument("--video", type=Path, default=Path("test_videos/KakaoTalk_20260509_190512638.mp4"))
    parser.add_argument("--model", type=Path, default=Path("yolov8m.pt"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--infer-conf", type=float, default=0.001)
    parser.add_argument("--eval-conf", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out", type=Path, default=Path("out_reports/yolov8m_eval"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    gt_by_frame, gt_tracks = parse_cvat_tracks(args.xml)
    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    preds_by_frame, processed_frames, inference_s = run_yolo(args.video, args.model, args.imgsz, args.infer_conf, args.device)
    metrics = compute_detection_metrics(gt_by_frame, preds_by_frame, 0.5, args.eval_conf)
    gt_risk = detect_risk_start_from_tracks(gt_tracks, fps=fps)
    risk_preds_by_frame = {
        frame: [p for p in items if p["conf"] >= args.eval_conf]
        for frame, items in preds_by_frame.items()
    }
    pred_tracks = build_pred_tracks_by_iou(risk_preds_by_frame)
    pred_risk = detect_risk_start_from_tracks(pred_tracks, fps=fps)
    lead_time = None
    if gt_risk["frame"] is not None and pred_risk["frame"] is not None:
        lead_time = (gt_risk["frame"] - pred_risk["frame"]) / fps

    result = {
        "video": str(args.video),
        "model": str(args.model),
        "frames_expected": expected_frames,
        "frames_processed": processed_frames,
        "source_fps": fps,
        "imgsz": args.imgsz,
        "inference_conf_threshold_for_ap": args.infer_conf,
        "confidence_threshold_for_precision_recall": args.eval_conf,
        "yolo_nms_iou": 0.45,
        "average_inference_fps": processed_frames / inference_s if inference_s > 0 else None,
        "inference_time_s": inference_s,
        "metrics": metrics,
        "risk_definition": {
            "horizon_s": 2.0,
            "distance_threshold": "predicted center distance <= person_box_height * 0.5",
            "sustain_frames": 3,
            "overlap_rule": "current person-obstacle box overlap is always risky",
            "obstacles": sorted(OBSTACLE_CLASSES),
        },
        "gt_risk_start": gt_risk,
        "yolo_risk_start": pred_risk,
        "warning_lead_time_s": lead_time,
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    save_predictions(args.out / "predictions.csv", preds_by_frame)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
