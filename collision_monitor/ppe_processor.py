"""
collision_monitor/ppe_processor.py
===================================
PPE 모델 추론, 결과 캐싱, 시각화를 담당하는 클래스.
FrameProcessor에서 분리.
"""
from __future__ import annotations

import cv2

from .config import CLASS_COLORS


class PPEProcessor:
    """
    PPE 모델 추론 + person track_id 기반 캐싱 + canvas 렌더링.

    frame_interval 프레임마다 한 번 추론하고, 나머지 프레임은 캐시를 재사용한다.
    """

    def __init__(self, model, frame_interval: int = 3):
        self.model = model
        self.class_names = getattr(model, "names", None) or model.model.names
        self._frame_interval = frame_interval
        self._frame_count    = 0
        self._cache: dict    = {}
        self._skip_classes       = {"none", "Person", "person"}
        self._violation_prefixes = ("no_",)
        # 클래스별 conf 임계값 (helmet은 낮게)
        self._conf_thresh = {
            "helmet":    0.25,
            "no_helmet": 0.25,
        }
        print(f"[PPEProcessor] 모델 로드 완료 | 클래스: {list(self.class_names.values())}")

    def process(
        self,
        canvas,
        color_image,
        people: list[dict],
        infer_device,
        use_half: bool,
        imgsz: int,
        default_conf: float,
    ):
        """PPE 추론 → 캐시 업데이트 → canvas에 결과 렌더링."""
        self._frame_count += 1
        run_ppe = (self._frame_count % self._frame_interval == 0)

        if run_ppe:
            ppe_results = self.model.predict(
                color_image,
                conf=0.25,
                imgsz=imgsz,
                verbose=False,
                device=infer_device,
                half=use_half,
            )[0]

            ppe_items = []
            for box in ppe_results.boxes:
                cls_id   = int(box.cls[0].item())
                ppe_name = (self.class_names.get(cls_id, str(cls_id))
                            if isinstance(self.class_names, dict)
                            else str(self.class_names[cls_id]))
                if ppe_name in self._skip_classes:
                    continue
                bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                ppe_conf = float(box.conf[0].item())
                if ppe_conf < self._conf_thresh.get(ppe_name, default_conf):
                    continue
                is_violation = ppe_name.startswith(self._violation_prefixes)
                box_color = CLASS_COLORS.get(
                    ppe_name, (60, 60, 220) if is_violation else (60, 200, 60)
                )
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), box_color, 2)
                cv2.putText(canvas, f"{ppe_name} {ppe_conf:.0%}",
                            (bx1, max(16, by1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
                ppe_items.append({
                    "name": ppe_name,
                    "conf": ppe_conf,
                    "cx":   (bx1 + bx2) // 2,
                    "cy":   (by1 + by2) // 2,
                })

            # 사람 bbox 기준으로 PPE 매핑 → 캐시 업데이트
            active_ids = {p.get("track_id") for p in people if p.get("track_id") is not None}
            for p in people:
                tid = p.get("track_id")
                if tid is None:
                    continue
                px1, py1, px2, py2 = p["bbox"]
                matched: dict[str, float] = {}
                for item in ppe_items:
                    if px1 <= item["cx"] <= px2 and py1 <= item["cy"] <= py2:
                        matched[item["name"]] = item["conf"]
                if matched:
                    self._cache[tid] = matched
            # 사라진 사람 캐시 정리
            for gone in list(self._cache):
                if gone not in active_ids:
                    del self._cache[gone]

        # 캐시된 PPE 결과를 사람 bbox 우측에 표시 (매 프레임)
        for p in people:
            tid = p.get("track_id")
            if tid is None:
                continue
            cached = self._cache.get(tid, {})
            if not cached:
                continue
            px1, py1, px2, py2 = p["bbox"]
            y_offset = py1
            for name, conf in cached.items():
                is_violation = name.startswith(self._violation_prefixes)
                box_color = CLASS_COLORS.get(
                    name, (60, 60, 220) if is_violation else (60, 200, 60)
                )
                mark = "[X]" if is_violation else "[O]"
                cv2.putText(canvas, f"{mark} {name} {conf:.0%}",
                            (px2 + 6, y_offset + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                y_offset += 22

        return canvas
