import numpy as np


def assign_tracks(items, prev_points, next_track_id, max_dist):
    used_ids = set()
    for item in items:
        if item.get("track_id") is not None:
            used_ids.add(int(item["track_id"]))

    used_prev_ids = set()
    for item in items:
        if item.get("track_id") is not None:
            continue
        best_id = None
        best_dist = max_dist
        for track_id, prev_point in prev_points.items():
            if track_id in used_prev_ids or track_id in used_ids:
                continue
            dist = float(np.linalg.norm(item["point_3d"] - prev_point))
            if dist < best_dist:
                best_dist = dist
                best_id = track_id
        if best_id is None:
            best_id = next_track_id
            next_track_id += 1
        used_prev_ids.add(best_id)
        used_ids.add(best_id)
        item["track_id"] = best_id

    curr_points = {int(item["track_id"]): item["point_3d"]
                   for item in items if item.get("track_id") is not None}
    return curr_points, next_track_id

