import math

import numpy as np

from .depth import collision_sample_depth
from .depth import depth_median_around_uv
from .depth import depth_near_around_uv


def pixel_to_3d(u, v, z, intrinsics):
    fx = intrinsics.fx
    fy = intrinsics.fy
    ppx = intrinsics.ppx
    ppy = intrinsics.ppy
    x = (u - ppx) / fx * z
    y = (v - ppy) / fy * z
    return np.array([x, y, z], dtype=np.float32)


def angle_from_forward_vector(forward_vec, person_3d, target_3d):
    vec = target_3d - person_3d
    vec_norm = np.linalg.norm(vec)
    f_norm = np.linalg.norm(forward_vec)
    if vec_norm < 1e-6 or f_norm < 1e-6:
        forward_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        f_norm = 1.0
    cos_theta = float(np.dot(vec / vec_norm, forward_vec / f_norm))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def bbox_grid_points(bbox, grid_x=3, grid_y=3):
    x1, y1, x2, y2 = bbox
    grid_x = max(1, int(grid_x))
    grid_y = max(1, int(grid_y))
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    points = []
    used = set()
    for j in range(grid_y):
        fy = (j + 0.5) / grid_y
        v = int(y1 + fy * box_h)
        for i in range(grid_x):
            fx = (i + 0.5) / grid_x
            u = int(x1 + fx * box_w)
            key = (u, v)
            if key in used:
                continue
            used.add(key)
            points.append(key)
    return points


def build_collision_samples(
    depth_image,
    bbox,
    intrinsics,
    depth_scale,
    grid_x,
    grid_y,
    win,
    near_percentile,
    near_weight,
    base_z,
    sample_z_max_offset,
):
    samples = []
    for u, v in bbox_grid_points(bbox, grid_x=grid_x, grid_y=grid_y):
        z_med = depth_median_around_uv(depth_image, u, v, depth_scale, win=win)
        z_near = depth_near_around_uv(
            depth_image, u, v, depth_scale, win=win, percentile=near_percentile, min_valid=10
        )
        z = collision_sample_depth(
            z_med=z_med,
            z_near=z_near,
            base_z=base_z,
            near_weight=near_weight,
            max_offset=sample_z_max_offset,
        )
        if z is None:
            continue
        samples.append({
            "uv": (u, v),
            "z": z,
            "point_3d": pixel_to_3d(u, v, z, intrinsics),
        })
    return samples


def min_distance_between_items(person_item, obs_item, distance_percentile=20.0):
    p_samples = person_item.get("collision_samples") or [{"uv": None, "point_3d": person_item["point_3d"]}]
    o_samples = obs_item.get("collision_samples") or [{"uv": None, "point_3d": obs_item["point_3d"]}]

    pairs = []
    for p_s in p_samples:
        p_pt = p_s["point_3d"]
        for o_s in o_samples:
            dist = float(np.linalg.norm(p_pt - o_s["point_3d"]))
            pairs.append((dist, p_s, o_s))

    if not pairs:
        return None, None, None

    distance_percentile = float(np.clip(distance_percentile, 1.0, 50.0))
    dists = np.array([x[0] for x in pairs], dtype=np.float32)
    target = float(np.percentile(dists, distance_percentile))
    best_dist, best_p, best_o = min(pairs, key=lambda x: (abs(x[0] - target), x[0]))
    return float(best_dist), best_p, best_o

