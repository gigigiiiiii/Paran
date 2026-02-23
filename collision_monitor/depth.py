import numpy as np


def _median_depth_in_rect(depth_image, x1, y1, x2, y2, depth_scale, min_valid=30):
    h, w = depth_image.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    valid = patch[patch > 0]
    if valid.size < min_valid:
        return None
    return float(np.median(valid) * depth_scale)


def _valid_depth_values_in_rect(depth_image, x1, y1, x2, y2, depth_scale):
    h, w = depth_image.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return valid.astype(np.float32) * float(depth_scale)


def depth_median_bottom_band(depth_image, bbox, depth_scale, band_ratio=0.22):
    x1, y1, x2, y2 = bbox
    band_ratio = float(np.clip(band_ratio, 0.05, 0.7))
    bh = max(1, int((y2 - y1) * band_ratio))
    by1 = max(y1, y2 - bh)
    return _median_depth_in_rect(depth_image, x1, by1, x2, y2, depth_scale, min_valid=30)


def depth_median_around_uv(depth_image, u, v, depth_scale, win=7):
    win = int(win)
    if win < 3:
        win = 3
    if win % 2 == 0:
        win += 1
    r = win // 2
    return _median_depth_in_rect(depth_image, u - r, v - r, u + r + 1, v + r + 1, depth_scale, min_valid=10)


def depth_near_around_uv(depth_image, u, v, depth_scale, win=7, percentile=30.0, min_valid=10):
    win = int(win)
    if win < 3:
        win = 3
    if win % 2 == 0:
        win += 1
    r = win // 2
    values = _valid_depth_values_in_rect(depth_image, u - r, v - r, u + r + 1, v + r + 1, depth_scale)
    if values is None or values.size < min_valid:
        return None
    percentile = float(np.clip(percentile, 1.0, 50.0))
    return float(np.percentile(values, percentile))


def collision_sample_depth(z_med, z_near, base_z, near_weight=0.25, max_offset=0.8):
    if z_med is None and z_near is None:
        return None
    near_weight = float(np.clip(near_weight, 0.0, 1.0))
    if z_med is None:
        z = z_near
    elif z_near is None:
        z = z_med
    else:
        z = (1.0 - near_weight) * z_med + near_weight * z_near

    if base_z is not None and max_offset > 0.0:
        base_z = float(base_z)
        max_offset = float(max_offset)
        if abs(float(z) - base_z) > max_offset:
            if z_med is not None and abs(float(z_med) - base_z) <= max_offset:
                z = z_med
            elif z_near is not None and abs(float(z_near) - base_z) <= max_offset:
                z = z_near
            else:
                return None
    return float(z)


def median_depth_from_bbox(depth_image, bbox, depth_scale):
    x1, y1, x2, y2 = bbox
    return _median_depth_in_rect(depth_image, x1, y1, x2, y2, depth_scale, min_valid=20)

