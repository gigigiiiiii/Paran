import numpy as np


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def risk_level(min_distance, ttc, warn_dist, danger_dist, warn_ttc, danger_ttc):
    dist_level = "SAFE"
    if min_distance is not None:
        if min_distance < danger_dist:
            dist_level = "DANGER"
        elif min_distance < warn_dist:
            dist_level = "WARNING"

    ttc_level = "SAFE"
    if ttc is not None:
        if ttc < danger_ttc:
            ttc_level = "DANGER"
        elif ttc < warn_ttc:
            ttc_level = "WARNING"

    rank = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
    return dist_level if rank[dist_level] >= rank[ttc_level] else ttc_level


def distance_score(min_distance, warn_dist, danger_dist):
    if min_distance is None:
        return 0.0
    warn_dist = float(warn_dist)
    danger_dist = float(danger_dist)
    if warn_dist <= danger_dist:
        return 1.0 if min_distance <= danger_dist else 0.0
    if min_distance <= danger_dist:
        return 1.0
    if min_distance >= warn_dist:
        return 0.0
    return _clip01((warn_dist - float(min_distance)) / (warn_dist - danger_dist))


def ttc_score(ttc, warn_ttc, danger_ttc):
    if ttc is None:
        return 0.0
    warn_ttc = float(warn_ttc)
    danger_ttc = float(danger_ttc)
    if warn_ttc <= danger_ttc:
        return 1.0 if ttc <= danger_ttc else 0.0
    if ttc <= danger_ttc:
        return 1.0
    if ttc >= warn_ttc:
        return 0.0
    return _clip01((warn_ttc - float(ttc)) / (warn_ttc - danger_ttc))


def closing_speed_score(closing_speed, ref_speed=1.2):
    if closing_speed is None or closing_speed <= 0.0:
        return 0.0
    ref = max(1e-3, float(ref_speed))
    return _clip01(float(closing_speed) / ref)


def compute_risk_score(
    min_distance,
    ttc,
    closing_speed,
    warn_dist,
    danger_dist,
    warn_ttc,
    danger_ttc,
    dist_weight=0.5,
    ttc_weight=0.35,
    close_weight=0.15,
    close_ref=1.2,
):
    s_dist = distance_score(min_distance, warn_dist=warn_dist, danger_dist=danger_dist)
    s_ttc = ttc_score(ttc, warn_ttc=warn_ttc, danger_ttc=danger_ttc)
    s_close = closing_speed_score(closing_speed, ref_speed=close_ref)

    wd = max(0.0, float(dist_weight))
    wt = max(0.0, float(ttc_weight))
    wc = max(0.0, float(close_weight))
    total = wd + wt + wc
    if total < 1e-6:
        wd, wt, wc = 0.5, 0.5, 0.0
    else:
        wd, wt, wc = wd / total, wt / total, wc / total

    score = _clip01(wd * s_dist + wt * s_ttc + wc * s_close)
    return score, {
        "distance": s_dist,
        "ttc": s_ttc,
        "closing": s_close,
        "w_dist": wd,
        "w_ttc": wt,
        "w_close": wc,
    }


def score_to_level(score, stable_level, warn_on=0.45, danger_on=0.75, warn_off=0.35, danger_off=0.65):
    score = _clip01(score)
    warn_on = _clip01(warn_on)
    danger_on = max(warn_on, _clip01(danger_on))
    warn_off = min(warn_on, _clip01(warn_off))
    danger_off = min(danger_on, max(warn_off, _clip01(danger_off)))

    if stable_level == "DANGER":
        if score >= danger_off:
            return "DANGER"
        if score >= warn_off:
            return "WARNING"
        return "SAFE"

    if stable_level == "WARNING":
        if score >= danger_on:
            return "DANGER"
        if score >= warn_off:
            return "WARNING"
        return "SAFE"

    if score >= danger_on:
        return "DANGER"
    if score >= warn_on:
        return "WARNING"
    return "SAFE"


def risk_color(level):
    if level == "DANGER":
        return (0, 0, 255)
    if level == "WARNING":
        return (0, 220, 255)
    return (0, 200, 0)


def ttc_los(person, obs):
    rel_vec = obs["point_3d"] - person["point_3d"]
    rel_dist = float(np.linalg.norm(rel_vec))
    if rel_dist < 1e-6:
        return None
    rel_dir = rel_vec / rel_dist

    p_vel = person["velocity"] if person["velocity"] is not None else np.zeros(3, dtype=np.float32)
    o_vel = obs["velocity"] if obs["velocity"] is not None else np.zeros(3, dtype=np.float32)
    rel_vel = o_vel - p_vel

    closing_speed = -float(np.dot(rel_vel, rel_dir))
    if closing_speed > 1e-3:
        return rel_dist / closing_speed
    return None


def closing_speed_los(person, obs):
    rel_vec = obs["point_3d"] - person["point_3d"]
    rel_dist = float(np.linalg.norm(rel_vec))
    if rel_dist < 1e-6:
        return None
    rel_dir = rel_vec / rel_dist

    p_vel = person["velocity"] if person["velocity"] is not None else np.zeros(3, dtype=np.float32)
    o_vel = obs["velocity"] if obs["velocity"] is not None else np.zeros(3, dtype=np.float32)
    rel_vel = o_vel - p_vel

    closing_speed = -float(np.dot(rel_vel, rel_dir))
    return closing_speed if closing_speed > 1e-3 else None


def ttc_forward(person, obs):
    f = person["velocity"]
    if f is None or float(np.linalg.norm(f)) < 1e-3:
        f = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    f_norm = float(np.linalg.norm(f))
    if f_norm < 1e-6:
        return None
    f_hat = f / f_norm

    rel = obs["point_3d"] - person["point_3d"]
    d_forward = float(np.dot(rel, f_hat))
    if d_forward <= 0.0:
        return None

    p_vel = person["velocity"] if person["velocity"] is not None else np.zeros(3, dtype=np.float32)
    o_vel = obs["velocity"] if obs["velocity"] is not None else np.zeros(3, dtype=np.float32)
    v_forward = float(np.dot(p_vel - o_vel, f_hat))
    if v_forward > 1e-3:
        return d_forward / v_forward
    return None
