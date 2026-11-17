"""End to end: tracking in, one label per video frame out.

    evidence  ->  emissions  ->  Viterbi per half  ->  labels (X outside the graded halves)

All tunable numbers live in PARAMS, with the physical reasoning beside each one. None of them
were fitted to any ground truth.
"""

import numpy as np

from . import decoder, features

PARAMS = {
    # Switching costs, in nats. A possession change has to be paid for by roughly this much
    # accumulated evidence. With ~1 nat per frame of clear control that is ~1/3 s of clear,
    # sustained control by the other team, the "controlled, not a touch" threshold.
    "switch_cost": 8.0,
    # Entering or leaving a stoppage: one stoppage needs both. Higher than a switch because a
    # stoppage has its own strong evidence (ball out, play frozen).
    "to_dead_cost": 12.0,
    "from_dead_cost": 12.0,
    # Ball-track control evidence (features.ball_team_logodds).
    "control_gain": 1.0,
    "near_gain": 0.1,
    # Placeholder dead-ball evidence (replaced by possession.deadball when present).
    "out_gain": 2.0,               # per frame with the ball past a line
    "activity_center": 0.9,        # m/s mean player speed where alive and dead are equally likely
    "activity_scale": 0.2,         # m/s
    "activity_gain": 0.15,         # per frame, after clipping to +/-4
    "activity_window": 25,         # frames, half-width of the moving average
}


def moving_nanmean(x, half):
    """Centred moving mean over +/- half frames, ignoring NaN."""
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half)
    hi = np.minimum(len(x), idx + half + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)


def placeholder_dead_logodds(m, feats, p):
    """Simple dead-ball evidence: ball past a line, or the whole pitch walking."""
    speed = np.hypot(*np.moveaxis(features.velocity(m.players), -1, 0))
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_speed = np.nanmean(speed, axis=1)
    activity = moving_nanmean(mean_speed, p["activity_window"])
    lo = p["activity_gain"] * np.clip(
        (p["activity_center"] - np.nan_to_num(activity, nan=p["activity_center"]))
        / p["activity_scale"], -4, 4)
    lo = lo + p["out_gain"] * feats["out"]
    return lo


def evidence(m, params=None):
    """Dict of per-frame log-odds streams. Each is a named, inspectable input to the decoder."""
    p = dict(PARAMS, **(params or {}))
    feats = features.ball_features(m)
    return {
        "team": features.ball_team_logodds(feats, p["control_gain"], p["near_gain"]),
        "dead": placeholder_dead_logodds(m, feats, p),
    }, feats


def run(m, params=None):
    p = dict(PARAMS, **(params or {}))
    ev, _ = evidence(m, p)
    em = decoder.emissions(ev["team"], ev["dead"])
    cost = decoder.transition_costs(p["switch_cost"], p["to_dead_cost"], p["from_dead_cost"])
    return decoder.decode_halves(em, m.halves, cost, m.n_frames)


def summarize(labels, halves, fps=25):
    """Possession count, mean duration and label shares over the graded halves. No GT involved."""
    n_poss, durs, counts = 0, [], {"H": 0, "A": 0, "D": 0}
    for _, first, last in halves:
        seg = labels[first:last + 1]
        for k in counts:
            counts[k] += int(np.sum(seg == k))
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.r_[0, change]
        stops = np.r_[change, len(seg)]
        for s, e in zip(starts, stops):
            if seg[s] in ("H", "A"):
                n_poss += 1
                durs.append((e - s) / fps)
    total = sum(counts.values())
    return {
        "possessions": n_poss,
        "mean_possession_s": float(np.mean(durs)) if durs else 0.0,
        "median_possession_s": float(np.median(durs)) if durs else 0.0,
        "share": {k: round(v / total, 3) for k, v in counts.items()},
    }
