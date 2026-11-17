"""End to end: tracking in, one label per video frame out.

    ball control  --+
    team shape    --+--> team log-odds --+
                                         +--> emissions --> Viterbi per half --> labels
    dead-ball detector -> dead log-odds -+

All tunable numbers live in PARAMS, with the reasoning beside each one. None of them was fitted to
any ground truth. They come from physical scales or from the components' own self-supervised
calibration, and tools/sensitivity.py checks they sit on plateaus.
"""

import hashlib
import os

import numpy as np

from . import decoder, deadball, features, team_shape
from .data import DATA_DIR

PARAMS = {
    # Switching costs, in nats: a change has to be paid for by roughly this much accumulated
    # evidence. With ~1 nat per frame of clear ball control, a possession change needs about 1/3 s
    # of clear, sustained control by the other team, the "controlled, not a touch" threshold.
    "switch_cost": 8.0,
    # Entering or leaving a stoppage. Dead-ball evidence is smooth and long-lived, so a stoppage
    # needs ~2-3 s of clear dead evidence at dead_weight. tools/deadball_calibration.py shows the
    # dead stream alone gives a stable ~50% dead share and ~120 stoppages for weight 0.1-0.2 with
    # a cost of 5-10 nats.
    "to_dead_cost": 8.0,
    "from_dead_cost": 8.0,
    # Ball-track control evidence (features.ball_team_logodds).
    "control_gain": 1.0,
    "near_gain": 0.1,
    # Player-only team-shape log-odds (team_shape). It is calibrated per frame but heavily
    # autocorrelated, so it is down-weighted. More weight where the ball is unseen, where it is the
    # only team evidence; little where ball control already decides the boundaries.
    "shape_clip": 4.0,
    "shape_weight_seen": 0.03,
    "shape_weight_unseen": 0.08,
    # Dead-ball log-odds (deadball), clipped so formation/kick-off stacks cannot drown the rest.
    "dead_clip": 6.0,
    "dead_weight": 0.15,
}

CACHE_VERSION = 2


def _shape_cache_path(m):
    """Cache key: every input team_shape reads, the code that produces them, and library versions.

    The tracking arrays, slot teams, halves, attack directions and pitch; the source of the shape
    model and of the loader; numpy / scipy / scikit-learn versions. Anything else that changes
    the model output would be a code change in one of those two files.
    """
    import scipy
    import sklearn
    from . import data
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(m.players).tobytes())
    h.update(np.ascontiguousarray(m.ball).tobytes())
    h.update(np.ascontiguousarray(m.team).tobytes())
    h.update(repr((m.halves, sorted(m.home_defends_left.items()), m.pitch)).encode())
    h.update(repr((np.__version__, scipy.__version__, sklearn.__version__)).encode())
    for module in (team_shape, data):
        with open(module.__file__, "rb") as handle:
            h.update(handle.read())
    return os.path.join(DATA_DIR, "_cache", "shape_v{}_{}.npy".format(CACHE_VERSION,
                                                                     h.hexdigest()[:16]))


def shape_logodds(m, use_cache=True):
    """team_shape.team_shape_logodds, cached (it trains a model: ~2 min)."""
    path = _shape_cache_path(m)
    if use_cache and os.path.isfile(path):
        return np.load(path)
    z = team_shape.team_shape_logodds(m)
    if use_cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, z)
    return z


def evidence(m, params=None, cache=None):
    """Dict of per-frame log-odds streams, each a named, inspectable decoder input.

    `cache` (a dict) keeps the expensive, parameter-free components between calls.
    """
    p = dict(PARAMS, **(params or {}))
    cache = {} if cache is None else cache
    if "feats" not in cache:
        cache["feats"] = features.ball_features(m)
        cache["shape"] = shape_logodds(m)
        cache["dead"] = deadball.dead_ball_logodds(m)
    feats = cache["feats"]

    ball = features.ball_team_logodds(feats, p["control_gain"], p["near_gain"])
    shape_w = np.where(feats["seen"], p["shape_weight_seen"], p["shape_weight_unseen"])
    shape = shape_w * np.clip(cache["shape"], -p["shape_clip"], p["shape_clip"])
    dead = p["dead_weight"] * np.clip(cache["dead"], -p["dead_clip"], p["dead_clip"])
    return {"ball": ball, "shape": shape, "team": ball + shape, "dead": dead}


def run(m, params=None, cache=None):
    p = dict(PARAMS, **(params or {}))
    ev = evidence(m, p, cache)
    em = decoder.emissions(ev["team"], ev["dead"])
    cost = decoder.transition_costs(p["switch_cost"], p["to_dead_cost"], p["from_dead_cost"])
    return decoder.decode_halves(em, m.halves, cost, m.n_frames)


def summarize(labels, halves, fps=25):
    """Possession count, mean duration and label shares over the graded halves. No GT involved."""
    n_poss, durs, counts = 0, [], {"H": 0, "A": 0, "D": 0}
    n_dead = 0
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
            elif seg[s] == "D":
                n_dead += 1
    total = sum(counts.values())
    return {
        "possessions": n_poss,
        "stoppages": n_dead,
        "mean_possession_s": round(float(np.mean(durs)), 2) if durs else 0.0,
        "median_possession_s": round(float(np.median(durs)), 2) if durs else 0.0,
        "share": {k: round(v / total, 3) for k, v in counts.items()},
    }
