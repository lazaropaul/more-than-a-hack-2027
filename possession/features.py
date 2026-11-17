"""Per-frame evidence from the ball track, on the video-frame axis.

Everything here is a physical quantity with a physically chosen scale. Nothing is fitted to labels.

Conventions: positions in metres, speeds in m/s, 25 fps. Log-odds are natural logs, positive
meaning HOME (team evidence) or DEAD (dead-ball evidence).
"""

import numpy as np

FPS = 25

# Velocity by central difference over +/- VEL_HALF frames (0.16 s): long enough to average
# tracking jitter, short enough to keep a pass's first frames.
VEL_HALF = 2

# Control model. A player controls the ball when it is within reach and moving with him rather
# than past him. CONTROL_SIGMA is reach (a stride and a half); a ball passing a player faster than
# REL_SPEED_PASS relative to him is in flight, not under control.
CONTROL_SIGMA = 1.2      # m
REL_SPEED_PASS = 6.0     # m/s
REL_SPEED_SCALE = 2.0    # m/s, softness of that gate
# Weak "who is nearest" term for when nobody is in reach: a loose ball drifting toward one team.
NEAR_SCALE = 3.0         # m

# Out of play: the ball is past a line by more than the tracker's typical error.
OUT_MARGIN = 0.75        # m


def velocity(xy, half=VEL_HALF):
    """Central-difference velocity of an (n, ..., 2) track, NaN where either end is missing."""
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    v[half:-half] = (xy[2 * half:] - xy[:-2 * half]) / (2.0 * half / FPS)
    return v


def nearest_by_team(players, team, point):
    """For each frame: distance from `point` to the nearest player of each team, and their slots.

    Returns (dist (n, 2), slot (n, 2)) with columns [home, away]; inf / -1 where none is tracked.
    """
    d = np.hypot(players[:, :, 0] - point[:, None, 0], players[:, :, 1] - point[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    dist = np.full((len(point), 2), np.inf, dtype=np.float32)
    slot = np.full((len(point), 2), -1, dtype=np.int64)
    for side in (0, 1):
        cols = np.flatnonzero(team == side)
        if len(cols) == 0:
            continue
        sub = d[:, cols]
        best = sub.argmin(axis=1)
        dist[:, side] = sub[np.arange(len(sub)), best]
        slot[:, side] = np.where(np.isfinite(dist[:, side]), cols[best], -1)
    return dist, slot


def ball_features(m):
    """Dict of per-frame ball features. NaN / 0 where the ball is not seen."""
    ball = m.ball.astype(np.float32)
    seen = ~np.isnan(ball[:, 0])
    ball_v = velocity(ball)
    player_v = velocity(m.players)

    dist, slot = nearest_by_team(m.players, m.team, ball)

    # Ball speed relative to the nearest player of each team.
    rel = np.full((len(ball), 2), np.nan, dtype=np.float32)
    rows = np.arange(len(ball))
    for side in (0, 1):
        ok = slot[:, side] >= 0
        pv = np.full((len(ball), 2), np.nan, dtype=np.float32)
        pv[ok] = player_v[rows[ok], slot[ok, side]]
        rel[:, side] = np.hypot(*(ball_v - pv).T)

    length, width = m.pitch
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        out = seen & ((x < -OUT_MARGIN) | (x > length + OUT_MARGIN)
                      | (y < -OUT_MARGIN) | (y > width + OUT_MARGIN))
    return {
        "seen": seen,
        "ball_speed": np.hypot(ball_v[:, 0], ball_v[:, 1]),
        "dist": dist,
        "rel_speed": rel,
        "out": out,
    }


def control_strength(dist, rel_speed):
    """0..1 per team: how much this looks like a player of that team has the ball at his feet."""
    reach = np.exp(-0.5 * (dist / CONTROL_SIGMA) ** 2)
    # Unknown relative speed (a velocity end missing) is neither confirmation nor veto.
    rel = np.where(np.isnan(rel_speed), REL_SPEED_PASS, rel_speed)
    with_him = 1.0 / (1.0 + np.exp(np.clip((rel - REL_SPEED_PASS) / REL_SPEED_SCALE, -30, 30)))
    return reach * with_him


def ball_team_logodds(feats, control_gain=1.0, near_gain=0.1):
    """Per-frame log-odds that HOME has the ball, from the ball track alone. 0 where unseen.

    control_gain * (control_home - control_away): close, slow-relative contact. This is what
    drives possession changes.
    near_gain * tanh((d_away - d_home) / NEAR_SCALE): a weak tilt toward whoever is nearer, for
    loose balls nobody is touching yet.
    """
    ctrl = control_strength(feats["dist"], feats["rel_speed"])
    d = feats["dist"]
    with np.errstate(invalid="ignore"):
        gap = np.where(np.isfinite(d[:, 0]) & np.isfinite(d[:, 1]), d[:, 1] - d[:, 0], 0.0)
    lo = control_gain * (ctrl[:, 0] - ctrl[:, 1]) + near_gain * np.tanh(gap / NEAR_SCALE)
    lo = np.where(feats["seen"] & ~feats["out"], lo, 0.0)
    return np.nan_to_num(lo).astype(np.float64)
