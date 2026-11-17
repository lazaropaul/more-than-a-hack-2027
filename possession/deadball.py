"""Dead-ball detector: per-frame log-odds that the ball is NOT in play.

Self-contained (numpy + scipy). Entry point:

    from possession.deadball import dead_ball_logodds
    lo = dead_ball_logodds(m)          # (n_frames,) float64, log-odds of DEAD, finite everywhere

`dead_ball_components(m)` returns the individual evidence terms (all in nats, + = dead) so a
decoder can re-weight them; `dead_ball_logodds` is their sum (the ball_context term carries the prior).

Every scale below is a physical one (metres, seconds, m/s) or a football prior; nothing refers to
specific frames of any match. 25 fps is read from FPS.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d

FPS = 25.0
HOME = 0

# ---------------------------------------------------------------------------------------------
# parameters (physical / football scales)
# ---------------------------------------------------------------------------------------------
P = dict(
    prior_dead=0.45,          # base rate where no context applies (football norm 40-50%)
    vel_half_window=5,        # frames each side for finite-difference speeds (0.4 s baseline)
    max_player_speed=11.0,    # m/s; faster = tracklet identity switch, discarded
    max_ball_speed=40.0,      # m/s; faster = detection teleport, discarded
    act_window=2.0,           # s, centred smoothing of collective activity
    # activity LLR: logistic in "top-5 mean player speed" (m/s); balanced anchor fit gave
    # slope 1.45 /(m/s), centre 3.17 m/s
    act_centre=3.15,
    act_slope=1.5,
    act_cap=2.5,
    # ball seen in bounds: context log-odds as a function of ball speed (m/s), piecewise linear,
    # from the self-supervised mixture estimate (shrunk towards 0, see notes)
    ctx_speed_knots=(0.4, 1.4, 2.75, 4.25, 6.5),
    ctx_speed_logit=(1.0, -0.2, -1.4, -2.3, -2.8),
    ball_speed_window=0.6,    # s smoothing of ball speed for the context
    # ball out of bounds
    out_margin=0.5,           # m beyond the line before 'out' counts
    out_full=2.0,             # m beyond the line for full weight
    out_far=12.0,             # m: beyond this a detection is mostly a false positive (no evidence)
    out_logit=3.0,            # context log-odds at full weight
    out_carry_weight=1.5,     # extra nats while the ball stays unseen after being seen out
    # stationary ball (set-piece placement)
    stat_speed=0.8,           # m/s
    stat_min=1.0,             # s before stationarity counts
    stat_weight=0.5,          # extra nats for a stationary run > stat_min (on top of context)
    stat_restart_bonus=1.0,   # extra nats when it sits at a restart location
    # ball-missing gaps: context log-odds = gap_lo + (gap_hi-gap_lo)*(1-exp(-(G-gap_t0)/gap_tau))
    gap_lo=-0.4, gap_hi=3.0, gap_t0=3.0, gap_tau=10.0,
    gap_edge=3.0,             # s; the first seconds of a gap are less certain ...
    gap_edge_pen=1.0,         # ... by up to this many nats
    # kickoff formation
    ko_min_players=7,         # per team
    ko_tol=1.0,               # m a player may stand over the halfway line
    ko_max_viol=1,            # players per team allowed on the wrong side (the kicker)
    ko_min_dur=1.0,           # s
    ko_weight=4.0,            # nats inside a formation run
    ko_merge=4.0,             # s; formation runs closer than this are merged
    prekick_search=8.0,       # minutes into a graded half to look for the opening kickoff
    prekick_weight=3.0,       # nats from half start to the opening kickoff
    # tracked-player count drop
    cnt_ref=20.0,             # players
    cnt_slope=0.15,           # nats per missing player below cnt_ref
    cnt_cap=0.7,
)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _nsmooth(x, w):
    """Centred moving average of x over w frames ignoring NaN; NaN where <20% support."""
    w = max(1, int(round(w)))
    ok = ~np.isnan(x)
    s = uniform_filter1d(np.where(ok, x, 0.0), w, mode="nearest")
    c = uniform_filter1d(ok.astype(np.float64), w, mode="nearest")
    return np.where(c > 0.2, s / np.maximum(c, 1e-9), np.nan)


def _runs(mask):
    """(starts, ends) of True runs, end exclusive."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _run_length(mask):
    """Per frame: length of the True run it belongs to (0 where False)."""
    out = np.zeros(len(mask), dtype=np.int64)
    s, e = _runs(mask)
    for a, b in zip(s, e):          # loop over runs, not frames
        out[a:b] = b - a
    return out


def _fill_short_false(mask, max_len):
    """Set False runs of length <= max_len that are bounded by True on both sides to True."""
    s, e = _runs(~mask)
    out = mask.copy()
    for a, b in zip(s, e):
        if a > 0 and b < len(mask) and b - a <= max_len:
            out[a:b] = True
    return out


def _speed(xy, k):
    """Centred finite-difference speed over +-k frames, NaN at the edges / missing."""
    v = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    v[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=-1) / (2 * k / FPS)
    return v


def _ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------
def player_activity(m, p=P):
    """Smoothed 'top-5 mean speed' of tracked players (m/s), mean speed, tracked count."""
    k = p["vel_half_window"]
    V = _speed(m.players, k)
    V[V > p["max_player_speed"]] = np.nan
    cnt = (~np.isnan(V)).sum(1)
    Vs = np.sort(np.where(np.isnan(V), -1.0, V), axis=1)
    top5 = Vs[:, -5:].mean(1).astype(np.float64)
    top5[cnt < 5] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        ms = np.where(cnt > 0, np.nansum(np.where(np.isnan(V), 0, V), 1) / np.maximum(cnt, 1), np.nan)
    w = p["act_window"] * FPS
    ntrack = (~np.isnan(m.players[..., 0])).sum(1).astype(np.float64)
    return _nsmooth(top5, w), _nsmooth(ms, w), _nsmooth(ntrack, w)


def ball_features(m, p=P):
    L, W = m.pitch
    B = m.ball.astype(np.float64)
    n = len(B)
    k = p["vel_half_window"]
    seen = ~np.isnan(B[:, 0])
    bv = _speed(B, k).astype(np.float64)
    # single-frame teleports: displacement per frame implies > max_ball_speed
    step = np.full(n, np.nan)
    step[1:] = np.linalg.norm(B[1:] - B[:-1], axis=1) * FPS
    tele = np.zeros(n, bool)
    tele[1:] |= step[1:] > p["max_ball_speed"]
    tele[:-1] |= step[1:] > p["max_ball_speed"]
    tele = maximum_filter1d(tele.astype(np.int8), 2 * k + 1).astype(bool)
    bv[tele | (bv > p["max_ball_speed"])] = np.nan
    dout = np.maximum.reduce([-B[:, 0], B[:, 0] - L, -B[:, 1], B[:, 1] - W])  # >0 outside
    return dict(B=B, seen=seen, bv=bv, dout=dout, tele=tele)


def restart_location(B, L, W):
    """Ball at a restart spot: touchline, corner, goal area, centre spot, penalty spot."""
    x, y = B[:, 0], B[:, 1]
    with np.errstate(invalid="ignore"):
        touch = (np.abs(y) < 1.5) | (np.abs(y - W) < 1.5)
        corner = (np.minimum(np.abs(x), np.abs(x - L)) < 2.5) & (np.minimum(np.abs(y), np.abs(y - W)) < 2.5)
        goal_area = (np.minimum(x, L - x) < 6.5) & (np.abs(y - W / 2) < 10.2) & (np.minimum(x, L - x) > -0.5)
        centre = np.hypot(x - L / 2, y - W / 2) < 2.0
        pen = (np.hypot(x - 11.0, y - W / 2) < 1.5) | (np.hypot(x - (L - 11.0), y - W / 2) < 1.5)
    return touch | corner | goal_area | centre | pen


def kickoff_formation(m, p=P):
    """Per frame: both teams in their own halves (orientation from m.home_defends_left)."""
    L = m.pitch[0]
    c = L / 2
    X = m.players[..., 0]
    home = m.team == HOME
    hx, ax = X[:, home], X[:, ~home]
    nh = (~np.isnan(hx)).sum(1)
    na = (~np.isnan(ax)).sum(1)
    tol = p["ko_tol"]
    with np.errstate(invalid="ignore"):
        h_right = (hx > c + tol).sum(1)   # home players right of the line
        h_left = (hx < c - tol).sum(1)
        a_right = (ax > c + tol).sum(1)
        a_left = (ax < c - tol).sum(1)
    half = m.half_mask()
    hdl = np.ones(len(X), bool)
    for h, v in m.home_defends_left.items():
        hdl[half == h] = v
    # outside graded halves accept either orientation
    viol_l = np.maximum(h_right, a_left)        # home on the left
    viol_r = np.maximum(h_left, a_right)
    viol = np.where(half == 0, np.minimum(viol_l, viol_r), np.where(hdl, viol_l, viol_r))
    ok = (nh >= p["ko_min_players"]) & (na >= p["ko_min_players"]) & (viol <= p["ko_max_viol"])
    # require persistence
    s, e = _runs(ok)
    form = np.zeros(len(X), bool)
    for a, b in zip(s, e):
        if b - a >= p["ko_min_dur"] * FPS:
            form[a:b] = True
    return form


def prekick_mask(m, form, p=P):
    """From the start of each graded half to the end of its opening kickoff formation.

    The opening kickoff is the first merged formation run that starts within prekick_search
    minutes of the half (or ends just after the half's first frame, when the formation sits
    before the graded block). No formation found -> nothing marked.
    """
    n = m.n_frames
    out = np.zeros(n, bool)
    merged = _fill_short_false(form, int(p["ko_merge"] * FPS))
    s, e = _runs(merged)
    for half, first, last in m.halves:
        lim = first + int(p["prekick_search"] * 60 * FPS)
        cand = [(a, b) for a, b in zip(s, e) if b > first and a < lim and b <= last]
        if cand:
            a, b = cand[0]
            out[first:b] = True
    return out


# ---------------------------------------------------------------------------------------------
# evidence terms
# ---------------------------------------------------------------------------------------------
def dead_ball_components(m, p=P):
    n = m.n_frames
    L, W = m.pitch
    comp = {}
    top5, ms, ntrack = player_activity(m, p)
    bf = ball_features(m, p)
    B, seen, bv, dout, tele = bf["B"], bf["seen"], bf["bv"], bf["dout"], bf["tele"]

    # 1. collective activity: players walk during stoppages
    a = -p["act_slope"] * (top5 - p["act_centre"])
    comp["activity"] = np.clip(np.nan_to_num(a, nan=0.0), -p["act_cap"], p["act_cap"])

    # 2. context log-odds from the ball's state (replaces a flat prior on every frame):
    #    seen in bounds -> f(ball speed); seen out -> out_logit; missing -> f(gap length);
    #    teleporting / far-out detections -> flat prior.
    prior = np.log(p["prior_dead"] / (1 - p["prior_dead"]))
    ctx = np.full(n, prior)
    bvs = _nsmooth(bv, p["ball_speed_window"] * FPS)
    with np.errstate(invalid="ignore"):
        inb = seen & (dout <= p["out_margin"]) & ~tele & ~np.isnan(bvs)
        w_out = _ramp(dout, p["out_margin"], p["out_full"])
        out = seen & (dout > p["out_margin"]) & (dout <= p["out_far"])
    ctx[inb] = np.interp(bvs[inb], p["ctx_speed_knots"], p["ctx_speed_logit"])
    # out: blend from the in-bounds value at the margin to out_logit at out_full
    base_out = np.interp(np.nan_to_num(bvs[out]), p["ctx_speed_knots"], p["ctx_speed_logit"])
    ctx[out] = (1 - w_out[out]) * base_out + w_out[out] * p["out_logit"]
    miss = ~seen
    glen = _run_length(miss) / FPS
    since = np.zeros(n)
    s_, e_ = _runs(miss)
    for a_, b_ in zip(s_, e_):
        since[a_:b_] = np.arange(b_ - a_) / FPS
    gctx = p["gap_lo"] + (p["gap_hi"] - p["gap_lo"]) * (1 - np.exp(-np.maximum(glen - p["gap_t0"], 0) / p["gap_tau"]))
    gctx -= p["gap_edge_pen"] * np.clip(1 - since / p["gap_edge"], 0, 1)
    ctx[miss] = gctx[miss]
    comp["ball_context"] = ctx

    # 3. carry: ball last seen out stays dead while unseen
    idx = np.where(seen, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    has_last = last >= 0
    last_dout = np.where(has_last, dout[np.maximum(last, 0)], np.nan)
    with np.errstate(invalid="ignore"):
        carry = miss & has_last & (last_dout > p["out_margin"]) & (last_dout < p["out_far"])
    comp["out_carry"] = p["out_carry_weight"] * carry

    # 4. stationary ball for > stat_min, extra at restart spots
    with np.errstate(invalid="ignore"):
        st = seen & (bv < p["stat_speed"]) & ~tele
    st = _fill_short_false(st, 5)
    rl = _run_length(st)
    st_long = st & (rl >= p["stat_min"] * FPS)
    rs = restart_location(B, L, W)
    rs = _fill_short_false(rs, 5)
    comp["ball_stationary"] = st_long * (p["stat_weight"] + p["stat_restart_bonus"] * rs)

    # 6. kickoff formation, and the pre-kickoff stretch of each half
    form = kickoff_formation(m, p)
    comp["kickoff_formation"] = p["ko_weight"] * form
    pre = prekick_mask(m, form, p)
    comp["pre_kickoff"] = p["prekick_weight"] * (pre & ~form)

    # 7. tracked-player count drop
    c = p["cnt_slope"] * (p["cnt_ref"] - np.nan_to_num(ntrack, nan=p["cnt_ref"]))
    comp["player_count"] = np.clip(c, 0.0, p["cnt_cap"])

    comp["_top5"] = top5
    comp["_form"] = form
    comp["_pre"] = pre
    return comp


def dead_ball_logodds(m, p=P):
    """(n_frames,) float64: log-odds that the ball is dead. Finite on every frame."""
    comp = dead_ball_components(m, p)
    lo = np.zeros(m.n_frames)
    for k, v in comp.items():
        if not k.startswith("_"):
            lo += v
    return lo
