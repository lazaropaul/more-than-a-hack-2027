"""Player-only team-possession model: "which team has the ball, from the 22 players alone".

Public entry point
------------------
    team_shape_logodds(m) -> np.ndarray (n_frames,) float64

log-odds that HOME is in possession, computed at inference from player positions / velocities
only (the ball is never an input feature). The ball is used for one thing: to build
self-supervised pseudo-labels on ball-visible frames, on which a classifier is trained
(transductively, on the match being labelled). No ground truth is read anywhere.

Design
------
* Team-symmetric ("us" vs "them") features expressed in the attacking direction of "us", so a
  single model serves both teams. Every frame is presented twice (home as "us", away as "us") and
  the final log-odds are antisymmetrised:  z = 0.5 * (z_home_as_us - z_away_as_us).
  That removes any home/away bias and makes the output exactly antisymmetric under swapping teams.
* Per-team shape statistics per frame (centroid, spread, length/width, lines, players in the
  opponent half / final third, velocity toward the opponent goal, pressing by the opponents,
  nearest-opponent spacing), then centred rolling means over 1 s, 2 s and 5 s, 2 s trends, and
  look-ahead / look-back 2 s windows plus a 0.5-5.5 s look-ahead window.
* Pseudo-labels: nearest player within 1.5 m of a visible ball, ball slow relative to him, no
  opponent also within 1.5 m, ball inside the pitch; centred +/-1 s majority (>= 90 %
  agreement) plus filling between two same-team controls <= 3 s apart with the ball visible.
* HistGradientBoostingClassifier (fixed seed); temperature fitted on out-of-fold predictions
  from blocked CV (contiguous 4-minute blocks, 5 folds, 8 s purge around test blocks; the
  widest feature window reaches 5.5 s into the future).

Ungraded frames (m.half_mask() == 0) and frames with < MIN_PLAYERS tracked return 0.0.
"""

import warnings

import numpy as np

FPS = 25.0
MIN_PLAYERS = 8          # total tracked players needed to emit a non-zero log-odds
VEL_K = 3                # velocity from +/-3 frames (0.24 s baseline)
MAX_SPEED = 11.0         # m/s; faster per-slot motion = identity switch, velocity set NaN
CONTROL_R = 1.5          # m, pseudo-label control radius
CONTROL_REL_V = 3.0      # m/s, ball speed relative to the controlling player
VOTE_HALF = 25           # +/-1 s majority vote
VOTE_MIN_FRAC = 0.9
VOTE_MIN_COUNT = 8
FILL_MAX_GAP = 75        # frames between same-team controls that may be filled
SEED = 0

BASE_NAMES = None        # filled by _team_stats (for notes / debugging)


# ----------------------------------------------------------------------------- helpers

def _nan_rolling_mean(a, w):
    """Centred rolling nan-mean along axis 0, window w (odd). NaN where the window is all-NaN."""
    a = np.asarray(a, dtype=np.float64)
    squeeze = a.ndim == 1
    if squeeze:
        a = a[:, None]
    n = a.shape[0]
    h = w // 2
    valid = ~np.isnan(a)
    cs = np.zeros((n + 1, a.shape[1]))
    cc = np.zeros((n + 1, a.shape[1]))
    np.cumsum(np.where(valid, a, 0.0), axis=0, out=cs[1:])
    np.cumsum(valid, axis=0, out=cc[1:])
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    s = cs[hi] - cs[lo]
    c = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c == 0] = np.nan
    return out[:, 0] if squeeze else out


def _shift(a, k):
    """a[t+k] with NaN padding (axis 0)."""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def _velocity(p, k=VEL_K, max_speed=MAX_SPEED):
    """Central-difference velocity (m/s) of (n, ..., 2) positions; NaN on gaps / switches."""
    v = (_shift(p, k) - _shift(p, -k)) * (FPS / (2.0 * k))
    sp = np.sqrt(np.sum(v * v, axis=-1))
    v[sp > max_speed] = np.nan
    return v


def _attack_sign(m, n):
    """Per-frame +1 if HOME attacks toward +x, -1 if toward -x, 0 on ungraded frames."""
    half = m.half_mask()
    s = np.zeros(n, dtype=np.float64)
    for h, _, _ in m.halves:
        s[half == h] = -1.0 if not m.home_defends_left[h] else 1.0
    return s, half


def _drop_extreme(xa, which):
    """Copy of xa (n, k) with the per-row min ('min') or max ('max') non-NaN entry set NaN."""
    out = xa.copy()
    has = ~np.all(np.isnan(xa), axis=1)
    if which == "min":
        idx = np.argmin(np.where(np.isnan(xa), np.inf, xa), axis=1)
    else:
        idx = np.argmax(np.where(np.isnan(xa), -np.inf, xa), axis=1)
    rows = np.flatnonzero(has)
    out[rows, idx[rows]] = np.nan
    return out


# ----------------------------------------------------------------------------- features

def _team_stats(xa, y, vxa, vy, press, nn_dist, L, W):
    """Per-frame statistics of one team, x in that team's attacking frame (0 = own goal).

    xa, y, vxa, vy: (n, k) for the team's slots (NaN untracked). press / nn_dist: (n, k) per
    player: speed at which the nearest opponent moves toward him, and distance to it.
    """
    names, cols = [], []

    def add(name, val):
        names.append(name)
        cols.append(np.asarray(val, dtype=np.float64))

    n_tr = np.sum(~np.isnan(xa), axis=1).astype(np.float64)
    # outfield = all but the deepest player (the keeper in nearly every frame)
    of = _drop_extreme(xa, "min")
    of_mask = ~np.isnan(of)
    yo = np.where(of_mask, y, np.nan)
    vxo = np.where(of_mask, vxa, np.nan)
    vyo = np.where(of_mask, vy, np.nan)
    n_of = of_mask.sum(1).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        add("n", n_tr)
        add("cx", np.nanmean(of, 1))
        add("cy_abs", np.abs(np.nanmean(yo, 1) - W / 2))
        add("sx", np.nanstd(of, 1))
        add("sy", np.nanstd(yo, 1))
        add("length", np.nanmax(of, 1) - np.nanmin(of, 1))
        add("width", np.nanmax(yo, 1) - np.nanmin(yo, 1))
        add("adv", np.nanmax(of, 1))                       # most advanced outfield player
        add("defline", np.nanmin(of, 1))                   # last outfield defender
        adv2 = np.nanmax(_drop_extreme(of, "max"), 1)      # 2nd most advanced
        add("adv2", adv2)
        add("deep_gk", np.nanmin(xa, 1))                   # deepest (keeper) position
        frac = lambda b: np.where(n_of > 0, np.sum(b & of_mask, 1) / np.maximum(n_of, 1), np.nan)
        add("f_opp_half", frac(of > L / 2))
        add("f_final3", frac(of > 2 * L / 3))
        add("f_own3", frac(of < L / 3))
        add("vx", np.nanmean(vxo, 1))                      # toward opponent goal
        add("vx_front", np.nanmean(np.where(of > np.nanmedian(of, 1)[:, None], vxo, np.nan), 1))
        add("speed", np.nanmean(np.sqrt(vxo ** 2 + vyo ** 2), 1))
        add("vy_spread", np.nanstd(vyo, 1))
        add("press_on", np.nanmean(press, 1))              # opponents closing on us
        add("press_on_max", np.nanmax(press, 1))
        add("nn_mean", np.nanmean(nn_dist, 1))             # spacing to nearest opponent
        add("nn_min", np.nanmin(nn_dist, 1))
        add("n_marked3", np.sum(nn_dist < 3.0, 1).astype(np.float64))
        add("n_free8", np.sum(nn_dist > 8.0, 1).astype(np.float64))
        add("area_box", (np.nanmax(of, 1) - np.nanmin(of, 1)) * (np.nanmax(yo, 1) - np.nanmin(yo, 1)))
    global BASE_NAMES
    BASE_NAMES = names
    return np.stack(cols, axis=1)


def _pair_terms(pu, pv, vv):
    """For each u player: distance to nearest v player and the speed at which it closes on him.

    pu (n, ku, 2), pv (n, kv, 2), vv (n, kv, 2). Returns (dist (n, ku), press (n, ku)).
    """
    n, ku = pu.shape[:2]
    dist = np.full((n, ku), np.nan)
    press = np.full((n, ku), np.nan)
    step = 20000
    for a in range(0, n, step):
        b = min(n, a + step)
        d = pu[a:b, :, None, :] - pv[a:b, None, :, :]             # u - v, (c, ku, kv, 2)
        dd = np.sqrt(np.sum(d * d, -1))
        dd_f = np.where(np.isnan(dd), np.inf, dd)
        j = np.argmin(dd_f, axis=2)                               # (c, ku)
        dmin = np.take_along_axis(dd_f, j[..., None], 2)[..., 0]
        ok = np.isfinite(dmin)
        r = np.take_along_axis(d, j[..., None, None].repeat(2, -1), 2)[:, :, 0, :]  # v -> u vector
        vj = np.take_along_axis(vv[a:b, None, :, :].repeat(ku, 1), j[..., None, None].repeat(2, -1), 2)[:, :, 0, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            pr = np.sum(r * vj, -1) / np.maximum(dmin, 0.3)
        dist[a:b] = np.where(ok, dmin, np.nan)
        press[a:b] = np.where(ok, pr, np.nan)
    return dist, press


def build_features(m):
    """Return (F_home_as_us, F_away_as_us, names, n_tracked, attack_sign, half).

    Each F is (n_frames, n_features) float32, NaN on ungraded frames.
    """
    n = m.n_frames
    L, W = float(m.pitch[0]), float(m.pitch[1])
    sign, half = _attack_sign(m, n)
    graded = half > 0
    P = m.players.astype(np.float64).copy()
    P[~graded] = np.nan
    V = _velocity(P)
    home = np.flatnonzero(m.team == 0)
    away = np.flatnonzero(m.team == 1)
    ph, pa, vh, va = P[:, home], P[:, away], V[:, home], V[:, away]

    dist_h, press_h = _pair_terms(ph, pa, va)   # pressing by away on home players
    dist_a, press_a = _pair_terms(pa, ph, vh)

    s = sign[:, None]
    # home attacking frame: x_att = x if s=+1 else L - x
    xh = np.where(s > 0, ph[..., 0], L - ph[..., 0])
    xa_ = np.where(s > 0, L - pa[..., 0], pa[..., 0])
    S_h = _team_stats(xh, ph[..., 1], s * vh[..., 0], vh[..., 1], press_h, dist_h, L, W)
    S_a = _team_stats(xa_, pa[..., 1], -s * va[..., 0], va[..., 1], press_a, dist_a, L, W)
    base = list(BASE_NAMES)

    # whole-frame context in the "us" frame: centre of play (all players) in us-attack coords
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        allx_h = np.nanmean(np.concatenate([xh, L - xa_], 1), 1)

    def persp(Su, St, play_x):
        # "them" positions re-expressed in the us frame where it matters
        feats = [Su, St, Su - St, play_x[:, None],
                 (Su[:, [base.index("cx")]] - (L - St[:, [base.index("cx")]])),   # cx_us - cx_them (us frame)
                 (Su[:, [base.index("adv")]] - (L - St[:, [base.index("defline")]])),  # our front vs their last line
                 ((L - St[:, [base.index("adv")]]) - Su[:, [base.index("defline")]])]  # their front vs our last line
        return np.concatenate(feats, 1)

    Fh = persp(S_h, S_a, allx_h)
    Fa = persp(S_a, S_h, L - allx_h)
    names0 = (["us_" + b for b in base] + ["them_" + b for b in base] + ["d_" + b for b in base]
              + ["play_x", "cx_gap", "front_vs_line_us", "front_vs_line_them"])

    # temporal context: rolling means (1 s, 2 s, 5 s) of a subset and 2 s trends
    key = ["us_cx", "them_cx", "d_cx", "d_vx", "us_vx", "them_vx", "d_press_on", "d_sx", "d_sy",
           "d_width", "d_length", "d_f_opp_half", "d_f_final3", "d_adv", "play_x", "cx_gap",
           "front_vs_line_us", "front_vs_line_them", "d_nn_mean", "d_n_free8", "d_n_marked3",
           "us_press_on", "them_press_on", "d_area_box", "d_vx_front", "d_speed"]
    ki = [names0.index(k) for k in key]
    out_h, out_a, names = [Fh], [Fa], list(names0)
    for w in (25, 51, 125):
        out_h.append(_nan_rolling_mean(Fh[:, ki], w))
        out_a.append(_nan_rolling_mean(Fa[:, ki], w))
        names += ["{}_r{}".format(k, w) for k in key]
    tk = [names0.index(k) for k in ("us_cx", "them_cx", "play_x", "d_width", "d_sx")]
    for F, out in ((Fh, out_h), (Fa, out_a)):
        sm = _nan_rolling_mean(F[:, tk], 25)
        out.append(_shift(sm, 25) - _shift(sm, -25))
    names += ["{}_trend2s".format(names0[i]) for i in tk]
    # asymmetric windows: shape reorganises 1-3 s AFTER a possession change, so the look-ahead
    # window [t, t+2 s] and the look-back [t-2 s, t] carry timing information; plus [t+0.5, t+5.5 s]
    for F, out in ((Fh, out_h), (Fa, out_a)):
        sm2 = _nan_rolling_mean(F[:, ki], 51)
        sm5 = _nan_rolling_mean(F[:, ki], 125)
        out += [_shift(sm2, 25), _shift(sm2, -25), _shift(sm5, 75)]
    names += ["{}_fwd2s".format(k) for k in key] + ["{}_back2s".format(k) for k in key]
    names += ["{}_fwd5s".format(k) for k in key]

    Fh = np.concatenate(out_h, 1).astype(np.float32)
    Fa = np.concatenate(out_a, 1).astype(np.float32)
    ntr = np.sum(~np.isnan(P[..., 0]), 1)
    return Fh, Fa, names, ntr, sign, half


# ----------------------------------------------------------------------------- pseudo-labels

def pseudo_labels(m, half=None):
    """Self-supervised possession labels from the ball on ball-visible graded frames.

    Returns (label (n,) int8 in {-1 unknown, 0 home, 1 away}, kind (n,) int8: 0 none, 1 core
    vote, 2 fill between same-team controls).
    """
    n = m.n_frames
    if half is None:
        half = m.half_mask()
    graded = half > 0
    L, W = float(m.pitch[0]), float(m.pitch[1])
    B = m.ball.astype(np.float64).copy()
    B[~graded] = np.nan
    P = m.players.astype(np.float64).copy()
    P[~graded] = np.nan
    vb = _velocity(B, max_speed=40.0)
    vp = _velocity(P)
    d = np.sqrt(np.sum((P - B[:, None, :]) ** 2, -1))          # (n, slots)
    df = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(df, 1)
    dmin = df[np.arange(n), j]
    team_j = m.team[j].astype(np.int8)
    # distance of the nearest player of the *other* team
    other = (m.team[None, :] != team_j[:, None])
    d_other = np.min(np.where(other, df, np.inf), 1)
    rel = vb - vp[np.arange(n), j]
    rel_sp = np.sqrt(np.sum(rel * rel, -1))
    bsp = np.sqrt(np.sum(vb * vb, -1))
    rel_sp = np.where(np.isnan(rel_sp), bsp, rel_sp)            # fallback: player velocity unknown
    inside = ((B[:, 0] > -0.5) & (B[:, 0] < L + 0.5) & (B[:, 1] > -0.5) & (B[:, 1] < W + 0.5))
    ctrl = (np.isfinite(dmin) & (dmin <= CONTROL_R) & (rel_sp < CONTROL_REL_V)
            & (d_other > CONTROL_R) & inside & graded)
    c = np.where(ctrl, team_j, -1).astype(np.int8)

    # centred majority vote
    w = 2 * VOTE_HALF + 1
    ch = _nan_rolling_mean((c == 0).astype(float), w) * w
    ca = _nan_rolling_mean((c == 1).astype(float), w) * w
    tot = ch + ca
    with np.errstate(invalid="ignore", divide="ignore"):
        fh = ch / tot
    lab = np.full(n, -1, np.int8)
    okv = (tot >= VOTE_MIN_COUNT) & graded & ~np.isnan(B[:, 0]) & inside
    lab[okv & (fh >= VOTE_MIN_FRAC)] = 0
    lab[okv & (fh <= 1 - VOTE_MIN_FRAC)] = 1
    kind = np.where(lab >= 0, 1, 0).astype(np.int8)

    # fill between consecutive control frames of the same team (ball visible all the way)
    idx = np.flatnonzero(c >= 0)
    if len(idx) > 1:
        seen = (~np.isnan(B[:, 0])) & inside & graded
        cs_unseen = np.concatenate([[0], np.cumsum(~seen)])
        a, b = idx[:-1], idx[1:]
        same = (c[a] == c[b]) & (b - a > 1) & (b - a <= FILL_MAX_GAP) & (half[a] == half[b])
        same &= (cs_unseen[b] - cs_unseen[a + 1]) == 0
        for s0, s1 in zip(a[same], b[same]):
            seg = slice(s0, s1 + 1)
            fill = lab[seg] < 0
            lab[seg][fill] = c[s0]
            kind[seg][fill] = 2
        # (python loop runs over control-pair events only, a few thousand)
    return lab, kind


# ----------------------------------------------------------------------------- model

def _blocked_folds(frames, half, n_folds=5, block_s=240, purge_s=8):
    """Yield (train_idx, test_idx) into `frames`: contiguous blocks assigned to folds round-robin."""
    block = (frames // int(block_s * FPS))
    # unique block id per half so blocks never straddle the half break
    bid = block * 10 + half[frames]
    ub, inv = np.unique(bid, return_inverse=True)
    fold_of_block = np.arange(len(ub)) % n_folds
    fold = fold_of_block[inv]
    purge = int(purge_s * FPS)
    for k in range(n_folds):
        te = np.flatnonzero(fold == k)
        tr_mask = fold != k
        # purge: training frames within `purge` frames of any test frame
        test_frames = np.zeros(int(frames.max()) + 2, bool)
        test_frames[frames[te]] = True
        near = _nan_rolling_mean(test_frames.astype(float), 2 * purge + 1) > 0
        tr_mask &= ~near[frames]
        yield np.flatnonzero(tr_mask), te


def _make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, max_features=0.5, early_stopping=False, random_state=SEED)


def _fit_sym(Fh, Fa, y, sub=3):
    """Fit on both perspectives: label 1 = 'us' in possession. y: 0 home / 1 away."""
    rows = np.arange(0, len(y), sub)
    X = np.concatenate([Fh[rows], Fa[rows]], 0)
    t = np.concatenate([(y[rows] == 0), (y[rows] == 1)]).astype(int)
    clf = _make_model()
    clf.fit(X, t)
    return clf


def _predict_sym(clf, Fh, Fa):
    """Antisymmetrised raw log-odds that home has the ball."""
    eps = 1e-6
    ph = np.clip(clf.predict_proba(Fh)[:, 1], eps, 1 - eps)
    pa = np.clip(clf.predict_proba(Fa)[:, 1], eps, 1 - eps)
    return 0.5 * (np.log(ph / (1 - ph)) - np.log(pa / (1 - pa)))


def _fit_temperature(z, y_home):
    """Scalar T minimising log-loss of sigmoid(z / T) (golden-section search on log T)."""
    from scipy.optimize import minimize_scalar

    def nll(logT):
        zz = z / np.exp(logT)
        return np.mean(np.logaddexp(0, -zz) * y_home + np.logaddexp(0, zz) * (1 - y_home))

    r = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def cross_validate(m, Fh=None, Fa=None, lab=None, kind=None, half=None):
    """Blocked CV on pseudo-labelled frames. Returns dict with OOF raw log-odds and metrics."""
    if Fh is None:
        Fh, Fa, _, _, _, half = build_features(m)
    if lab is None:
        lab, kind = pseudo_labels(m, half)
    frames = np.flatnonzero(lab >= 0)
    y = lab[frames]
    z = np.full(len(frames), np.nan)
    for tr, te in _blocked_folds(frames, half):
        clf = _fit_sym(Fh[frames[tr]], Fa[frames[tr]], y[tr])
        z[te] = _predict_sym(clf, Fh[frames[te]], Fa[frames[te]])
    return {"frames": frames, "y": y, "z_raw": z, "kind": kind[frames]}


def team_shape_logodds(m, return_details=False):
    """Per-frame calibrated log-odds that HOME has the ball, from players only.

    0.0 on ungraded frames and on frames with fewer than MIN_PLAYERS tracked players.
    """
    np.random.seed(SEED)
    Fh, Fa, names, ntr, sign, half = build_features(m)
    lab, kind = pseudo_labels(m, half)
    cv = cross_validate(m, Fh, Fa, lab, kind, half)
    yh = (cv["y"] == 0).astype(float)
    T = _fit_temperature(cv["z_raw"], yh)

    frames = cv["frames"]
    clf = _fit_sym(Fh[frames], Fa[frames], cv["y"])
    graded = half > 0
    z = np.zeros(m.n_frames)
    gi = np.flatnonzero(graded & (ntr >= MIN_PLAYERS))
    z[gi] = _predict_sym(clf, Fh[gi], Fa[gi]) / T
    z[~np.isfinite(z)] = 0.0
    if return_details:
        return z, {"T": T, "cv": cv, "names": names, "lab": lab, "kind": kind, "half": half,
                   "ntr": ntr, "Fh": Fh, "Fa": Fa, "clf": clf}
    return z
