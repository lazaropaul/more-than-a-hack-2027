"""Where the dead-ball constants in possession/deadball.py come from. ATD only, no ground truth.

    python tools/deadball_calibration.py

Everything below is computed from the tracking of the match being labelled, using anchor sets
defined by unambiguous physical situations, never by a label:

  DEAD anchors   ball seen 2-12 m outside a line; unseen frames < 4 s after the ball was last
                 seen out; kick-off formation (both teams in their own halves).
  ALIVE anchors  ball >= 3 m inside the pitch moving 8-35 m/s and cleanly seen; or the five
                 fastest players averaging > 4.8 m/s.

It prints:
  1. leave-one-cue-out AUC: each cue family is scored only on anchors defined by the OTHER family
     (ball cues on player-defined anchors and vice versa), so no cue grades itself;
  2. the activity term's slope and centre from a class-balanced logistic fit on the ball-defined
     anchors -> act_slope / act_centre (the module uses 1.5 and 3.15);
  3. a self-supervised mixture estimate of P(dead) per ball context, by EM on the activity
     distribution with class densities taken from the anchors -> ctx_speed_logit and the gap
     curve (the module shrinks these toward 0 by ~1 nat at the extremes, because the estimate
     is partly circular);
  4. the dead-share and stoppage count this implies under a 2-state Viterbi, which is where the
     pipeline's dead_weight (0.1-0.2) and alive<->dead cost (5-10 nats) recommendation comes from.
"""

import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import deadball as db  # noqa: E402
from possession.data import load_match  # noqa: E402
from possession.decoder import viterbi  # noqa: E402

FPS = 25


def auc(score, pos, neg):
    """Rank AUC of score separating pos from neg (Mann-Whitney), ties averaged."""
    x = np.r_[score[pos], score[neg]]
    y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(xs):  # average ranks over ties
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def anchors(m, comp, bf):
    n = m.n_frames
    g = m.half_mask() > 0
    seen, dout, bv = bf["seen"], bf["dout"], bf["bv"]
    top5 = comp["_top5"]
    seen_frac = uniform_filter1d(seen.astype(float), FPS)
    idx = np.where(seen, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    last_out = np.where(last >= 0, dout[np.maximum(last, 0)], np.nan)
    with np.errstate(invalid="ignore"):
        dead_out = g & seen & (dout > 2) & (dout < 12)
        dead_postout = (g & ~seen & (last >= 0) & (last_out > 2) & (last_out < 12)
                        & ((np.arange(n) - last) < 4 * FPS))
        alive_ball = g & seen & (dout < -3) & (bv > 8) & (bv < 35) & (seen_frac > 0.8)
        alive_act = g & (top5 > 4.8)
    return {
        "dead_ball": dead_out | dead_postout,
        "dead_player": g & comp["_form"],
        "alive_ball": alive_ball,
        "alive_player": alive_act,
    }


def balanced_logistic(x, y, iters=200):
    """1-D logistic regression with class-balanced weights, by Newton's method. Returns (w, b)."""
    wts = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    w, b = 0.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(w * x + b)))
        r = wts * (p - y)
        h = wts * p * (1 - p)
        g = np.array([np.sum(r * x), np.sum(r)])
        H = np.array([[np.sum(h * x * x), np.sum(h * x)], [np.sum(h * x), np.sum(h)]])
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        w, b = w - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return w, b


def main():
    m = load_match()
    comp = db.dead_ball_components(m)
    bf = db.ball_features(m)
    a = anchors(m, comp, bf)
    for k, v in a.items():
        print("anchor {:<13} {:>7d} frames".format(k, int(v.sum())))

    # 1. leave-one-cue-out AUC
    ball_terms = ["ball_context", "out_carry", "ball_stationary"]
    player_terms = ["activity", "player_count"]
    ball_score = sum(comp[t] for t in ball_terms)
    player_score = sum(comp[t] for t in player_terms)
    print("\n1. leave-one-cue-out AUC")
    print("   player cues on ball-defined anchors   {:.3f}".format(
        auc(player_score, a["dead_ball"], a["alive_ball"])))
    print("   ball cues on player-defined anchors   {:.3f}".format(
        auc(ball_score, a["dead_player"], a["alive_player"])))
    print("   activity: formation vs fast ball      {:.3f}".format(
        auc(comp["activity"], a["dead_player"], a["alive_ball"])))

    # 2. activity term: balanced logistic on ball-defined anchors (activity did not define them)
    top5 = comp["_top5"]
    sel = (a["dead_ball"] | a["alive_ball"]) & ~np.isnan(top5)
    w, b = balanced_logistic(top5[sel], a["dead_ball"][sel].astype(float))
    print("\n2. activity logistic: slope {:.2f} /(m/s), centre {:.2f} m/s   (module: {} / {})".format(
        -w, -b / w, db.P["act_slope"], db.P["act_centre"]))

    # 3. mixture estimate of P(dead) per context
    dead = a["dead_ball"] | a["dead_player"]
    alive = a["alive_ball"]
    bins = np.r_[0, np.arange(1.0, 6.01, 0.25), 20]

    def density(mask):
        h, _ = np.histogram(top5[mask & ~np.isnan(top5)], bins)
        h = h + 0.5
        return h / h.sum()

    f_dead, f_alive = density(dead), density(alive)

    def em(mask):
        v = top5[mask & ~np.isnan(top5)]
        k = np.clip(np.digitize(v, bins) - 1, 0, len(f_dead) - 1)
        f1, f0, pi = f_dead[k], f_alive[k], 0.5
        for _ in range(500):
            pi = np.mean(pi * f1 / (pi * f1 + (1 - pi) * f0))
        return pi, len(v)

    g = m.half_mask() > 0
    seen, dout, bv = bf["seen"], bf["dout"], bf["bv"]
    print("\n3. mixture P(dead) by context (logit in brackets)")
    pi, _ = em(g)
    print("   all graded frames  {:.3f}".format(pi))
    inb = g & seen & (dout < 0) & ~bf["tele"]
    for lo, hi in [(0, 0.8), (0.8, 2), (2, 3.5), (3.5, 5), (5, 8), (8, 40)]:
        pi, n = em(inb & (bv >= lo) & (bv < hi))
        print("   ball seen {:>4}-{:<4} m/s  {:.3f} ({:+.2f})  n={}".format(
            lo, hi, pi, np.log(pi / (1 - pi)), n))
    glen = db._run_length(~seen) / FPS
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]:
        pi, n = em(g & ~seen & (glen >= lo) & (glen < hi))
        print("   ball missing, gap {:>3}-{:<5} s  {:.3f} ({:+.2f})  n={}".format(
            lo, hi if hi < 1e9 else "inf", pi, np.log(pi / (1 - pi)), n))

    # 4. what the dead stream alone implies under a 2-state Viterbi
    lo = np.clip(db.dead_ball_logodds(m), -6, 6)
    print("\n4. 2-state Viterbi on dead log-odds alone (alpha = weight, C = switch cost)")
    for alpha, cost in [(0.1, 5), (0.15, 8), (0.2, 10), (0.1, 10), (0.5, 5)]:
        dead_frames, stoppages = 0, 0
        for _, first, last in m.halves:
            em2 = np.zeros((last + 1 - first, 3))
            em2[:, 0] = -0.5 * alpha * lo[first:last + 1]
            em2[:, 1] = -np.inf  # unused third state of the 3-state decoder
            em2[:, 2] = 0.5 * alpha * lo[first:last + 1]
            c = np.full((3, 3), 1e9)
            c[0, 0] = c[2, 2] = 0
            c[0, 2] = c[2, 0] = cost
            path = viterbi(em2, c)
            dead_frames += int(np.sum(path == 2))
            stoppages += int(np.sum(np.diff(np.r_[0, (path == 2).astype(int)]) == 1))
        print("   alpha {:.2f}  C {:>4}: dead share {:.3f}, stoppages {}".format(
            alpha, cost, dead_frames / g.sum(), stoppages))


if __name__ == "__main__":
    main()
