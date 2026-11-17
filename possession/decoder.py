"""Three-state segmenter: Viterbi over {H, A, D} with explicit switching costs.

Why a state model rather than per-frame labels plus smoothing: the ranking metric matches whole
possessions, so the thing to get right is *where the boundaries are* and *how many there are*.
Viterbi finds the single best segmentation under a clear trade-off. Every state change costs a
fixed amount, and evidence has to pay for it. One touch by the other team does not buy a
possession change, but sustained control does, which is how CHALLENGE.md defines "controlled"
possession. Where there is no evidence (ball unseen), nothing pays for a change, so the
current state carries through, as "possession stays with the last team" says.

Scores are in log-likelihood units (nats). Emissions are per-frame scores for each state; costs are
subtracted on a change.
"""

import numpy as np

H, A, D = 0, 1, 2
STATE_LABELS = np.array(["H", "A", "D"])


def transition_costs(switch, to_dead, from_dead):
    """3x3 cost matrix, rows = from, cols = to."""
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = switch
    c[H, D] = c[A, D] = to_dead
    c[D, H] = c[D, A] = from_dead
    return c


def viterbi(emission, cost, start=None):
    """Best state path for one contiguous block.

    emission: (n, 3) per-frame scores, higher = more likely.
    cost: (3, 3) non-negative transition costs.
    start: optional (3,) initial scores.

    Pure-Python inner loop over three states: ~1 s per 100k frames and no dependency, which is
    plenty for 160k graded frames.
    """
    n = len(emission)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    neg = -np.asarray(cost, dtype=float)
    t00, t01, t02 = neg[0]
    t10, t11, t12 = neg[1]
    t20, t21, t22 = neg[2]
    em = np.asarray(emission, dtype=float)
    back = np.empty((n, 3), dtype=np.int8)

    s = np.zeros(3) if start is None else np.asarray(start, dtype=float)
    s0, s1, s2 = s[0] + em[0, 0], s[1] + em[0, 1], s[2] + em[0, 2]
    back[0] = (0, 1, 2)
    e0, e1, e2 = em[:, 0].tolist(), em[:, 1].tolist(), em[:, 2].tolist()
    for t in range(1, n):
        # into H
        a, b, c = s0 + t00, s1 + t10, s2 + t20
        if a >= b and a >= c:
            n0, p0 = a, 0
        elif b >= c:
            n0, p0 = b, 1
        else:
            n0, p0 = c, 2
        # into A
        a, b, c = s0 + t01, s1 + t11, s2 + t21
        if b >= a and b >= c:
            n1, p1 = b, 1
        elif a >= c:
            n1, p1 = a, 0
        else:
            n1, p1 = c, 2
        # into D
        a, b, c = s0 + t02, s1 + t12, s2 + t22
        if c >= a and c >= b:
            n2, p2 = c, 2
        elif a >= b:
            n2, p2 = a, 0
        else:
            n2, p2 = b, 1
        back[t, 0], back[t, 1], back[t, 2] = p0, p1, p2
        s0, s1, s2 = n0 + e0[t], n1 + e1[t], n2 + e2[t]

    path = np.empty(n, dtype=np.int8)
    last = int(np.argmax((s0, s1, s2)))
    for t in range(n - 1, -1, -1):
        path[t] = last
        last = back[t, last]
    return path


def emissions(team_logodds, dead_logodds):
    """(n, 3) emission scores from two log-odds streams.

    team_logodds: log P(evidence | H) / P(evidence | A), positive = home.
    dead_logodds: log P(evidence | D) / P(evidence | alive), positive = dead.
    Split symmetrically so that H vs A compares the team term and alive vs D compares the dead
    term.
    """
    team = np.nan_to_num(np.asarray(team_logodds, dtype=float))
    dead = np.nan_to_num(np.asarray(dead_logodds, dtype=float))
    em = np.empty((len(team), 3))
    em[:, H] = 0.5 * team - 0.5 * dead
    em[:, A] = -0.5 * team - 0.5 * dead
    em[:, D] = 0.5 * dead
    return em


def decode_halves(emission, halves, cost, n_frames):
    """Labels for the whole video: Viterbi inside each graded half, X outside."""
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in halves:
        path = viterbi(emission[first:last + 1], cost)
        labels[first:last + 1] = STATE_LABELS[path]
    return labels
