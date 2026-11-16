"""Possession inference from tactical ATD features and Smart Tagging."""

import numpy as np

HOME, AWAY, DEAD, UNKNOWN = 0, 1, 2, 3

_TEAM_TAGS = {
    "barcelona": HOME,
    "atletic club": AWAY,
}
_POSSESSION_PHASES = {
    "BUILD UP",
    "PROGRESSION",
    "CREATING CHANCES",
    "ATTACKING TRANSITION",
    "DEFENSIVE TRANSITION",
}


def _tagged_team(tag):
    name = (tag.get("tags", {}).get("Team") or "").strip().lower()
    team = _TEAM_TAGS.get(name)
    if team is None:
        return None
    code = (tag.get("code") or "").strip().upper()
    if code == "DEFENSIVE TRANSITION":
        return AWAY if team == HOME else HOME
    return team


def _mark_boundary_gaps(features, dead, pitch):
    """Treat a sustained ball-tracking gap after a near-boundary observation as a stoppage."""
    visible = features["ball_visible"].to_numpy(dtype=bool)
    ball_x = features["ball_x_m"].to_numpy(dtype=np.float64)
    ball_y = features["ball_y_m"].to_numpy(dtype=np.float64)
    length_m, width_m = pitch
    missing = np.flatnonzero(~visible)
    if missing.size == 0:
        return

    gaps = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
    for gap in gaps:
        if gap.size < 75 or gap[0] == 0:
            continue
        previous = gap[0] - 1
        if not visible[previous]:
            continue
        x, y = ball_x[previous], ball_y[previous]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        near_boundary = (
            x <= 3.0 or x >= length_m - 3.0
            or y <= 3.0 or y >= width_m - 3.0
        )
        if near_boundary:
            dead[gap] = True


def _find_restarts(features, smart_tags, n_frames, pitch):
    """Return stoppage masks and inferred restart frames from set-piece intervals."""
    dead = np.zeros(n_frames, dtype=bool)
    restarts = []
    visible = features["ball_visible"].to_numpy(dtype=bool)
    speed = features["ball_speed_mps"].to_numpy(dtype=np.float64)
    _mark_boundary_gaps(features, dead, pitch)

    for tag in smart_tags:
        if (tag.get("code") or "").strip().upper() != "SET PIECES":
            continue
        start = max(0, min(n_frames, int(tag["start_frame"])))
        end = max(start, min(n_frames, int(tag["end_frame"])))
        if end <= start:
            continue

        # A restart is the first short burst of ball movement in the tagged interval.
        moving = visible[start:end] & np.isfinite(speed[start:end]) & (speed[start:end] >= 1.0)
        moving_pair = moving[:-1] & moving[1:]
        candidates = np.flatnonzero(moving_pair)
        if candidates.size == 0:
            continue
        restart = start + int(candidates[0])
        if restart - start > 150:
            continue
        if restart - start >= 5:
            dead[start:restart] = True
        restarts.append((restart, min(end, restart + 75), _tagged_team(tag)))

    return dead, restarts


def _assign_visible_possession(features, dead):
    n_frames = len(features)
    labels = np.full(n_frames, UNKNOWN, dtype=np.int8)
    labels[dead] = DEAD

    visible = features["ball_visible"].to_numpy(dtype=bool)
    home = features["home_nearest_ball_m"].to_numpy(dtype=np.float64)
    away = features["away_nearest_ball_m"].to_numpy(dtype=np.float64)
    usable = visible & ~dead & np.isfinite(home) & np.isfinite(away)

    nearest = np.minimum(home, away)
    margin = np.abs(home - away)
    decisive = usable & (nearest <= 5.0) & ~(
        (home <= 3.0) & (away <= 3.0) & (margin < 0.75)
    )
    labels[decisive] = np.where(home[decisive] <= away[decisive], HOME, AWAY)
    return labels


def _apply_tag_evidence(labels, smart_tags, restarts):
    """Use automatic tactical tags only to fill frames without a tracking decision."""
    result = labels.copy()
    for tag in smart_tags:
        code = (tag.get("code") or "").strip().upper()
        if code not in _POSSESSION_PHASES:
            continue
        team = _tagged_team(tag)
        if team is None:
            continue
        start = max(0, min(len(result), int(tag["start_frame"])))
        end = max(start, min(len(result), int(tag["end_frame"])))
        interval = result[start:end]
        interval[interval == UNKNOWN] = team

    for start, end, team in restarts:
        if team is None:
            continue
        interval = result[start:end]
        interval[interval == UNKNOWN] = team
    return result


def _confirm_switches(labels, min_switch_frames=50):
    """Ignore brief opposing-team blips; backdate sustained changes to their first frame."""
    result = labels.copy()
    current = UNKNOWN
    pending_team = UNKNOWN
    pending_start = -1
    pending_count = 0

    for frame, label in enumerate(labels):
        label = int(label)
        if label == DEAD:
            current = DEAD
            pending_team = UNKNOWN
            pending_start = -1
            pending_count = 0
            result[frame] = DEAD
            continue

        if label == UNKNOWN:
            result[frame] = UNKNOWN
            continue

        if current in (UNKNOWN, DEAD):
            current = label
            pending_team = UNKNOWN
            pending_start = -1
            pending_count = 0
            result[frame] = label
            continue

        if label == current:
            pending_team = UNKNOWN
            pending_start = -1
            pending_count = 0
            result[frame] = current
            continue

        if label != pending_team:
            pending_team = label
            pending_start = frame
            pending_count = 1
        else:
            pending_count += 1

        result[frame] = current
        if pending_count >= min_switch_frames:
            result[pending_start:frame + 1] = pending_team
            current = pending_team
            pending_team = UNKNOWN
            pending_start = -1
            pending_count = 0

    return result


def _fill_play_unknowns(labels):
    """Carry the previous state through unknown gaps, backfilling the opening gap."""
    result = labels.copy()
    known = np.flatnonzero(result != UNKNOWN)
    if known.size == 0:
        return result

    previous = np.maximum.accumulate(
        np.where(result != UNKNOWN, np.arange(len(result)), -1)
    )
    previous[previous < 0] = known[0]
    return result[previous]


def predict_possession(
    features,
    smart_tags,
    video_blocks,
    pitch=(105.3, 68.0),
    min_switch_frames=50,
):
    """Return H/A/D/X labels, with smoothing isolated to each half of play."""
    n_frames = len(features)
    dead, restarts = _find_restarts(features, smart_tags, n_frames, pitch)
    labels = _assign_visible_possession(features, dead)
    labels = _apply_tag_evidence(labels, smart_tags, restarts)

    output = np.full(n_frames, UNKNOWN, dtype=np.int8)
    for _, first, last in video_blocks:
        start = max(0, first)
        stop = min(n_frames, last + 1)
        if start < stop:
            half_labels = _confirm_switches(
                labels[start:stop], min_switch_frames=min_switch_frames)
            output[start:stop] = _fill_play_unknowns(half_labels)
    return output


def label_strings(codes):
    """Convert integer state codes to the submission's H/A/D/X labels."""
    mapping = np.array(["H", "A", "D", "X"])
    if np.any((codes < 0) | (codes >= len(mapping))):
        raise ValueError("possession labels contain an unknown state code")
    return mapping[codes]
