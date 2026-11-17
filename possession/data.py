"""Load the tactical ATD into numpy arrays, in metres, indexed by video frame.

The raw file is ~200 MB of text. It is parsed once with pandas' C engine, after turning both field
separators (':' and ';') into commas so that every line becomes one flat CSV row:

    frame_id, x1, y1, s1, ..., xN, yN, sN, ball_x, ball_y, ball_s

and cached as a compressed .npz under data/_cache/ (gitignored, like everything in data/).

Slot -> team
------------
Every track slot belongs to one team. The authoritative map is the ATD metadata XML
(<Player teamId=...>, home team listed first). When that file is absent we derive the map from
the tracking itself, using the metadata's own layout convention -- one contiguous home block, then
one contiguous away block, each ending with its goalkeeper. Goalkeepers are the only slots that sit
next to one goal line for a whole half, so they are found from the data, and the home block is
everything up to and including the lower-indexed keeper. See infer_slot_teams().
"""

import io
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, asset_paths, half_video_blocks, source_frames  # noqa: E402

HOME, AWAY = 0, 1
CACHE_VERSION = 1


class Match(object):
    """Everything the pipeline reads, as plain arrays on the video-frame axis.

    players   (n_frames, n_slots, 2) float32, metres, NaN where the slot is not tracked
    ball      (n_frames, 2) float32, metres, NaN where the ball is not seen
    team      (n_slots,) int8, HOME or AWAY
    halves    [(half, first_video_frame, last_video_frame)] -- the graded blocks
    pitch     (length, width) in metres
    home_defends_left  {half: bool} -- which goal each side defends, derived from the keepers
    """

    def __init__(self, players, ball, team, halves, pitch, home_defends_left, team_source):
        self.players = players
        self.ball = ball
        self.team = team
        self.halves = halves
        self.pitch = pitch
        self.home_defends_left = home_defends_left
        self.team_source = team_source

    @property
    def n_frames(self):
        return self.ball.shape[0]

    def half_mask(self):
        """Per-frame half number, 0 outside the graded halves."""
        out = np.zeros(self.n_frames, dtype=np.int8)
        for half, first, last in self.halves:
            out[first:last + 1] = half
        return out


def _cache_path(raw_path):
    stat = os.stat(raw_path)
    name = "atd_v{}_{}_{}.npz".format(CACHE_VERSION, stat.st_size, int(stat.st_mtime))
    return os.path.join(DATA_DIR, "_cache", name)


def parse_raw(raw_path, use_cache=True):
    """(frame_id, players_norm (n, S, 3), ball_norm (n, 3)) as float32, normalised 0-1 coords."""
    cache = _cache_path(raw_path)
    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        return z["frame_id"], z["players"], z["ball"]

    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, na_values=["NaN"],
                        dtype=np.float32, engine="c").to_numpy()
    del text
    frame_id = table[:, 0].astype(np.int64)
    n_slots = (table.shape[1] - 1 - 3) // 3
    if 1 + 3 * n_slots + 3 != table.shape[1]:
        raise ValueError("unexpected ATD width {}".format(table.shape[1]))
    players = np.ascontiguousarray(table[:, 1:1 + 3 * n_slots].reshape(len(table), n_slots, 3))
    ball = np.ascontiguousarray(table[:, -3:])

    if use_cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, frame_id=frame_id, players=players, ball=ball)
    return frame_id, players, ball


def slot_teams_from_xml(meta_path):
    """Slot -> team from the ATD metadata, home team = first <Team> in the file."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def goalkeeper_slots(players_x, halves, min_presence=0.5, min_offset=0.3):
    """Slots that sit next to one goal line for a whole half: (slot, half, mean_x) triples.

    players_x is normalised 0-1. A keeper's mean x over a half is within ~0.15 of a goal line and
    the slot is tracked most of the half; no outfield tracklet comes close to that.
    """
    found = []
    for half, first, last in halves:
        x = players_x[first:last + 1]
        present = ~np.isnan(x)
        frac = present.mean(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # never-tracked slots: all-NaN mean
            mean_x = np.nanmean(x, axis=0)
        for slot in np.flatnonzero((frac >= min_presence) & (np.abs(mean_x - 0.5) >= min_offset)):
            found.append((int(slot), half, float(mean_x[slot])))
    return found


def infer_slot_teams(players_x, halves):
    """Slot -> team without the metadata: home block = slots up to the first keeper slot."""
    keepers = sorted({slot for slot, _, _ in goalkeeper_slots(players_x, halves)})
    if len(keepers) != 2:
        raise ValueError("expected 2 goalkeeper slots, found {}".format(keepers))
    n_slots = players_x.shape[1]
    team = np.full(n_slots, AWAY, dtype=np.int8)
    team[:keepers[0] + 1] = HOME
    return team


def home_defends_left(players_x, team, halves):
    """{half: True if the home keeper is near x=0}, from the data."""
    out = {}
    for slot, half, mean_x in goalkeeper_slots(players_x, halves):
        if team[slot] == HOME:
            out[half] = mean_x < 0.5
    for half, _, _ in halves:
        if half not in out:  # keeper not found this half: assume ends switch at halftime
            other = [v for h, v in out.items() if h != half]
            out[half] = (not other[0]) if other else True
    return out


def load_match(game=GAME, data_dir=None, use_cache=True):
    paths = asset_paths(game, data_dir)
    frame_id, players_n, ball_n = parse_raw(paths["atd_raw"], use_cache=use_cache)

    n_frames = source_frames("tactical", game)
    if len(frame_id) != n_frames or frame_id[0] != 1 or np.any(np.diff(frame_id) != 1):
        raise ValueError("ATD is not one record per video frame numbered from 1")

    halves = half_video_blocks(game)
    if os.path.isfile(paths["atd_meta"]):
        team, source = slot_teams_from_xml(paths["atd_meta"]), "metadata xml"
        if len(team) != players_n.shape[1]:
            raise ValueError("metadata lists {} slots, raw has {}".format(
                len(team), players_n.shape[1]))
    else:
        team, source = infer_slot_teams(players_n[:, :, 0], halves), "inferred (no metadata xml)"

    length, width = GAMES[game]["pitch"]
    scale = np.array([length, width], dtype=np.float32)
    players = players_n[:, :, :2] * scale
    ball = ball_n[:, :2] * scale
    return Match(players=players, ball=ball, team=team, halves=halves, pitch=(length, width),
                 home_defends_left=home_defends_left(players_n[:, :, 0], team, halves),
                 team_source=source)


if __name__ == "__main__":
    m = load_match()
    print("frames {}  slots {}  teams from {}".format(m.n_frames, len(m.team), m.team_source))
    print("home slots", np.flatnonzero(m.team == HOME) + 1)
    print("away slots", np.flatnonzero(m.team == AWAY) + 1)
    print("home defends left:", m.home_defends_left)
    print("ball seen on {:.1%} of frames".format(np.mean(~np.isnan(m.ball[:, 0]))))
