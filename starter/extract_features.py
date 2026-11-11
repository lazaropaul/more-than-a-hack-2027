"""Extract per-frame tracking features for the possession pipeline.

The tactical source is the default because its frame indices are the submission indices.
Coordinates are converted from the ATD's normalised pitch coordinates to metres; speeds are
already in metres per second.

    python starter/extract_features.py
    python starter/extract_features.py --source broadcast --out features/broadcast.csv
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, DEFAULT_SOURCE, require, source_frames

HOME, AWAY = 0, 1
TEAM_NAMES = ("home", "away")


def _track_teams(meta_path):
    """Return each tracking slot's team code, preserving the metadata slot order."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2 or not all(team.get("id") for team in teams):
        raise ValueError("{}: expected exactly two teams with IDs".format(meta_path))
    home_id, away_id = (team.get("id") for team in teams)
    if home_id == away_id:
        raise ValueError("{}: home and away team IDs are identical".format(meta_path))

    players = list(root.iter("Player"))
    if not players:
        raise ValueError("{}: no Player entries".format(meta_path))
    unknown = {player.get("teamId") for player in players} - {home_id, away_id}
    if unknown:
        raise ValueError("{}: tracking slots reference unknown team IDs {}".format(
            meta_path, sorted(unknown)))
    return np.asarray(
        [HOME if player.get("teamId") == home_id else AWAY for player in players],
        dtype=np.int8,
    )


def _empty_features(n_frames):
    columns = {
        "frame": np.arange(n_frames, dtype=np.int64),
        "ball_x_m": np.full(n_frames, np.nan),
        "ball_y_m": np.full(n_frames, np.nan),
        "ball_visible": np.zeros(n_frames, dtype=bool),
        "ball_speed_mps": np.full(n_frames, np.nan),
        "ball_acceleration_mps2": np.full(n_frames, np.nan),
        "ball_gap_frames": np.full(n_frames, np.nan),
    }
    for team in TEAM_NAMES:
        columns.update({
            "{}_centroid_x_m".format(team): np.full(n_frames, np.nan),
            "{}_centroid_y_m".format(team): np.full(n_frames, np.nan),
            "{}_spread_m".format(team): np.full(n_frames, np.nan),
            "{}_n_players".format(team): np.zeros(n_frames, dtype=np.int16),
            "{}_nearest_ball_m".format(team): np.full(n_frames, np.nan),
            "{}_players_within_5m_ball".format(team): np.full(n_frames, np.nan),
            "{}_team_speed_mean_mps".format(team): np.full(n_frames, np.nan),
            "{}_centroid_ball_m".format(team): np.full(n_frames, np.nan),
        })
    columns["distance_ratio_home_away"] = np.full(n_frames, np.nan)
    columns["formation_compactness_diff_m"] = np.full(n_frames, np.nan)
    return columns


def extract_features(raw_path, meta_path, n_frames, pitch):
    """Read an ATD raw file and return one feature row per zero-based video frame.

    Ball speed and acceleration are calculated only across consecutive frames with an observed
    ball position. `ball_gap_frames` is NaN before the first observation, zero on observed frames,
    and the number of consecutive missing frames otherwise.
    """
    length_m, width_m = pitch
    team_codes = _track_teams(meta_path)
    columns = _empty_features(n_frames)
    previous_ball = None
    previous_ball_speed = None
    missing_ball_frames = None

    with open(raw_path, "r") as handle:
        for expected_frame, line in enumerate(handle, start=1):
            if expected_frame > n_frames:
                raise ValueError("{}: more than {} ATD records".format(raw_path, n_frames))
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                raise ValueError("{}: malformed ATD record at line {}".format(
                    raw_path, expected_frame))
            try:
                frame_id = int(parts[0])
            except ValueError as error:
                raise ValueError("{}: invalid frame ID at line {}".format(
                    raw_path, expected_frame)) from error
            if frame_id != expected_frame:
                raise ValueError(
                    "{}: expected frame ID {}, found {} at line {}".format(
                        raw_path, expected_frame, frame_id, expected_frame))

            players = np.fromstring(parts[1].replace(";", ","), sep=",")
            if players.size != team_codes.size * 3:
                raise ValueError(
                    "{}: frame {} has {} player values; metadata describes {} slots".format(
                        raw_path, frame_id, players.size, team_codes.size))
            players = players.reshape((-1, 3))
            player_xy = players[:, :2]
            player_valid = np.isfinite(player_xy).all(axis=1)
            player_speed = players[:, 2]

            ball_values = np.fromstring(parts[2], sep=",")
            if ball_values.size != 3:
                raise ValueError("{}: frame {} has malformed ball values".format(
                    raw_path, frame_id))
            ball_x, ball_y = ball_values[:2]
            ball_visible = bool(np.isfinite(ball_x) and np.isfinite(ball_y))
            frame = frame_id - 1
            ball_xy_m = None

            if ball_visible:
                ball_xy_m = np.asarray([ball_x * length_m, ball_y * width_m])
                columns["ball_x_m"][frame] = ball_xy_m[0]
                columns["ball_y_m"][frame] = ball_xy_m[1]
                columns["ball_visible"][frame] = True
                columns["ball_gap_frames"][frame] = 0
                missing_ball_frames = 0

                if previous_ball is not None:
                    displacement = float(np.linalg.norm(ball_xy_m - previous_ball))
                    speed = displacement * GAMES[GAME]["fps"]
                    columns["ball_speed_mps"][frame] = speed
                    if previous_ball_speed is not None:
                        columns["ball_acceleration_mps2"][frame] = (
                            (speed - previous_ball_speed) * GAMES[GAME]["fps"]
                        )
                    previous_ball_speed = speed
                else:
                    previous_ball_speed = None
                previous_ball = ball_xy_m
            else:
                missing_ball_frames = (
                    1 if missing_ball_frames is None else missing_ball_frames + 1
                )
                columns["ball_gap_frames"][frame] = missing_ball_frames
                previous_ball = None
                previous_ball_speed = None

            team_nearest = []
            team_spreads = []
            team_centroid_distances = []
            for team_index, team in enumerate(TEAM_NAMES):
                slot_mask = (team_codes == team_index) & player_valid
                positions = player_xy[slot_mask]
                count = positions.shape[0]
                columns["{}_n_players".format(team)][frame] = count
                if count == 0:
                    team_nearest.append(np.nan)
                    team_spreads.append(np.nan)
                    team_centroid_distances.append(np.nan)
                    continue

                positions_m = positions * np.asarray([length_m, width_m])
                centroid = positions_m.mean(axis=0)
                radial_distances = np.linalg.norm(positions_m - centroid, axis=1)
                spread = float(np.sqrt(np.mean(radial_distances ** 2)))
                columns["{}_centroid_x_m".format(team)][frame] = centroid[0]
                columns["{}_centroid_y_m".format(team)][frame] = centroid[1]
                columns["{}_spread_m".format(team)][frame] = spread
                team_spreads.append(spread)

                speeds = player_speed[slot_mask]
                finite_speeds = speeds[np.isfinite(speeds)]
                if finite_speeds.size:
                    columns["{}_team_speed_mean_mps".format(team)][frame] = (
                        finite_speeds.mean()
                    )

                if ball_xy_m is None:
                    team_nearest.append(np.nan)
                    team_centroid_distances.append(np.nan)
                else:
                    distances = np.linalg.norm(positions_m - ball_xy_m, axis=1)
                    nearest = float(distances.min())
                    columns["{}_nearest_ball_m".format(team)][frame] = nearest
                    columns["{}_players_within_5m_ball".format(team)][frame] = int(
                        np.count_nonzero(distances <= 5.0)
                    )
                    centroid_distance = float(np.linalg.norm(centroid - ball_xy_m))
                    columns["{}_centroid_ball_m".format(team)][frame] = centroid_distance
                    team_nearest.append(nearest)
                    team_centroid_distances.append(centroid_distance)

            if np.isfinite(team_nearest).all() and team_nearest[AWAY] > 0:
                columns["distance_ratio_home_away"][frame] = (
                    team_nearest[HOME] / team_nearest[AWAY]
                )
            if np.isfinite(team_spreads).all():
                columns["formation_compactness_diff_m"][frame] = (
                    team_spreads[HOME] - team_spreads[AWAY]
                )

    if expected_frame != n_frames:
        raise ValueError("{}: expected {} ATD records, found {}".format(
            raw_path, n_frames, expected_frame))
    return pd.DataFrame(columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("tactical", "broadcast"), default=DEFAULT_SOURCE)
    parser.add_argument(
        "--out",
        help="output CSV path (default: features/BAR-ATH_<source>_features.csv)",
    )
    args = parser.parse_args()

    paths = require(["atd_meta", "atd_raw"], source=args.source)
    frame_count = source_frames(args.source)
    features = extract_features(
        paths[1], paths[0], frame_count, GAMES[GAME]["pitch"])
    output = args.out or os.path.join(
        "features", "BAR-ATH_{}_features.csv".format(args.source))
    output_path = os.path.abspath(output)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    features.to_csv(output_path, index=False)
    print("wrote {} rows x {} features to {}".format(
        len(features), len(features.columns) - 1, output_path))
    print("ball visible: {:.1f}%".format(100.0 * features["ball_visible"].mean()))


if __name__ == "__main__":
    main()
