"""Generate and validate a provisional submission from ATD tracking features.

This currently wires feature extraction to the repository's nearest-player baseline. Replace
`predict_baseline` with the team's possession model when it is ready.

    python run.py
    python run.py --window 25 --out evaluation/results/member1_provisional.csv
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from games import GAME, GAMES, require, source_frames
from starter.baseline import AWAY, HOME, NOBODY, smooth, write_submission
from starter.extract_features import extract_features

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(
    ROOT, "evaluation", "results", "member1_provisional.csv")


def predict_baseline(features):
    """Assign the closest tracked team to visible balls; leave missing balls as D."""
    labels = np.full(len(features), NOBODY, dtype=np.int8)
    visible = features["ball_visible"].to_numpy(dtype=bool)
    home = features["home_nearest_ball_m"].to_numpy(dtype=np.float64)
    away = features["away_nearest_ball_m"].to_numpy(dtype=np.float64)
    has_home = np.isfinite(home)
    has_away = np.isfinite(away)

    labels[visible & has_home & ~has_away] = HOME
    labels[visible & has_away & ~has_home] = AWAY
    both = visible & has_home & has_away
    labels[both] = np.where(home[both] <= away[both], HOME, AWAY)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="centred majority-vote half-window in frames (default: 50)",
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="output CSV path")
    args = parser.parse_args()
    if args.window < 0:
        parser.error("--window must be zero or greater")

    print("extracting tactical tracking features...", flush=True)
    paths = require(["atd_meta", "atd_raw"])
    features = extract_features(
        paths[1],
        paths[0],
        source_frames("tactical", GAME),
        GAMES[GAME]["pitch"],
    )
    print("applying provisional nearest-player baseline...", flush=True)
    labels = smooth(predict_baseline(features), args.window)

    output_path = os.path.abspath(args.out)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    write_submission(output_path, labels)
    print("wrote provisional nearest-player baseline: {}".format(output_path), flush=True)

    validator = os.path.join(ROOT, "evaluation", "validate_submission.py")
    subprocess.run([sys.executable, validator, output_path], check=True)


if __name__ == "__main__":
    main()
