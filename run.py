"""Generate and validate the Furbito possession submission.

    python run.py
    python run.py --out evaluation/results/furbito.csv
"""

import argparse
import os
import subprocess
import sys

from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import write_submission
from starter.extract_features import extract_features
from starter.load_smart_tagging import load as load_smart_tagging
from src.possession import label_strings, predict_possession

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(
    ROOT, "evaluation", "results", "furbito.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="output CSV path")
    args = parser.parse_args()

    print("extracting tactical tracking features...", flush=True)
    paths = require(["atd_meta", "atd_raw"])
    features = extract_features(
        paths[1],
        paths[0],
        source_frames("tactical", GAME),
        GAMES[GAME]["pitch"],
    )
    print("loading automatic Smart Tagging...", flush=True)
    smart_tags = load_smart_tagging()
    print("classifying possession, restarts and temporal switches...", flush=True)
    labels = label_strings(predict_possession(
        features, smart_tags, half_video_blocks(GAME), GAMES[GAME]["pitch"]))

    output_path = os.path.abspath(args.out)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    write_submission(output_path, labels)
    print("wrote Furbito submission: {}".format(output_path), flush=True)

    validator = os.path.join(ROOT, "evaluation", "validate_submission.py")
    subprocess.run([sys.executable, validator, output_path], check=True)


if __name__ == "__main__":
    main()
