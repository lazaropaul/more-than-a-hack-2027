"""Regenerate the submission from the provided data. The one command a reviewer needs:

    python make_submission.py                       # -> evaluation/results/<team>.csv, validated
    python make_submission.py --name my_try         # a different file name

Deterministic: the same data gives the same file, byte for byte.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from possession import pipeline  # noqa: E402
from possession.data import load_match  # noqa: E402

DEFAULT_NAME = "furbito"
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=DEFAULT_NAME, help="output file stem")
    parser.add_argument("--out", default=RESULTS_DIR, help="output folder")
    parser.add_argument("--params", default=None,
                        help="JSON object overriding pipeline.PARAMS (for sensitivity checks)")
    args = parser.parse_args()

    match = load_match()
    print("loaded {} frames, slot teams from {}".format(match.n_frames, match.team_source))
    params = json.loads(args.params) if args.params else None
    labels = pipeline.run(match, params)

    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    print("wrote", os.path.relpath(path, ROOT))
    print(json.dumps(pipeline.summarize(labels, match.halves)))

    check = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation",
                                                         "validate_submission.py"), path])
    sys.exit(check.returncode)


if __name__ == "__main__":
    main()
