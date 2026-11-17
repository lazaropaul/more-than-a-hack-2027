"""Parameter sensitivity without any ground truth.

For each parameter, scale it by a few factors, rerun the pipeline and compare the result with the
base output, using the official metric code with the BASE OUTPUT standing in for the truth. A
parameter on a plateau gives agreement near 1 across the sweep; a parameter on a cliff does not.
Values should sit on plateaus, chosen from priors, never from a peak on any labelled window.

    python tools/sensitivity.py                         # all numeric params
    python tools/sensitivity.py switch_cost to_dead_cost --factors 0.5 0.75 1.5 2

No ground truth is read. The only output is self-agreement, plus counts and shares.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import pipeline  # noqa: E402
from possession.data import load_match  # noqa: E402


def as_codes(labels):
    return pm.label_codes(np.asarray(labels, dtype=object))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("params", nargs="*", help="parameter names (default: all numeric)")
    parser.add_argument("--factors", type=float, nargs="+", default=[0.5, 0.75, 1.5, 2.0])
    args = parser.parse_args()

    match = load_match()
    base_params = dict(pipeline.PARAMS)
    cache = {}
    base = pipeline.run(match, base_params, cache)
    base_codes = as_codes(base)
    s = pipeline.summarize(base, match.halves)
    print("base: {} possessions, mean {:.1f} s, shares {}".format(
        s["possessions"], s["mean_possession_s"], s["share"]))
    print("{:<22} {:>6} {:>8} {:>8} {:>6} {:>6}".format(
        "param", "factor", "selfSpell", "selfMacro", "poss", "D"))

    names = args.params or [k for k, v in base_params.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for name in names:
        for f in args.factors:
            value = base_params[name] * f
            if isinstance(base_params[name], int):
                value = int(round(value))
            labels = pipeline.run(match, dict(base_params, **{name: value}), cache)
            r = pm.evaluate(base_codes, as_codes(labels))
            s = pipeline.summarize(labels, match.halves)
            print("{:<22} {:>6.2f} {:>8.3f} {:>8.3f} {:>6d} {:>6.3f}".format(
                name, f, r["spell_f1_mean"], r["macro_f1"], s["possessions"], s["share"]["D"]))


if __name__ == "__main__":
    main()
