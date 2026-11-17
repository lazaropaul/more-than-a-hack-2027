# Running the yeeclaw pipeline

How to go from the match data to the submitted file, `evaluation/results/yeeclaw.csv`, and how to
check and explore it. For how the method works, see [FEATURES.md](FEATURES.md); for the summary,
see [TECHNICAL.md](../TECHNICAL.md).

**In short:**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-pipeline.txt
.venv/bin/python make_submission.py
```

---

## 1. Requirements

| | |
|---|---|
| Python | tested on **3.15** |
| packages | `numpy`, `pandas`, `scipy`, `scikit-learn`, pinned in `requirements-pipeline.txt` |
| memory | ~2.7 GB peak on the first run |
| time | ~70 s on the first run (8 cores); ~4 s after that |
| disk | ~300 MB of cache in `data/_cache/` |

No GPU, no video decoding, no internet access at run time.

## 2. Install

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-pipeline.txt
```

On Windows, the interpreter is `.venv\Scripts\python.exe`. Use it wherever this guide says
`.venv/bin/python`.

`requirements-pipeline.txt` pins the exact versions the submitted file was produced with. Other
versions will very likely give the same labels, but only the pinned ones are guaranteed to
reproduce the file byte for byte. The organisers' own `requirements.txt` (numpy and pandas) is
enough for the scorer, not for the pipeline.

## 3. Put the data in place

The pipeline needs **one file**, the tactical tracking:

```
data/
└─ tactical/
   ├─ BAR-ATH_tactical_atd.txt      required
   └─ BAR-ATH_tactical_atd.xml      optional: slot → team map (derived from the data if absent)
```

The other assets (videos, homography, Smart Tagging, broadcast) are not used.

**Unpacked somewhere else?** Point `MTAH_DATA_DIR` at the folder that *contains* `tactical/`:

```bash
export MTAH_DATA_DIR=/path/to/mtah2027          # macOS / Linux
$env:MTAH_DATA_DIR = "D:\mtah2027"              # Windows PowerShell
```

**Downloaded as split zips** (folders such as `tactical-20261111T110123Z-1-004/tactical/`)? Either
move the inner `tactical/` folder to `data/tactical/`, or link it:

```bash
ln -sfn tactical-20261111T110123Z-1-004/tactical data/tactical      # run inside data/
```

Check the data is found and read correctly:

```bash
.venv/bin/python -m possession.data
```

```
frames 199319  slots 46  teams from inferred (no metadata xml)
home slots [ 1 ... 24]
away slots [25 ... 46]
home defends left: {1: True, 2: False}
ball seen on 56.9% of frames
```

With the metadata XML present, the first line says `teams from metadata xml` and the slot lists
should be the same.

## 4. Generate the submission

```bash
.venv/bin/python make_submission.py
```

This loads the tracking, computes the three evidence streams, decodes, writes
`evaluation/results/yeeclaw.csv`, and runs the official `validate_submission.py` on it. The end of
the output should look like:

```
wrote evaluation/results/yeeclaw.csv
{"possessions": 248, "stoppages": 114, "mean_possession_s": 14.07, "median_possession_s": 8.96, "share": {"H": 0.332, "A": 0.217, "D": 0.451}}
Valid. Note this checks the format only -- it says nothing about whether the labels are right.
```

The command exits with code 0 only if the file is valid. **That file is the one to upload**, under
the name `yeeclaw.csv`.

### Options

| option | default | what it does |
|---|---|---|
| `--name NAME` | `yeeclaw` | output file stem: writes `NAME.csv` |
| `--out DIR` | `evaluation/results` | output folder |
| `--params JSON` | none | override decoder parameters from `possession/pipeline.py` `PARAMS`, for experiments |

Example experiment (writes a separate file and leaves `yeeclaw.csv` alone):

```bash
.venv/bin/python make_submission.py --name try_switch12 --params '{"switch_cost": 12}'
```

Never upload an experiment file. The submitted file must be what the plain command produces,
because the organisers may re-run it to confirm.

## 5. Check and score it

```bash
.venv/bin/python evaluation/validate_submission.py evaluation/results/yeeclaw.csv   # format check
.venv/bin/python evaluation/run_evaluation.py                                      # scoreboard
```

`run_evaluation.py` scores every `.csv` in `evaluation/results/` against the **public 5 minutes**.
That only proves the pipeline is wired correctly. Those 17 possessions are too few to rank
anything, and the rules forbid tuning on them (see `RULES.md`).

## 6. Reproducibility

- **Deterministic.** The same data and the same pinned packages give a byte-identical file.
  Checked on a clean clone with a fresh environment and an empty cache, with the data folder set
  through `MTAH_DATA_DIR`.
- **Cache.** The first run parses the tracking (~2 s) and trains the team-shape model (~60 s),
  then saves both in `data/_cache/`. Later runs reuse them.
  - The cache key includes the tracking data, the code that produces it and the library versions,
    so a stale cache is never reused.
  - To force a full cold run, delete the folder: `rm -rf data/_cache`.
- **Nothing to commit.** `data/` (including the cache) and `evaluation/results/*.csv` are
  gitignored, because the match data and full-match labels must not be published.

## 7. Extra tools (optional)

Neither tool reads any ground truth.

| command | what it tells you | time |
|---|---|---|
| `.venv/bin/python tools/sensitivity.py` | how much the output changes when each parameter is scaled ×0.5 … ×2; values should sit on plateaus | ~1 min |
| `.venv/bin/python tools/sensitivity.py switch_cost --factors 0.8 1.2` | the same for chosen parameters and factors | seconds |
| `.venv/bin/python tools/deadball_calibration.py` | re-derives the dead-ball constants from the tracking: anchor sets, AUCs, activity fit, P(dead) per context | ~10 s |

## 8. Troubleshooting

| symptom | cause and fix |
|---|---|
| `match data not found: .../data/tactical/BAR-ATH_tactical_atd.txt` | data not where expected; see §3 (`data/tactical/` or `MTAH_DATA_DIR`) |
| `ModuleNotFoundError: No module named 'numpy'` (or `sklearn`) | running the system Python; use `.venv/bin/python`, after §2 |
| `expected 2 goalkeeper slots` | only without the metadata XML, on unusually poor tracking; add `BAR-ATH_tactical_atd.xml` to `data/tactical/` |
| file has ~164k rows | the broadcast tracking was used; the pipeline reads `tactical/` only, so check the data folder |
| killed or out of memory on the first run | the shape-model training peaks at ~2.7 GB; close other programs and re-run (later runs need far less) |
| `RuntimeWarning: Mean of empty slice` | harmless: track slots that are empty on some frames |

## 9. Where things are

| path | what |
|---|---|
| `make_submission.py` | the one command |
| `possession/` | the pipeline: `data`, `features`, `team_shape`, `deadball`, `decoder`, `pipeline` |
| `tools/` | sensitivity and calibration checks |
| `docs/FEATURES.md` | how every feature works |
| `TECHNICAL.md` | the technical document for the submission |
| `evaluation/` | the organisers' validator and scorer (unchanged) |
