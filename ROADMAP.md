# Roadmap: ball possession from automatic tracking

Goal: for each of the 199,319 tactical frames, output `H` / `A` / `D` / `X`. Ranked on **spellF1**
(whole possessions matched by temporal IoU). Floor to beat: **0.2433** (`baseline_smooth50`).

## Constraints that shape the design

- **spellF1 rewards whole, correctly bounded possessions.** Per-frame flicker is fatal, so the
  core of the system is a *segmenter* (a state model with switching costs), not a per-frame
  classifier.
- **The public 5 minutes are a pipeline check, not a tuning set** (RULES.md §2). Parameters come
  from football and physics priors (control radius, ball speeds, restart geometry), and we choose
  between variants with unsupervised checks: how stable the output is under small parameter
  changes, whether the counts are plausible for a football match, and whether the output agrees
  with Smart Tagging, which is itself algorithmic. We do not choose by public-window score.
- **No hand labels and no frame-specific constants.** Every rule has to work on another match too.
  The second-half kickoff, for example, must be *detected*, not hard-coded.
- **Reproducible**: one command regenerates the same file from the data bundle.

## Data actually on disk (2026-11-17)

| asset | status | used for |
|---|---|---|
| tactical ATD `.txt` | present | everything |
| tactical ATD `.xml` | **missing** | slot→team map. Fallback: derived from the data (home block ends at the home GK slot) |
| Smart Tagging | **missing** | restart and team prior (optional; used when present) |
| homography / video | **missing** | not needed for v1 |
| broadcast ATD | xml only | not used |

The download came as split Google-Drive zips (`…-1-002`, `…-1-004`); parts `-001`/`-003`
probably hold the missing files. Getting them unlocks Phase 5.

## Phases

### Phase 0: setup ✅
venv (`numpy`, `pandas`, `scipy`), data symlinks into the layout `games.py` expects, and the slot
teams checked against the committed `baseline_raw.csv` (slots 1–24 home, 25–46 away).

### Phase 1: foundation
- `possession/data.py`: fast ATD parse (pandas C engine, ~2 s) into a cached `.npz`, slot teams
  from the XML or the fallback, and half boundaries.
- `possession/features.py`: per-frame features in metres: the ball with short gaps interpolated,
  ball velocity, nearest-player distance per team, player activity (mean speed), in-bounds flag,
  and player count.
- `make_submission.py`: the single entry point, ending with validation.

### Phase 2: dead-ball detector
Evidence for `D`: the ball outside the pitch (plus a margin); the ball unseen for a long stretch;
a collective stoppage (low mean player speed); and a **restart setup** (the ball stationary at a
restart location: touchline, corner, goal area, centre spot). Kick-off formation (both teams in
their own halves, ball on the centre spot) catches the pre-kickoff `D` stretch at the start of
the 2nd half without hard-coding a frame. A `D` interval ends when the ball is struck from the
restart spot.

### Phase 3: possession decoder (the core)
- **Control evidence** per frame: the ball within a control radius of a player of team T and
  moving slowly *relative to that player*. In flight, a pass or a loose ball gives no evidence,
  so the previous possession is kept (that is the rule "contested, so stays with the last team").
- **3-state HMM {H, A, D}, decoded with Viterbi.** Emissions are log-likelihoods from the
  evidence above. Transitions carry explicit switching costs: H↔A costs C_switch, alive↔dead costs
  C_dead. To switch, the evidence has to pile up past the cost, so one touch or deflection
  does not flip possession but sustained control does. That is how CHALLENGE.md defines
  "controlled".
- Missing ball means neutral evidence for H/A, so the state carries through occlusions. This
  matches how the scorer forward-fills `X`.

### Phase 4: restarts
After a `D` stretch, the next possession goes to the team whose player is at the ball when it is
struck (and with Smart Tagging, to its `SET PIECES` team when it has one).

### Phase 5: Smart Tagging prior (needs the missing file)
`SET PIECES` intervals give strong `D`→alive anchors. Phase codes (Build Up, Progression,
Creating Chances, Transitions) add soft team evidence over their interval, only as a prior,
because phases are not possessions.

### Phase 6: robustness and self-checks (no GT)
- Sensitivity sweeps: the output segmentation should change smoothly with each parameter. We
  pick values on plateaus, not peaks.
- Plausibility: possession count and mean duration in line with published football statistics
  from *other* matches, and the D share.
- Agreement with Smart Tagging (algorithmic labels, so allowed for validation).

### Phase 7: deliverables
Pinned `requirements.txt`, `python make_submission.py` → `evaluation/results/<team>.csv`, then
`validate_submission.py`, and `TECHNICAL.md` (approach, assets used or dropped, noise handling,
possession rules, limitations, reproduction).

## Stretch ideas (if time allows)
- Learn emission weights self-supervised: train on the high-confidence frames the decoder
  already produces, then decode again (pseudo-labels from an algorithm are allowed).
- Ball trajectory reconstruction through occlusions (a Kalman filter plus a player-carry
  hypothesis).
- Use the broadcast ATD as a second view of the ball once it is aligned to tactical.
