# Ball possession from automatic tracking: technical document

Team **furbito**, More than a Hack 2027, Phase 1.

## 1. Approach in one paragraph

We treat possession as a **segmentation** problem, not a per-frame classification, because the
ranking metric (spellF1) scores whole possessions. Three independent evidence streams are computed
per frame, each as a log-odds in nats. They are: **ball control** (who has the ball at their feet),
**team shape** (which team the 22 players' configuration says is attacking), and **dead ball**
(whether play is stopped). A three-state model {H, A, D} decodes them with **Viterbi**, with an
explicit cost on every state change. A possession change happens only when accumulated evidence
pays for it. That is our reading of "controlled" possession: a touch or a deflection does not pay
for a change, sustained control does. Nothing is fitted to human labels. The one learned component
is trained on pseudo-labels that the algorithm derives from the tracking itself.

```
ATD ──► ball control evidence ─┐
    ──► team-shape model ──────┼─► team log-odds ─┐
    ──► dead-ball detector ────┴──────────────────┴─► emissions ─► Viterbi per half ─► H/A/D (X outside halves)
```

## 2. Assets

| asset | used? | how |
|---|---|---|
| tactical ATD (players + ball) | **yes** | everything |
| tactical ATD metadata XML | yes when present | slot→team map; without it the map is derived from the data (§4) |
| Smart Tagging | no | not available to us in time; it would be a natural prior for restarts (§7) |
| homography, videos | no | not needed: the tracking is already in pitch coordinates |
| broadcast ATD | no | own timeline, not aligned; its ball would add little in the time available |
| public 5-min ground truth | **pipeline check only** | frame alignment and file format; never used to tune or choose |

## 3. The data problem that shaped the design

The ball is missing on 41% of graded frames, but **not as flicker**. It is missing in about 150
long gaps (median 11 s, p90 34 s). Gaps of 1–3 s do not exist; the tracker bridges short losses
itself. Interpolating the ball is therefore pointless, and two questions have to be answered
**without the ball**:

1. *Is play stopped?* Most long gaps are stoppages: by collective player activity, 81% of gap time
   looks like a stoppage.
2. *If not, which team has the ball?* This needs a player-only model.

## 4. Components

### 4.1 Loading and slot→team (`possession/data.py`)
The raw ATD is parsed with the pandas C engine (~2 s) into metres on the video-frame axis
(`video_frame = frame_id − 1`) and cached. Slots map to teams from the metadata XML. If the XML is
absent, the map is derived from the data, following the metadata's own layout (one contiguous home
block, then one away block, each ending with its keeper). The keepers are the only slots that sit
next to a goal line for a whole half, so the home block ends at the lower-indexed keeper slot.
On this match this agrees exactly with the published baseline's assignment. Each team's attacking
direction per half also comes from the keepers' positions.

### 4.2 Ball-control evidence (`possession/features.py`)
For each team: the distance from the ball to its nearest player, and the ball's speed relative to
that player. Control strength = `exp(−d²/2σ²)` with σ = 1.2 m (reach), gated by a logistic on
relative speed centred at 6 m/s. A ball passing a player faster than that is in flight, not under
control. Team log-odds = control(home) − control(away), plus a weak 0.1·tanh(Δd/3 m) "nearer team"
tilt for loose balls. When the ball is unseen or out of bounds the evidence is 0, so the decoder
keeps its state.

### 4.3 Team-shape model (`possession/team_shape.py`)
- **Features.** Team-symmetric "us vs them" features in the attacking frame of "us":
  - centroid, spread, length and width, and the most advanced and deepest lines;
  - share of players in the opponent half and the final third;
  - velocity toward the opponent goal;
  - pressing (opponents closing in) and nearest-opponent spacing;
  - home-minus-away differences, with 1/2/5 s rolling means, trends and look-ahead/look-back
    windows. The look-ahead windows matter because a team's shape reorganises 1–3 s after a turnover.
- **Pseudo-labels.** On ball-visible frames, a frame is labelled when:
  - the nearest player is within 1.5 m;
  - the ball is slower than 3 m/s relative to him;
  - no opponent is within 1.5 m;
  - the ball is in the pitch;
  - and the frame is in a ±1 s window with ≥ 90% agreement.

  Frames between two same-team controls ≤ 3 s apart are filled in. That gives 54k frames (34% of
  graded). No human decided any label.
- **Model.** Gradient-boosted trees (`HistGradientBoostingClassifier`, fixed seed). Every frame is
  presented from both perspectives, and the output is antisymmetrised, so there is no home bias by
  construction.
- **Validation.** Blocked 5-fold CV (contiguous 4-min blocks, 8 s purge) against the pseudo-labels:
  accuracy 0.89, AUC 0.955. On ball-missing gaps bracketed by the same team: 27/27 gaps correct,
  frame accuracy 0.94. Temperature scaling (T ≈ 2) calibrates the log-odds out-of-fold.
- **Inference.** Runs on every graded frame, ball or not.

### 4.4 Dead-ball detector (`possession/deadball.py`)
A naive-Bayes sum of physically scaled terms, in nats:
- **ball context.** In bounds, a function of ball speed: a still ball suggests a set piece, a fast
  ball live play. Seen 0.5–12 m outside a line: dead. Missing: a function of the whole gap's length,
  since long gaps are almost always stoppages.
- **collective activity.** The 2 s mean speed of the five fastest players, logistic around
  3.15 m/s. This centre was confirmed by a balanced logistic fit on algorithmic anchors.
- **out-carry.** The ball stays dead while unseen after last being seen out.
- **stationary ball.** Over 1 s, with a bonus at restart locations (touchline, corner, goal area,
  centre and penalty spots).
- **kick-off formation.** Both teams in their own halves with the ball at the centre spot.
- **pre-kick-off stretch.** From the start of each graded half to the end of the first detected
  kick-off line-up. Formation runs less than 30 s apart are merged, because a line-up breaks and
  re-forms before the whistle. This is how the 2nd half's long dead opening is found **without
  hard-coding any frame**.
- **tracked-player count drop.**

Validated without labels by leave-one-cue-out AUC: player cues on ball-defined anchors 0.86, ball
cues on player-defined anchors 0.93. `tools/deadball_calibration.py` re-derives every
data-estimated constant from the ATD alone: the anchor sets, these AUCs, the activity logistic fit,
the mixture estimate of P(dead) per ball context, and the weight/cost behaviour.

### 4.5 Decoder (`possession/decoder.py`, `possession/pipeline.py`)
- **Emissions:** H = ½·team − ½·dead, A = −½·team − ½·dead, D = ½·dead.
- **Transition costs:** H↔A = 8 nats; alive↔dead = 8 nats each way.
- **Weights:** shape enters at 0.03 per frame when the ball is seen (ball control sets boundaries)
  and 0.08 when it is unseen (shape is then the only team evidence). Dead log-odds enter at 0.15,
  clipped to ±6. These weights are small because the shape and dead streams are smooth and heavily
  autocorrelated. Their per-frame values are calibrated, but adding them up frame by frame would
  count the same evidence many times.
- **Decoding:** Viterbi runs separately on each graded half. Frames outside the halves are `X`
  (never graded).

## 5. Possession rules implemented

- **Change of possession:** the other team must show enough accumulated control evidence to pay
  the switch cost, about 1/3 s of clear, close, slow-relative control. A touch, a deflection, or a
  ball flying past a player (relative speed > 6 m/s) does not pay for it.
- **Loose and contested balls:** with both teams near the ball, control evidence roughly cancels and
  possession stays with the last team, as CHALLENGE.md specifies.
- **Ball unseen:** team evidence comes only from shape, at a low weight. Possession changes in a gap
  only when the shape evidence is sustained.
- **Stoppages** begin when dead evidence accumulates (ball out, play frozen, ball set at a restart
  spot). They end when the evidence turns (ball struck, players accelerate). The next possession
  goes to whichever team's control evidence comes first. We never label dead-ball ownership:
  dead is `D`.
- **No `X` on graded frames.** The ranking metric forward-fills `X`, so a decoded state is never
  worse than abstaining.

## 6. How parameters were chosen (no ground truth)

- **Physical scales:** reach, pass speed, walking speed and pitch geometry.
- **The components' own self-supervised calibration:** CV temperature, and anchor logistic fits.
- **A sensitivity sweep** (`tools/sensitivity.py`). It scales each parameter ×0.5 to ×2 and
  compares the result with the base output using the official metric code, the base output acting
  as the reference. Values sit on plateaus (self-spellF1 ≥ 0.83 at ×2, ≥ 0.91 at ×0.75/×1.5).
- **Football plausibility of the output:** 45% dead time (LaLiga effective playing time is
  ~51–55 min of ~95), 114 stoppages (typical 100–120), 248 possessions.

The public 5 minutes were scored only to check that the pipeline was wired correctly. They were
never used to choose between versions or values.

## 7. Limitations and what we would do next

- **Turnovers inside ball-missing gaps** rely on the shape model. It lags 1–3 s at a change and
  is less sure in loose or aerial phases.
- **Slow build-up and a goalkeeper holding the ball** can look like a stoppage: walking players
  and a still ball.
- **Quick restarts** (a throw-in taken within 3–4 s) can be merged into the surrounding play.
- **Next steps:**
  - Smart Tagging as a restart and team prior;
  - a ghost-ball estimator (ball position from players) for carrier evidence inside gaps. We
    prototyped it at 0.75 carrier-team accuracy on simulated gaps, below the shape model, so we
    left it out;
  - re-estimating emission weights by EM on the match itself.

## 8. Training data

No labelled dataset was used. The only learned component (the team-shape classifier) is trained
on pseudo-labels computed by an algorithm from the provided tracking (§4.3). No human labelled
any frame of any match.

## 9. Reproduction

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-pipeline.txt
# data bundle unpacked into ./data (data/tactical/BAR-ATH_tactical_atd.txt [+ .xml]) or MTAH_DATA_DIR
.venv/bin/python make_submission.py            # -> evaluation/results/furbito.csv, then validates it
```

About 70 s cold, of which ~60 s is training the shape model (cached in `data/_cache/`), and about
5 s warm. The output is deterministic: byte-identical across cold runs.
