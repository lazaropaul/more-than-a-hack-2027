# Furbito — Technical Document

## Approach

The pipeline combines the tactical automatic tracking data (ATD) with automatic Smart Tagging.
For visible-ball frames, it assigns the team of the nearest player when that player is within
5 metres. Frames with a close, contested pair of opposing players are left uncertain instead of
forcing a noisy choice. A team change must persist for 50 frames (2 seconds) before the transition
is accepted; once confirmed, the transition is placed at the start of the sustained evidence.
This temporal hysteresis reduces short tracker-induced possession flips without training a model
on the match.

The detector is deliberately heuristic: the repository provides no labelled training set from
other matches, and the rules prohibit using hand labels from this match to train or tune it.

## Data used

- **Tactical ATD:** player and ball coordinates, team membership, nearest-player distances and
  ball speed.
- **Smart Tagging:** possession-phase team tags fill uncertain frames only. Set-piece intervals
  identify a possible restart when the ball is subsequently observed moving; the interval before
  that movement is labelled dead ball, and its team tag can support the first 75 frames after
  restart.
- **Not used:** broadcast video/tracking, tactical video frames, homography, external labelled
  data or private ground truth.

The public five-minute labels may be used only to check that the pipeline and output format work;
they are not used to fit parameters or select between variants. No training data is used.

## Missing ball, dead balls and possession rules

Missing ball observations are not automatically treated as dead ball. A set-piece tag contributes
`D` only from its start until the first detected two-frame movement burst, and only when that
restart is found within six seconds of the tag start. A team tag can fill uncertain frames, but
never overrides a tracking-based decision. `N/A` team tags are ignored.

The closest team is assigned only when at least one tracked player is within 5 metres. If both
teams have a player within 3 metres and their nearest distances differ by less than 0.75 metres,
the frame is treated as uncertain. Sustained opposing-team evidence is required for 50 frames
before a possession change is accepted. A confirmed change is backdated to its first evidence
frame. Remaining uncertain gaps are filled with the preceding state within each half (the opening
gap is filled from the first known state), so `X` is reserved for frames outside play. Smoothing
and filling are performed independently on each half.

## Limitations and future work

The ball and player tracking are automatic and can be missing or misplaced. Smart Tagging is also
automatic, and its phase intervals are not possession ground truth. The restart detector can miss
stoppages without a suitable set-piece tag. A missing-ball gap of at least 3 seconds is called
dead only when the last observed ball position was within 3 metres of a pitch boundary; other
long gaps remain uncertain. This can miss stoppages and can mistake an in-play boundary occlusion
for a stoppage.
Contested balls, loose balls and restarts therefore remain uncertain. With more time, we would add
a broader out-of-play detector and assess it on labelled data from other matches.

## Reproduction

From the repository root, with the enrolled match bundle unpacked under `data/`:

```bash
pip install -r requirements.txt
python run.py
python evaluation/validate_submission.py evaluation/results/furbito.csv
```

The runner extracts features from the tactical source, generates
`evaluation/results/furbito.csv`, and invokes the validator. The match data and generated CSV are
not committed to the public repository.
