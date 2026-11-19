# World Model verification gates / 60-D defense observation

## Unit tests (must be 100% passing)
- `pytest -q test_worldmodel_stage.py`
- obs 60/action 7; world block transform = relative + EEF world
- no cross-episode window; no future observed states supplied to `forecast`
- time-limit truncation continuation=1; true termination continuation=0
- multi-step prior output shapes and gradients finite
- posterior filter resets at episode boundaries
- checkpoint records frozen train/val/test file names

## Benchmark split
- train / validation / test: 70 / 15 / 15 entire episodes, with immutable split
- hold back new, never-in-training evaluation seeds for each trajectory family
- preserve real modes and source file paths in experiment manifest

## Prediction metrics
A) Open-loop block world-position L2 error (m), at 10/20/40 steps
   report mean/median/P90, per block; do not use future observations
   (recorded future actions are allowed for action-conditional offline test).
B) Report separate active+never-grasped, grasping-contact and cleared groups.
C) Compare to constant velocity, constant acceleration, and EKF.
D) Breach within horizon: PR-AUC, recall, precision, calibration, false-negative rate.
   Tune thresholds only on validation. Rare positives -> `null`, not 0.
E) Report event time error and ordering of the 5 threats by predicted time to breach.
F) Constrain comparison to cases where an attack is physically interceptable.
G) Test object-index permutation and contact-swap augmentation where applicable.

## Test distributions
1. same sine family, unseen seeds and unseen phase/frequency combinations
2. same family but speed out-of-range
3. constant linear radial
4. pulsed/accelerating or turning with phase changes
5. gripper grasp/release and surprising object contact
6. switch from heuristic/random mixture to a learned Actor

## Suggested research milestones (NOT measured thresholds)
- first: RSSM filters and prior train without NaN/shape errors
- second: on withheld same-distribution data, compare 0.5,1.0,2.0s open-loop
- third: demonstrate where RSSM improves over a CV or EKF predictor;
  if it does not, investigate context length, data coverage and latent KL.
- fourth: event/uncertainty calibration and action-conditioned contact response
- fifth: freeze a world model checkpoint and integrate with explicit arm model

## Known limitations of this reference
- Gaussian latent, not exact DreamerV3 categorical latent.
- No explicit gripper contact physics; condition on actions/history only.
- Gaussian prior initialized with mean for deterministic evaluation;
  sample multiple prior particles for uncertainty calibration study.
- Offline trajectories do not test on-policy imagination bias.
- No robot simulator gradient path; not yet a full DiffDreamer implementation.
