# Defense World Model — next stage after pre-WM data collection

This reference starts from your real `.npz` trajectories, NOT raw images. It trains a small **Gaussian RSSM** in PyTorch.

**Important:** It is a transparent reference implementation and not a claim of exact DreamerV3 or DiffDreamer reproduction. It deliberately decodes external block **world positions** (and object status, discrete reward, continuation), not the full high-frequency mechanical arm state. The future robot physical rollout will be a separate module.

## Install

Run from your existing robosuite Python environment:
```bash
pip install "numpy>=1.24" "torch>=2.2" "pytest>=8"
```

## Stage 0 - inspect replay
```bash
python worldmodel_stage.py inspect \
  --data /home/pan/projects/robosuite/artifacts/defense_pre_wm/datasets/sine5
```
The script assumes that the 60-D observation order is:
- [0:14] 7 joint positions + 7 joint velocities
- [14:20] EEF world position XYZ + EEF linear velocity XYZ
- [20:50] 5 blocks each [relative XYZ, attacking, grasped, cleared]
- [50:55] 5 categorical statuses [0 attacking, 1 cleared, 2 breached]
- [55] normalized time
- [56:60] quaternion

This matches **the user-reported schema**, not verified direct access to that local path.
`block_pos_gt`, `block_vel_gt`, hidden motion params and future trajectories are NOT used for model training.

## Stage 1 - train
```bash
python worldmodel_stage.py train \
  --data /home/pan/projects/robosuite/artifacts/defense_pre_wm/datasets/sine5 \
  --out runs/defense_rssm_v0 \
  --device cuda --steps 5000 --batch 16 --length 32
```

Train/val/test split is by entire episode, seeded (default 42).
Only training episodes are used for normalization.
Losses include posterior/prior object-position prediction, status CE, symlog discrete reward,
weighted continuation BCE, and stopped-gradient KL regularizers (free-nat).
The numeric weights are starter settings, not original DiffDreamer hyperparameters.
`world_model.pt` saves the best checkpoint under teacher-forced validation loss.

## Stage 2 - independent open-loop evaluation
```bash
python worldmodel_stage.py eval \
  --data /home/pan/projects/robosuite/artifacts/defense_pre_wm/datasets/sine5 \
  --out runs/defense_rssm_v0 --device cuda
```
Evaluation uses observed states only in the 20-step context. Afterward, it advances the prior using
**recorded future actions** and no future observations. This is action-conditioned open-loop prediction,
not closed-loop RL evaluation. Results compare against a constant-velocity forecast and show breach
event detection (precision/recall at probability 0.5). Rare-event metrics are `null`
when the test split has no positive examples. Change the threshold based on
validation data, not the held-out test split.
`always_active_wm_m` scores blocks that stay attacking AND ungrasped through the entire prediction window, separating
their free motion from complex grasped-object motion.

Predictions at horizons 10,20,40 correspond to 0.5,1,2 seconds at 20 Hz.

## Stage 3 - test each distribution
Use completely disjoint episodes for:
- sine same-range velocities and different seeds
- slower/faster sine
- straight-line
- pulsed/variable-speed
- contact and post-grasp stages
No claims of cross-distribution generalization without this evaluation.

## Stage 4 - future integration
The checkpoint freezes train/val/test episode filenames, so `eval --seed`
cannot accidentally change the held-out split.
The `initialize` + `transition` contract exposes posterior inference and prior imagination.
When the differentiable robot model is ready, feed **predicted explicit robot state** to
an interaction-aware RSSM transition or contact model. Do not use future robot ground truth
as if available at deployment.

## Cautions
- This example uses a Gaussian latent, not the exact DreamerV3 categorical latent and two-hot heads.
- Sparse discrete events are hard to fit. Evaluate breach **recall**, PR-AUC and calibration separately.
- Large errors during grasping may indicate missing robot-object interaction modeling, not merely a need for more replay.
- For exact DreamerV3 components, prefer the official JAX implementation or TorchRL.
- `--data` requires your actual episode files; a synthetic `test_worldmodel_stage.py` tests interfaces only.
