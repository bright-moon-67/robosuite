"""Action-conditioned offline prediction metrics; future states are labels only."""

import numpy as np
import torch


def world_positions(obs):
    blocks = obs[..., 20:50].reshape(*obs.shape[:-1], 5, 6)
    return blocks[..., :3] + obs[..., 14:17][..., None, :]


def constant_velocity(context, horizon):
    positions = world_positions(context)
    velocity = positions[-1] - positions[-2]
    return positions[-1] + np.arange(1, horizon + 1)[:, None, None] * velocity


def constant_acceleration(context, horizon):
    positions = world_positions(context)
    acceleration = positions[-1] - 2 * positions[-2] + positions[-3]
    steps = np.arange(1, horizon + 1)[:, None, None]
    return constant_velocity(context, horizon) + steps * (steps + 1) / 2 * acceleration


def error_summary(values):
    values = np.asarray(values)
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()) if values.size else None,
        "median_m": float(np.median(values)) if values.size else None,
        "p90_m": float(np.percentile(values, 90)) if values.size else None,
    }


def event_summary(truth, scores):
    truth, scores = np.asarray(truth, dtype=bool), np.asarray(scores)
    predicted = scores > 0.5
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    # Average precision (step-integrated PR area); aggregate tied scores together.
    ap = None
    if truth.any():
        order = np.argsort(-scores, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
        cumulative = np.cumsum(truth[order])[ends]
        precision = cumulative / (ends + 1)
        recall = cumulative / truth.sum()
        ap = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    return {
        "count": len(truth),
        "positives": int(truth.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_negative_rate": fn / (tp + fn) if tp + fn else None,
        "pr_auc_average_precision": ap,
        "brier": float(np.mean((scores - truth) ** 2)) if len(truth) else None,
    }


@torch.no_grad()
def openloop_report(
    model, episodes, stats, horizons=(10, 20, 40), burn=20, windows_per_ep=2, seed=777, device="cpu", stride=None
):
    """Report endpoint and trajectory errors, contact strata and breach risk.

    Future observations are used only for scoring after forecasting. Contact labels
    are the dataset's grasped flag, not a reconstruction of all MuJoCo contacts.
    Random windows estimate normal prevalence; stride mode includes terminal windows.
    """
    if burn < 2 or windows_per_ep < 1 or not horizons or min(horizons) < 1 or (stride is not None and stride < 1):
        raise ValueError("invalid context, horizon or window sampling arguments")
    model.eval()
    report = {}
    for horizon in horizons:
        rng = np.random.default_rng(seed)
        groups = {
            name: {baseline: [] for baseline in ("wm", "cv", "ca")}
            for name in ("all", "active_ungrasped", "grasped", "cleared")
        }
        trajectories = {name: [] for name in ("wm", "cv", "ca")}
        per_block = [[] for _ in range(5)]
        truths, risks, block_truths, block_risks = [], [], [], []
        time_errors, threat_correct = [], []
        reward_errors, cont_errors = [], []
        windows, skipped = 0, 0
        for ep in episodes:
            last_start = len(ep["action"]) - burn - horizon
            if last_start < 0:
                skipped += 1
                continue
            if stride is None:
                starts = rng.choice(last_start + 1, min(windows_per_ep, last_start + 1), replace=False)
            else:
                starts = sorted(set(range(0, last_start + 1, stride)) | {last_start})
            for start in starts:
                b = start + burn
                context = ep["states"][start : b + 1]
                predicted, statuses, reward, cont = model.forecast(
                    torch.as_tensor(context, device=device)[None],
                    torch.as_tensor(ep["action"][start:b], device=device)[None],
                    torch.as_tensor(ep["action"][b : b + horizon], device=device)[None],
                    stats,
                )
                predictions = {
                    "wm": predicted[0].cpu().numpy(),
                    "cv": constant_velocity(context, horizon),
                    "ca": constant_acceleration(context, horizon),
                }
                future = ep["states"][b + 1 : b + horizon + 1]
                truth = world_positions(future)
                current_active = context[-1, 50:55] == 0
                flags = ep["states"][b : b + horizon + 1, 20:50].reshape(-1, 5, 6)
                grasped = flags[:, :, 4].astype(bool).any(axis=0)
                active = current_active & (future[:, 50:55] == 0).all(axis=0) & ~grasped
                masks = {
                    "all": np.ones(5, dtype=bool),
                    "active_ungrasped": active,
                    "grasped": grasped,
                    "cleared": context[-1, 50:55] == 1,
                }
                for name, prediction in predictions.items():
                    errors = np.linalg.norm(prediction - truth, axis=-1)
                    trajectories[name].extend(errors.ravel().tolist())
                    for group, mask in masks.items():
                        groups[group][name].extend(errors[-1, mask].tolist())
                    if name == "wm":
                        for i in range(5):
                            per_block[i].append(float(errors[-1, i]))
                probability = statuses[0, :, :, 2].cpu().numpy()
                breached = future[:, 50:55] == 2
                actual = breached.any(axis=0) & current_active
                scores = probability.max(axis=0)
                if current_active.any():
                    truths.append(bool(actual.any()))
                    risks.append(float(scores[current_active].max()))
                    block_truths.extend(actual[current_active].tolist())
                    block_risks.extend(scores[current_active].tolist())
                true_times = np.where(actual, breached.argmax(axis=0) + 1, horizon + 1)
                pred_times = np.where(
                    (probability > 0.5).any(axis=0) & current_active,
                    (probability > 0.5).argmax(axis=0) + 1,
                    horizon + 1,
                )
                detected = actual & (pred_times <= horizon)
                time_errors.extend((np.abs(pred_times[detected] - true_times[detected]) / 20).tolist())
                if actual.any() and current_active.sum() > 1:
                    # No threshold crossing is a missed threat, not a correct tie.
                    first = np.flatnonzero(pred_times == pred_times.min())
                    threat_correct.append(
                        float(pred_times.min() <= horizon and np.all(true_times[first] == true_times.min()))
                    )
                reward_errors.extend(np.abs(reward[0].cpu().numpy() - ep["r_disc"][b : b + horizon]).tolist())
                cont_errors.extend(((cont[0].cpu().numpy() - ep["continuation"][b : b + horizon]) ** 2).tolist())
                windows += 1
        report[str(horizon)] = {
            "windows": windows,
            "skipped_short_episodes": skipped,
            "horizon_seconds": horizon / 20,
            "endpoint": {g: {k: error_summary(v) for k, v in values.items()} for g, values in groups.items()},
            "trajectory": {k: error_summary(v) for k, v in trajectories.items()},
            "wm_per_block": [error_summary(v) for v in per_block],
            "breach_window": event_summary(truths, risks),
            "breach_block": event_summary(block_truths, block_risks),
            "detected_breach_time_mae_seconds": float(np.mean(time_errors)) if time_errors else None,
            "detected_breach_count": len(time_errors),
            "first_threat_count": len(threat_correct),
            "first_threat_accuracy": float(np.mean(threat_correct)) if threat_correct else None,
            "reward_mae": float(np.mean(reward_errors)) if reward_errors else None,
            "continuation_brier": float(np.mean(cont_errors)) if cont_errors else None,
        }
    return report
