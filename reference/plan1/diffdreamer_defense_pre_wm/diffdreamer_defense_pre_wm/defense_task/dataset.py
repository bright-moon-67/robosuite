"""Stable replay episode I/O before RSSM training."""
from pathlib import Path
import json
import numpy as np


OBS_KEYS = ("defense_robot", "defense_blocks", "defense_status", "defense_clock")


def vectorize_observation(obs):
    """Fixed flat state vector, with no future path parameters or simulator oracle."""
    chunks = []
    for k in OBS_KEYS:
        if k not in obs:
            raise KeyError(f"Missing observation {k}. Import DefenseTask before making env.")
        arr = np.asarray(obs[k], dtype=np.float32).ravel()
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite observation: {k}")
        chunks.append(arr)
    return np.concatenate(chunks, axis=0)


def validate_episode(ep):
    keys = ("obs", "action", "next_obs", "reward", "r_disc", "r_diff", "done")
    for k in keys:
        if k not in ep:
            raise KeyError(k)
    n = len(ep["action"])
    if n < 1:
        raise ValueError("Empty episode")
    if any(len(ep[k]) != n for k in keys):
        raise ValueError("Episode rows not aligned: o_t, a_t, o_{t+1}")
    if ep["obs"].shape != ep["next_obs"].shape:
        raise ValueError("obs and next_obs shape mismatch")
    if not np.isfinite(ep["obs"]).all() or not np.isfinite(ep["next_obs"]).all():
        raise ValueError("NaN / inf observations")
    if not np.allclose(ep["reward"], ep["r_disc"] + ep["r_diff"], atol=1e-5):
        raise ValueError("reward decomposition inconsistent")
    if np.any(ep["done"][:-1]):
        raise ValueError("No transitions may follow termination")
    return True


def save_episode(path, transitions, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ep = {}
    if not transitions:
        raise ValueError("No transitions")
    keys = ("obs", "action", "next_obs", "reward", "r_disc", "r_diff", "done",
            "status", "cleared_count", "breached_count",
            "block_pos_gt", "block_vel_gt")
    for k in keys:
        ep[k] = np.asarray([tr[k] for tr in transitions])
    validate_episode(ep)
    np.savez_compressed(path, **ep)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf8")
    return ep


def load_episode(path):
    with np.load(path, allow_pickle=False) as f:
        ep = {k: f[k] for k in f.files}
    validate_episode(ep)
    return ep
