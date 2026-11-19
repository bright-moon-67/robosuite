"""Versioned, strictly aligned episode storage and episode-safe sequence replay.

Transition row t is (obs_t, action_t, reward_{t+1}, continuation_{t+1}, obs_{t+1}).
Time limits end data episodes but retain continuation=1 for value bootstrapping.
Ground-truth diagnostics are NEVER returned as policy/world-model input.
"""

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
OBS_KEYS = ("defense_robot", "defense_blocks", "defense_status", "defense_clock", "defense_eef_quat")
MODEL_KEYS = (
    "obs",
    "action",
    "next_obs",
    "reward",
    "r_disc",
    "r_diff",
    "done",
    "terminated",
    "truncated",
    "continuation",
    "is_first",
    "is_last",
)
DIAGNOSTIC_KEYS = (
    "status",
    "cleared_count",
    "breached_count",
    "block_pos_gt",
    "block_vel_gt",
    "prev_block_pos_gt",
    "prev_block_vel_gt",
    "desired_vel_gt",
    "time",
    "next_time",
    "action_source",
)


def observation_schema(n_blocks):
    if int(n_blocks) != n_blocks or n_blocks < 1:
        raise ValueError("n_blocks must be a positive integer")
    sizes = (20, 6 * int(n_blocks), int(n_blocks), 1, 4)
    start, fields = 0, {}
    for key, size in zip(OBS_KEYS, sizes):
        fields[key] = {"start": start, "stop": start + size, "size": size}
        start += size
    return {
        "version": SCHEMA_VERSION,
        "size": start,
        "fields": fields,
        "robot": "7 arm q, 7 arm dq, world eef xyz, world eef linear velocity",
        "blocks": "per block: xyz relative to eef, active, grasped, cleared",
        "eef_quat": "xyzw in world frame; robot proprioception",
    }


def vectorize_observation(obs):
    for key in OBS_KEYS:
        if key not in obs:
            raise KeyError(f"Missing observation {key}")
    n = np.asarray(obs["defense_status"]).size
    schema = observation_schema(n)
    chunks = []
    for key in OBS_KEYS:
        value = np.asarray(obs[key], dtype=np.float32).ravel()
        if value.size != schema["fields"][key]["size"] or not np.isfinite(value).all():
            raise ValueError(f"Invalid shape or non-finite observation: {key}")
        chunks.append(value)
    # Explicit allowlist: never flatten the full obs dictionary or info.
    return np.concatenate(chunks)


def validate_episode(ep):
    for key in MODEL_KEYS + DIAGNOSTIC_KEYS:
        if key not in ep:
            raise KeyError(key)
    n = len(ep["action"])
    if n < 1:
        raise ValueError("Empty episode")
    for key in MODEL_KEYS + DIAGNOSTIC_KEYS:
        value = np.asarray(ep[key])
        if value.ndim < 1 or len(value) != n:
            raise ValueError(f"Episode rows not aligned: {key}")
        if value.dtype.kind not in "buif" or not np.isfinite(value).all():
            raise ValueError(f"Non-numeric / non-finite values in {key}")
    obs, nxt, action = ep["obs"], ep["next_obs"], ep["action"]
    if obs.ndim != 2 or obs.shape != nxt.shape or action.shape != (n, 7):
        raise ValueError("Invalid obs/next_obs/action shape")
    if np.max(np.abs(action)) > 1.000001:
        raise ValueError("Action outside normalized OSC_POSE bounds")
    if not np.allclose(nxt[:-1], obs[1:], rtol=0, atol=1e-6):
        raise ValueError("Broken transition chain: next_obs[t] != obs[t+1]")
    scalar_keys = (
        "reward",
        "r_disc",
        "r_diff",
        "done",
        "terminated",
        "truncated",
        "continuation",
        "is_first",
        "is_last",
        "time",
        "next_time",
        "cleared_count",
        "breached_count",
        "action_source",
    )
    if any(ep[k].shape != (n,) for k in scalar_keys):
        raise ValueError("Scalar transition channels must have shape (T,)")
    if not np.allclose(ep["reward"], ep["r_disc"] + ep["r_diff"], rtol=0, atol=1e-5):
        raise ValueError("Reward decomposition inconsistent")
    for key in ("done", "terminated", "truncated", "continuation", "is_first", "is_last", "action_source"):
        if not np.isin(ep[key], [0, 1]).all():
            raise ValueError(f"Invalid binary channel: {key}")
    term, trunc, done = (ep[k].astype(bool) for k in ("terminated", "truncated", "done"))
    if np.any(term & trunc) or not np.array_equal(done, term | trunc):
        raise ValueError("done must equal terminated OR truncated (mutually exclusive)")
    if np.any(done[:-1]) or not done[-1] or not np.array_equal(ep["is_last"], done):
        raise ValueError("Episode must end once, on its final row")
    if not ep["is_first"][0] or np.any(ep["is_first"][1:]):
        raise ValueError("is_first must mark only the initial transition")
    if not np.array_equal(ep["continuation"], (~term).astype(float)):
        raise ValueError("Continuation must be 1-terminated, including time-limit bootstrapping")
    if ep["status"].ndim != 2 or not np.isin(ep["status"], [0, 1, 2]).all():
        raise ValueError("Invalid status labels")
    blocks = ep["status"].shape[1]
    schema = observation_schema(blocks)
    if obs.shape[1] != schema["size"]:
        raise ValueError("Observation width does not match block count/schema")
    sf = schema["fields"]["defense_status"]
    if not np.array_equal(nxt[:, sf["start"] : sf["stop"]], ep["status"]):
        raise ValueError("Status labels must describe next_obs")
    bf = schema["fields"]["defense_blocks"]
    for state in (obs, nxt):
        state_status = state[:, sf["start"] : sf["stop"]]
        b = state[:, bf["start"] : bf["stop"]].reshape(n, blocks, 6)
        if not np.isin(state_status, [0, 1, 2]).all() or not np.isin(b[:, :, 3:], [0, 1]).all():
            raise ValueError("Invalid observation status/mask values")
        if not np.array_equal(b[:, :, 3], state_status == 0) or not np.array_equal(b[:, :, 5], state_status == 1):
            raise ValueError("Active/cleared masks inconsistent with status")
    before_status = obs[:, sf["start"] : sf["stop"]]
    if np.any((before_status != 0) & (before_status != ep["status"])):
        raise ValueError("Cleared/breached statuses must remain latched")
    for status, count in ((1, "cleared_count"), (2, "breached_count")):
        if not np.array_equal((ep["status"] == status).sum(axis=1), ep[count]):
            raise ValueError("Status counts inconsistent")
    for key in ("block_pos_gt", "block_vel_gt", "prev_block_pos_gt", "prev_block_vel_gt"):
        if ep[key].shape != (n, blocks, 3):
            raise ValueError(f"Invalid diagnostic shape: {key}")
    if ep["desired_vel_gt"].shape != (n, blocks, 2):
        raise ValueError("Invalid desired velocity shape")
    if np.any(ep["next_time"] <= ep["time"]) or not np.allclose(ep["next_time"][:-1], ep["time"][1:], atol=1e-8):
        raise ValueError("Timestamps are not continuous")
    for label in ("pos", "vel"):
        if not np.allclose(ep[f"block_{label}_gt"][:-1], ep[f"prev_block_{label}_gt"][1:], atol=1e-7):
            raise ValueError("Diagnostic transition chain broken")
    # State labels are t+1, and their relative positions must match the actual observation.
    for state, label in ((obs, "prev_block_pos_gt"), (nxt, "block_pos_gt")):
        relative = state[:, bf["start"] : bf["stop"]].reshape(n, blocks, 6)[:, :, :3]
        if not np.allclose(relative + state[:, None, 14:17], ep[label], atol=2e-6):
            raise ValueError("Ground-truth position labels are misaligned")
    return True


def save_episode(path, transitions, metadata):
    path = Path(path)
    if path.suffix != ".npz" or path.exists() or path.with_suffix(".json").exists():
        raise ValueError("Use an unused .npz filename; episode overwrite is forbidden")
    if not transitions:
        raise ValueError("No transitions")
    ep = {k: np.asarray([row[k] for row in transitions]) for k in MODEL_KEYS + DIAGNOSTIC_KEYS}
    validate_episode(ep)
    meta = dict(
        metadata,
        schema=observation_schema(ep["status"].shape[1]),
        transition_convention="obs_t,action_t,reward_t+1,continuation_t+1,next_obs_t+1",
        label_convention="block_*_gt are t+1; prev_block_*_gt are t; diagnostics only",
        action_convention="7D normalized OSC_POSE world-frame delta + gripper",
    )
    serialized = json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **ep)
    path.with_suffix(".json").write_text(serialized, encoding="utf8")
    return ep


def load_episode(path):
    with np.load(path, allow_pickle=False) as f:
        ep = {k: f[k] for k in f.files}
    validate_episode(ep)
    return ep


class ReplayBuffer:
    """In-memory episode replay. Random contiguous windows never cross resets.

    ``sample`` returns state [batch,L+1,D], action/reward/continuation [batch,L,...],
    plus obs/next_obs. Only model channels are exposed; no GT or motion configs.
    Use a history/burn-in prefix when initializing a recurrent model mid-episode.
    Capacity is measured in whole episodes; mixed observation widths are rejected.
    """

    def __init__(self, capacity_episodes=1000, seed=0):
        if int(capacity_episodes) != capacity_episodes or capacity_episodes < 1:
            raise ValueError("capacity_episodes must be positive integer")
        self.capacity = int(capacity_episodes)
        self.rng = np.random.default_rng(seed)
        self.episodes = []
        self.shape = None

    def add(self, episode):
        ep = load_episode(episode) if isinstance(episode, (str, Path)) else episode
        validate_episode(ep)
        shape = (ep["obs"].shape[1], ep["action"].shape[1])
        if self.shape is not None and shape != self.shape:
            raise ValueError("Mixed schemas/block counts require separate replay buffers")
        self.shape = shape
        self.episodes.append({k: np.array(ep[k], copy=True) for k in MODEL_KEYS})
        self.episodes = self.episodes[-self.capacity :]

    def sample(self, batch_size, length):
        if any(int(x) != x or x < 1 for x in (batch_size, length)):
            raise ValueError("batch_size and length must be positive integers")
        windows = np.array([max(0, len(ep["action"]) - length + 1) for ep in self.episodes])
        if windows.sum() == 0:
            raise ValueError("No episode long enough for this sequence length")
        rows = []
        for _ in range(int(batch_size)):
            index = int(self.rng.choice(len(windows), p=windows / windows.sum()))
            start = int(self.rng.integers(windows[index]))
            rows.append({k: v[start : start + length] for k, v in self.episodes[index].items()})
        result = {k: np.stack([r[k] for r in rows]) for k in MODEL_KEYS}
        result["state"] = np.concatenate([result["obs"][:, :1], result["next_obs"]], axis=1)
        return result
