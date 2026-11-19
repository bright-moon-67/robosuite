import copy

import numpy as np
import pytest
from defense_task.dataset import (
    DIAGNOSTIC_KEYS,
    MODEL_KEYS,
    OBS_KEYS,
    ReplayBuffer,
    load_episode,
    observation_schema,
    save_episode,
    validate_episode,
    vectorize_observation,
)


def observation(n=2, time=0):
    robot = np.zeros(20)
    robot[14:17] = [0, 0, 1]
    blocks = np.zeros((n, 6))
    blocks[:, :3] = [0.35 - 0.01 * time, 0, -0.178]
    blocks[:, 3] = 1
    return dict(
        defense_robot=robot,
        defense_blocks=blocks.ravel(),
        defense_status=np.zeros(n),
        defense_clock=[time / 70],
        defense_eef_quat=[0, 0, 0, 1],
    )


def transitions(n=2, length=8):
    rows = []
    for t in range(length):
        before, after = observation(n, t * 0.05), observation(n, (t + 1) * 0.05)
        pos = after["defense_blocks"].reshape(n, 6)[:, :3] + after["defense_robot"][14:17]
        prev = before["defense_blocks"].reshape(n, 6)[:, :3] + before["defense_robot"][14:17]
        done = t == length - 1
        rows.append(
            dict(
                obs=vectorize_observation(before),
                next_obs=vectorize_observation(after),
                action=np.zeros(7),
                reward=-0.2,
                r_disc=0.0,
                r_diff=-0.2,
                done=done,
                terminated=False,
                truncated=done,
                continuation=1.0,
                is_first=t == 0,
                is_last=done,
                status=np.zeros(n),
                cleared_count=0,
                breached_count=0,
                block_pos_gt=pos,
                block_vel_gt=np.zeros((n, 3)),
                prev_block_pos_gt=prev,
                prev_block_vel_gt=np.zeros((n, 3)),
                desired_vel_gt=np.zeros((n, 2)),
                time=t * 0.05,
                next_time=(t + 1) * 0.05,
                action_source=0,
            )
        )
    return rows


@pytest.fixture
def ep():
    rows = transitions()
    return {k: np.asarray([r[k] for r in rows]) for k in MODEL_KEYS + DIAGNOSTIC_KEYS}


def test_roundtrip_and_no_overwrite(tmp_path):
    path = tmp_path / "episode.npz"
    save_episode(path, transitions(), {"seed": 3})
    assert validate_episode(load_episode(path))
    with pytest.raises(ValueError, match="overwrite"):
        save_episode(path, transitions(), {})
    assert path.with_suffix(".json").exists()


def test_input_allowlist_and_shapes():
    obs = observation()
    original = vectorize_observation(obs)
    obs.update(block_vel_gt=np.ones((2, 3)) * 999, future_path=np.arange(50))
    np.testing.assert_array_equal(original, vectorize_observation(obs))
    assert original.shape == (observation_schema(2)["size"],)
    obs["defense_blocks"] = np.zeros(18)
    with pytest.raises(ValueError):
        vectorize_observation(obs)
    with pytest.raises(KeyError):
        vectorize_observation({})


@pytest.mark.parametrize("key", ["action", "r_diff", "block_vel_gt", "obs", "continuation"])
def test_reject_nonfinite(ep, key):
    ep[key] = ep[key].astype(float)
    ep[key].flat[0] = np.nan
    with pytest.raises(ValueError):
        validate_episode(ep)


def test_broken_alignment(ep):
    ep["obs"][3, 0] += 1
    with pytest.raises(ValueError, match="transition chain"):
        validate_episode(ep)


def test_bad_rewards(ep):
    ep["reward"][0] += 1
    with pytest.raises(ValueError, match="decomposition"):
        validate_episode(ep)


def test_continuation_semantics(ep):
    assert validate_episode(ep)
    ep["continuation"][-1] = 0  # timeout must still bootstrap
    with pytest.raises(ValueError, match="Continuation"):
        validate_episode(ep)
    ep["terminated"][-1] = True
    ep["truncated"][-1] = False
    assert validate_episode(ep)


def test_no_transitions_after_done(ep):
    ep["done"][2] = True
    ep["terminated"][2] = True
    with pytest.raises(ValueError):
        validate_episode(ep)


def test_diagnostics_align_to_next_state(ep):
    ep["block_pos_gt"][-1, 0, 0] += 0.1
    with pytest.raises(ValueError, match="misaligned"):
        validate_episode(ep)


def test_replay_is_contiguous_and_diagnostic_free(ep):
    replay = ReplayBuffer(seed=17)
    replay.add(ep)
    replay.add(copy.deepcopy(ep))
    sample = replay.sample(12, 4)
    assert sample["state"].shape == (12, 5, 39)
    np.testing.assert_array_equal(sample["state"][:, :-1], sample["obs"])
    np.testing.assert_array_equal(sample["state"][:, 1:], sample["next_obs"])
    assert not sample["done"][:, :-1].any()
    assert not set(DIAGNOSTIC_KEYS) & set(sample)
    other = ReplayBuffer(seed=17)
    other.add(ep)
    other.add(ep)
    np.testing.assert_array_equal(sample["state"], other.sample(12, 4)["state"])
    with pytest.raises(ValueError):
        replay.sample(1, 50)
    rows = transitions(n=1)
    with pytest.raises(ValueError, match="Mixed schemas"):
        replay.add({k: np.asarray([r[k] for r in rows]) for k in MODEL_KEYS + DIAGNOSTIC_KEYS})


def test_capacity_and_copy(ep):
    replay = ReplayBuffer(capacity_episodes=1)
    replay.add(ep)
    ep["obs"][0, 0] = 999
    assert replay.episodes[0]["obs"][0, 0] == 0
