"""Real MuJoCo tests: contact, force tracking, observations, termination and resets."""

import numpy as np
import pytest

pytest.importorskip("robosuite")
from defense_task.dataset import vectorize_observation
from defense_task.environment import DefenseTask
from defense_task.policy import DefenseHeuristic


@pytest.fixture
def env():
    e = DefenseTask(num_blocks=1, seed=12, horizon=600)
    e.reset()
    yield e
    e.close()


def idle(env):
    action = np.zeros(7)
    action[-1] = -1
    return action


def place(env, pos):
    env.sim.data.set_joint_qpos(env.blocks[0].joints[0], np.r_[pos, 1, 0, 0, 0])
    env.sim.data.set_joint_qvel(env.blocks[0].joints[0], np.zeros(6))
    env.sim.forward()


def test_current_fork_controller_and_state(env):
    obs = env.reset()
    assert env.action_dim == 7
    assert obs["defense_robot"].shape == (20,)
    assert obs["defense_blocks"].shape == (6,)
    assert vectorize_observation(obs).shape == (32,)
    c = env.robots[0].part_controllers["right"]
    assert c.input_ref_frame == "world" and c.input_type == "delta"
    for _ in range(5):
        obs, r, done, info = env.step(idle(env))
        assert r == pytest.approx(info["r_disc"] + info["r_diff"])
        np.testing.assert_allclose(obs["defense_blocks"][:3] + obs["defense_robot"][14:17], info["block_pos_gt"][0])
        np.testing.assert_array_equal(obs["defense_status"], info["status"])
    assert info["continuation"] == 1 and not done


@pytest.mark.parametrize("mode", ["linear", "sine", "pulsed"])
def test_free_motion_velocity_tracking(mode):
    e = DefenseTask(num_blocks=1, motion_mode=mode, attack_speed=(0.008, 0.016), horizon=200, seed=33)
    try:
        e.reset()
        initial = e._block_pos(0).copy()
        errors = []
        for _ in range(160):
            _, _, done, info = e.step(idle(e))
            assert not done
            force = np.linalg.norm(e.sim.data.xfrc_applied[e.block_body_ids[0], :2])
            assert force <= e.force_limit + 1e-9
            if info["time"] < e.start_delay:
                np.testing.assert_allclose(e._block_pos(0)[:2], initial[:2], atol=2e-4)
            if info["time"] > e._motion[0]["start"] + 0.3:
                errors.append(np.linalg.norm(e._block_v(0)[:2] - e._desired_velocity(0)))
        assert np.linalg.norm(e._block_pos(0)[:2]) < np.linalg.norm(initial[:2]) - 0.025
        assert np.sqrt(np.mean(np.square(errors))) < 0.002
    finally:
        e.close()


def test_height_without_grasp_never_clears(env):
    pos = env._block_pos(0)
    pos[2] = env.table_top + 0.25
    place(env, pos)
    for _ in range(8):
        env._update_events()
    assert env.status[0] == 0 and env.held_counts[0] == 0
    env._pre_action(idle(env), True)
    np.testing.assert_array_equal(env.sim.data.xfrc_applied[env.block_body_ids[0]], 0)


def test_boundary_is_latched_and_observation_current(env):
    place(env, [0.22, 0, 0.822])
    obs, reward, done, info = env.step(idle(env))
    assert done and info["terminated"] and not info["truncated"] and info["continuation"] == 0
    assert info["failed"] and info["r_disc"] == -12 and info["new_breached"] == [0]
    assert np.asarray(obs["defense_status"]).reshape(-1)[0] == 2
    env._update_events()
    assert not env._last_events["breached"]
    assert env.reward(idle(env)) == pytest.approx(env._last_reward["continuous"])


def test_timeout_not_terminal(env):
    env.horizon = 2
    env.step(idle(env))
    _, _, done, info = env.step(idle(env))
    assert done and info["truncated"] and not info["terminated"] and info["continuation"] == 1


def test_seeded_resets_and_hidden_motion(env):
    env.rng = np.random.default_rng(17)
    obs1 = env.reset()
    initial = env._block_pos(0)
    motion = dict(env._motion[0])
    env.rng = np.random.default_rng(17)
    obs2 = env.reset()
    np.testing.assert_array_equal(vectorize_observation(obs1), vectorize_observation(obs2))
    assert motion["speed"] == env._motion[0]["speed"]
    env.reset()
    assert not np.allclose(initial, env._block_pos(0))
    state = vectorize_observation(env._get_observations(force_update=True))
    env._motion[0]["speed"] = 0.999
    env._motion[0]["phase"] = 999
    np.testing.assert_array_equal(state, vectorize_observation(env._get_observations(force_update=True)))


@pytest.mark.parametrize("count", [1, 3, 5, 8])
def test_counts_and_random_geometry(count):
    from defense_task.geometry import circle_intrusion

    e = DefenseTask(num_blocks=count, seed=4)
    try:
        obs = e.reset()
        assert len(vectorize_observation(obs)) == 25 + 7 * count
        positions = np.stack([e._block_pos(i) for i in range(count)])
        for i, pos in enumerate(positions):
            assert not circle_intrusion(pos, [1, 0, 0, 0], e.block_half_size, [0, 0], 0.2, 0.8)
            assert np.all(np.abs(pos[:2]) + 0.022 < np.asarray(e.table_full_size[:2]) / 2)
            for j in range(i):
                assert np.linalg.norm(pos[:2] - positions[j, :2]) > 2 * np.sqrt(2) * 0.022
        e.hard_reset = True
        e.reset()
        e.step(idle(e))
    finally:
        e.close()


def test_real_grasp_lift_and_drive_stop(env):
    # One fixed reproducible case checks real contacts, not a fabricated grasp flag.
    env.rng = np.random.default_rng(43)
    obs = env.reset()
    policy = DefenseHeuristic(1)
    saw_grasp = False
    for _ in range(450):
        action = policy.act(obs)
        obs, _, done, info = env.step(action)
        if env.grasp_flags[0]:
            saw_grasp = True
            assert env._check_grasp(env.robots[0].gripper, env.blocks[0])
            env._pre_action(action, True)
            np.testing.assert_array_equal(env.sim.data.xfrc_applied[env.block_body_ids[0]], 0)
        if done:
            break
    assert saw_grasp and info["success"] and info["cleared_count"] == 1
    assert info["r_disc"] == 11  # one clear and whole-episode success
    assert env.held_counts[0] >= env.lift_hold_steps


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_blocks=1.5),
        dict(attack_speed=(-1, 0.01)),
        dict(spawn_radius=(0.19, 0.3)),
        dict(table_friction=(-1, 0, 0)),
        dict(stagger_seconds=-1),
        dict(lift_hold_steps=0),
        dict(robots="Baxter"),
        dict(attack_speed=(float("nan"), 0.02)),
    ],
)
def test_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        DefenseTask(**kwargs)


def test_invalid_actions_do_not_advance_physics(env):
    initial_time = env.sim.data.time
    for action in (np.zeros(6), np.full(7, np.nan), np.full(7, 2)):
        with pytest.raises(ValueError):
            env.step(action)
    assert env.sim.data.time == initial_time


def test_hold_counter_requires_unbroken_grasp(env, monkeypatch):
    place(env, [-0.3, 0, 1.0])
    monkeypatch.setattr(env, "_check_grasp", lambda *args, **kwargs: True)
    env._update_events()
    env._update_events()
    assert env.held_counts[0] == 2 and env.status[0] == 0
    monkeypatch.setattr(env, "_check_grasp", lambda *args, **kwargs: False)
    env._update_events()
    assert env.held_counts[0] == 0 and env.status[0] == 0


def test_multiple_event_rewards_and_no_repeat():
    env = DefenseTask(num_blocks=3, seed=6, failure_on_first_breach=False)
    try:
        env.reset()
        for i, x in enumerate([0.22, 0.21, 0.20]):
            env.sim.data.set_joint_qpos(env.blocks[i].joints[0], [x, 0, 0.822, 1, 0, 0, 0])
        env.sim.forward()
        env._update_events()
        assert np.all(env.status == 2)
        env.reward(np.zeros(7))
        assert env._last_reward["discrete"] == -36
        env._update_events()
        env.reward(np.zeros(7))
        assert env._last_reward["discrete"] == 0
    finally:
        env.close()
