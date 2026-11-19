"""Headless integration tests for moving attackers and task outcomes."""

import numpy as np
import pytest

import robosuite as suite


@pytest.fixture
def env():
    environment = suite.make("TableDefense", has_renderer=False, has_offscreen_renderer=False,
                             renderer="mujoco", hard_reset=False, seed=7)
    environment.reset()
    yield environment
    environment.close()


def idle(env):
    action = np.zeros(env.action_dim)
    action[-1] = -1
    return action


def place(env, i, position, quaternion=(1, 0, 0, 0)):
    env.sim.data.set_joint_qpos(env.attackers[i].joints[0], np.r_[position, quaternion])
    env.sim.data.set_joint_qvel(env.attackers[i].joints[0], np.zeros(6))
    env.sim.forward()


def test_initial_placement_and_observations(env):
    assert env.num_attackers == 5
    assert np.all(env.task_info()["clearances"] > 0.10)
    assert np.all(np.abs(env.attacker_positions[:, :2]) + env.cube_half_size < env.table_full_size[:2] / 2)
    obs = env.reset()
    assert obs["attackers_pos"].shape == (15,)
    assert obs["attackers_vel"].shape == (15,)
    assert obs["attackers_status"].shape == (10,)
    assert np.isfinite(obs["object-state"]).all()


def test_launch_delay_radial_motion_and_speed(env):
    initial = env.attacker_positions.copy()
    action = idle(env)
    for _ in range(30):
        env.step(action)
    np.testing.assert_allclose(env.attacker_positions[:, :2], initial[:, :2], atol=2e-4)
    for _ in range(50):
        env.step(action)
    travel = initial[0, :2] - env.attacker_positions[0, :2]
    assert 0.012 < np.linalg.norm(travel) < 0.020  # 2 seconds at approximately 0.008 m/s
    assert abs(travel @ env._tangents[0]) < 0.001
    np.testing.assert_allclose(env.attacker_positions[1:, :2], initial[1:, :2], atol=2e-4)


def test_footprint_breach_and_latched_failure(env):
    # Center is outside; the front edge is inside.
    place(env, 0, [env.defense_center[0] + env.defense_radius + 0.01, 0, 0.82])
    obs, reward, done, info = env.step(idle(env))
    assert done and info["failed"] and info["breach_count"] == 1
    assert reward == -1
    assert obs["attackers_status"][1] == 1
    place(env, 0, [0.30, 0, 1.1])
    env._update_task_state()
    assert env.breached[0] and not env.removed[0]


def test_lift_hold_and_force_disengagement(env):
    env.sim.data.time = 3.0
    place(env, 0, [*env.spawn_positions[0], 1.04])
    env._update_task_state()
    assert not env.removed[0]
    env.sim.data.time += env.removal_hold_time + 0.01
    env._update_task_state()
    assert env.removed[0] and not env.breached[0]
    env._pre_action(idle(env), policy_step=True)
    dof = env._dof_starts[0]
    np.testing.assert_array_equal(env.sim.data.qfrc_applied[dof:dof + 2], 0)


def test_grasp_disengages_propulsion(env):
    env.sim.data.time = 3.0
    env._grasped[0] = True
    env._pre_action(idle(env), policy_step=False)
    dof = env._dof_starts[0]
    np.testing.assert_array_equal(env.sim.data.qfrc_applied[dof:dof + 2], 0)


def test_lift_timer_resets_when_put_down(env):
    place(env, 0, [*env.spawn_positions[0], 1.04])
    env._update_task_state()
    env.sim.data.time += 0.1
    place(env, 0, [*env.spawn_positions[0], 0.82])
    env._update_task_state()
    assert np.isnan(env._lift_since[0]) and not env.removed[0]


def test_outside_removal_success_and_reset(env):
    for i in range(env.num_attackers):
        place(env, i, [env.defense_center[0] + env.removal_radius + 0.06, (i - 2) * 0.06, 0.82])
    env._update_task_state()
    assert env._check_success()
    assert env.task_info()["removed_count"] == 5
    obs, reward, done, info = env.step(idle(env))
    assert done and info["success"]
    env.reset()
    assert not env._check_success()
    assert not env.removed.any() and not env.breached.any()
    np.testing.assert_allclose(env.attacker_positions[:, :2], env.spawn_positions)


def test_rotated_cube_footprint(env):
    # A rotated corner breaches even though an unrotated face would be outside.
    position = [env.defense_center[0] + env.defense_radius + 0.025, 0, 0.82]
    place(env, 0, position)
    assert env._footprint_clearance(0)[0] > 0
    place(env, 0, position, [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)])
    assert env._footprint_clearance(0)[0] < 0


def test_outside_threshold_uses_entire_footprint(env):
    place(env, 0, [env.defense_center[0] + env.removal_radius + 0.01, 0, 0.82])
    env._update_task_state()
    assert not env.removed[0]
    place(env, 0, [env.defense_center[0] + env.removal_radius + 0.03, 0, 0.82])
    env._update_task_state()
    assert env.removed[0]
    assert env.reward() == pytest.approx(0.2)
    env._update_task_state()
    assert env.reward() == 0  # event reward cannot be repeatedly collected


def test_failed_preview_continues_without_becoming_success(env):
    env.terminate_on_breach = False
    place(env, 0, [*env.defense_center, 0.82])
    _, _, done, info = env.step(idle(env))
    assert info["failed"] and not done
    env.removed[1:] = True
    assert not env._check_success()


def test_hard_reset_rebuilds_task_references(env):
    env.hard_reset = True
    env.removed[:] = True
    obs = env.reset()
    assert not env.removed.any()
    assert obs["attackers_status"].sum() == 0
    np.testing.assert_allclose(env.attacker_positions[:, :2], env.spawn_positions)


def test_mixed_motion_and_custom_count():
    env = suite.make("TableDefense", num_attackers=3, motion="mixed", start_delay=0, launch_interval=0,
                     has_renderer=False, has_offscreen_renderer=False, renderer="mujoco", hard_reset=False)
    try:
        env.reset()
        initial = env.attacker_positions.copy()
        for _ in range(30):
            env.step(idle(env))
        offset = env.attacker_positions[:, :2] - initial[:, :2]
        assert abs(offset[0] @ env._tangents[0]) < 0.001
        assert abs(offset[1] @ env._tangents[1]) > 0.005
    finally:
        env.close()


@pytest.mark.parametrize("kwargs", [
    {"num_attackers": 0}, {"num_attackers": 2.5}, {"defense_radius": -0.2},
    {"attack_speed": -1}, {"spawn_radius": 0.1}, {"motion": "unknown"},
    {"defense_center": [1, 0]}, {"removal_radius": 0.3}, {"cube_size": 0.2},
    {"attack_speed": float("nan")},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        suite.make("TableDefense", **kwargs)
