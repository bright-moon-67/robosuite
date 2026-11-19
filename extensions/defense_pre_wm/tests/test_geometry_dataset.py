import numpy as np
import pytest
from defense_task.dataset import load_episode, save_episode, validate_episode, vectorize_observation
from defense_task.geometry import circle_intrusion, footprint_distance, quat_to_matrix, sample_spawns


def test_quaternion_and_footprint():
    assert np.allclose(quat_to_matrix([1, 0, 0, 0]), np.eye(3))
    with pytest.raises(ValueError):
        quat_to_matrix([0, 0, 0, 0])
    h = [0.02, 0.02, 0.02]
    p = [0.25, 0, 0.822]
    assert np.isclose(footprint_distance(p, [1, 0, 0, 0], h, [0, 0]), 0.23)
    assert not circle_intrusion(p, [1, 0, 0, 0], h, [0, 0], 0.2, 0.8)
    # Boundary touch (center x = 0.22, half width 0.02)
    assert circle_intrusion([0.22, 0, 0.822], [1, 0, 0, 0], h, [0, 0], 0.2, 0.8)
    assert circle_intrusion([0, 0, 0.822], [1, 0, 0, 0], h, [0, 0], 0.2, 0.8)
    # Safe above-table carry.
    assert not circle_intrusion([0, 0, 1.0], [1, 0, 0, 0], h, [0, 0], 0.2, 0.8)
    # Rotation by 45 degrees around z.
    q = [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)]
    d = footprint_distance([0.25, 0, 0.822], q, h, [0, 0])
    assert d < 0.23


@pytest.mark.parametrize(
    "n,arc",
    [
        (1, (np.pi / 2, 3 * np.pi / 2)),
        (5, (np.pi / 2, 3 * np.pi / 2)),
        (5, (0, 2 * np.pi)),
    ],
)
def test_spawns(n, arc):
    a = sample_spawns(42, n, [0, 0], (0.38, 0.49), arc, [0.022, 0.022], [0.6, 0.6])
    b = sample_spawns(42, n, [0, 0], (0.38, 0.49), arc, [0.022, 0.022], [0.6, 0.6])
    assert len(a) == n
    assert all(np.allclose(p[0], q[0]) for p, q in zip(a, b))
    for i, (p, ang) in enumerate(a):
        assert 0.38 <= np.linalg.norm(p) <= 0.49
        for other, _ in a[:i]:
            assert np.linalg.norm(p - other) > 2 * np.linalg.norm([0.022, 0.022]) + 0.018


def test_invalid_spawn():
    with pytest.raises(ValueError):
        sample_spawns(0, 0, [0, 0], (0.4, 0.5), (0, 1), [0.02, 0.02], [0.6, 0.6])
    with pytest.raises(RuntimeError):
        sample_spawns(0, 30, [0, 0], (0.4, 0.45), (0, 0.1), [0.02, 0.02], [0.5, 0.5])


def test_policy_selection_and_recovery():
    from defense_task.policy import DefenseHeuristic

    policy = DefenseHeuristic(n_blocks=2)
    # 7 joints q and dq + EEF and EEF speed => 20 components
    robot = np.zeros(20)
    robot[-6:-3] = [0.0, 0.0, 1.0]
    blocks = np.zeros((2, 6))
    blocks[0, :3] = [0.3, 0, -0.178]
    blocks[1, :3] = [0.4, 0, -0.178]
    blocks[:, 3] = 1

    obs = {"defense_robot": robot, "defense_blocks": blocks.ravel()}
    act = policy.act(obs)
    assert policy.target_id == 0 and act.shape == (7,) and act[-1] == -1
    # If object is cleared, policy does not continue attempting to grasp it.
    blocks[0, 3] = 0
    obs["defense_blocks"] = blocks.ravel()
    act2 = policy.act(obs)
    assert policy.stage == "release" and act2[-1] == -1
