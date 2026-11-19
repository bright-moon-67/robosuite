import numpy as np
import pytest
from defense_task.geometry import (
    footprint_distance, circle_intrusion, sample_spawns, quat_to_matrix,
)
from defense_task.dataset import (
    vectorize_observation, save_episode, load_episode, validate_episode,
)


def test_quaternion_and_footprint():
    assert np.allclose(quat_to_matrix([1,0,0,0]), np.eye(3))
    with pytest.raises(ValueError):
        quat_to_matrix([0,0,0,0])
    h = [0.02, 0.02, 0.02]
    p = [0.25, 0, 0.822]
    assert np.isclose(footprint_distance(p, [1,0,0,0], h, [0,0]), 0.23)
    assert not circle_intrusion(p, [1,0,0,0], h, [0,0], 0.2, 0.8)
    # Boundary touch (center x = 0.22, half width 0.02)
    assert circle_intrusion([0.22,0,0.822], [1,0,0,0], h, [0,0], 0.2, 0.8)
    assert circle_intrusion([0,0,0.822], [1,0,0,0], h, [0,0], 0.2, 0.8)
    # Safe above-table carry.
    assert not circle_intrusion([0,0,1.0], [1,0,0,0], h, [0,0], 0.2, 0.8)
    # Rotation by 45 degrees around z.
    q = [np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)]
    d = footprint_distance([0.25,0,0.822], q, h, [0,0])
    assert d < 0.23


@pytest.mark.parametrize("n,arc", [
    (1,(np.pi/2,3*np.pi/2)),
    (5,(np.pi/2,3*np.pi/2)),
    (5,(0,2*np.pi)),
])
def test_spawns(n,arc):
    a = sample_spawns(42,n,[0,0],(.38,.49),arc,[.022,.022],[.6,.6])
    b = sample_spawns(42,n,[0,0],(.38,.49),arc,[.022,.022],[.6,.6])
    assert len(a) == n
    assert all(np.allclose(p[0], q[0]) for p,q in zip(a,b))
    for i, (p, ang) in enumerate(a):
        assert .38 <= np.linalg.norm(p) <= .49
        for other,_ in a[:i]:
            assert np.linalg.norm(p-other) > 2*np.linalg.norm([.022,.022]) + .018


def test_invalid_spawn():
    with pytest.raises(ValueError):
        sample_spawns(0,0,[0,0],(.4,.5),(0,1),[.02,.02],[.6,.6])
    with pytest.raises(RuntimeError):
        sample_spawns(0,30,[0,0],(.4,.45),(0,.1),[.02,.02],[.5,.5])


def test_dataset(tmp_path):
    obs = {"defense_robot": np.zeros(20),
           "defense_blocks": np.zeros(9),
           "defense_status": np.zeros(1),
           "defense_clock": np.zeros(1)}
    x = vectorize_observation(obs)
    tr = [{"obs":x, "action":np.zeros(7), "next_obs":x+0.001,
           "reward":1.5, "r_disc":2., "r_diff":-.5, "done":False,
           "status":np.array([0]), "cleared_count":0,
           "breached_count":0,
           "block_pos_gt":np.zeros((1,3)),
           "block_vel_gt":np.zeros((1,3))},
          {"obs":x+0.001, "action":np.ones(7),
           "next_obs":x+0.002, "reward":-2., "r_disc":-2.,
           "r_diff":0., "done":True, "status":np.array([2]),
           "cleared_count":0, "breached_count":1,
           "block_pos_gt":np.zeros((1,3)),
           "block_vel_gt":np.zeros((1,3))}]
    path=tmp_path/"one.npz"
    save_episode(path,tr,{"test":True})
    got=load_episode(path)
    assert got["obs"].shape == (2,len(x))
    assert validate_episode(got)
    assert path.with_suffix(".json").exists()
    bad=dict(got)
    bad["reward"]=np.array([123.,-2.])
    with pytest.raises(ValueError):
        validate_episode(bad)
    with pytest.raises(KeyError):
        vectorize_observation({})


def test_policy_selection_and_recovery():
    from defense_task.policy import DefenseHeuristic
    policy = DefenseHeuristic(n_blocks=2)
    # 7 joints q and dq + EEF and EEF speed => 20 components
    robot = np.zeros(20)
    robot[-6:-3] = [0., 0., 1.0]
    blocks = np.zeros((2,6))
    blocks[0,:3] = [.3,0,-.178]
    blocks[1,:3] = [.4,0,-.178]
    blocks[:,3] = 1
    
    obs = {"defense_robot":robot,"defense_blocks":blocks.ravel()}
    act=policy.act(obs)
    assert policy.target_id == 0 and act.shape == (7,) and act[-1] == -1
    # If object is cleared, policy does not continue attempting to grasp it.
    blocks[0,3] = 0
    obs["defense_blocks"] = blocks.ravel()
    act2=policy.act(obs)
    assert policy.stage == "release" and act2[-1] == -1
