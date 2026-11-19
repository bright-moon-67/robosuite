"""Runs only on machines where the user's robosuite fork is installed."""
import numpy as np
import pytest

pytest.importorskip("robosuite")
from defense_task.environment import DefenseTask


def test_mujoco_single_block():
    env = DefenseTask(robots="Panda", num_blocks=1,
                      has_renderer=False, has_offscreen_renderer=False,
                      horizon=30, seed=12)
    try:
        o=env.reset()
        assert np.asarray(o["defense_blocks"]).shape == (6,)
        assert env.action_dim >= 7
        for _ in range(3):
            a=np.zeros(env.action_dim)
            a[-1]=-1
            o,r,done,info=env.step(a)
            assert np.isfinite(r)
            assert np.isclose(r,info["r_disc"]+info["r_diff"])
            assert len(info["status"])==1
            if done:
                break
    finally:
        env.close()
