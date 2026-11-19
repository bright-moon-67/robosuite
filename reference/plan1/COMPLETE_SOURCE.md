# Complete source: DiffDreamer Defense (pre-World-Model)
All files below are included verbatim and also in the ZIP archive.
Runtime compatibility target: robosuite 1.5.x; MuJoCo smoke test was not runnable here.

## `README.md`

```markdown
# DiffDreamer Defense / Pre-World-Model code

本项目是**World Model 训练之前的完整阶段**，不包含 RSSM、Actor-Critic 或可微分接触模拟。

## 已完成

- `DefenseTask(ManipulationEnv)`：桌面半径 0.20 m 防御圆、n 个自由刚体物块
- `linear / sine / pulsed` 速度模式（每次 episode 的轨迹参数独立随机）
- 进攻驱动力在 MuJoCo 物理子步施加，不强制改物块 `qpos`
- 抓取 `_check_grasp` + 连续抬升判定清除；桌面物块外轮廓进入禁区判定突破
- 独立 `r_disc` / `r_diff` 和事件/终止统计
- OSC_POSE 规则抓取基线（只供数据采集，不保证每次成功）
- 统一固定向量状态数据集，记录 `(o_t,a_t,r_t,c_t,o_{t+1})`

## 环境要求

本包针对 **robosuite 1.5.x 官方 API**。你的 fork 具体提交未能通过公开 URL 读取，首次运行需要检查它是否与此 API 兼容。
示例 Python 3.10/3.11；先安装你 fork 中自带的依赖：

```bash
git clone https://github.com/bright-moon-67/robosuite.git
cd robosuite
pip install -e .
cd /path/to/diffdreamer_defense_pre_wm
pip install -e ".[test]"
```

不要同时再安装另一个 PyPI robosuite 版本覆盖你 fork。

## 运行

```bash
# 无渲染最小烟雾测试
python -m defense_task.cli smoke --blocks 1 --horizon 100
# 图形界面手工检查（需桌面 OpenGL 环境）
python -m defense_task.cli demo --blocks 1 --horizon 400
# 批量评估规则控制策略（在引入 World Model 之前的必要基线）
python -m defense_task.cli evaluate --blocks 5 --episodes 20
# 五物块采样，速度范围按需调
python -m defense_task.cli collect --blocks 5 --episodes 20 --horizon 1400 \
    --out datasets/defense --min-speed .008 --max-speed .016 --stagger 6
pytest -q tests/test_geometry_dataset.py
pytest -q tests/test_sim_smoke.py
```

所有数值为**初始实验值**，不是论文参数。特别需要在真实仿真中检查：
`Panda` 的机械臂可达空间、夹爪默认朝向、实际单次抓取耗时、
物块力控速度是否与目标速度一致。

## 物理与事件约束

- 物块 `qpos` 仅在 `reset()` 时设定；进攻阶段施加 `xfrc_applied` 水平外力
- 接触和自由刚体运动由 MuJoCo 求解；抓取后不再提供外部推进力
- `status=0` active, `status=1` lifted/cleared, `status=2` entered defense circle
- 夹持且物块中心高于桌面 `0.11 m` 并连续保持 3 个策略步才清除
- 非清除物块，只要其**投影外轮廓**触碰防御圆且底部靠近桌面，即判突破
- 默认首次突破即 episode 失败；全部清除成功
- 已清除物块**不会消失**，后续仍可能与环境物体碰撞，但不再进攻、不计突破

## 数据集格式

每个 `episode_XXXXX.npz` 保存：
- `obs`, `action`, `next_obs`: 每一行动严格对应 `o_t`, `a_t`, `o_{t+1}`
- `reward`, `r_disc`, `r_diff`, `done`
- `status`, `cleared_count`, `breached_count`

`defense_robot` = arm q, dq, eef xyz, eef velocity。
`defense_blocks` = 每物块6维：相对末端位置xyz、active、grasped、cleared。
`defense_status` = 每物块 status code；`defense_clock` = 时间归一化。
固定顺序拼接的 obs 的 schema 在 `defense_task/dataset.py`。

`r_diff` 目前只是一个**连续函数形式的奖励值**；
要真正实现 DiffDreamer BPTT，需要随后加入**可微分机械臂动力学**与同步 rollout。
`r_disc` 将来用于 RSSM reward head 和 Critic，但当前并未训练网络。

## 建议的执行顺序

1. 把 `smoke --blocks 1` 跑通，查看 ring/table/box 坐标
2. `demo --blocks 1`，验证抓取不受进攻外力强制牵引
3. 统计几十次单块防御成功/失败，调 OSC 增益和落点
4. 单块线性 → 单块变速 → 五块慢速错峰
5. 收集 Replay，验证所有通道无 NaN、无状态泄漏
6. **下一阶段**才训练 RSSM，并接入可微分 robot dynamics

## 与原始 DiffDreamer 的联系

原文的 Eq. (9)–(11) 指出：
RSSM 负责潜在环境预测，可微分物理负责显式自身动力学，
联合 Actor 同步预测，并分别处理离散任务奖励与连续运动奖励。

本阶段仅完成数据、真实仿真、规则基线。不要把 MuJoCo 普通 step 误称为可微分 rollout。

**速度标签与避免信息泄漏**：`block_pos_gt`、`block_vel_gt` 只在离线数据集
作为诊断/评估标签保存，绝不能拼接进 RSSM/Actor 的输入。规则基线根据前后两帧相对位置
和机器人末端世界位置估计物块速度。仿真器使用隐藏的每物块速度与轨迹参数驱动进攻。

```


## `defense_task/__init__.py`

```python
"""Pre-World-Model defense environment for robosuite 1.5.x."""
from .geometry import footprint_distance, circle_intrusion, sample_spawns
from .dataset import vectorize_observation, save_episode, validate_episode

__all__ = [
    "footprint_distance", "circle_intrusion", "sample_spawns",
    "vectorize_observation", "save_episode", "validate_episode",
]

```


## `defense_task/cli.py`

```python
"""python -m defense_task.cli smoke|demo|collect ..."""
import argparse
from pathlib import Path
import json
import numpy as np

from .dataset import vectorize_observation, save_episode, OBS_KEYS
from .policy import DefenseHeuristic


def make_env(args, render=False):
    import robosuite as suite
    # Import class first to trigger robosuite's auto registration via EnvMeta.
    from .environment import DefenseTask  # noqa: F401

    if hasattr(suite, "load_composite_controller_config"):
        controller = suite.load_composite_controller_config(controller="BASIC", robot="Panda")
    else:
        raise RuntimeError("This package targets robosuite 1.5.x composite controllers.")
    # BASIC defaults may change, so explicitly set OSC_POSE for Panda.
    for item in controller.get("body_parts", {}).values():
        if isinstance(item, dict) and item.get("type") in ("OSC_POSITION", "OSC_POSE"):
            item["type"] = "OSC_POSE"
    env = suite.make(
        env_name="DefenseTask", robots="Panda", controller_configs=controller,
        num_blocks=args.blocks, motion_mode=args.motion,
        attack_speed=(args.min_speed, args.max_speed),
        stagger_seconds=args.stagger, seed=args.seed,
        has_renderer=render, has_offscreen_renderer=False,
        horizon=args.horizon, control_freq=20,
    )
    if env.action_dim != 7:
        raise RuntimeError(
            f"Expected OSC_POSE+1 gripper action (7D); got {env.action_dim}. "
            "Inspect controller body_parts for Panda and set OSC_POSE.")
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("smoke", "demo", "collect", "evaluate"))
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--motion", choices=("linear", "sine", "pulsed"), default="linear")
    p.add_argument("--min-speed", type=float, default=0.008)
    p.add_argument("--max-speed", type=float, default=0.016)
    p.add_argument("--stagger", type=float, default=6.0)
    p.add_argument("--horizon", type=int, default=1400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.15,
                   help="Probability of random action for exploration data")
    p.add_argument("--out", default="datasets/defense")
    args = p.parse_args()
    if args.min_speed <= 0 or args.min_speed > args.max_speed:
        p.error("invalid speed interval")
    if not 0 <= args.epsilon <= 1:
        p.error("epsilon must be in [0,1]")
    if args.blocks < 1:
        p.error("blocks must be positive")
    env = make_env(args, render=args.mode == "demo")
    rng = np.random.default_rng(args.seed)
    try:
        totals = {"success":0, "breach":0, "cleared":0, "steps":0}
        n_episodes = args.episodes if args.mode in ("collect","evaluate") else 1
        out = Path(args.out)
        for e in range(n_episodes):
            obs = env.reset()
            policy = DefenseHeuristic(
                env.n_blocks, table_top=env.table_top,
                defense_center=env.defense_center, defense_radius=env.defense_radius,
                dt=1./env.control_freq)
            transitions = []
            done, info = False, {}
            for t in range(args.horizon):
                action = policy.act(obs)
                if args.mode == "smoke":
                    action = np.zeros(env.action_dim)
                    action[-1] = -1
                elif args.mode == "collect" and rng.random() < args.epsilon:
                    action = rng.uniform(-0.5, 0.5, env.action_dim)
                    action[-1] = rng.choice([-1., 1.])
                before = vectorize_observation(obs)
                new_obs, rew, done, info = env.step(action)
                after = vectorize_observation(new_obs)
                transitions.append(dict(
                    obs=before, action=np.asarray(action, dtype=np.float32),
                    next_obs=after, reward=rew,
                    r_disc=info["r_disc"], r_diff=info["r_diff"],
                    done=done, status=np.asarray(info["status"]),
                    cleared_count=info["cleared_count"],
                    breached_count=info["breached_count"],
                    block_pos_gt=info["block_pos_gt"],
                    block_vel_gt=info["block_vel_gt"]))
                obs = new_obs
                if args.mode == "demo":
                    env.render()
                if done:
                    break
            if args.mode == "collect":
                meta = dict(
                    base_seed=args.seed, episode_index=e, mode=args.motion,
                    block_count=env.n_blocks,
                    obs_keys=list(OBS_KEYS), control_freq=env.control_freq,
                    action_dim=env.action_dim, episode_idx=e,
                    status_codes={"attacking":0,"cleared":1,"breached":2},
                )
                path = out/f"episode_{e:05d}.npz"
                ep = save_episode(path, transitions, meta)
                print(f"Saved {path} : {len(ep['action'])} steps")
            totals["success"] += int(info.get("success", False))
            totals["breach"] += int(info.get("failed", False))
            totals["cleared"] += int(info.get("cleared_count", 0))
            totals["steps"] += len(transitions)
            print(f"episode={e} steps={len(transitions)} cleared={info.get('cleared_count')} "
                  f"breached={info.get('breached_count')} success={info.get('success')}")
        if args.mode in ("collect","evaluate"):
            print(f"summary: episodes={n_episodes}, wins={totals['success']}, "
                  f"breached_episodes={totals['breach']}, "
                  f"cleared_blocks={totals['cleared']}, "
                  f"mean_steps={totals['steps']/max(n_episodes,1):.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

```


## `defense_task/dataset.py`

```python
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

```


## `defense_task/environment.py`

```python
"""Physics-based multi-block defense in robosuite 1.5.x.

Run: from defense_task.environment import DefenseTask; env = DefenseTask(robots="Panda")
The ONLY extra forces move freely moving attackers; collisions and grasping stay MuJoCo-physical.
"""
import xml.etree.ElementTree as ET
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

from .geometry import circle_intrusion, sample_spawns


class DefenseTask(ManipulationEnv):
    """Defend a table circle against approaching MuJoCo free-joint blocks.

    Status codes: 0=attacking, 1=cleared by gripper lift, 2=breached.
    No camera images are necessary for this first *state-observation* benchmark.
    """

    def __init__(self, robots="Panda", controller_configs=None,
                 num_blocks=5, defense_radius=0.20,
                 table_full_size=(1.20, 1.20, 0.05),
                 spawn_radius=(0.38, 0.49),
                 spawn_angle=(np.pi/2, 3*np.pi/2),
                 attack_speed=(0.008, 0.016), motion_mode="linear",
                 stagger_seconds=6.0, target_height=0.11,
                 lift_hold_steps=3, failure_on_first_breach=True,
                 control_freq=20, horizon=1400, seed=None,
                 has_renderer=False, has_offscreen_renderer=False,
                 **kwargs):
        if int(num_blocks) < 1:
            raise ValueError("num_blocks must be >= 1")
        if isinstance(robots, (list, tuple)) and len(robots) != 1:
            raise ValueError("DefenseTask currently supports one arm robot only")
        if defense_radius <= 0 or target_height <= 0:
            raise ValueError("defense_radius and target_height must be positive")
        if motion_mode not in ("linear", "sine", "pulsed"):
            raise ValueError("motion_mode must be linear, sine or pulsed")
        self.n_blocks = int(num_blocks)
        self.defense_radius = float(defense_radius)
        self.defense_center = np.array([0., 0.], dtype=float)
        self.table_full_size = tuple(table_full_size)
        self.table_friction = (0.9, 0.005, 0.0001)
        self.table_offset = np.array([0., 0., 0.80])
        self.spawn_radius = tuple(spawn_radius)
        self.spawn_angle = tuple(spawn_angle)
        self.speed_range = tuple(attack_speed)
        self.motion_mode = motion_mode
        self.stagger_seconds = float(stagger_seconds)
        self.target_height = float(target_height)
        self.lift_hold_steps = int(lift_hold_steps)
        self.failure_on_first_breach = bool(failure_on_first_breach)
        self.block_half_size = np.array([0.022, 0.022, 0.022])
        self.status = np.zeros(self.n_blocks, dtype=np.int8)
        self.held_counts = np.zeros(self.n_blocks, dtype=np.int32)
        self.was_grasped = np.zeros(self.n_blocks, dtype=bool)
        self.grasp_flags = np.zeros(self.n_blocks, dtype=bool)
        self._last_events = {"cleared": [], "breached": []}
        self._last_action = None
        self._last_reward = {"discrete": 0., "continuous": 0., "total": 0.}
        self._motion = []
        self._eef_vel = np.zeros(3)
        self._last_eef = None
        self._last_distance = None
        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            initialization_noise=None,
            gripper_types="default", base_types="default",
            use_camera_obs=False, has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            control_freq=control_freq, horizon=horizon,
            ignore_done=False, hard_reset=True, seed=seed,
            **kwargs,
        )

    @property
    def table_top(self):
        return float(self.table_offset[2])

    def _load_model(self):
        super()._load_model()
        # Same robot/table convention as robosuite Lift.
        base = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(base)
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        # Non-colliding ring visual, supported by MuJoCo MJCF geoms.
        n_ring = 72
        for j in range(n_ring):
            t0, t1 = 2*np.pi*j/n_ring, 2*np.pi*(j+1)/n_ring
            a = self.defense_center + self.defense_radius*np.array([np.cos(t0), np.sin(t0)])
            b = self.defense_center + self.defense_radius*np.array([np.cos(t1), np.sin(t1)])
            z = self.table_top + 0.003
            ET.SubElement(arena.worldbody, "geom", {
                "name": f"defense_ring_{j}", "type": "capsule",
                "fromto": f"{a[0]} {a[1]} {z} {b[0]} {b[1]} {z}",
                "size": "0.0025", "rgba": "0.95 0.15 0.12 0.95",
                "contype": "0", "conaffinity": "0", "group": "1",
            })
        palette = [
            [0.95, 0.24, 0.22, 1], [0.93, 0.57, 0.20, 1],
            [0.23, 0.68, 0.93, 1], [0.68, 0.38, 0.88, 1],
            [0.34, 0.74, 0.36, 1],
        ]
        self.blocks = [
            BoxObject(
                name=f"attacker_{i}",
                size_min=self.block_half_size,
                size_max=self.block_half_size,
                rgba=palette[i % len(palette)],
                rng=self.rng,
            ) for i in range(self.n_blocks)
        ]
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=self.blocks,
        )

    def _setup_references(self):
        super()._setup_references()
        self.block_body_ids = [self.sim.model.body_name2id(o.root_body) for o in self.blocks]
        self.block_dof_addrs = [
            int(self.sim.model.jnt_dofadr[self.sim.model.joint_name2id(o.joints[0])])
            for o in self.blocks
        ]
        self.eef_site_id = self.robots[0].eef_site_id[self.robots[0].arms[0]]

    def _eef(self):
        return np.array(self.sim.data.site_xpos[self.eef_site_id], dtype=float)

    def _block_pos(self, i):
        return np.array(self.sim.data.body_xpos[self.block_body_ids[i]], dtype=float)

    def _block_v(self, i):
        adr = self.block_dof_addrs[i]
        return np.array(self.sim.data.qvel[adr:adr+3], dtype=float)

    def _setup_observables(self):
        obs = super()._setup_observables()

        @sensor(modality="defense")
        def defense_robot(cache):
            robot = self.robots[0]
            q = np.array(self.sim.data.qpos[robot._ref_arm_joint_pos_indexes])
            dq = np.array(self.sim.data.qvel[robot._ref_arm_joint_vel_indexes])
            return np.concatenate([q, dq, self._eef(), self._eef_vel])

        @sensor(modality="defense")
        def defense_blocks(cache):
            # Each block: relative xyz to robot EEF, active, grasped, cleared.
            # No oracle object velocity, path ID or future trajectory in actor/RSSM obs.
            out = np.zeros((self.n_blocks, 6), dtype=np.float32)
            eef = self._eef()
            for i in range(self.n_blocks):
                p = self._block_pos(i)
                out[i] = np.r_[p-eef,
                               float(self.status[i] == 0),
                               float(self.grasp_flags[i]),
                               float(self.status[i] == 1)]
            return out.reshape(-1)

        @sensor(modality="defense")
        def defense_status(cache):
            return self.status.astype(np.float32).copy()

        @sensor(modality="defense")
        def defense_clock(cache):
            return np.array([self.cur_time / (self.horizon / self.control_freq)], dtype=np.float32)

        for fn in (defense_robot, defense_blocks, defense_status, defense_clock):
            obs[fn.__name__] = Observable(
                name=fn.__name__, sensor=fn, sampling_rate=self.control_freq)
        return obs

    def _reset_internal(self):
        super()._reset_internal()
        self.status[:] = 0
        self.held_counts[:] = 0
        self.was_grasped[:] = False
        self.grasp_flags[:] = False
        self._last_events = {"cleared": [], "breached": []}
        self._last_action = None
        self._last_distance = None
        self._last_reward = {"discrete": 0., "continuous": 0., "total": 0.}
        positions = sample_spawns(
            self.rng, self.n_blocks, self.defense_center,
            self.spawn_radius, self.spawn_angle,
            self.block_half_size[:2], np.asarray(self.table_full_size[:2]) / 2,
        )
        self._motion = []
        for i, (xy, angle) in enumerate(positions):
            q = np.array([1., 0., 0., 0.])
            p = np.r_[xy, self.table_top + self.block_half_size[2] + 0.001]
            self.sim.data.set_joint_qpos(self.blocks[i].joints[0], np.r_[p, q])
            self.sim.data.qvel[self.block_dof_addrs[i]:self.block_dof_addrs[i]+6] = 0.
            self.sim.data.xfrc_applied[self.block_body_ids[i], :] = 0.
            self._motion.append({
                "inward": -np.array([np.cos(angle), np.sin(angle)]),
                "speed": float(self.rng.uniform(*self.speed_range)),
                "start": float(i * self.stagger_seconds),
                "phase": float(self.rng.uniform(0., 2*np.pi)),
            })
        self.sim.forward()
        self._last_eef = self._eef()
        self._eef_vel = np.zeros(3, dtype=float)

    def _desired_velocity(self, i):
        cfg = self._motion[i]
        if self.cur_time < cfg["start"]:
            return np.zeros(2)
        offset = self.defense_center - self._block_pos(i)[:2]
        u = offset / max(np.linalg.norm(offset), 1e-8)
        t = self.cur_time - cfg["start"]
        vel = cfg["speed"]*u
        if self.motion_mode == "sine":
            perp = np.array([-u[1], u[0]])
            vel = vel + 0.005*np.cos(0.8*t + cfg["phase"])*perp
        elif self.motion_mode == "pulsed":
            vel = vel*(0.75 + 0.5*(1 + np.sin(0.5*t + cfg["phase"]))/2)
        return vel

    def _pre_action(self, action, policy_step=False):
        # Called once PER MuJoCo physics substep by robosuite.
        # Never teleport blocks: free-joint dynamics + collisions remain physical.
        super()._pre_action(action, policy_step=policy_step)
        for i, bid in enumerate(self.block_body_ids):
            self.sim.data.xfrc_applied[bid, :] = 0.
            if self.status[i] != 0:
                continue
            desired = self._desired_velocity(i)
            if np.linalg.norm(desired) < 1e-9:
                continue
            p = self._block_pos(i)
            # Pause programmed attack forces after genuine two-finger grasp or lift.
            if self.grasp_flags[i] or p[2] > self.table_top + 0.055:
                continue
            measured = self._block_v(i)[:2]
            mass = max(1e-4, float(self.sim.model.body_mass[bid]))
            # Friction feed-forward and bounded P velocity tracking.
            unit = desired / np.linalg.norm(desired)
            friction = 1.05*self.table_friction[0]*mass*9.81
            force = friction*unit + 5.0*(desired - measured)
            norm = float(np.linalg.norm(force))
            if norm > 1.8:
                force *= 1.8/norm
            self.sim.data.xfrc_applied[bid, :2] = force

    def _update_events(self):
        self._last_events = {"cleared": [], "breached": []}
        for i in range(self.n_blocks):
            self.grasp_flags[i] = bool(self._check_grasp(
                gripper=self.robots[0].gripper, object_geoms=self.blocks[i]))
            if self.status[i] != 0:
                continue
            p = self._block_pos(i)
            q = np.array(self.sim.data.body_xquat[self.block_body_ids[i]])
            if circle_intrusion(p, q, self.block_half_size,
                                self.defense_center, self.defense_radius, self.table_top):
                self.status[i] = 2
                self._last_events["breached"].append(i)
                continue
            if self.grasp_flags[i]:
                self.was_grasped[i] = True
            if self.was_grasped[i] and self.grasp_flags[i] and (
                    p[2] >= self.table_top + self.target_height):
                self.held_counts[i] += 1
            else:
                self.held_counts[i] = 0
            if self.held_counts[i] >= self.lift_hold_steps:
                self.status[i] = 1
                self._last_events["cleared"].append(i)

    def _nearest_distance(self):
        active = np.flatnonzero(self.status == 0)
        if not len(active):
            return 0.
        ee = self._eef()
        return float(min(np.linalg.norm(ee - self._block_pos(i)) for i in active))

    def reward(self, action=None):
        # Event reward is sparse and non-differentiable (future Critic pathway).
        disc = (3.*len(self._last_events["cleared"])
                - 12.*len(self._last_events["breached"]))
        if len(self._last_events["cleared"]) and np.all(self.status == 1):
            disc += 8.
        # This is a continuous *reward function*, NOT yet an autodiff rollout.
        dist = self._nearest_distance()
        ctrl = np.asarray(action) if action is not None else np.zeros(1)
        cont = -0.18*np.tanh(5.*dist) - 0.0005*np.sum(ctrl*ctrl)
        self._last_distance = dist
        self._last_reward = {
            "discrete": float(disc),
            "continuous": float(cont),
            "total": float(disc + cont),
        }
        return float(disc + cont)

    def _post_action(self, action):
        self._update_events()
        old = np.asarray(self._last_eef if self._last_eef is not None else self._eef())
        now = self._eef()
        self._eef_vel = (now-old)/self.control_timestep
        self._last_eef = now
        reward, done, info = super()._post_action(action)
        failed = bool(np.any(self.status == 2))
        success = bool(np.all(self.status == 1))
        if success or (failed and self.failure_on_first_breach):
            self.done = True
            done = True
        info.update({
            "r_disc": self._last_reward["discrete"],
            "r_diff": self._last_reward["continuous"],
            "new_cleared": list(self._last_events["cleared"]),
            "new_breached": list(self._last_events["breached"]),
            "status": self.status.copy(),
            "cleared_count": int(np.sum(self.status == 1)),
            "breached_count": int(np.sum(self.status == 2)),
            "success": success, "failed": failed,
            "time": float(self.cur_time),
            "block_pos_gt": np.stack([self._block_pos(i) for i in range(self.n_blocks)]),
            "block_vel_gt": np.stack([self._block_v(i) for i in range(self.n_blocks)]),
        })
        # robosuite updates observables BEFORE _post_action; refresh status sensors.
        self._update_observables(force=True)
        return reward, done, info

    def _check_success(self):
        return bool(np.all(self.status == 1))

```


## `defense_task/geometry.py`

```python
"""Pure geometry helpers (no robosuite dependency)."""
import numpy as np


def quat_to_matrix(q):
    """MuJoCo quaternion order is w, x, y, z."""
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = q / n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _convex_hull(points):
    """Monotonic chain for the horizontal projection of the 3-D box."""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(pts) <= 1:
        return np.asarray(pts, dtype=float)
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def footprint_distance(position, quat_wxyz, half_size, center_xy):
    """Exact min XY distance from circle center to projected oriented box.

    3D box's 8 corners are projected into XY; their convex hull is tested.
    Supports tipped cubes as well as yaw rotations. Returns meters >= 0.
    """
    p = np.asarray(position, dtype=float).reshape(3)
    h = np.asarray(half_size, dtype=float).reshape(3)
    if np.any(h <= 0):
        raise ValueError("half_size must be positive")
    c = np.asarray(center_xy, dtype=float).reshape(2)
    rot = quat_to_matrix(quat_wxyz)
    corners = np.array([[sx*h[0], sy*h[1], sz*h[2]]
                        for sx in (-1, 1) for sy in (-1, 1)
                        for sz in (-1, 1)])
    poly = _convex_hull((p + corners @ rot.T)[:, :2])
    if len(poly) < 3:
        return float(np.linalg.norm(p[:2] - c))
    # Convex CCW polygon contains c iff all cross products >= 0.
    inside = True
    best = np.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i+1) % len(poly)]
        v = b-a
        signed = v[0]*(c[1]-a[1]) - v[1]*(c[0]-a[0])
        if signed < -1e-10:
            inside = False
        t = float(np.clip(np.dot(c-a, v) / max(np.dot(v,v), 1e-15), 0, 1))
        best = min(best, np.linalg.norm(c-(a+t*v)))
    return 0.0 if inside else float(best)


def circle_intrusion(position, quat_wxyz, half_size, center_xy, radius,
                     table_top, height_tolerance=0.035):
    """True iff a non-lifted block's projected footprint enters the circle.

    A cube carried wholly above table_top+height_tolerance is safe.
    Boundary contact counts as a breach.
    """
    p = np.asarray(position, dtype=float)
    h = np.asarray(half_size, dtype=float)
    rot = quat_to_matrix(quat_wxyz)
    vertical_extent = float(np.sum(np.abs(rot[2, :]) * h))
    bottom = p[2] - vertical_extent
    if bottom > table_top + height_tolerance:
        return False
    return footprint_distance(p, quat_wxyz, h, center_xy) <= float(radius) + 1e-9


def sample_spawns(rng, n, center_xy, radius_range, angle_range,
                  half_xy, table_half_xy, min_gap=0.018):
    """Collision-free polar placements on tabletop; random but seedable.

    half_xy is the attacker half-size in XY. table_half_xy are table half-extents.
    """
    rng = np.random.default_rng(rng) if not hasattr(rng, "uniform") else rng
    if n < 1:
        raise ValueError("n must be positive")
    rmin, rmax = map(float, radius_range)
    a0, a1 = map(float, angle_range)
    if not (0 < rmin <= rmax and a0 < a1):
        raise ValueError("invalid radius or angle range")
    center_xy = np.asarray(center_xy, dtype=float).reshape(2)
    half_xy = np.asarray(half_xy, dtype=float).reshape(2)
    table_half_xy = np.asarray(table_half_xy, dtype=float).reshape(2)
    result = []
    min_center_gap = 2*np.linalg.norm(half_xy) + min_gap
    for i in range(n):
        found = False
        # Evenly spaced angular sectors with randomized offset, avoiding clustering.
        sector = (a1-a0)/n
        for attempt in range(2000):
            ang = float(rng.uniform(a0+i*sector+0.07*sector,
                                    a0+(i+1)*sector-0.07*sector))
            r = float(rng.uniform(rmin, rmax))
            pos = center_xy + r*np.array([np.cos(ang), np.sin(ang)])
            if np.any(np.abs(pos) + np.linalg.norm(half_xy) + 0.01 >= table_half_xy):
                continue
            if all(np.linalg.norm(pos - p) > min_center_gap for p, _ in result):
                result.append((pos, ang))
                found = True
                break
        if not found:
            raise RuntimeError("Cannot place all blocks; enlarge table or reduce n/range")
    return result

```


## `defense_task/policy.py`

```python
"""Rule-based OSC_POSE baseline: estimates approach speeds from relative positions."""
import numpy as np


class DefenseHeuristic:
    """Observation-only target selection + finite-state grasping.

    Per block observation: block XYZ relative EEF, active, grasped, cleared.
    The policy cannot access the simulator's future trajectories or true velocity.
    """

    def __init__(self, n_blocks, table_top=0.8, defense_center=(0.,0.),
                 defense_radius=0.2, dt=0.05, gain=8.0):
        self.n = int(n_blocks)
        self.table_top = float(table_top)
        self.center = np.asarray(defense_center, dtype=float)
        self.radius = float(defense_radius)
        self.dt = float(dt)
        self.gain = float(gain)
        self.reset()

    def reset(self):
        self.target_id = None
        self.stage = "above"
        self.wait = 0
        self._prev_positions = None
        self._vel_ema = np.zeros((self.n, 2), dtype=float)

    def act(self, obs):
        robot = np.asarray(obs["defense_robot"], dtype=float).reshape(-1)
        ee = robot[-6:-3]
        blocks = np.asarray(obs["defense_blocks"], dtype=float).reshape(self.n, 6)
        positions = blocks[:, :3] + ee[None,:]
        if self._prev_positions is not None:
            raw_v = (positions[:, :2]-self._prev_positions[:, :2])/self.dt
            self._vel_ema = .6*self._vel_ema + .4*raw_v
        self._prev_positions = positions.copy()
        active = np.flatnonzero(blocks[:, 3] > 0.5)
        action = np.zeros(7, dtype=float)
        action[-1] = -1.  # Open gripper.
        if self.target_id is not None and self.target_id not in active:
            self.target_id = None
            self.stage, self.wait = "release", 0
        if self.stage == "release":
            action[:3] = np.clip(
                self.gain*(np.array([ee[0],ee[1],self.table_top+.25])-ee),-1.,1.)
            self.wait += 1
            if self.wait >= 12:
                self.stage, self.wait = "above", 0
            return action
        if self.target_id is None:
            if not len(active):
                return action
            scores = []
            for j in active:
                d = positions[j,:2]-self.center
                unit = d / max(np.linalg.norm(d),1e-8)
                v_in = max(0.,-float(np.dot(self._vel_ema[j],unit)))
                remaining = max(0.,np.linalg.norm(d)-self.radius-.032)
                ttb = remaining/max(v_in,.006)  # estimated time-to-boundary
                travel = .25*np.linalg.norm(positions[j]-ee)
                scores.append(ttb+travel)
            self.target_id = int(active[int(np.argmin(scores))])
        b = blocks[self.target_id]
        pos = positions[self.target_id]
        if self.stage == "above":
            waypoint = pos + [0,0,.13]
            if np.linalg.norm(ee-waypoint)<.035:
                self.stage="descend"
        elif self.stage == "descend":
            waypoint = pos + [0,0,.012]
            if np.linalg.norm(ee-waypoint)<.024:
                self.stage,self.wait="close",0
        elif self.stage == "close":
            waypoint = pos + [0,0,.012]
            action[-1]=+1.  # Close.
            self.wait += 1
            if b[4]>.5:
                self.stage="lift"
            elif self.wait>=40:
                self.stage,self.wait="above",0
        else:
            waypoint = np.array([pos[0],pos[1],self.table_top+.27])
            action[-1]=+1.
            if b[4]<.5 and ee[2]<self.table_top+.12:
                self.stage,self.wait="above",0
        action[:3]=np.clip(self.gain*(waypoint-ee),-1.,1.)
        return action

```


## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "diffdreamer-defense-prewm"
version = "0.1.0"
description = "Robosuite moving-block defense environment and pre-world-model data"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
test = ["pytest>=7"]

[tool.setuptools.packages.find]
include = ["defense_task*"]

```


## `tests/test_geometry_dataset.py`

```python
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

```


## `tests/test_sim_smoke.py`

```python
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

```
