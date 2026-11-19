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
