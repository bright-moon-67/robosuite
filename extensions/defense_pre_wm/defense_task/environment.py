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

    def __init__(
        self,
        robots="Panda",
        controller_configs=None,
        num_blocks=5,
        defense_radius=0.20,
        table_full_size=(1.00, 1.10, 0.05),
        spawn_radius=(0.32, 0.40),
        spawn_angle=(np.deg2rad(100), np.deg2rad(260)),
        attack_speed=(0.008, 0.016),
        motion_mode="linear",
        stagger_seconds=6.0,
        start_delay=2.0,
        target_height=0.11,
        table_friction=(0.025, 0.005, 0.0001),
        force_limit=0.12,
        velocity_time_constant=0.035,
        lift_hold_steps=3,
        failure_on_first_breach=True,
        control_freq=20,
        horizon=1400,
        seed=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        **kwargs,
    ):
        if isinstance(num_blocks, bool) or int(num_blocks) != num_blocks or not 1 <= num_blocks <= 12:
            raise ValueError("num_blocks must be an integer in [1, 12]")
        if isinstance(robots, (list, tuple)) and len(robots) != 1:
            raise ValueError("DefenseTask currently supports one arm robot only")
        if defense_radius <= 0 or target_height <= 0:
            raise ValueError("defense_radius and target_height must be positive")
        if motion_mode not in ("linear", "sine", "pulsed"):
            raise ValueError("motion_mode must be linear, sine or pulsed")
        if robots not in ("Panda", ["Panda"], ("Panda",)):
            raise ValueError("This calibrated baseline supports one Panda only")
        if int(lift_hold_steps) != lift_hold_steps or lift_hold_steps < 1:
            raise ValueError("lift_hold_steps must be a positive integer")
        values = np.r_[
            defense_radius,
            table_full_size,
            spawn_radius,
            spawn_angle,
            attack_speed,
            stagger_seconds,
            start_delay,
            target_height,
            table_friction,
            force_limit,
            velocity_time_constant,
            control_freq,
            horizon,
        ]
        if not np.isfinite(values).all():
            raise ValueError("All task parameters must be finite")
        if len(table_full_size) != 3 or np.any(np.asarray(table_full_size) <= 0):
            raise ValueError("table_full_size must contain three positive dimensions")
        if len(attack_speed) != 2 or not 0 <= attack_speed[0] <= attack_speed[1]:
            raise ValueError("attack_speed must be a nonnegative increasing interval")
        if len(spawn_radius) != 2 or not defense_radius + np.sqrt(2) * 0.022 < spawn_radius[0] <= spawn_radius[1]:
            raise ValueError("Spawn footprints must start outside the defense circle")
        if len(spawn_angle) != 2 or not 0 < spawn_angle[1] - spawn_angle[0] <= 2 * np.pi:
            raise ValueError("spawn_angle must span (0, 2*pi] radians")
        if min(stagger_seconds, start_delay) < 0 or min(force_limit, velocity_time_constant, control_freq) <= 0:
            raise ValueError("Invalid motion or timing parameters")
        if len(table_friction) != 3 or np.any(np.asarray(table_friction) < 0):
            raise ValueError("table_friction must contain three nonnegative values")
        if defense_radius >= min(table_full_size[:2]) / 2 or target_height <= 0.022 + 0.035:
            raise ValueError("Circle must fit on table; clearance height must exceed breach height")
        if int(horizon) != horizon or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.start_delay = float(start_delay)
        self.force_limit = float(force_limit)
        self.velocity_time_constant = float(velocity_time_constant)
        self.n_blocks = int(num_blocks)
        self.defense_radius = float(defense_radius)
        self.defense_center = np.array([0.0, 0.0], dtype=float)
        self.table_full_size = tuple(table_full_size)
        self.table_friction = tuple(table_friction)
        self.table_offset = np.array([0.0, 0.0, 0.80])
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
        self._last_reward = {"discrete": 0.0, "continuous": 0.0, "total": 0.0}
        self._motion = []
        self._eef_vel = np.zeros(3)
        self._last_eef = None
        self._last_distance = None
        if controller_configs is None:
            from .controllers import panda_osc_config

            controller_configs = panda_osc_config()
        kwargs.setdefault("initialization_noise", None)
        kwargs.setdefault("gripper_types", "default")
        kwargs.setdefault("base_types", "default")
        kwargs.setdefault("hard_reset", False)
        kwargs.setdefault("renderer", "mujoco" if not has_renderer else "mjviewer")
        if kwargs.pop("use_camera_obs", False):
            raise ValueError("DefenseTask is a structured-state task; camera images are for rendering only")
        if kwargs.pop("ignore_done", False):
            raise ValueError("DefenseTask requires episode boundaries for replay alignment")
        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            use_camera_obs=False,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=False,
            seed=seed,
            **kwargs,
        )

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1 + 1e-6):
            raise ValueError("Action must be a finite normalized 7D OSC_POSE delta command")
        return super().step(action)

    @property
    def table_top(self):
        return float(self.table_offset[2])

    def _load_model(self):
        super()._load_model()
        if len(self.robots) != 1 or len(self.robots[0].arms) != 1:
            raise ValueError("DefenseTask needs one single-arm robot")
        # Same robot/table convention as robosuite Lift.
        base = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(base)
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        arena.table_visual.attrib.pop("material", None)
        arena.table_visual.set("rgba", "0.16 0.20 0.25 1")
        # Non-colliding ring visual, supported by MuJoCo MJCF geoms.
        n_ring = 72
        for j in range(n_ring):
            t0, t1 = 2 * np.pi * j / n_ring, 2 * np.pi * (j + 1) / n_ring
            a = self.defense_center + self.defense_radius * np.array([np.cos(t0), np.sin(t0)])
            b = self.defense_center + self.defense_radius * np.array([np.cos(t1), np.sin(t1)])
            z = self.table_top + 0.003
            ET.SubElement(
                arena.worldbody,
                "geom",
                {
                    "name": f"defense_ring_{j}",
                    "type": "capsule",
                    "fromto": f"{a[0]} {a[1]} {z} {b[0]} {b[1]} {z}",
                    "size": "0.0025",
                    "rgba": "0.15 0.9 0.98 1",
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                },
            )
        palette = [
            [0.95, 0.24, 0.22, 1],
            [0.93, 0.57, 0.20, 1],
            [0.23, 0.68, 0.93, 1],
            [0.68, 0.38, 0.88, 1],
            [0.34, 0.74, 0.36, 1],
        ]
        self.blocks = [
            BoxObject(
                name=f"attacker_{i}",
                size=self.block_half_size,
                density=600,
                friction=self.table_friction,
                rgba=palette[i % len(palette)],
                rng=self.rng,
            )
            for i in range(self.n_blocks)
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
            int(self.sim.model.jnt_dofadr[self.sim.model.joint_name2id(o.joints[0])]) for o in self.blocks
        ]
        self.ring_ids = [self.sim.model.geom_name2id(f"defense_ring_{i}") for i in range(72)]
        self.eef_site_id = self.robots[0].eef_site_id[self.robots[0].arms[0]]

    def _eef(self):
        return np.array(self.sim.data.site_xpos[self.eef_site_id], dtype=float)

    def _block_pos(self, i):
        return np.array(self.sim.data.body_xpos[self.block_body_ids[i]], dtype=float)

    def _block_v(self, i):
        adr = self.block_dof_addrs[i]
        return np.array(self.sim.data.qvel[adr : adr + 3], dtype=float)

    def _setup_observables(self):
        obs = super()._setup_observables()

        @sensor(modality="defense")
        def defense_robot(cache):
            robot = self.robots[0]
            q = np.array(self.sim.data.qpos[robot._ref_arm_joint_pos_indexes])
            dq = np.array(self.sim.data.qvel[robot._ref_arm_joint_vel_indexes])
            return np.concatenate(
                [
                    q,
                    dq,
                    self._eef(),
                    self.sim.data.get_site_xvelp(self.robots[0].gripper["right"].important_sites["grip_site"]),
                ]
            )

        @sensor(modality="defense")
        def defense_blocks(cache):
            # Each block: relative xyz to robot EEF, active, grasped, cleared.
            # No oracle object velocity, path ID or future trajectory in actor/RSSM obs.
            out = np.zeros((self.n_blocks, 6), dtype=np.float32)
            eef = self._eef()
            for i in range(self.n_blocks):
                p = self._block_pos(i)
                out[i] = np.r_[
                    p - eef, float(self.status[i] == 0), float(self.grasp_flags[i]), float(self.status[i] == 1)
                ]
            return out.reshape(-1)

        @sensor(modality="defense")
        def defense_status(cache):
            return self.status.astype(np.float32).copy()

        @sensor(modality="defense")
        def defense_clock(cache):
            return np.array([self.cur_time / (self.horizon / self.control_freq)], dtype=np.float32)

        @sensor(modality="defense")
        def defense_eef_quat(cache):
            from robosuite.utils.transform_utils import mat2quat

            return mat2quat(self.sim.data.site_xmat[self.eef_site_id].reshape(3, 3))  # xyzw

        for fn in (defense_robot, defense_blocks, defense_status, defense_clock, defense_eef_quat):
            obs[fn.__name__] = Observable(name=fn.__name__, sensor=fn, sampling_rate=self.control_freq)
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
        self._last_reward = {"discrete": 0.0, "continuous": 0.0, "total": 0.0}
        self.events = []
        self.sim.model.geom_rgba[self.ring_ids] = [0.15, 0.9, 0.98, 1]
        positions = sample_spawns(
            self.rng,
            self.n_blocks,
            self.defense_center,
            self.spawn_radius,
            self.spawn_angle,
            self.block_half_size[:2],
            np.asarray(self.table_full_size[:2]) / 2,
        )
        self._motion = []
        for i, (xy, angle) in enumerate(positions):
            q = np.array([1.0, 0.0, 0.0, 0.0])
            p = np.r_[xy, self.table_top + self.block_half_size[2] + 0.001]
            self.sim.data.set_joint_qpos(self.blocks[i].joints[0], np.r_[p, q])
            self.sim.data.qvel[self.block_dof_addrs[i] : self.block_dof_addrs[i] + 6] = 0.0
            self.sim.data.xfrc_applied[self.block_body_ids[i], :] = 0.0
            self._motion.append(
                {
                    "inward": -np.array([np.cos(angle), np.sin(angle)]),
                    "speed": float(self.rng.uniform(*self.speed_range)),
                    "start": float(self.start_delay + i * self.stagger_seconds + self.rng.uniform(0, 0.5)),
                    "frequency": float(self.rng.uniform(0.5, 1.0)),
                    "lateral_speed": float(self.rng.uniform(0.002, 0.005)),
                    "phase": float(self.rng.uniform(0.0, 2 * np.pi)),
                }
            )
        self.sim.forward()
        self._last_eef = self._eef()
        self._eef_vel = np.zeros(3, dtype=float)

    def _desired_velocity(self, i):
        cfg = self._motion[i]
        if self.sim.data.time < cfg["start"]:
            return np.zeros(2)
        u = cfg["inward"]
        t = self.sim.data.time - cfg["start"]
        vel = cfg["speed"] * u
        if self.motion_mode == "sine":
            perp = np.array([-u[1], u[0]])
            vel = vel + cfg["lateral_speed"] * np.cos(cfg["frequency"] * t + cfg["phase"]) * perp
        elif self.motion_mode == "pulsed":
            vel = vel * (0.75 + 0.5 * (1 + np.sin(cfg["frequency"] * t + cfg["phase"])) / 2)
        return vel

    def _pre_action(self, action, policy_step=False):
        # Called once PER MuJoCo physics substep by robosuite.
        # Never teleport blocks: free-joint dynamics + collisions remain physical.
        super()._pre_action(action, policy_step=policy_step)
        for i, bid in enumerate(self.block_body_ids):
            self.sim.data.xfrc_applied[bid, :] = 0.0
            if self.status[i] != 0:
                continue
            desired = self._desired_velocity(i)
            if np.linalg.norm(desired) < 1e-9:
                continue
            p = self._block_pos(i)
            # Contacts are current for this physics substep, not last policy step.
            grasped = self._check_grasp(self.robots[0].gripper, self.blocks[i])
            if grasped or p[2] > self.table_top + self.block_half_size[2] + 0.008 or p[2] < self.table_top - 0.022:
                continue
            measured = self._block_v(i)[:2]
            mass = max(1e-4, float(self.sim.model.body_mass[bid]))
            # Friction feed-forward and bounded P velocity tracking.
            unit = desired / np.linalg.norm(desired)
            friction = self.table_friction[0] * mass * 9.81
            force = friction * unit + mass * (desired - measured) / self.velocity_time_constant
            norm = float(np.linalg.norm(force))
            if norm > self.force_limit:
                force *= self.force_limit / norm
            self.sim.data.xfrc_applied[bid, :2] = force

    def _update_events(self):
        self._last_events = {"cleared": [], "breached": []}
        for i in range(self.n_blocks):
            self.grasp_flags[i] = bool(self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.blocks[i]))
            if self.status[i] != 0:
                continue
            p = self._block_pos(i)
            q = np.array(self.sim.data.body_xquat[self.block_body_ids[i]])
            if circle_intrusion(p, q, self.block_half_size, self.defense_center, self.defense_radius, self.table_top):
                self.status[i] = 2
                self._last_events["breached"].append(i)
                continue
            if self.grasp_flags[i]:
                self.was_grasped[i] = True
            if self.was_grasped[i] and self.grasp_flags[i] and (p[2] >= self.table_top + self.target_height):
                self.held_counts[i] += 1
            else:
                self.held_counts[i] = 0
            if self.held_counts[i] >= self.lift_hold_steps:
                self.status[i] = 1
                self._last_events["cleared"].append(i)

    def _nearest_distance(self):
        active = np.flatnonzero(self.status == 0)
        if not len(active):
            return 0.0
        ee = self._eef()
        return float(min(np.linalg.norm(ee - self._block_pos(i)) for i in active))

    def reward(self, action=None):
        # Event reward is sparse and non-differentiable (future Critic pathway).
        disc = 3.0 * len(self._last_events["cleared"]) - 12.0 * len(self._last_events["breached"])
        if len(self._last_events["cleared"]) and np.all(self.status == 1):
            disc += 8.0
        # This is a continuous *reward function*, NOT yet an autodiff rollout.
        dist = self._nearest_distance()
        ctrl = np.asarray(action) if action is not None else np.zeros(1)
        cont = -0.18 * np.tanh(5.0 * dist) - 0.0005 * np.sum(ctrl * ctrl)
        self._last_distance = dist
        self._last_reward = {
            "discrete": float(disc),
            "continuous": float(cont),
            "total": float(disc + cont),
        }
        return float(disc + cont)

    def _post_action(self, action):
        self._update_events()
        for event, ids in self._last_events.items():
            self.events.extend({"time": float(self.sim.data.time), "event": event, "object": int(i)} for i in ids)
        if np.any(self.status == 2):
            self.sim.model.geom_rgba[self.ring_ids] = [1, 0.22, 0.18, 1]
        old = np.asarray(self._last_eef if self._last_eef is not None else self._eef())
        now = self._eef()
        self._eef_vel = (now - old) / self.control_timestep
        self._last_eef = now
        reward, done, info = super()._post_action(action)
        failed = bool(np.any(self.status == 2))
        success = bool(np.all(self.status == 1))
        if success or (failed and self.failure_on_first_breach):
            self.done = True
            done = True
        terminated = success or (failed and self.failure_on_first_breach)
        truncated = bool(self.timestep >= self.horizon and not terminated)
        info.update(
            {
                "terminated": bool(terminated),
                "truncated": truncated,
                "continuation": float(not terminated),
                "r_disc": self._last_reward["discrete"],
                "r_diff": self._last_reward["continuous"],
                "new_cleared": list(self._last_events["cleared"]),
                "new_breached": list(self._last_events["breached"]),
                "status": self.status.copy(),
                "cleared_count": int(np.sum(self.status == 1)),
                "breached_count": int(np.sum(self.status == 2)),
                "success": success,
                "failed": failed,
                "time": float(self.cur_time),
                "block_pos_gt": np.stack([self._block_pos(i) for i in range(self.n_blocks)]),
                "block_vel_gt": np.stack([self._block_v(i) for i in range(self.n_blocks)]),
                "desired_vel_gt": np.stack(
                    [self._desired_velocity(i) if self.status[i] == 0 else np.zeros(2) for i in range(self.n_blocks)]
                ),
            }
        )
        # robosuite updates observables BEFORE _post_action; refresh status sensors.
        self._update_observables(force=True)
        return reward, done, info

    def _check_success(self):
        return bool(np.all(self.status == 1))
