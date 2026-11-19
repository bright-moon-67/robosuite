"""A configurable moving-object defense task on the standard manipulation table.

Attackers are free-joint rigid bodies, driven by bounded horizontal forces.
The robot always uses the ordinary robosuite action / controller interface.
"""

import xml.etree.ElementTree as ET

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import mat2quat


class TableDefense(ManipulationEnv):
    """Protect a tabletop circle from slowly approaching, graspable cubes.

    Distances are meters; speeds are meters/second; delays are seconds.
    ``motion`` is ``linear``, ``weave`` or ``mixed``. The default five
    attackers approach from the front semicircle accessible to a Panda.
    Removal requires lifting the object's *bottom* ``removal_height`` above
    the tabletop for ``removal_hold_time``, or moving its entire footprint
    beyond ``removal_radius``. Removed objects remain physical but stop moving.
    A breach is latched when any projected cube corner/edge overlaps the disk
    while the object is near the table. Success requires every object removed
    without a breach. With ``terminate_on_breach=False``, previews can continue.

    Propulsion is a small planar velocity servo with friction compensation,
    not pose teleportation. It disengages while grasped or lifted. This is a
    task-level moving-target abstraction, not a model of a specific actuator.
    """

    COLORS = (
        (0.95, 0.28, 0.22, 1), (1.0, 0.65, 0.12, 1),
        (0.20, 0.55, 0.98, 1), (0.68, 0.36, 0.94, 1), (0.15, 0.78, 0.57, 1),
    )

    def __init__(
        self, robots="Panda", num_attackers=5, defense_radius=0.20,
        defense_center=(-0.12, 0.0), spawn_radius=(0.34, 0.38),
        attack_speed=0.008, start_delay=2.0, launch_interval=4.0,
        motion="linear", weave_amplitude=0.018, weave_period=8.0,
        cube_size=0.04, table_full_size=(0.8, 1.0, 0.05),
        removal_height=0.12, removal_radius=0.46, removal_hold_time=0.15,
        terminate_on_breach=True, use_object_obs=True, reward_scale=1.0,
        reward_shaping=False, **kwargs,
    ):
        if isinstance(num_attackers, bool) or int(num_attackers) != num_attackers or not 1 <= num_attackers <= 12:
            raise ValueError("num_attackers must be an integer from 1 to 12")
        self.num_attackers = int(num_attackers)
        self.defense_center = np.asarray(defense_center, dtype=float)
        self.defense_radius = float(defense_radius)
        self.table_full_size = np.asarray(table_full_size, dtype=float)
        self.table_offset = np.array([0.0, 0.0, 0.8])
        self.cube_half_size = float(cube_size) / 2
        self.spawn_radius = np.broadcast_to(np.asarray(spawn_radius, dtype=float), (2,)).copy()
        self.attack_speeds = np.broadcast_to(np.asarray(attack_speed, dtype=float), (self.num_attackers,)).copy()
        self.start_delay, self.launch_interval = float(start_delay), float(launch_interval)
        self.motion = motion
        self.weave_amplitude, self.weave_period = float(weave_amplitude), float(weave_period)
        self.removal_height, self.removal_radius = float(removal_height), float(removal_radius)
        self.removal_hold_time = float(removal_hold_time)
        self.terminate_on_breach = bool(terminate_on_breach)
        self.use_object_obs = use_object_obs
        self.reward_scale, self.reward_shaping = reward_scale, reward_shaping
        self.drive_friction = 0.025
        if self.defense_center.shape != (2,) or self.table_full_size.shape != (3,):
            raise ValueError("defense_center must be length 2; table_full_size must be length 3")
        scalars = np.r_[self.defense_center, self.table_full_size, self.spawn_radius, self.attack_speeds,
                        defense_radius, cube_size, start_delay, launch_interval, weave_amplitude,
                        weave_period, removal_height, removal_radius, removal_hold_time]
        if not np.all(np.isfinite(scalars)):
            raise ValueError("Task parameters must be finite")
        if min(defense_radius, cube_size, weave_period, removal_height, removal_hold_time) <= 0:
            raise ValueError("Radii, dimensions and time constants must be positive")
        if np.any(self.table_full_size <= 0) or np.any(self.attack_speeds < 0):
            raise ValueError("Table dimensions must be positive and speeds nonnegative")
        if min(start_delay, launch_interval, weave_amplitude) < 0 or motion not in {"linear", "weave", "mixed"}:
            raise ValueError("Invalid delay, weave amplitude, or motion mode")
        if self.spawn_radius[0] > self.spawn_radius[1]:
            raise ValueError("spawn_radius must be an increasing interval")
        if self.spawn_radius[0] <= defense_radius + np.sqrt(2) * self.cube_half_size:
            raise ValueError("Spawn positions must lie fully outside the defense disk")
        if removal_radius <= self.spawn_radius[1] + np.sqrt(2) * self.cube_half_size:
            raise ValueError("removal_radius must exceed the initial object footprints")
        if np.any(np.abs(self.defense_center) + defense_radius >= self.table_full_size[:2] / 2):
            raise ValueError("Defense disk must fit on the table")
        angles = np.deg2rad(np.linspace(-80, 80, self.num_attackers)) if self.num_attackers > 1 else np.array([0.0])
        self._outward = np.column_stack([np.cos(angles), np.sin(angles)])
        self._tangents = np.column_stack([-np.sin(angles), np.cos(angles)])
        self._radii = np.linspace(*self.spawn_radius, self.num_attackers)
        self.spawn_positions = self.defense_center + self._outward * self._radii[:, None]
        if np.any(np.abs(self.spawn_positions) + np.sqrt(2) * self.cube_half_size >= self.table_full_size[:2] / 2):
            raise ValueError("Spawned objects would extend beyond the table; reduce spawn_radius")
        if self.num_attackers > 1:
            separations = np.linalg.norm(self.spawn_positions[:, None] - self.spawn_positions[None, :], axis=2)
            np.fill_diagonal(separations, np.inf)
            if separations.min() < 2 * np.sqrt(2) * self.cube_half_size + 0.005:
                raise ValueError("Too many / too large attackers for collision-free initial placement")
        kwargs.setdefault("initialization_noise", None)
        kwargs.setdefault("control_freq", 20)
        kwargs.setdefault("horizon", 1200)
        kwargs.setdefault("use_camera_obs", False)
        kwargs.setdefault("camera_names", "defenseview")
        super().__init__(robots=robots, **kwargs)

    def _load_model(self):
        super()._load_model()
        if len(self.robots) != 1 or len(self.robots[0].arms) != 1:
            raise ValueError("TableDefense expects one single-arm robot")
        robot = self.robots[0].robot_model
        robot.set_base_xpos(robot.base_xpos_offset["table"](self.table_full_size[0]))
        arena = TableArena(table_full_size=self.table_full_size, table_friction=(self.drive_friction, 0.005, 0.0001),
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        arena.table_visual.attrib.pop("material", None)
        arena.table_visual.set("rgba", "0.16 0.20 0.25 1")
        z = self.table_offset[2] + 0.0015
        # Flat non-colliding markings: never obstruct or support a block.
        for i in range(96):
            a, b = 2 * np.pi * np.array([i, i + 1]) / 96
            p = self.defense_center + self.defense_radius * np.array([np.cos(a), np.sin(a)])
            q = self.defense_center + self.defense_radius * np.array([np.cos(b), np.sin(b)])
            arena.worldbody.append(ET.Element("geom", name=f"defense_ring_{i}", type="capsule",
                fromto=array_to_string([*p, z, *q, z]), size="0.002", rgba="0.15 0.9 0.98 1",
                contype="0", conaffinity="0", group="1"))
        for i, start in enumerate(self.spawn_positions):
            end = self.defense_center + self._outward[i] * (self.defense_radius + 0.025)
            for j in range(6):
                p = start + (end - start) * j / 6
                q = start + (end - start) * (j + 0.55) / 6
                arena.worldbody.append(ET.Element("geom", name=f"attack_lane_{i}_{j}", type="capsule",
                    fromto=array_to_string([*p, z, *q, z]), size="0.0009", rgba="0.55 0.60 0.65 1",
                    contype="0", conaffinity="0", group="1"))
        for name, pos, target, fovy in (
            ("defenseview", [1.20, -1.45, 1.95], [-0.22, 0, 1.00], 44),
            ("defensetop", [-0.12, 0, 2.15], [-0.12, 0, 0.80], 49),
        ):
            backward = np.array(pos) - target
            backward /= np.linalg.norm(backward)
            up = np.array([0, 1, 0]) if name == "defensetop" else np.array([0, 0, 1])
            right = np.cross(up, backward)
            right /= np.linalg.norm(right)
            rotation = np.column_stack([right, np.cross(backward, right), backward])
            arena.set_camera(name, pos, mat2quat(rotation)[[3, 0, 1, 2]], {"fovy": str(fovy)})
        self.attackers = [BoxObject(name=f"attacker_{i}", size=[self.cube_half_size] * 3,
            rgba=self.COLORS[i % len(self.COLORS)], density=600,
            friction=[self.drive_friction, 0.005, 0.0001]) for i in range(self.num_attackers)]
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[robot], mujoco_objects=self.attackers)

    def _setup_references(self):
        super()._setup_references()
        self.attacker_body_ids = np.array([self.sim.model.body_name2id(obj.root_body) for obj in self.attackers])
        self._dof_starts = [self.sim.model.get_joint_qvel_addr(obj.joints[0])[0] for obj in self.attackers]
        self._ring_geom_ids = [self.sim.model.geom_name2id(f"defense_ring_{i}") for i in range(96)]

    def _reset_internal(self):
        super()._reset_internal()
        self.removed = np.zeros(self.num_attackers, dtype=bool)
        self.breached = np.zeros(self.num_attackers, dtype=bool)
        self._lift_since = np.full(self.num_attackers, np.nan)
        self._new_removed = self._new_breached = 0
        self._grasped = np.zeros(self.num_attackers, dtype=bool)
        self.launch_times = self.start_delay + np.arange(self.num_attackers) * self.launch_interval
        self.events = []
        if not self.deterministic_reset:
            for i, obj in enumerate(self.attackers):
                self.sim.data.set_joint_qpos(obj.joints[0], [*self.spawn_positions[i],
                    self.table_offset[2] + self.cube_half_size + 0.001, 1, 0, 0, 0])
                self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        self.sim.model.geom_rgba[self._ring_geom_ids] = [0.15, 0.9, 0.98, 1]

    @property
    def attacker_positions(self):
        return self.sim.data.body_xpos[self.attacker_body_ids].copy()

    def _footprint(self, i):
        """XY convex hull and lowest height of the oriented physical cube."""
        from scipy.spatial import ConvexHull

        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        body = self.attacker_body_ids[i]
        vertices = self.sim.data.body_xpos[body] + self.cube_half_size * corners @ self.sim.data.body_xmat[body].reshape(3, 3).T
        xy = vertices[:, :2]
        return xy[ConvexHull(xy).vertices], vertices[:, 2].min()

    def _footprint_clearance(self, i):
        """Exact distance from projected cube polygon to defense disk; <=0 means breach."""
        polygon, bottom = self._footprint(i)
        edges = np.roll(polygon, -1, axis=0) - polygon
        relative = self.defense_center - polygon
        cross = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
        inside = np.all(cross >= -1e-10)
        t = np.clip(np.sum(relative * edges, axis=1) / np.sum(edges * edges, axis=1), 0, 1)
        distances = np.linalg.norm(polygon + t[:, None] * edges - self.defense_center, axis=1)
        distance = 0.0 if inside else float(distances.min())
        return distance - self.defense_radius, bottom, distance

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if policy_step:
            self._grasped = np.array([self._check_grasp(self.robots[0].gripper, obj) for obj in self.attackers])
        now = self.sim.data.time
        for i, dof in enumerate(self._dof_starts):
            self.sim.data.qfrc_applied[dof:dof + 2] = 0
            pos = self.sim.data.body_xpos[self.attacker_body_ids[i]]
            if self.removed[i] or self.breached[i] or self._grasped[i]:
                continue
            if now < self.launch_times[i] or pos[2] > self.table_offset[2] + self.cube_half_size + 0.008:
                continue
            if pos[2] < self.table_offset[2] - self.cube_half_size:
                continue
            desired = -self.attack_speeds[i] * self._outward[i]
            if self.motion == "weave" or (self.motion == "mixed" and i % 2):
                phase = 2 * np.pi * (now - self.launch_times[i]) / self.weave_period
                desired = desired + self._tangents[i] * self.weave_amplitude * 2 * np.pi / self.weave_period * np.cos(phase)
            speed = np.linalg.norm(desired)
            if speed == 0:
                continue
            mass = self.sim.model.body_mass[self.attacker_body_ids[i]]
            velocity = self.sim.data.qvel[dof:dof + 2]
            force = mass * (desired - velocity) / 0.035 + self.drive_friction * mass * 9.81 * desired / speed
            self.sim.data.qfrc_applied[dof:dof + 2] = np.clip(force, -0.12, 0.12)

    def _update_task_state(self):
        self._new_removed = self._new_breached = 0
        for i in range(self.num_attackers):
            if self.removed[i] or self.breached[i]:
                continue
            clearance, bottom, distance = self._footprint_clearance(i)
            raised = bottom >= self.table_offset[2] + self.removal_height
            if raised and np.isnan(self._lift_since[i]):
                self._lift_since[i] = self.sim.data.time
            elif not raised:
                self._lift_since[i] = np.nan
            held_up = raised and self.sim.data.time - self._lift_since[i] >= self.removal_hold_time
            moved_out = distance > self.removal_radius
            if held_up or moved_out:
                self.removed[i] = True
                self._new_removed += 1
                self.events.append({"time": float(self.sim.data.time), "object": i,
                                    "event": "removed", "reason": "lift" if held_up else "outside"})
            elif clearance <= 0 and bottom < self.table_offset[2] + self.removal_height:
                self.breached[i] = True
                self._new_breached += 1
                self.events.append({"time": float(self.sim.data.time), "object": i, "event": "breach"})
        if np.any(self.breached):
            self.sim.model.geom_rgba[self._ring_geom_ids] = [1, 0.22, 0.18, 1]

    def _post_action(self, action):
        self._update_task_state()
        # Task flags change after the final physics substep; expose them now.
        self._update_observables(force=True)
        reward, done, info = super()._post_action(action)
        success = self._check_success()
        task_done = success or (self.terminate_on_breach and bool(np.any(self.breached)))
        self.done = done or (task_done and not self.ignore_done)
        info.update(self.task_info())
        return reward, self.done, info

    def task_info(self):
        clearances = np.array([self._footprint_clearance(i)[0] for i in range(self.num_attackers)])
        active = (~self.removed) & (~self.breached) & (self.sim.data.time >= self.launch_times)
        return {"success": self._check_success(), "failed": bool(np.any(self.breached)),
                "removed_count": int(self.removed.sum()), "breach_count": int(self.breached.sum()),
                "active_count": int(active.sum()), "clearances": clearances,
                "removed": self.removed.copy(), "breached": self.breached.copy(),
                "time": float(self.sim.data.time)}

    def reward(self, action=None):
        value = self._new_removed / self.num_attackers - float(self._new_breached)
        return float(value if self.reward_scale is None else self.reward_scale * value)

    def _check_success(self):
        return bool(np.all(self.removed) and not np.any(self.breached))

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            @sensor(modality="object")
            def attackers_pos(obs_cache):
                return self.attacker_positions.ravel()

            @sensor(modality="object")
            def attackers_vel(obs_cache):
                return np.concatenate([self.sim.data.qvel[d:d + 3] for d in self._dof_starts])

            @sensor(modality="object")
            def attackers_status(obs_cache):
                return np.column_stack([self.removed, self.breached]).astype(float).ravel()

            @sensor(modality="object")
            def defense_geometry(obs_cache):
                return np.r_[self.defense_center, self.defense_radius]

            for fn in (attackers_pos, attackers_vel, attackers_status, defense_geometry):
                observables[fn.__name__] = Observable(name=fn.__name__, sensor=fn, sampling_rate=self.control_freq)
        return observables
