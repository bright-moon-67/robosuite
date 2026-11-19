"""Rule-based OSC_POSE baseline: estimates approach speeds from relative positions."""

import numpy as np


class DefenseHeuristic:
    """Observation-only target selection + finite-state grasping.

    Per block observation: block XYZ relative EEF, active, grasped, cleared.
    The policy cannot access the simulator's future trajectories or true velocity.
    """

    def __init__(self, n_blocks, table_top=0.8, defense_center=(0.0, 0.0), defense_radius=0.2, dt=0.05, gain=8.0):
        if int(n_blocks) != n_blocks or n_blocks < 1 or not np.isfinite([dt, gain]).all() or min(dt, gain) <= 0:
            raise ValueError("Invalid policy count, dt or gain")
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
        self._desired_quat = None
        self._stage_steps = 0
        self._previous_stage = None
        self._prev_positions = None
        self._vel_ema = np.zeros((self.n, 2), dtype=float)

    def act(self, obs):
        if self.stage == self._previous_stage:
            self._stage_steps += 1
        else:
            self._stage_steps = 0
        self._previous_stage = self.stage
        if self._stage_steps > int(8 / self.dt):
            self.target_id = None
            self.stage, self.wait = "release", 0
        robot = np.asarray(obs["defense_robot"], dtype=float).reshape(-1)
        ee = robot[-6:-3]
        blocks = np.asarray(obs["defense_blocks"], dtype=float).reshape(self.n, 6)
        positions = blocks[:, :3] + ee[None, :]
        if self._prev_positions is not None:
            raw_v = (positions[:, :2] - self._prev_positions[:, :2]) / self.dt
            self._vel_ema = 0.6 * self._vel_ema + 0.4 * raw_v
        self._prev_positions = positions.copy()
        active = np.flatnonzero(blocks[:, 3] > 0.5)
        action = np.zeros(7, dtype=float)
        action[-1] = -1.0  # Open gripper.
        if "defense_eef_quat" in obs:
            from scipy.spatial.transform import Rotation

            quat = np.asarray(obs["defense_eef_quat"])
            if self._desired_quat is None:
                self._desired_quat = quat.copy()
            error = (Rotation.from_quat(self._desired_quat) * Rotation.from_quat(quat).inv()).as_rotvec()
            # Config maps normalized rotation delta to +/-0.5 radians.
            action[3:6] = np.clip(error / 0.5, -1, 1)
        if self.target_id is not None and self.target_id not in active:
            self.target_id = None
            self.stage, self.wait = "release", 0
        if self.stage == "release":
            action[:3] = np.clip(self.gain * (np.array([ee[0], ee[1], self.table_top + 0.25]) - ee), -1.0, 1.0)
            self.wait += 1
            if self.wait >= 12:
                self.stage, self.wait = "above", 0
            return action
        if self.target_id is None:
            if not len(active):
                return action
            scores = []
            for j in active:
                d = positions[j, :2] - self.center
                unit = d / max(np.linalg.norm(d), 1e-8)
                v_in = max(0.0, -float(np.dot(self._vel_ema[j], unit)))
                remaining = max(0.0, np.linalg.norm(d) - self.radius - 0.032)
                ttb = remaining / max(v_in, 0.006)  # estimated time-to-boundary
                travel = 0.25 * np.linalg.norm(positions[j] - ee)
                scores.append(ttb + travel)
            self.target_id = int(active[int(np.argmin(scores))])
        b = blocks[self.target_id]
        pos = positions[self.target_id]
        if self.stage == "above":
            waypoint = pos + [0, 0, 0.13]
            if np.linalg.norm(ee - waypoint) < 0.035:
                self.stage = "descend"
        elif self.stage == "descend":
            waypoint = pos + [0, 0, 0.012]
            if np.linalg.norm(ee - waypoint) < 0.024:
                self.stage, self.wait = "close", 0
        elif self.stage == "close":
            waypoint = pos + [0, 0, 0.012]
            action[-1] = +1.0  # Close.
            self.wait += 1
            if b[4] > 0.5:
                self.stage = "lift"
            elif self.wait >= 40:
                self.stage, self.wait = "above", 0
        else:
            waypoint = np.array([pos[0], pos[1], self.table_top + 0.27])
            action[-1] = +1.0
            if b[4] < 0.5 and ee[2] < self.table_top + 0.12:
                self.stage, self.wait = "above", 0
        action[:3] = np.clip(self.gain * (waypoint - ee), -1.0, 1.0)
        return action
