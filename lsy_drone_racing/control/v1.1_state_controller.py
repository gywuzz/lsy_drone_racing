
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class V1StateController(Controller):
    """Follow a gate-derived trajectory using the full-state command interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._freq = float(config.env.freq)
        self._cruise_speed = 0.55  # Conservative Level-0 starting value [m/s].
        self._gate_offset = 0.45  # Approach/exit distance from the gate plane [m].
        self._gate_alignment_offset = 0.12  # Keep the local path normal to the gate [m].
        self._gate_bypass_offset = 0.65  # Cross back outside the physical gate frame [m].
        self._takeoff_height = 0.45  # Initial height before flying towards gate 1 [m].
        self._minimum_segment_time = 0.45  # Avoid very short, aggressive spline segments [s].
        self._obstacle_clearance = 0.30  # Horizontal clearance from obstacle centers [m].
        self._avoidance_margin = 0.08  # Extra room for tracking error and spline curvature [m].

        self._tick = 0
        self._finished = False
        self._obstacles = np.asarray(obs["obstacles_pos"], dtype=np.float64)
        self._waypoints = self._build_waypoints(obs)
        self._waypoint_times = self._allocate_waypoint_times(self._waypoints)
        self._t_total = float(self._waypoint_times[-1])

        # A global cubic spline can overshoot badly after avoidance points and create loops that
        # approach a gate from the wrong side.  PCHIP is shape-preserving in each coordinate: it
        # passes through every waypoint without introducing coordinate-wise overshoot.
        self._position_spline = PchipInterpolator(
            self._waypoint_times, self._waypoints, axis=0
        )

    def _build_waypoints(
        self, obs: dict[str, NDArray[np.floating]]
    ) -> NDArray[np.float64]:
        """Create takeoff and approach/center/exit waypoints for the configured gate order."""
        start = np.asarray(obs["pos"], dtype=np.float64)
        takeoff = start.copy()
        takeoff[2] = max(float(start[2]), self._takeoff_height)
        waypoints = [start, takeoff]

        gate_sequence = np.asarray(obs["gate_sequence"], dtype=np.int64)
        gate_directions = np.asarray(obs["gate_sequence_direction"], dtype=np.float64)
        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
        gates_quat = np.asarray(obs["gates_quat"], dtype=np.float64)

        gate_plans = []
        saved_gate_entries: dict[int, NDArray[np.float64]] = {}
        for gate_id, direction in zip(gate_sequence, gate_directions, strict=True):
            center = gates_pos[gate_id]

            # The environment defines a gate crossing along the gate frame's local x-axis.
            normal = R.from_quat(gates_quat[gate_id]).apply([1.0, 0.0, 0.0])
            normal *= direction

            approach = center - self._gate_offset * normal
            exit_point = center + self._gate_offset * normal
            approach = self._move_staging_point_away_from_obstacles(approach, normal)
            exit_point = self._move_staging_point_away_from_obstacles(exit_point, normal)

            # Repeated occurrences of a physical gate must use exactly the same entry point.  In
            # the default track this makes the fifth target (gate 2 again) return directly to the
            # entry used during the first pass instead of inventing another route behind the gate.
            gate_key = int(gate_id)
            if gate_key in saved_gate_entries:
                approach = saved_gate_entries[gate_key].copy()
            else:
                saved_gate_entries[gate_key] = approach.copy()

            # The outer staging points may move sideways to avoid an obstacle.  These two inner
            # points remain on the gate normal and force the spline to become straight immediately
            # before and after crossing the gate plane.  This is especially important when the
            # same physical gate occurs more than once in the gate sequence.
            alignment_entry = center - self._gate_alignment_offset * normal
            alignment_exit = center + self._gate_alignment_offset * normal
            gate_plans.append(
                (approach, alignment_entry, center, alignment_exit, exit_point, normal)
            )

        for index, gate_plan in enumerate(gate_plans):
            approach, alignment_entry, center, alignment_exit, exit_point, normal = gate_plan
            waypoints.extend(
                (approach, alignment_entry, center, alignment_exit, exit_point)
            )

            # After a valid pass, the next gate may lie back on the entry side of the current gate.
            # Returning directly would send the trajectory backwards through the same opening.
            # Instead, cross the current gate plane again outside its physical frame.
            if index + 1 < len(gate_plans):
                next_approach = gate_plans[index + 1][0]
                next_is_on_entry_side = np.dot(next_approach - center, normal) < 0.0
                if next_is_on_entry_side:
                    waypoints.extend(self._build_gate_return_bypass(center, normal))

        # Staging points have already been moved away from nearby obstacles.  Do not insert
        # additional detours between gates: for a repeated gate, the next target must remain the
        # exact entry point saved during its first occurrence.
        return np.asarray(waypoints, dtype=np.float64)

    def _build_gate_return_bypass(
        self, center: NDArray[np.float64], normal: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Create two points that return across a gate plane outside the gate frame."""
        lateral = np.array([-normal[1], normal[0], 0.0], dtype=np.float64)
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm < 1e-6:
            lateral = np.array([0.0, 1.0, 0.0])
        else:
            lateral /= lateral_norm

        candidates = []
        for sign in (1.0, -1.0):
            side_offset = sign * self._gate_bypass_offset * lateral
            exit_side = center + self._gate_offset * normal + side_offset
            entry_side = center - self._gate_offset * normal + side_offset
            candidates.append((exit_side, entry_side))

        # Choose the side that stays farthest from all known obstacle centers.
        if len(self._obstacles):
            clearances = [
                min(
                    np.min(np.linalg.norm(self._obstacles[:, :2] - pair[0][:2], axis=1)),
                    np.min(np.linalg.norm(self._obstacles[:, :2] - pair[1][:2], axis=1)),
                )
                for pair in candidates
            ]
            return candidates[int(np.argmax(clearances))]
        return candidates[0]

    def _move_staging_point_away_from_obstacles(
        self, point: NDArray[np.float64], gate_normal: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Move an approach/exit point sideways if an obstacle occupies that point.

        Gate centers are never moved.  Only the staging point is shifted within the gate plane, so
        the resulting trajectory still passes through the exact gate center.
        """
        if not len(self._obstacles):
            return point

        required_clearance = self._obstacle_clearance + self._avoidance_margin
        if np.min(np.linalg.norm(self._obstacles[:, :2] - point[:2], axis=1)) >= required_clearance:
            return point

        lateral = np.array([-gate_normal[1], gate_normal[0]], dtype=np.float64)
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm < 1e-6:
            lateral = np.array([0.0, 1.0])
        else:
            lateral /= lateral_norm

        candidates = []
        for sign in (1.0, -1.0):
            candidate = point.copy()
            candidate[:2] += sign * required_clearance * lateral
            candidates.append(candidate)

        # Pick the side with the largest distance to the closest obstacle.
        clearances = [
            np.min(np.linalg.norm(self._obstacles[:, :2] - candidate[:2], axis=1))
            for candidate in candidates
        ]
        return candidates[int(np.argmax(clearances))]

    def _allocate_waypoint_times(self, waypoints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Assign more time to long segments and keep short segments numerically well behaved."""
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        durations = np.maximum(distances / self._cruise_speed, self._minimum_segment_time)
        return np.concatenate(([0.0], np.cumsum(durations)))

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.float32]:
        """Return the desired 16-dimensional full-state command."""
        del obs, info  # V1 follows its Level-0 trajectory open-loop in the planning layer.

        t = min(self._tick / self._freq, self._t_total)
        desired_position = self._position_spline(t)
        desired_velocity = self._position_spline(t, 1)
        desired_acceleration = self._position_spline(t, 2)

        # Point the drone along the horizontal direction of travel.  Near zero velocity, keep the
        # previous valid yaw to avoid atan2 noise at the beginning and end of the trajectory.
        horizontal_velocity = desired_velocity[:2]
        if np.linalg.norm(horizontal_velocity) > 1e-4:
            self._desired_yaw = float(np.arctan2(horizontal_velocity[1], horizontal_velocity[0]))
        elif not hasattr(self, "_desired_yaw"):
            self._desired_yaw = 0.0

        desired_quaternion = R.from_euler("z", self._desired_yaw).as_quat()
        desired_angular_velocity = np.zeros(3)

        return np.concatenate(
            (
                desired_position,
                desired_velocity,
                desired_acceleration,
                desired_quaternion,
                desired_angular_velocity,
            ),
            dtype=np.float32,
        )

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance trajectory time and stop after all configured gate passes are complete."""
        del action, reward, terminated, truncated, info

        self._tick += 1
        all_gates_passed = int(obs["n_gates_passed"]) >= len(obs["gate_sequence"])
        trajectory_timed_out = self._tick / self._freq >= self._t_total + 2.0
        self._finished = all_gates_passed or trajectory_timed_out
        return self._finished

    def episode_callback(self):
        """Reset time-dependent state when an environment reuses this controller instance."""
        self._tick = 0
        self._finished = False

    def render_callback(self, sim: Sim):
        """Draw the planned path in green and the current setpoint in red."""
        t = min(self._tick / self._freq, self._t_total)
        setpoint = self._position_spline(t).reshape(1, -1)
        trajectory = self._position_spline(np.linspace(0.0, self._t_total, 200))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.025)
