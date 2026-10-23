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


class V2StateController(Controller):
    """Fly to each recorded entry, pass the gate, then select the next ordered entry."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._freq = float(config.env.freq)
        self._cruise_speed = 0.80
        self._alignment_speed = 0.45
        self._gate_speed = 0.36
        self._gate_offset = 0.45
        self._gate_alignment_start_offset = 0.30
        self._gate_alignment_offset = 0.12
        self._gate_bypass_offset = 0.62
        self._takeoff_height = 0.45
        self._minimum_segment_time = 0.24
        self._obstacle_clearance = 0.30
        self._avoidance_margin = 0.08
        self._avoidance_lookahead = 0.28
        self._entry_tolerance = 0.09
        self._alignment_tolerance = 0.10
        self._exit_tolerance = 0.18
        self._tracking_slowdown_distance = 0.45

        self._pass_index = int(obs["n_gates_passed"])
        self._pending_pass_index: int | None = None
        self._phase = "TO_ENTRY"
        self._finished = False
        self._desired_yaw = 0.0

        self._known_gate_poses: dict[
            int, tuple[NDArray[np.float64], NDArray[np.float64]]
        ] = {}
        self._recorded_entries: dict[int, NDArray[np.float64]] = {}
        self._known_obstacles: dict[int, NDArray[np.float64]] = {}
        self._gate_visited = np.asarray(obs["gates_visited"], dtype=bool).copy()
        self._obstacles_visited = np.asarray(obs["obstacles_visited"], dtype=bool).copy()
        self._update_known_objects(obs)

        self._current_gate_id = -1
        self._current_center = np.zeros(3)
        self._current_normal = np.array([1.0, 0.0, 0.0])
        self._current_entry = np.asarray(obs["pos"], dtype=np.float64).copy()
        self._current_exit = self._current_entry.copy()

        self._local_spline: PchipInterpolator | None = None
        self._local_waypoints = np.empty((0, 3), dtype=np.float64)
        self._local_times = np.empty(0, dtype=np.float64)
        self._trajectory_time = 0.0
        self._trajectory_duration = 0.0
        self._last_desired_position = np.asarray(obs["pos"], dtype=np.float64).copy()

        self._select_current_gate(obs)
        self._plan_to_entry(obs, include_takeoff=True)

    def _update_known_objects(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Cache true object poses after they enter sensor range."""
        for gate_id in np.flatnonzero(np.asarray(obs["gates_visited"], dtype=bool)):
            gate_id = int(gate_id)
            self._known_gate_poses[gate_id] = (
                np.asarray(obs["gates_pos"][gate_id], dtype=np.float64).copy(),
                np.asarray(obs["gates_quat"][gate_id], dtype=np.float64).copy(),
            )

        for obstacle_id in np.flatnonzero(
            np.asarray(obs["obstacles_visited"], dtype=bool)
        ):
            obstacle_id = int(obstacle_id)
            self._known_obstacles[obstacle_id] = np.asarray(
                obs["obstacles_pos"][obstacle_id], dtype=np.float64
            ).copy()

    def _available_obstacles(
        self, obs: dict[str, NDArray[np.floating]]
    ) -> NDArray[np.float64]:
        obstacles = np.asarray(obs["obstacles_pos"], dtype=np.float64).copy()
        for obstacle_id, position in self._known_obstacles.items():
            obstacles[obstacle_id] = position
        return obstacles

    def _select_current_gate(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Select exactly the gate specified by the ordered pass sequence."""
        self._current_gate_id = int(obs["gate_sequence"][self._pass_index])
        direction = float(obs["gate_sequence_direction"][self._pass_index])

        if self._current_gate_id in self._known_gate_poses:
            center, quaternion = self._known_gate_poses[self._current_gate_id]
        else:
            center = np.asarray(
                obs["gates_pos"][self._current_gate_id], dtype=np.float64
            )
            quaternion = np.asarray(
                obs["gates_quat"][self._current_gate_id], dtype=np.float64
            )

        normal = R.from_quat(quaternion).apply([1.0, 0.0, 0.0]) * direction
        calculated_entry = center - self._gate_offset * normal

        # The first true observation records a physical gate's entry.  A repeated occurrence such
        # as the fifth pass through gate 2 reuses this exact point and direction.
        if self._current_gate_id in self._recorded_entries:
            entry = self._recorded_entries[self._current_gate_id].copy()
        else:
            entry = calculated_entry
            if self._current_gate_id in self._known_gate_poses:
                self._recorded_entries[self._current_gate_id] = entry.copy()

        obstacles = self._available_obstacles(obs)
        entry = self._move_staging_point(entry, normal, obstacles)
        exit_point = self._move_staging_point(
            center + self._gate_offset * normal, normal, obstacles
        )

        self._current_center = center.copy()
        self._current_normal = normal
        self._current_entry = entry
        self._current_exit = exit_point

    def _refresh_current_gate_if_observed(
        self, obs: dict[str, NDArray[np.floating]]
    ) -> bool:
        """Record the active gate's true entry and report whether the entry plan changed."""
        old_gate_visited = self._gate_visited.copy()
        old_obstacles_visited = self._obstacles_visited.copy()
        self._gate_visited = np.asarray(obs["gates_visited"], dtype=bool).copy()
        self._obstacles_visited = np.asarray(obs["obstacles_visited"], dtype=bool).copy()
        self._update_known_objects(obs)

        gate_newly_seen = (
            self._gate_visited[self._current_gate_id]
            and not old_gate_visited[self._current_gate_id]
        )
        obstacle_newly_seen = np.any(self._obstacles_visited & ~old_obstacles_visited)
        if not (gate_newly_seen or obstacle_newly_seen):
            return False

        old_entry = self._current_entry.copy()
        if gate_newly_seen:
            # Replace a nominal entry by the true entry.  For gate 2 this recorded point is reused
            # for the fifth pass later in the same episode.
            self._recorded_entries.pop(self._current_gate_id, None)
        self._select_current_gate(obs)
        return bool(
            gate_newly_seen
            or obstacle_newly_seen
            or np.linalg.norm(self._current_entry - old_entry) > 0.01
        )

    def _plan_to_entry(
        self,
        obs: dict[str, NDArray[np.floating]],
        include_takeoff: bool = False,
        departure_gate: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    ) -> None:
        """Plan only the transfer from the current position to the current gate entry."""
        start = np.asarray(obs["pos"], dtype=np.float64)
        waypoints = [start.copy()]
        if include_takeoff and start[2] < self._takeoff_height - 0.05:
            takeoff = start.copy()
            takeoff[2] = self._takeoff_height
            waypoints.append(takeoff)

        route_start = waypoints[-1]
        if departure_gate is not None:
            previous_center, previous_normal = departure_gate
            next_entry_is_behind_previous_gate = (
                np.dot(self._current_entry - previous_center, previous_normal) < 0.0
            )
            if next_entry_is_behind_previous_gate:
                bypass = self._build_departure_bypass(
                    route_start,
                    previous_center,
                    previous_normal,
                    self._available_obstacles(obs),
                )
                waypoints.extend(bypass)
                route_start = waypoints[-1]
        waypoints.extend(
            self._obstacle_aware_transfer(
                route_start, self._current_entry, self._available_obstacles(obs)
            )
        )
        self._phase = "TO_ENTRY"
        self._set_local_plan(waypoints, self._cruise_speed)

    def _build_departure_bypass(
        self,
        start: NDArray[np.float64],
        center: NDArray[np.float64],
        normal: NDArray[np.float64],
        obstacles: NDArray[np.float64],
    ) -> list[NDArray[np.float64]]:
        """Cross back over a passed gate's plane outside its physical frame."""
        lateral = np.array([-normal[1], normal[0], 0.0], dtype=np.float64)
        lateral /= max(np.linalg.norm(lateral), 1e-9)
        candidates = []
        for sign in (1.0, -1.0):
            side = sign * self._gate_bypass_offset * lateral
            outside_exit = center + self._gate_offset * normal + side
            outside_entry = center - self._gate_offset * normal + side
            path_length = np.linalg.norm(outside_exit - start) + np.linalg.norm(
                outside_entry - outside_exit
            )
            if len(obstacles):
                clearance = min(
                    np.min(np.linalg.norm(obstacles[:, :2] - outside_exit[:2], axis=1)),
                    np.min(np.linalg.norm(obstacles[:, :2] - outside_entry[:2], axis=1)),
                )
            else:
                clearance = np.inf
            candidates.append((clearance, path_length, outside_exit, outside_entry))

        safe = [item for item in candidates if item[0] >= self._obstacle_clearance]
        selected = min(safe or candidates, key=lambda item: item[1])
        return [selected[2], selected[3]]

    def _plan_through_gate(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Create a short crossing constrained to the gate normal."""
        start = np.asarray(obs["pos"], dtype=np.float64)
        alignment_entry = (
            self._current_center - self._gate_alignment_offset * self._current_normal
        )
        alignment_exit = (
            self._current_center + self._gate_alignment_offset * self._current_normal
        )
        alignment_after = (
            self._current_center
            + self._gate_alignment_start_offset * self._current_normal
        )
        waypoints = [
            start,
            alignment_entry,
            self._current_center,
            alignment_exit,
            alignment_after,
            self._current_exit,
        ]
        self._phase = "THROUGH_GATE"
        self._set_local_plan(waypoints, self._gate_speed)

    def _plan_gate_alignment(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Move to a centered point in front of the gate before starting the crossing."""
        start = np.asarray(obs["pos"], dtype=np.float64)
        alignment_start = (
            self._current_center
            - self._gate_alignment_start_offset * self._current_normal
        )
        self._phase = "ALIGN_GATE"
        self._set_local_plan([start, alignment_start], self._alignment_speed)

    def _obstacle_aware_transfer(
        self,
        start: NDArray[np.float64],
        target: NDArray[np.float64],
        obstacles: NDArray[np.float64],
    ) -> list[NDArray[np.float64]]:
        """Route to an entry, inserting a two-point detour around the first blocking pole."""
        if not len(obstacles):
            return [target.copy()]

        segment = target[:2] - start[:2]
        segment_length = np.linalg.norm(segment)
        if segment_length < 1e-6:
            return [target.copy()]
        direction = segment / segment_length
        required_clearance = self._obstacle_clearance + self._avoidance_margin

        conflicts = []
        for obstacle in obstacles:
            parameter = float(
                np.clip((obstacle[:2] - start[:2]) @ direction / segment_length, 0.0, 1.0)
            )
            closest = start[:2] + parameter * segment
            distance = np.linalg.norm(obstacle[:2] - closest)
            if distance < required_clearance and 0.05 < parameter < 0.95:
                conflicts.append((parameter, obstacle))
        if not conflicts:
            return [target.copy()]

        parameter, obstacle = min(conflicts, key=lambda item: item[0])
        lateral = np.array([-direction[1], direction[0]])
        height = start[2] + parameter * (target[2] - start[2])
        candidates = []
        for sign in (1.0, -1.0):
            side = sign * required_clearance * lateral
            before = np.array(
                [
                    *(obstacle[:2] - self._avoidance_lookahead * direction + side),
                    height,
                ]
            )
            after = np.array(
                [
                    *(obstacle[:2] + self._avoidance_lookahead * direction + side),
                    height,
                ]
            )
            clearance = min(
                np.min(np.linalg.norm(obstacles[:, :2] - before[:2], axis=1)),
                np.min(np.linalg.norm(obstacles[:, :2] - after[:2], axis=1)),
            )
            path_length = (
                np.linalg.norm(before - start)
                + np.linalg.norm(after - before)
                + np.linalg.norm(target - after)
            )
            candidates.append((clearance, path_length, before, after))

        safe = [item for item in candidates if item[0] >= self._obstacle_clearance]
        selected = min(safe or candidates, key=lambda item: item[1])
        return [selected[2], selected[3], target.copy()]

    def _move_staging_point(
        self,
        point: NDArray[np.float64],
        normal: NDArray[np.float64],
        obstacles: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Move an entry or exit sideways when a pole occupies the nominal point."""
        if not len(obstacles):
            return point.copy()
        required_clearance = self._obstacle_clearance + self._avoidance_margin
        if np.min(np.linalg.norm(obstacles[:, :2] - point[:2], axis=1)) >= required_clearance:
            return point.copy()

        lateral = np.array([-normal[1], normal[0]], dtype=np.float64)
        lateral /= max(np.linalg.norm(lateral), 1e-9)
        candidates = []
        for sign in (1.0, -1.0):
            candidate = point.copy()
            candidate[:2] += sign * required_clearance * lateral
            clearance = np.min(np.linalg.norm(obstacles[:, :2] - candidate[:2], axis=1))
            candidates.append((clearance, candidate))
        return max(candidates, key=lambda item: item[0])[1]

    def _set_local_plan(
        self, waypoints: list[NDArray[np.float64]], speed: float
    ) -> None:
        """Install a shape-preserving local trajectory."""
        cleaned = [np.asarray(waypoints[0], dtype=np.float64)]
        for waypoint in waypoints[1:]:
            waypoint = np.asarray(waypoint, dtype=np.float64)
            if np.linalg.norm(waypoint - cleaned[-1]) > 1e-5:
                cleaned.append(waypoint)
        if len(cleaned) == 1:
            cleaned.append(cleaned[0].copy())
        self._local_waypoints = np.asarray(cleaned)

        distances = np.linalg.norm(np.diff(self._local_waypoints, axis=0), axis=1)
        durations = np.maximum(distances / speed, self._minimum_segment_time)
        self._local_times = np.concatenate(([0.0], np.cumsum(durations)))
        self._trajectory_time = 0.0
        self._trajectory_duration = float(self._local_times[-1])
        self._local_spline = PchipInterpolator(
            self._local_times, self._local_waypoints, axis=0
        )
        self._last_desired_position = self._local_waypoints[0].copy()

    def _local_plan_finished(self) -> bool:
        """Return whether the active local trajectory has reached its final setpoint."""
        return self._trajectory_time >= self._trajectory_duration - 1e-6

    def _advance_phase(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Advance phases without changing the configured gate order."""
        position = np.asarray(obs["pos"], dtype=np.float64)
        if self._phase == "TO_ENTRY":
            entry_reached = np.linalg.norm(position - self._current_entry) < self._entry_tolerance
            # The entry is only a staging target.  If disturbances or an obstacle prevent the
            # drone from reaching its strict radius, move on to the centered alignment target once
            # the entry trajectory has completed instead of holding forever.
            if entry_reached or self._local_plan_finished():
                self._plan_gate_alignment(obs)
        elif self._phase == "ALIGN_GATE":
            alignment_start = (
                self._current_center
                - self._gate_alignment_start_offset * self._current_normal
            )
            alignment_error = np.linalg.norm(position - alignment_start)
            alignment_reached = alignment_error < self._alignment_tolerance
            settled_close_enough = self._local_plan_finished() and alignment_error < 0.16
            if alignment_reached or settled_close_enough:
                self._plan_through_gate(obs)
        elif self._phase == "TO_EXIT" and self._pending_pass_index is not None:
            exit_reached = np.linalg.norm(position - self._current_exit) < self._exit_tolerance
            if exit_reached or self._local_plan_finished():
                previous_gate = (self._current_center.copy(), self._current_normal.copy())
                self._pass_index = self._pending_pass_index
                self._pending_pass_index = None
                self._select_current_gate(obs)
                self._plan_to_entry(obs, departure_gate=previous_gate)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.float32]:
        """Update the ordered state machine and return a full-state command."""
        del info

        if self._phase == "TO_ENTRY" and self._refresh_current_gate_if_observed(obs):
            self._plan_to_entry(obs)
        self._advance_phase(obs)

        tracking_error = np.linalg.norm(
            np.asarray(obs["pos"], dtype=np.float64) - self._last_desired_position
        )
        time_scale = np.clip(
            1.0 - tracking_error / self._tracking_slowdown_distance, 0.40, 1.0
        )
        self._trajectory_time = min(
            self._trajectory_time + time_scale / self._freq, self._trajectory_duration
        )

        assert self._local_spline is not None
        local_plan_ended = self._trajectory_time >= self._trajectory_duration - 1e-6
        if local_plan_ended:
            # PCHIP generally has a non-zero endpoint derivative.  Commanding that derivative
            # together with a fixed final position makes the drone circle around the target and
            # can prevent the phase distance threshold from ever being met.  Hold every local
            # endpoint as a true hover while waiting for the next phase.
            desired_position = self._local_waypoints[-1].copy()
            desired_velocity = np.zeros(3)
            desired_acceleration = np.zeros(3)
        else:
            desired_position = self._local_spline(self._trajectory_time)
            desired_velocity = self._local_spline(self._trajectory_time, 1) * time_scale
            desired_acceleration = self._local_spline(self._trajectory_time, 2) * time_scale**2
        self._last_desired_position = desired_position.copy()

        horizontal_velocity = desired_velocity[:2]
        if np.linalg.norm(horizontal_velocity) > 1e-4:
            self._desired_yaw = float(np.arctan2(horizontal_velocity[1], horizontal_velocity[0]))
        desired_quaternion = R.from_euler("z", self._desired_yaw).as_quat()

        return np.concatenate(
            (
                desired_position,
                desired_velocity,
                desired_acceleration,
                desired_quaternion,
                np.zeros(3),
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
        """Use the environment's gate counter as the sole source of race progress."""
        del action, reward, terminated, truncated, info
        gates_passed = int(obs["n_gates_passed"])
        total_passes = len(obs["gate_sequence"])
        self._finished = gates_passed >= total_passes

        if not self._finished and gates_passed > self._pass_index:
            self._pending_pass_index = gates_passed
            self._phase = "TO_EXIT"
        return self._finished

    def episode_callback(self):
        """No cross-episode learning is used by this baseline."""

    def render_callback(self, sim: Sim):
        """Draw only the currently active local trajectory."""
        assert self._local_spline is not None
        trajectory = self._local_spline(
            np.linspace(0.0, self._trajectory_duration, 100)
        )
        setpoint = self._local_spline(self._trajectory_time).reshape(1, -1)
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.025)
