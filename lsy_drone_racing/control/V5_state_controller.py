"""V5: observed-map 3D routing and smooth full-state gate crossings.

Transit paths use A* with inflated gate/pole geometry and line-of-sight
simplification. Quintic references supply position, velocity and acceleration;
a bounded altitude bias compensates randomized dynamics. Transit waypoints
carry a continuous tangent velocity; each gate retains a straight crossing. Gate progress is
accepted only from the environment, with an exit segment before the next turn.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class V5StateController(Controller):
    """Plan collision-aware transfers and align before crossing each gate."""

    # Speeds [m/s]
    TRANSIT_SPEED = 1.80
    CROSS_SPEED = 1.05
    APPROACH_GAIN = 2.0  # Velocity command per metre of remaining distance [1/s]
    # Geometry [m]
    STAGING_DIST = 0.65
    STAGING_OFFSETS = (0.0, 0.18, -0.18, 0.30, -0.30)
    EXIT_DIST = 0.38
    CRUISE_HEIGHT = 1.78
    DRONE_RADIUS = 0.09
    OBSTACLE_CLEARANCE = 0.50
    GATE_HALF_EXTENT = 0.55
    GATE_BYPASS = 0.85
    # Tolerances
    STOP_TOLERANCE = 0.10
    PASS_TOLERANCE = 0.07
    STOP_TIMEOUT = 2.0
    PASS_TIMEOUT = 1.50
    MAX_RETRIES = 5
    POS_LIMIT_LOW = np.array([-2.40, -1.42, 0.12])
    POS_LIMIT_HIGH = np.array([2.40, 1.42, 1.92])

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialise the controller and plan the first transit.

        Args:
            obs: Initial observation of the environment.
            info: Initial environment information.
            config: Race configuration.
        """
        super().__init__(obs, info, config)
        self._freq = float(config.env.freq)
        self._dt = 1.0 / self._freq

        self._gate_ids = np.asarray(obs["gate_sequence"], dtype=int)
        self._directions = np.asarray(obs["gate_sequence_direction"], dtype=float)
        self._n_passes = int(len(self._gate_ids))
        self._pass_index = int(obs["n_gates_passed"])
        self._finished = False
        self._pending_pass = None

        self._seen_gates = np.asarray(obs["gates_visited"], dtype=bool).copy()
        self._seen_obstacles = np.asarray(obs["obstacles_visited"], dtype=bool).copy()

        self._center = np.zeros(3)
        self._staging = np.zeros(3)
        self._exit = np.zeros(3)

        self._phase = "TRANSIT"
        self._waypoints = np.zeros((1, 3))
        self._stop_flags = np.zeros(1, dtype=bool)
        self._index = 0
        self._phase_time = 0.0
        self._retries = 0
        self._desired_yaw = 0.0
        self._integral = np.zeros(3)
        self._path_points = np.zeros((2, 3))

        self._select_gate(obs)
        self._plan_transit(obs, np.asarray(obs["pos"], dtype=np.float64))

    # ---------------------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------------------
    @staticmethod
    def _horizontal(vector: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return the normalised horizontal component of a 3D vector."""
        horizontal = np.array([vector[0], vector[1], 0.0], dtype=np.float64)
        norm = float(np.linalg.norm(horizontal))
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return horizontal / norm

    @staticmethod
    def _segment_distance(
        start: NDArray[np.floating], end: NDArray[np.floating], points: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        """Distance of every point (xy) to the segment start-end (xy)."""
        if not len(points):
            return np.full(0, np.inf)
        segment = end[:2] - start[:2]
        length_sq = float(segment @ segment)
        if length_sq < 1e-12:
            return np.linalg.norm(points[:, :2] - start[:2], axis=1)
        parameter = np.clip((points[:, :2] - start[:2]) @ segment / length_sq, 0.0, 1.0)
        closest = start[:2] + parameter[:, None] * segment
        return np.linalg.norm(points[:, :2] - closest, axis=1)

    def _gate_pose(
        self, obs: dict[str, NDArray[np.floating]], gate_id: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the best known pose of a gate (true pose once seen, nominal before)."""
        center = np.asarray(obs["gates_pos"][gate_id], dtype=np.float64).copy()
        quaternion = np.asarray(obs["gates_quat"][gate_id], dtype=np.float64).copy()
        return center, quaternion

    def _obstacles(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.float64]:
        """Return the best known pole positions."""
        return np.asarray(obs["obstacles_pos"], dtype=np.float64).copy()

    # ---------------------------------------------------------------------------------------
    # Gate geometry
    # ---------------------------------------------------------------------------------------
    def _select_gate(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Choose the staging point of the active gate and derive the crossing line."""
        gate_id = int(self._gate_ids[self._pass_index])
        direction = float(self._directions[self._pass_index])
        center, quaternion = self._gate_pose(obs, gate_id)
        normal = R.from_quat(quaternion).apply([1.0, 0.0, 0.0]) * direction
        normal_h = self._horizontal(normal)
        lateral = np.array([-normal_h[1], normal_h[0], 0.0])
        obstacles = self._obstacles(obs)

        best_staging = center - self.STAGING_DIST * normal_h
        best_staging[2] = center[2]
        best_score = -np.inf
        for offset in self.STAGING_OFFSETS:
            staging = center - self.STAGING_DIST * normal_h + offset * lateral
            staging[2] = center[2]
            crossing = self._line_direction(staging, center)
            exit_point = center + self.EXIT_DIST * crossing
            if len(obstacles):
                line_clearance = float(
                    np.min(self._segment_distance(staging, exit_point, obstacles))
                )
                point_clearance = float(
                    np.min(np.linalg.norm(obstacles[:, :2] - staging[:2], axis=1))
                )
            else:
                line_clearance = self.OBSTACLE_CLEARANCE
                point_clearance = self.OBSTACLE_CLEARANCE
            score = 3.0 * min(point_clearance, 0.45) + min(line_clearance, 0.30)
            score -= 0.25 * abs(offset)
            if point_clearance < 0.30:
                score -= 3.0
            if line_clearance < 0.28:
                score -= 1.0
            if score > best_score:
                best_score = score
                best_staging = staging.copy()

        crossing = self._line_direction(best_staging, center)
        self._center = center.copy()
        self._staging = best_staging.copy()
        self._exit = center + self.EXIT_DIST * crossing

    @staticmethod
    def _line_direction(start: NDArray[np.float64], end: NDArray[np.float64]) -> NDArray:
        """Unit vector from start to end, with a safe fallback for coincident points."""
        direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return direction / norm

    # ---------------------------------------------------------------------------------------
    # Route planning
    # ---------------------------------------------------------------------------------------
    def _plan_transit(
        self, obs: dict[str, NDArray[np.floating]], start: NDArray[np.float64]
    ) -> None:
        """Search free space, then simplify the route to the gate staging point."""
        start = np.asarray(start, dtype=np.float64)
        staging = self._staging
        axes = [np.arange(-2.2, 2.21, 0.2), np.arange(-1.2, 1.21, 0.2), np.arange(0.4, 1.81, 0.2)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        shape = grid.shape[:3]

        def collision(points: NDArray[np.float64]) -> NDArray[np.bool_]:
            points = np.asarray(points)
            occupied = np.zeros(points.shape[:-1], dtype=bool)
            for k, center in enumerate(obs["gates_pos"]):
                local = (
                    R.from_quat(obs["gates_quat"][k])
                    .inv()
                    .apply((points - center).reshape(-1, 3))
                    .reshape(points.shape)
                )
                margin = 0.10 if obs["gates_visited"][k] else 0.24
                occupied |= (
                    (np.abs(local[..., 0]) < 0.10 + margin)
                    & (np.abs(local[..., 1]) < 0.36 + margin)
                    & (points[..., 2] < center[2] + 0.36 + margin)
                )
            for k, pole in enumerate(obs["obstacles_pos"]):
                radius = 0.16 if obs["obstacles_visited"][k] else 0.31
                occupied |= (np.linalg.norm(points[..., :2] - pole[:2], axis=-1) < radius) & (
                    points[..., 2] < 1.73
                )
            return occupied

        def clear(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
            samples = np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 0.04) + 1))
            return not np.any(collision(samples))

        blocked = collision(grid)
        flat = grid.reshape(-1, 3)
        free = np.flatnonzero(~blocked.ravel())

        def attach(point: NDArray[np.float64]) -> tuple[int, int, int] | None:
            for idx in free[np.argsort(np.linalg.norm(flat[free] - point, axis=1))]:
                if clear(point, flat[idx]):
                    return tuple(np.unravel_index(idx, shape))
            return None

        source, goal = attach(start), attach(staging)
        points = [start.copy(), staging.copy()]
        if source is not None and goal is not None:
            queue = [(0.0, source)]
            cost = {source: 0.0}
            parent = {}
            while queue:
                _, node = heapq.heappop(queue)
                if node == goal:
                    route = [staging.copy(), grid[node]]
                    while node in parent:
                        node = parent[node]
                        route.append(grid[node])
                    route.append(start.copy())
                    route.reverse()
                    points = [route[0]]
                    i = 0
                    while i < len(route) - 1:
                        j = len(route) - 1
                        while j > i + 1 and not clear(route[i], route[j]):
                            j -= 1
                        points.append(route[j])
                        i = j
                    break
                for axis in range(3):
                    for sign in (-1, 1):
                        neighbor = list(node)
                        neighbor[axis] += sign
                        neighbor = tuple(neighbor)
                        if (
                            any(n < 0 or n >= shape[k] for k, n in enumerate(neighbor))
                            or blocked[neighbor]
                        ):
                            continue
                        g = cost[node] + 0.2
                        if g >= cost.get(neighbor, float("inf")):
                            continue
                        cost[neighbor] = g
                        parent[neighbor] = node
                        h = float(np.linalg.norm(grid[neighbor] - grid[goal]))
                        heapq.heappush(queue, (g + h, neighbor))
        self._waypoints = np.asarray(points, dtype=np.float64)
        self._stop_flags = np.ones(len(points), dtype=bool)
        self._index = 1
        self._phase = "TRANSIT"
        self._phase_time = 0.0
        self._path_points = self._waypoints.copy()

    def _plan_cross(self) -> None:
        """Straight crossing through the gate centre to the exit point."""
        self._waypoints = np.asarray([self._center.copy(), self._exit.copy()], dtype=np.float64)
        self._stop_flags = np.asarray([False, False])
        self._index = 1
        self._phase = "CROSS"
        self._phase_time = 0.0
        self._path_points = np.asarray(
            [self._staging.copy(), self._center.copy(), self._exit.copy()]
        )

    # ---------------------------------------------------------------------------------------
    # Control
    # ---------------------------------------------------------------------------------------
    def _active_target(self) -> NDArray[np.float64]:
        """Return the waypoint currently chased."""
        return self._waypoints[min(self._index, len(self._waypoints) - 1)]

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.float32]:
        """Advance the waypoint list and return the next full-state command."""
        del info
        self._latest_obs = obs
        position = np.asarray(obs["pos"], dtype=np.float64)
        velocity = np.asarray(obs["vel"], dtype=np.float64)
        self._phase_time += self._dt

        if not self._finished:
            self._maybe_refine(obs, position)
            self._advance_waypoint(position, velocity)

        target = self._active_target()
        delta = target - position
        distance = float(np.linalg.norm(delta))

        if self._finished:
            max_speed = 0.0
        elif self._phase == "CROSS":
            max_speed = self.CROSS_SPEED
        else:
            max_speed = self.TRANSIT_SPEED

        if (
            not hasattr(self, "_trajectory_target")
            or np.linalg.norm(target - self._trajectory_target) > 1e-5
        ):
            self._trajectory_target = target.copy()
            self._trajectory_start = position.copy()
            # Carry bounded measured velocity through transfer corners.
            self._trajectory_velocity = velocity.copy()
            speed = float(np.linalg.norm(velocity))
            if speed > 0.6:
                self._trajectory_velocity *= 0.6 / speed
            if self._phase == "CROSS":
                self._trajectory_velocity[:] = 0.0
            self._trajectory_end_velocity = np.zeros(3)
            if self._phase == "TRANSIT" and self._index + 1 < len(self._waypoints):
                incoming = self._line_direction(position, target)
                outgoing = self._line_direction(target, self._waypoints[self._index + 1])
                tangent = incoming + outgoing
                if incoming @ outgoing > -0.2:
                    self._trajectory_end_velocity = (
                        0.45 * tangent / max(np.linalg.norm(tangent), 1e-6)
                    )
            self._trajectory_duration = max(
                0.85, distance * 1.8 / max(max_speed, 0.5), np.sqrt(5.8 * distance / 4.5)
            )
            self._trajectory_tick = 0.0
        duration = self._trajectory_duration
        self._trajectory_tick = min(duration, self._trajectory_tick + self._dt)
        u = self._trajectory_tick / duration
        displacement = target - self._trajectory_start
        v0 = self._trajectory_velocity * duration
        v1 = self._trajectory_end_velocity * duration
        c3 = 10 * displacement - 6 * v0 - 4 * v1
        c4 = -15 * displacement + 8 * v0 + 7 * v1
        c5 = 6 * displacement - 3 * v0 - 3 * v1
        position_command = self._trajectory_start + v0 * u + c3 * u**3 + c4 * u**4 + c5 * u**5
        velocity_command = (v0 + 3 * c3 * u**2 + 4 * c4 * u**3 + 5 * c5 * u**4) / duration
        acceleration = (6 * c3 * u + 12 * c4 * u**2 + 20 * c5 * u**3) / duration**2
        self._integral = np.clip(
            self._integral + 1.5 * (position_command - position) * self._dt, -0.5, 0.5
        )
        # Horizontal bias left over from a turn displaces the narrow crossing line.
        self._integral[:2] = 0.0
        position_command += self._integral
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])

        position_command = np.clip(position_command, self.POS_LIMIT_LOW, self.POS_LIMIT_HIGH)
        return np.concatenate(
            (position_command, velocity_command, acceleration, quaternion, np.zeros(3)),
            dtype=np.float32,
        )

    def _advance_waypoint(
        self, position: NDArray[np.float64], velocity: NDArray[np.floating]
    ) -> None:
        """Move to the next waypoint when the current one is reached."""
        if self._finished:
            return
        target = self._active_target()
        distance = float(np.linalg.norm(target - position))
        speed = float(np.linalg.norm(velocity))
        stop = bool(self._stop_flags[min(self._index, len(self._stop_flags) - 1)])

        if stop:
            if self._index + 1 < len(self._waypoints):
                reached = distance < 0.16
            else:
                reached = distance < self.STOP_TOLERANCE and speed < 0.30
        else:
            reached = distance < self.PASS_TOLERANCE and speed < 0.60
        if not reached:
            return
        self._phase_time = 0.0

        if self._index + 1 < len(self._waypoints):
            self._index += 1
            return

        if self._phase == "TRANSIT":
            self._plan_cross()
        elif self._pending_pass is not None:
            self._pass_index = self._pending_pass
            self._pending_pass = None
            self._select_gate(self._latest_obs)
            self._plan_transit(self._latest_obs, position)
        else:
            self._retries += 1
            self._select_gate(self._latest_obs)
            self._plan_transit(self._latest_obs, position)

    def _maybe_refine(
        self, obs: dict[str, NDArray[np.floating]], position: NDArray[np.float64]
    ) -> None:
        """Re-plan when the true pose of the gate or of a pole becomes visible."""
        gate_id = int(self._gate_ids[self._pass_index])
        visited = np.asarray(obs["gates_visited"], dtype=bool)
        obstacles_visited = np.asarray(obs["obstacles_visited"], dtype=bool)
        gate_new = bool(visited[gate_id]) and not bool(self._seen_gates[gate_id])
        pole_new = bool(np.any(obstacles_visited & ~self._seen_obstacles))
        self._seen_gates = visited.copy()
        self._seen_obstacles = obstacles_visited.copy()
        if not (gate_new or pole_new) or (self._phase == "CROSS" and not gate_new):
            return
        self._select_gate(obs)
        if self._phase == "CROSS" or self._index == 3:
            self._waypoints = np.asarray([position.copy(), self._staging.copy()])
            self._stop_flags = np.asarray([False, True])
            self._index = 1
            self._phase = "TRANSIT"
            self._phase_time = 0.0
        else:
            self._plan_transit(obs, position)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Track gate progress reported by the environment."""
        del action, reward, terminated, truncated, info
        gates_passed = int(obs["n_gates_passed"])
        if gates_passed > self._pass_index:
            self._pending_pass = gates_passed
            self._retries = 0
        self._finished = gates_passed >= self._n_passes
        return self._finished

    def episode_callback(self):
        """No cross-episode learning is used."""

    def render_callback(self, sim: Sim):
        """Draw the route and the active target."""
        if len(self._path_points) >= 2:
            draw_line(sim, self._path_points, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, self._active_target().reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.03)
