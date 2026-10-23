"""Improved V1StateController.

Every modification w.r.t. the original file is tagged ``# [CHANGE n]`` in the code:

[CHANGE 1]  PCHIP (C1) -> minimum-snap QP (handout Sec. 4, Eq. 27-49), C3-continuous.
[CHANGE 2]  Time allocation: lower minimum segment time + sampling-based feasibility check
            (peak speed / acceleration) that rescales segment durations.
[CHANGE 3]  Whole-trajectory obstacle check; detour waypoints are inserted where the spline
            actually gets too close (not only for staging points).
[CHANGE 4]  Replanning from the current state if gates/obstacles change in the observation.
[CHANGE 5]  Progress-adaptive reference clock (reference slows down when tracking error grows).
[CHANGE 6]  Yaw: continuous (unwrapped), rate-limited, with yaw-rate feed-forward.
[CHANGE 7]  Engineering: parameters from config, dataclass for gate plans, shared helpers,
            yaw state initialised in __init__ and reset in episode_callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


# ----------------------------------------------------------------------------------------------
# [CHANGE 1] Minimum-snap trajectory (QP of the handout, solved via its KKT system)
# ----------------------------------------------------------------------------------------------
_ORDER = 7  # polynomial order N per segment (8 coefficients, enough for min-snap, r = 4)
_SNAP = 4  # derivative order r that is minimised


def _falling(i: int, order: int) -> float:
    """c(i) = i (i-1) ... (i-order+1), the factor in handout Eq. (31)/(37)."""
    return float(np.prod(np.arange(i, i - order, -1)))  # empty product (order 0) = 1


def _poly_row(order: int, tau: float) -> NDArray[np.float64]:
    """Row vector a with a @ p = d^order/dt^order of sum_n p_n t^n, evaluated at t = tau."""
    row = np.zeros(_ORDER + 1)
    for i in range(order, _ORDER + 1):
        row[i] = _falling(i, order) * tau ** (i - order)
    return row


def _snap_hessian(duration: float) -> NDArray[np.float64]:
    """Hessian Q_s of J_s = int_0^T ||p^(r)(t)||^2 dt = p^T Q p (handout Eq. 28-45).

    Local time per segment (t_{s-1} = 0, t_s = T) -> well conditioned numerics.
    """
    q = np.zeros((_ORDER + 1, _ORDER + 1))
    for i in range(_SNAP, _ORDER + 1):
        for j in range(_SNAP, _ORDER + 1):
            power = i + j - 2 * _SNAP + 1
            q[i, j] = _falling(i, _SNAP) * _falling(j, _SNAP) * duration**power / power
    return q


class MinSnapTrajectory:
    """Piecewise polynomial position trajectory (ascending coefficients, local segment time)."""

    def __init__(self, coeffs: NDArray[np.float64], durations: NDArray[np.float64]):
        self.coeffs = coeffs  # (S, N+1, 3)
        self.knots = np.concatenate(([0.0], np.cumsum(durations)))
        self.t_total = float(self.knots[-1])

    def __call__(self, t: float | NDArray[np.float64], order: int = 0) -> NDArray[np.float64]:
        scalar = np.ndim(t) == 0
        t_arr = np.clip(np.atleast_1d(np.asarray(t, dtype=np.float64)), 0.0, self.t_total)
        seg = np.clip(np.searchsorted(self.knots, t_arr, side="right") - 1, 0, len(self.coeffs) - 1)
        tau = t_arr - self.knots[seg]
        out = np.zeros((len(t_arr), 3))
        for i in range(order, _ORDER + 1):
            out += _falling(i, order) * (tau ** (i - order))[:, None] * self.coeffs[seg, i, :]
        return out[0] if scalar else out


def solve_min_snap(
    waypoints: NDArray[np.float64], durations: NDArray[np.float64], v0: NDArray[np.float64]
) -> MinSnapTrajectory:
    """min sum_s p_s^T Q_s p_s  s.t.  waypoint interpolation + C^3 continuity (handout Eq. 49).

    All three axes share the same Hessian and constraint matrix, so one KKT solve with three
    right-hand sides is enough.  Start: velocity v0, zero acceleration/jerk.  End: rest.
    """
    n_seg = len(durations)
    n_var = (_ORDER + 1) * n_seg
    zero = np.zeros(3)
    hessian = np.zeros((n_var, n_var))
    rows: list[NDArray[np.float64]] = []
    rhs: list[NDArray[np.float64]] = []

    def block(seg: int) -> slice:
        return slice(seg * (_ORDER + 1), (seg + 1) * (_ORDER + 1))

    def add(seg: int, order: int, tau: float, value: NDArray[np.float64]) -> None:
        row = np.zeros(n_var)
        row[block(seg)] = _poly_row(order, tau)
        rows.append(row)
        rhs.append(value)

    for s in range(n_seg):
        hessian[block(s), block(s)] = _snap_hessian(durations[s])
        add(s, 0, 0.0, waypoints[s])  # start of segment hits waypoint s
        add(s, 0, durations[s], waypoints[s + 1])  # end of segment hits waypoint s+1
    add(0, 1, 0.0, v0)  # boundary conditions at start
    add(0, 2, 0.0, zero)
    add(0, 3, 0.0, zero)
    for order in (1, 2, 3):  # boundary conditions at end (rest)
        add(n_seg - 1, order, durations[-1], zero)
    for s in range(n_seg - 1):  # C^3 continuity at inner knots
        for order in (1, 2, 3):
            row = np.zeros(n_var)
            row[block(s)] = _poly_row(order, durations[s])
            row[block(s + 1)] = -_poly_row(order, 0.0)
            rows.append(row)
            rhs.append(zero)

    a_mat = np.asarray(rows)
    b_mat = np.asarray(rhs)
    n_con = len(a_mat)
    kkt = np.block(
        [[2.0 * hessian + 1e-9 * np.eye(n_var), a_mat.T], [a_mat, np.zeros((n_con, n_con))]]
    )
    rhs_full = np.vstack([np.zeros((n_var, 3)), b_mat])
    try:
        sol = np.linalg.solve(kkt, rhs_full)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(kkt, rhs_full, rcond=None)[0]
    coeffs = sol[:n_var].reshape(n_seg, _ORDER + 1, 3)
    return MinSnapTrajectory(coeffs, np.asarray(durations, dtype=np.float64))


def _get_param(config, name: str, default):
    """[CHANGE 7] Read optional tuning values from ``config.controller_params`` (if present)."""
    try:
        return type(default)(config.controller_params[name])
    except (AttributeError, KeyError, TypeError):
        return default


@dataclass
class GatePlan:
    """[CHANGE 7] Replaces the anonymous 6-tuple of the original code."""

    approach: NDArray[np.float64]
    alignment_entry: NDArray[np.float64]
    center: NDArray[np.float64]
    alignment_exit: NDArray[np.float64]
    exit_point: NDArray[np.float64]
    normal: NDArray[np.float64]


class V1StateController(Controller):
    """Follow a gate-derived minimum-snap trajectory using the full-state command interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._freq = float(config.env.freq)
        # [CHANGE 7] All tuning values can be overridden in the config ([controller_params]).
        self._cruise_speed = _get_param(config, "cruise_speed", 0.55)  # [m/s]
        self._gate_offset = _get_param(config, "gate_offset", 0.45)  # [m]
        self._gate_alignment_offset = _get_param(config, "gate_alignment_offset", 0.12)  # [m]
        self._gate_bypass_offset = _get_param(config, "gate_bypass_offset", 0.65)  # [m]
        self._takeoff_height = _get_param(config, "takeoff_height", 0.45)  # [m]
        self._obstacle_clearance = _get_param(config, "obstacle_clearance", 0.30)  # [m]
        self._avoidance_margin = _get_param(config, "avoidance_margin", 0.08)  # [m]

        # [CHANGE 2] Smaller minimum segment time (0.45 -> 0.15 s): the old value forced the
        # drone to crawl through the short 0.12 m gate alignment segments.  Aggressiveness is now
        # bounded by the sampled speed/acceleration limits below instead.
        self._minimum_segment_time = _get_param(config, "minimum_segment_time", 0.15)  # [s]
        self._max_speed = _get_param(config, "max_speed", 1.5)  # [m/s]
        self._max_accel = _get_param(config, "max_accel", 4.0)  # [m/s^2]

        # [CHANGE 3] Whole-trajectory collision refinement.
        self._max_refine_iters = _get_param(config, "max_refine_iters", 10)
        self._check_dt = 0.02  # [s] sampling step for limit / collision checks

        # [CHANGE 4] Replanning.
        self._replan_threshold = _get_param(config, "replan_threshold", 0.03)  # [m]
        self._replan_cooldown = _get_param(config, "replan_cooldown", 1.0)  # [s]

        # [CHANGE 5] Progress-adaptive reference clock.
        self._err_deadband = _get_param(config, "err_deadband", 0.10)  # [m]
        self._err_max = _get_param(config, "err_max", 0.50)  # [m]
        self._min_time_scale = _get_param(config, "min_time_scale", 0.30)
        self._timeout_factor = _get_param(config, "timeout_factor", 2.0)

        # [CHANGE 6] / [CHANGE 7] Yaw state lives in __init__ (no hasattr hack any more).
        self._max_yaw_rate = _get_param(config, "max_yaw_rate", 2.0)  # [rad/s]
        self._desired_yaw = self._initial_yaw(obs)
        self._yaw_rate = 0.0

        self._tick = 0
        self._finished = False
        self._last_replan_tick = -int(self._replan_cooldown * self._freq)
        self._plan(obs, replan=False)
        # [CHANGE 4] Remember the initial plan so episode_callback can restore it.
        self._initial_plan = (self._traj, self._planned_gates, self._planned_quats,
                              self._planned_obstacles, self._obstacles)

    # ------------------------------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------------------------------
    def _plan(self, obs: dict[str, NDArray[np.floating]], replan: bool) -> None:
        """[CHANGE 4] Planning factored out of __init__ so it can be re-run in flight."""
        self._obstacles = np.asarray(obs["obstacles_pos"], dtype=np.float64)
        self._planned_gates = np.asarray(obs["gates_pos"], dtype=np.float64).copy()
        self._planned_quats = np.asarray(obs["gates_quat"], dtype=np.float64).copy()
        self._planned_obstacles = self._obstacles.copy()

        start = np.asarray(obs["pos"], dtype=np.float64)
        v0 = np.asarray(obs["vel"], dtype=np.float64) if replan else np.zeros(3)
        first_target = int(obs["n_gates_passed"]) if replan else 0

        waypoints = self._build_waypoints(obs, start, first_target, replan)
        self._traj = self._optimize(waypoints, v0)
        self._t_total = self._traj.t_total
        self._t_ref = 0.0  # [CHANGE 5] reference clock restarts with every plan
        self._plan_tick0 = self._tick

    def _build_waypoints(
        self,
        obs: dict[str, NDArray[np.floating]],
        start: NDArray[np.float64],
        first_target: int,
        replan: bool,
    ) -> NDArray[np.float64]:
        """Create takeoff and approach/center/exit waypoints for the (remaining) gate order."""
        waypoints = [start]
        if not replan:  # [CHANGE 4] no takeoff point when replanning in flight
            takeoff = start.copy()
            takeoff[2] = max(float(start[2]), self._takeoff_height)
            waypoints.append(takeoff)

        # [CHANGE 4] Only the gates that are still to be passed.
        gate_sequence = np.asarray(obs["gate_sequence"], dtype=np.int64)[first_target:]
        gate_directions = np.asarray(obs["gate_sequence_direction"], dtype=np.float64)[first_target:]
        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
        gates_quat = np.asarray(obs["gates_quat"], dtype=np.float64)

        gate_plans: list[GatePlan] = []
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

            # Repeated occurrences of a physical gate reuse exactly the same entry point.
            gate_key = int(gate_id)
            if gate_key in saved_gate_entries:
                approach = saved_gate_entries[gate_key].copy()
            else:
                saved_gate_entries[gate_key] = approach.copy()

            # Inner points stay on the gate normal -> straight crossing of the gate plane.
            gate_plans.append(
                GatePlan(
                    approach=approach,
                    alignment_entry=center - self._gate_alignment_offset * normal,
                    center=center,
                    alignment_exit=center + self._gate_alignment_offset * normal,
                    exit_point=exit_point,
                    normal=normal,
                )
            )

        for index, plan in enumerate(gate_plans):
            points = [
                plan.approach,
                plan.alignment_entry,
                plan.center,
                plan.alignment_exit,
                plan.exit_point,
            ]
            if replan and index == 0:
                # [CHANGE 4] Do not fly backwards to staging points that are already behind us.
                points = self._drop_passed_staging_points(points, plan, start)
            waypoints.extend(points)

            # Next gate on the entry side of this one -> cross the gate plane outside the frame.
            if index + 1 < len(gate_plans):
                next_approach = gate_plans[index + 1].approach
                if np.dot(next_approach - plan.center, plan.normal) < 0.0:
                    waypoints.extend(self._build_gate_return_bypass(plan.center, plan.normal))

        return self._dedupe(np.asarray(waypoints, dtype=np.float64))

    @staticmethod
    def _drop_passed_staging_points(
        points: list[NDArray[np.float64]], plan: GatePlan, start: NDArray[np.float64]
    ) -> list[NDArray[np.float64]]:
        """[CHANGE 4] Remove approach/alignment-entry points the drone is already beyond."""
        if np.dot(start - plan.center, plan.normal) >= 0.0:
            return points  # already behind the gate plane, keep everything
        kept = []
        for k, point in enumerate(points):
            if k < 2 and np.dot(point - start, plan.normal) < 0.0:
                continue
            kept.append(point)
        return kept

    @staticmethod
    def _dedupe(waypoints: NDArray[np.float64], tol: float = 1e-3) -> NDArray[np.float64]:
        """[CHANGE 1] Zero-length segments make the QP needlessly ill-conditioned."""
        kept = [waypoints[0]]
        for point in waypoints[1:]:
            if np.linalg.norm(point - kept[-1]) > tol:
                kept.append(point)
        return np.asarray(kept)

    @staticmethod
    def _lateral_unit(normal: NDArray[np.float64]) -> NDArray[np.float64]:
        """[CHANGE 7] Shared helper (was duplicated in two methods): horizontal unit normal."""
        lateral = np.array([-normal[1], normal[0], 0.0], dtype=np.float64)
        norm = np.linalg.norm(lateral)
        if norm < 1e-6:
            return np.array([0.0, 1.0, 0.0])
        return lateral / norm

    def _build_gate_return_bypass(
        self, center: NDArray[np.float64], normal: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Create two points that return across a gate plane outside the gate frame."""
        lateral = self._lateral_unit(normal)  # [CHANGE 7]

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

        Gate centers are never moved.  (Residual collisions of the moved point or of the spline
        between waypoints are handled by the trajectory-level check, [CHANGE 3].)
        """
        if not len(self._obstacles):
            return point

        required_clearance = self._obstacle_clearance + self._avoidance_margin
        if np.min(np.linalg.norm(self._obstacles[:, :2] - point[:2], axis=1)) >= required_clearance:
            return point

        lateral = self._lateral_unit(gate_normal)  # [CHANGE 7]
        candidates = []
        for sign in (1.0, -1.0):
            candidate = point.copy()
            candidate[:2] += sign * required_clearance * lateral[:2]
            candidates.append(candidate)

        clearances = [
            np.min(np.linalg.norm(self._obstacles[:, :2] - candidate[:2], axis=1))
            for candidate in candidates
        ]
        return candidates[int(np.argmax(clearances))]

    # ------------------------------------------------------------------------------------------
    # [CHANGE 2] / [CHANGE 3] Time allocation, feasibility scaling and collision refinement
    # ------------------------------------------------------------------------------------------
    def _allocate_waypoint_times(self, waypoints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Segment durations (not cumulative any more): distance / cruise speed, lower-bounded."""
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        return np.maximum(distances / self._cruise_speed, self._minimum_segment_time)

    def _sample_times(self, traj: MinSnapTrajectory) -> NDArray[np.float64]:
        return np.linspace(0.0, traj.t_total, max(int(traj.t_total / self._check_dt), 2))

    def _fit_with_limits(
        self, waypoints: NDArray[np.float64], v0: NDArray[np.float64]
    ) -> MinSnapTrajectory:
        """Solve the QP, then stretch all durations until speed/acceleration limits hold.

        Scaling all durations by f scales velocities by 1/f and accelerations by 1/f^2.
        """
        durations = self._allocate_waypoint_times(waypoints)
        traj = solve_min_snap(waypoints, durations, v0)
        for _ in range(6):
            ts = self._sample_times(traj)
            peak_speed = float(np.max(np.linalg.norm(traj(ts, 1), axis=1)))
            peak_accel = float(np.max(np.linalg.norm(traj(ts, 2), axis=1)))
            factor = max(peak_speed / self._max_speed, np.sqrt(peak_accel / self._max_accel))
            if factor <= 1.0:
                break
            durations = durations * min(float(factor), 1.5)
            traj = solve_min_snap(waypoints, durations, v0)
        return traj

    def _first_obstacle_violation(self, traj: MinSnapTrajectory):
        """Return (time, point, obstacle index) of the worst clearance violation, or None."""
        if not len(self._obstacles):
            return None
        ts = self._sample_times(traj)
        points = traj(ts)
        dist = np.linalg.norm(points[:, None, :2] - self._obstacles[None, :, :2], axis=2)
        i, k = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[i, k] >= self._obstacle_clearance:
            return None
        return float(ts[i]), points[i], int(k)

    def _insert_detour(
        self, waypoints: NDArray[np.float64], traj: MinSnapTrajectory, hit
    ) -> NDArray[np.float64]:
        """Insert a waypoint pushed radially away from the offending obstacle."""
        t, point, k = hit
        seg = int(np.clip(np.searchsorted(traj.knots, t, side="right") - 1, 0, len(waypoints) - 2))
        away = point[:2] - self._obstacles[k, :2]
        norm = np.linalg.norm(away)
        if norm < 1e-6:  # directly on the obstacle center: step sideways of the segment
            direction = waypoints[seg + 1] - waypoints[seg]
            away = np.array([-direction[1], direction[0]])
            norm = max(np.linalg.norm(away), 1e-6)
        detour = point.copy()
        detour[:2] = self._obstacles[k, :2] + away / norm * (
            self._obstacle_clearance + self._avoidance_margin
        )
        return np.insert(waypoints, seg + 1, detour, axis=0)

    def _optimize(
        self, waypoints: NDArray[np.float64], v0: NDArray[np.float64]
    ) -> MinSnapTrajectory:
        """QP -> limit scaling -> collision check -> detour insertion, iterated a few times."""
        traj = self._fit_with_limits(waypoints, v0)
        for _ in range(self._max_refine_iters):
            hit = self._first_obstacle_violation(traj)
            if hit is None:
                break
            waypoints = self._insert_detour(waypoints, traj, hit)
            traj = self._fit_with_limits(waypoints, v0)
        self._waypoints = waypoints
        return traj

    # ------------------------------------------------------------------------------------------
    # Online part
    # ------------------------------------------------------------------------------------------
    def _maybe_replan(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """[CHANGE 4] Replan from the current state if gates/obstacles moved noticeably."""
        if self._tick - self._last_replan_tick < int(self._replan_cooldown * self._freq):
            return
        gates = np.asarray(obs["gates_pos"], dtype=np.float64)
        quats = np.asarray(obs["gates_quat"], dtype=np.float64)
        obstacles = np.asarray(obs["obstacles_pos"], dtype=np.float64)
        if (
            gates.shape != self._planned_gates.shape
            or obstacles.shape != self._planned_obstacles.shape
        ):
            return
        shift = max(
            float(np.max(np.abs(gates - self._planned_gates))),
            float(np.max(np.abs(obstacles - self._planned_obstacles))) if len(obstacles) else 0.0,
        )
        rotated = float(np.max(np.abs(quats - self._planned_quats))) > 0.02
        if shift < self._replan_threshold and not rotated:
            return
        if int(obs["n_gates_passed"]) >= len(obs["gate_sequence"]):
            return
        self._plan(obs, replan=True)
        self._last_replan_tick = self._tick

    @staticmethod
    def _initial_yaw(obs: dict[str, NDArray[np.floating]]) -> float:
        """[CHANGE 6] Start from the real yaw so the first command does not jump."""
        try:
            return float(R.from_quat(np.asarray(obs["quat"], dtype=np.float64)).as_euler("zyx")[0])
        except (KeyError, ValueError):
            return 0.0

    def _update_yaw(self, desired_velocity: NDArray[np.float64]) -> None:
        """[CHANGE 6] Continuous, rate-limited yaw along the travel direction + yaw rate."""
        dt = 1.0 / self._freq
        horizontal_velocity = desired_velocity[:2]
        if np.linalg.norm(horizontal_velocity) > 0.05:
            target = float(np.arctan2(horizontal_velocity[1], horizontal_velocity[0]))
            error = (target - self._desired_yaw + np.pi) % (2.0 * np.pi) - np.pi  # wrap to +-pi
            step = float(np.clip(error, -self._max_yaw_rate * dt, self._max_yaw_rate * dt))
            self._desired_yaw += step  # stays continuous: no +-pi jump
            self._yaw_rate = step / dt
        else:
            self._yaw_rate = 0.0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.float32]:
        """Return the desired 16-dimensional full-state command."""
        del info
        self._maybe_replan(obs)  # [CHANGE 4] (obs was ignored before)

        t = self._t_ref  # [CHANGE 5] adaptive reference time instead of tick / freq
        desired_position = self._traj(t)
        desired_velocity = self._traj(t, 1)
        desired_acceleration = self._traj(t, 2)

        self._update_yaw(desired_velocity)  # [CHANGE 6]
        desired_quaternion = R.from_euler("z", self._desired_yaw).as_quat()
        desired_angular_velocity = np.array([0.0, 0.0, self._yaw_rate])  # [CHANGE 6] feed-forward

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
        """Advance the reference clock and stop after all gate passes (or on timeout)."""
        del action, reward, terminated, truncated, info

        self._tick += 1

        # [CHANGE 5] Slow the reference down while the drone lags behind it.  Otherwise the
        # reference keeps running away and the position error (and thus tilt) keeps growing.
        error = float(
            np.linalg.norm(np.asarray(obs["pos"], dtype=np.float64) - self._traj(self._t_ref))
        )
        span = max(self._err_max - self._err_deadband, 1e-6)
        scale = float(np.clip(1.0 - (error - self._err_deadband) / span, self._min_time_scale, 1.0))
        self._t_ref = min(self._t_ref + scale / self._freq, self._t_total)

        all_gates_passed = int(obs["n_gates_passed"]) >= len(obs["gate_sequence"])
        # [CHANGE 5] Timeout is based on real time since the last plan and tolerates slow-down.
        elapsed = (self._tick - self._plan_tick0) / self._freq
        trajectory_timed_out = elapsed >= self._timeout_factor * self._t_total + 2.0
        self._finished = all_gates_passed or trajectory_timed_out
        return self._finished

    def episode_callback(self):
        """Reset time-dependent state when an environment reuses this controller instance."""
        self._tick = 0
        self._finished = False
        # [CHANGE 4] / [CHANGE 5] / [CHANGE 6] restore the initial plan and reset new state.
        (self._traj, self._planned_gates, self._planned_quats,
         self._planned_obstacles, self._obstacles) = self._initial_plan
        self._t_total = self._traj.t_total
        self._t_ref = 0.0
        self._plan_tick0 = 0
        self._yaw_rate = 0.0
        self._last_replan_tick = -int(self._replan_cooldown * self._freq)

    def render_callback(self, sim: Sim):
        """Draw the planned path in green and the current setpoint in red."""
        setpoint = self._traj(self._t_ref).reshape(1, -1)
        trajectory = self._traj(np.linspace(0.0, self._t_total, 200))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.025)
