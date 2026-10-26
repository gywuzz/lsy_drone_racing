"""V3: one global minimum-snap trajectory through all remaining gates.

Structure (replaces the V2 state machine TO_ENTRY -> ALIGN_GATE -> THROUGH_GATE -> TO_EXIT):

1. Nodes      Start, then per gate  pre (-L n) / gate center / post (+L n); L is short (0.15 m).
              If the next gate lies behind the gate just passed, an over-the-top (or side)
              bypass is inserted.  No takeoff point, no 0.45 m staging points, no alignment phase.
2. QP         Minimum-snap (handout Sec. 4): piecewise degree-7 polynomials, C^3 continuity,
              waypoint interpolation, velocity of the gate center fixed along the gate normal,
              start from the current state, rest at the end.  The velocity does not drop to zero
              at corners (unlike PCHIP).
3. Time       Segment durations are tuned so that the sampled speed / acceleration of every
              segment sits at (not far below) the limits -> as fast as the limits allow.
4. Safety     The whole sampled trajectory is checked against poles (horizontal clearance), gate
              frames (unintended crossings), floor and ceiling.  Violations get a detour node
              and the QP is solved again.
5. Online     Replan from the current reference state if a gate / pole pose changes (sensor
              range), if a gate was missed, or if the drone is far off the reference.
              The reference clock slows down while the tracking error is large.

Tuning values can be overridden in the config under ``[controller_params]``.

STATUS (prototype, same crude point-mass simulator for all numbers, 16 random level2 seeds):
  V2.2 / V2.3 : 8/16 finished, mean 25.5 s
  V3 (this)   : 10/16 finished, mean 18.1 s;  nominal track: finished, about 12 s, no collision
Remaining failures of V3: frame hits at gate 1 / gate 4 (gate pose is corrected late: it is only
seen from 0.7 m) and leaving the arena after tight turns.  The difference between 8/16 and 10/16
is within the noise of this small test.  Not validated in the real simulator.
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
# Minimum-snap QP (handout Eq. 27-49), per segment in normalised time u = tau / T
# ----------------------------------------------------------------------------------------------
_ORDER = 7  # degree of every polynomial segment
_N = _ORDER + 1
_SNAP = 4  # minimised derivative (snap)


def _falling(i: int, order: int) -> float:
    """c(i) = i (i-1) ... (i-order+1)   (handout Eq. 37)."""
    return float(np.prod(np.arange(i, i - order, -1)))  # empty product = 1


def _derivative_row(order: int, u: float) -> NDArray[np.float64]:
    """Row r with  T^order * d^order p / dt^order (at u = tau/T)  =  r @ c."""
    row = np.zeros(_N)
    for i in range(order, _N):
        row[i] = _falling(i, order) * u ** (i - order)
    return row


def _snap_hessian(duration: float) -> NDArray[np.float64]:
    """Q_s with  int_0^T ||p''''||^2 dt = c^T Q c   (handout Eq. 28/44/45, normalised time)."""
    q = np.zeros((_N, _N))
    for i in range(_SNAP, _N):
        for j in range(_SNAP, _N):
            q[i, j] = (
                _falling(i, _SNAP)
                * _falling(j, _SNAP)
                / (i + j - 2 * _SNAP + 1)
                * duration ** (1 - 2 * _SNAP)
            )
    return q


class MinSnapTrajectory:
    """Piecewise polynomial position trajectory (coefficients in metres, normalised time)."""

    def __init__(self, coeffs: NDArray[np.float64], durations: NDArray[np.float64]):
        self.coeffs = coeffs  # (S, N, 3)
        self.durations = np.asarray(durations, dtype=np.float64)
        self.knots = np.concatenate(([0.0], np.cumsum(self.durations)))
        self.t_total = float(self.knots[-1])

    def segment_index(self, t: NDArray[np.float64]) -> NDArray[np.int64]:
        return np.clip(np.searchsorted(self.knots, t, side="right") - 1, 0, len(self.durations) - 1)

    def __call__(self, t: float | NDArray[np.float64], order: int = 0) -> NDArray[np.float64]:
        scalar = np.ndim(t) == 0
        t_arr = np.clip(np.atleast_1d(np.asarray(t, dtype=np.float64)), 0.0, self.t_total)
        seg = self.segment_index(t_arr)
        duration = self.durations[seg]
        u = (t_arr - self.knots[seg]) / duration
        out = np.zeros((len(t_arr), 3))
        for i in range(order, _N):
            out += (_falling(i, order) * u ** (i - order) / duration**order)[:, None] * (
                self.coeffs[seg, i, :]
            )
        return out[0] if scalar else out


def solve_min_snap(
    waypoints: NDArray[np.float64],
    durations: NDArray[np.float64],
    v_start: NDArray[np.float64],
    a_start: NDArray[np.float64],
    knot_velocity: dict[int, NDArray[np.float64]],
) -> MinSnapTrajectory:
    """min sum_s c_s^T Q_s c_s  s.t. interpolation, C^3 continuity, boundary and gate velocities.

    All axes share Hessian and constraint matrix: one KKT solve with three right-hand sides.
    knot_velocity maps an interior waypoint index to its prescribed velocity vector.
    """
    n_seg = len(durations)
    n_var = _N * n_seg
    zero = np.zeros(3)
    hessian = np.zeros((n_var, n_var))
    rows: list[NDArray[np.float64]] = []
    rhs: list[NDArray[np.float64]] = []

    def block(seg: int) -> slice:
        return slice(seg * _N, (seg + 1) * _N)

    def add(seg: int, order: int, u: float, value: NDArray[np.float64]) -> None:
        row = np.zeros(n_var)
        row[block(seg)] = _derivative_row(order, u)
        rows.append(row)
        rhs.append(value * durations[seg] ** order)

    for s in range(n_seg):
        hessian[block(s), block(s)] = _snap_hessian(durations[s])
        add(s, 0, 0.0, waypoints[s])
        add(s, 0, 1.0, waypoints[s + 1])
    add(0, 1, 0.0, v_start)
    add(0, 2, 0.0, a_start)
    add(n_seg - 1, 1, 1.0, zero)
    add(n_seg - 1, 2, 1.0, zero)
    for j, velocity in knot_velocity.items():
        if 0 < j < n_seg:
            add(j - 1, 1, 1.0, velocity)
    for s in range(n_seg - 1):  # C^3 continuity at inner knots
        ref = min(durations[s], durations[s + 1])
        for order in (1, 2, 3):
            row = np.zeros(n_var)
            row[block(s)] = (ref / durations[s]) ** order * _derivative_row(order, 1.0)
            row[block(s + 1)] = -((ref / durations[s + 1]) ** order) * _derivative_row(order, 0.0)
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
    return MinSnapTrajectory(sol[:n_var].reshape(n_seg, _N, 3), durations)


def _get_param(config, name: str, default):
    """Read an optional tuning value from ``config.controller_params``."""
    try:
        return type(default)(config.controller_params[name])
    except (AttributeError, KeyError, TypeError):
        return default


@dataclass
class Node:
    pos: NDArray[np.float64]
    kind: str  # start | pre | gate | post | free | finish
    normal: NDArray[np.float64] | None = None  # gate nodes: crossing direction
    gate_id: int = -1
    pass_index: int = -1  # gate nodes: index in the ordered pass sequence


@dataclass
class Plan:
    traj: MinSnapTrajectory
    nodes: list[Node]
    gate_times: dict[int, float]  # pass index -> time of the gate-center node


class V3GlobalController(Controller):
    """Global minimum-snap race controller for the full-state command interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = float(config.env.freq)

        def p(name: str, default):
            return _get_param(config, name, default)

        # Speed / acceleration limits and time allocation.  Raise step by step!
        self._cruise_speed = p("cruise_speed", 1.5)  # initial guess for open segments [m/s]
        self._gate_speed = p("gate_speed", 1.2)  # initial guess near gates [m/s]
        self._max_speed = p("max_speed", 2.5)  # [m/s]
        self._max_accel = p("max_accel", 5.0)  # norm of the commanded acceleration [m/s^2]
        self._min_segment_time = p("min_segment_time", 0.08)  # [s]
        # Speed near gates: a gate is only seen from 0.7 m, so the pose is corrected late and the
        # lateral tracking error must stay small at the crossing.
        self._gate_speed_limit = p("gate_speed_limit", 9.0)  # within gate_slow_radius [m/s]; 9.0 = off (in my tests the limits made things worse)
        self._gate_slow_radius = p("gate_slow_radius", 0.40)  # [m]
        self._unseen_gate_speed = p("unseen_gate_speed", 9.0)  # approaching an unseen gate [m/s]
        self._unseen_gate_radius = p("unseen_gate_radius", 1.2)  # [m]
        self._time_iterations = p("time_iterations", 2)
        self._check_dt = 0.02  # sampling step for limits / collision checks [s]

        # Geometry (level2.toml: frame 0.72 m, opening 0.40 m, poles 0.03 m wide / 1.52 m high).
        self._gate_straight = p("gate_straight_length", 0.2)  # optional pre/post nodes each side [m]
        self._behind_test_lead = 0.45  # distance of the virtual entry used for the "behind" test [m]
        self._finish_distance = p("finish_distance", 0.30)  # braking distance after last gate [m]
        self._frame_half = 0.36
        self._opening_core = 0.14  # allowed |lateral|, |vertical| offset at a planned crossing
        self._frame_margin = 0.06
        self._obstacle_clearance = p("obstacle_clearance", 0.20)  # horizontal [m]
        self._detour_margin = p("detour_margin", 0.08)  # [m]
        # Objects outside the 0.7 m sensor range only report their nominal pose; level2.toml
        # randomises poles by +-0.15 m and gates by +-0.15 m, so unseen objects get more room.
        self._pole_uncertainty = p("pole_uncertainty", 0.0)  # [m]
        self._gate_uncertainty = p("gate_uncertainty", 0.0)  # [m]
        self._pole_top_margin = 0.12  # fly over a pole only above top + margin [m]
        self._floor_height = p("floor_height", 0.0)  # trajectory must stay above [m]
        self._max_flight_height = p("max_flight_height", 1.85)  # pos_limit_high z = 2.0 [m]
        self._max_refine = p("max_refine", 10)
        self._stretch_retries = p("stretch_retries", 0)  # slow-down retries for residual violations
        # Arena limits (level2.toml safety_limits: x +-2.5, y +-1.5, z up to 2.0); stay inside.
        try:
            low = np.asarray(config.env.track.safety_limits.pos_limit_low, dtype=np.float64)
            high = np.asarray(config.env.track.safety_limits.pos_limit_high, dtype=np.float64)
        except (AttributeError, KeyError, TypeError):
            low, high = np.array([-2.5, -1.5, -1e-3]), np.array([2.5, 1.5, 2.0])
        margin = p("arena_margin", 0.20)
        self._arena_low = low[:2] + margin
        self._arena_high = high[:2] - margin

        # Return route when the next gate lies behind the one just passed.
        self._bypass_mode = p("bypass_mode", "over")  # "over" | "side"
        self._over_clearance = p("over_gate_clearance", 0.25)  # above the frame top [m]
        self._behind_distance = p("over_gate_behind_distance", 0.35)  # [m]
        self._side_offset = p("gate_bypass_offset", 0.62)  # [m]

        # Online behaviour.
        self._replan_threshold = p("replan_threshold", 0.03)  # [m]
        self._replan_cooldown = p("replan_cooldown", 0.30)  # [s]
        self._miss_tolerance = p("miss_tolerance", 0.35)  # gate not counted this long after [s]
        self._ref_replan_error = p("ref_replan_error", 0.06)  # replan from the measured state above [m]
        self._derail_distance = p("derail_distance", 0.60)  # replan if farther from reference [m]
        self._err_deadband = p("err_deadband", 0.10)  # no slow-down below this error [m]
        self._err_max = p("err_max", 0.50)  # strongest slow-down at this error [m]
        self._min_time_scale = p("min_time_scale", 0.30)
        self._scale_rise = 0.02  # max increase of the clock rate per step
        self._scale_fall = 0.10  # max decrease of the clock rate per step
        self._max_yaw_rate = p("max_yaw_rate", 2.0)  # [rad/s]

        self._start_speed = 0.0  # current speed at the beginning of the plan [m/s]
        self._known_gates: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        self._known_poles: dict[int, NDArray[np.float64]] = {}
        self._update_known(obs)
        self._gates_pos, self._gates_quat, self._poles = self._effective(obs)

        self._desired_yaw = self._initial_yaw(obs)
        self._yaw_rate = 0.0
        self._tick = 0
        self._finished = False
        self._t_ref = 0.0
        self._time_scale = 1.0
        self._last_replan_tick = -int(self._replan_cooldown * self._freq)
        self._n_passed = int(obs["n_gates_passed"])
        pos = np.asarray(obs["pos"], dtype=np.float64)
        self._last_ref = (pos.copy(), np.zeros(3), np.zeros(3))
        self._plan_start_tick = 0
        self.replan_count = 0
        self.failed_replans = 0
        self.last_failure = ""
        self.last_trigger = ""

        zero = np.zeros(3)
        self._plan = self._make_plan(obs, pos, zero, zero, replan=False) or self._fallback_plan(
            obs, pos, zero, zero, replan=False
        )
        self._initial_plan = (self._plan, self._gates_pos.copy(), self._gates_quat.copy(),
                              self._poles.copy())

    # ------------------------------------------------------------------------------------------
    # Objects: sensor-range handling (unvisited objects report nominal poses)
    # ------------------------------------------------------------------------------------------
    def _update_known(self, obs: dict[str, NDArray[np.floating]]) -> None:
        # Cache a pose only the first time an object is seen: afterwards the observation may fall
        # back to the nominal pose when the object leaves the sensor range, which must not
        # overwrite the true pose (and must not trigger replanning).
        for gate_id in np.flatnonzero(np.asarray(obs["gates_visited"], dtype=bool)):
            gate_id = int(gate_id)
            if gate_id in self._known_gates:
                continue
            self._known_gates[gate_id] = (
                np.asarray(obs["gates_pos"][gate_id], dtype=np.float64).copy(),
                np.asarray(obs["gates_quat"][gate_id], dtype=np.float64).copy(),
            )
        for pole_id in np.flatnonzero(np.asarray(obs["obstacles_visited"], dtype=bool)):
            pole_id = int(pole_id)
            if pole_id in self._known_poles:
                continue
            self._known_poles[pole_id] = np.asarray(
                obs["obstacles_pos"][pole_id], dtype=np.float64
            ).copy()

    def _effective(self, obs: dict[str, NDArray[np.floating]]):
        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64).copy()
        gates_quat = np.asarray(obs["gates_quat"], dtype=np.float64).copy()
        poles = np.asarray(obs["obstacles_pos"], dtype=np.float64).copy()
        for gate_id, (position, quat) in self._known_gates.items():
            gates_pos[gate_id] = position
            gates_quat[gate_id] = quat
        for pole_id, position in self._known_poles.items():
            poles[pole_id] = position
        return gates_pos, gates_quat, poles

    # ------------------------------------------------------------------------------------------
    # 1. Nodes
    # ------------------------------------------------------------------------------------------
    def _gate_normal(self, gate_id: int, direction: float) -> NDArray[np.float64]:
        return R.from_quat(self._gates_quat[gate_id]).apply([1.0, 0.0, 0.0]) * direction

    def _build_nodes(
        self, obs: dict[str, NDArray[np.floating]], start: NDArray[np.float64], replan: bool
    ) -> list[Node]:
        first = int(obs["n_gates_passed"])
        sequence = np.asarray(obs["gate_sequence"], dtype=np.int64)
        directions = np.asarray(obs["gate_sequence_direction"], dtype=np.float64)
        length = self._gate_straight

        gates = []
        for pass_index in range(first, len(sequence)):
            gate_id = int(sequence[pass_index])
            normal = self._gate_normal(gate_id, float(directions[pass_index]))
            gates.append((pass_index, gate_id, self._gates_pos[gate_id].copy(), normal))

        nodes = [Node(start.copy(), "start")]
        if not gates:
            return nodes

        for k, (pass_index, gate_id, center, normal) in enumerate(gates):
            pre = center - length * normal if length > 1e-6 else None
            post = center + length * normal if length > 1e-6 else None
            if k == 0 and replan:
                side = float(np.dot(start - center, normal))
                if side > 0.05:  # already beyond the gate plane but not counted: re-enter
                    nodes += self._return_bypass(start, center, normal)
                elif pre is not None and side > -length and float(np.dot(pre - start, normal)) < 0.0:
                    pre = None  # already between pre point and the gate center
            if pre is not None:
                nodes.append(Node(pre, "pre", gate_id=gate_id))
            nodes.append(Node(center, "gate", normal, gate_id, pass_index))
            if post is not None:
                nodes.append(Node(post, "post", gate_id=gate_id))

            if k + 1 < len(gates):
                _, _, next_center, next_normal = gates[k + 1]
                next_entry = next_center - self._behind_test_lead * next_normal
                if float(np.dot(next_entry - center, normal)) < 0.0:
                    nodes += self._return_bypass(post if post is not None else center, center, normal)

        last_normal = gates[-1][3]
        nodes.append(Node(nodes[-1].pos + self._finish_distance * last_normal, "finish"))
        return nodes

    def _return_bypass(
        self, start: NDArray[np.float64], center: NDArray[np.float64], normal: NDArray[np.float64]
    ) -> list[Node]:
        """Nodes that bring the drone back to the entry side of a gate it has just left."""
        normal_h = np.array([normal[0], normal[1], 0.0])
        normal_h /= max(float(np.linalg.norm(normal_h)), 1e-9)

        over_z = center[2] + self._frame_half + self._over_clearance
        if self._bypass_mode == "over" and over_z <= self._max_flight_height:
            over_z = max(over_z, float(start[2]))
            lateral = np.array([-normal_h[1], normal_h[0], 0.0])
            need = self._obstacle_clearance + self._detour_margin
            start_behind = float(np.dot(start - center, normal_h))
            chosen = None
            # Climb point: as close behind the gate as possible, but clear of poles.  Try the
            # centered line first, then shift sideways (the pole next to the exit of gate 3).
            for behind in (self._behind_distance, 0.50, 0.65):
                behind = max(behind, start_behind)
                options = []
                for offset in (0.0, 0.30, -0.30):
                    climb = center + behind * normal_h + offset * lateral
                    clearance = (
                        float(np.min(np.linalg.norm(self._poles[:, :2] - climb[:2], axis=1)))
                        if len(self._poles)
                        else 9.0
                    )
                    options.append((offset, climb, clearance))
                ok = [o for o in options if o[2] >= need]
                if ok:
                    chosen = ok[0] if ok[0][0] == 0.0 else max(ok, key=lambda o: o[2])
                    break
                if chosen is None or max(o[2] for o in options) > chosen[2]:
                    chosen = max(options, key=lambda o: o[2])
            climb = chosen[1].copy()
            climb[2] = over_z
            above = center.copy()
            above[2] = over_z
            front = center - self._behind_distance * normal_h
            front[2] = over_z
            return [Node(climb, "free"), Node(above, "free"), Node(front, "free")]

        lateral = np.array([-normal_h[1], normal_h[0], 0.0])
        candidates = []
        for sign in (1.0, -1.0):
            offset = sign * self._side_offset * lateral
            candidates.append(
                (center + 0.45 * normal_h + offset, center - 0.45 * normal_h + offset)
            )
        clearance = [
            min(
                float(np.min(np.linalg.norm(self._poles[:, :2] - pair[0][:2], axis=1))),
                float(np.min(np.linalg.norm(self._poles[:, :2] - pair[1][:2], axis=1))),
            )
            for pair in candidates
        ]
        a, b = candidates[int(np.argmax(clearance))]
        return [Node(a, "free"), Node(b, "free")]

    # ------------------------------------------------------------------------------------------
    # 2./3. QP and time allocation
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _positions(nodes: list[Node]) -> NDArray[np.float64]:
        return np.asarray([node.pos for node in nodes], dtype=np.float64)

    def _initial_durations(self, nodes: list[Node]) -> NDArray[np.float64]:
        dist = np.linalg.norm(np.diff(self._positions(nodes), axis=0), axis=1)
        speed = np.full(len(dist), self._cruise_speed)
        for i in range(len(dist)):
            if nodes[i].kind == "gate" or nodes[i + 1].kind == "gate":
                speed[i] = self._gate_speed
        speed[0] = max(min(speed[0], 0.6 * self._cruise_speed), self._start_speed)  # from rest / flight
        speed[-1] = min(speed[-1], 0.6 * self._cruise_speed)  # end at rest
        return np.maximum(dist / speed, self._min_segment_time)

    def _solve(
        self,
        nodes: list[Node],
        durations: NDArray[np.float64],
        state: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> MinSnapTrajectory:
        pos = self._positions(nodes)
        knot_velocity: dict[int, NDArray[np.float64]] = {}
        for j, node in enumerate(nodes):
            if node.kind == "gate" and 0 < j < len(nodes) - 1 and node.normal is not None:
                length = float(
                    np.linalg.norm(pos[j] - pos[j - 1]) + np.linalg.norm(pos[j + 1] - pos[j])
                )
                # Gate speed follows from the local durations (self-consistent, no wiggles).
                knot_velocity[j] = node.normal * (length / (durations[j - 1] + durations[j]))
        return solve_min_snap(pos, durations, state[0], state[1], knot_velocity)

    def _sample_times(self, traj: MinSnapTrajectory) -> NDArray[np.float64]:
        return np.linspace(0.0, traj.t_total, max(int(traj.t_total / self._check_dt) + 1, 2))

    def _limit_factors(self, traj: MinSnapTrajectory) -> NDArray[np.float64]:
        """Per segment: >1 means speed or acceleration limit exceeded."""
        ts = self._sample_times(traj)
        seg = traj.segment_index(ts)
        speed = np.linalg.norm(traj(ts, 1), axis=1)
        accel = np.linalg.norm(traj(ts, 2), axis=1)
        peak_speed = np.zeros(len(traj.durations))
        peak_accel = np.zeros(len(traj.durations))
        np.maximum.at(peak_speed, seg, speed)
        np.maximum.at(peak_accel, seg, accel)
        return np.maximum(np.sqrt(peak_accel / self._max_accel), peak_speed / self._max_speed)

    def _is_sane(self, traj: MinSnapTrajectory) -> bool:
        """Reject exploded / ill-conditioned solutions (huge or non-finite coefficients)."""
        if not np.all(np.isfinite(traj.coeffs)) or not (0.0 < traj.t_total < 90.0):
            return False
        ts = np.linspace(0.0, traj.t_total, 300)
        return bool(
            np.abs(traj(ts)).max() < 20.0
            and np.linalg.norm(traj(ts, 1), axis=1).max() < 3.0 * self._max_speed
        )

    def _profile_durations(
        self, traj: MinSnapTrajectory, n_segments: int, start_speed: float = 0.0
    ) -> NDArray[np.float64]:
        """Time-optimal speed along the current path (forward/backward pass), per segment.

        v <= v_max, centripetal v^2*kappa and tangential acceleration share the budget a_max;
        start and end at rest.  Returns the segment durations that realise this profile.
        """
        ts = np.linspace(0.0, traj.t_total, max(int(traj.t_total / 0.01), 200) + 1)
        pts, vel, acc = traj(ts), traj(ts, 1), traj(ts, 2)
        speed = np.linalg.norm(vel, axis=1)
        kappa = np.linalg.norm(np.cross(vel, acc), axis=1) / np.maximum(speed, 1e-3) ** 3
        kappa = np.minimum(kappa, 20.0)
        ds = np.maximum(np.linalg.norm(np.diff(pts, axis=0), axis=1), 1e-6)
        budget = 0.9 * self._max_accel  # margin for the deviation of the re-solved QP
        v_lim = np.minimum(self._max_speed, np.sqrt(budget * 0.95 / np.maximum(kappa, 1e-6)))
        for g in range(len(self._gates_pos)):  # slower near gates, slowest near unseen ones
            known = g in self._known_gates
            radius = self._gate_slow_radius if known else self._unseen_gate_radius
            limit = self._gate_speed_limit if known else self._unseen_gate_speed
            near = np.linalg.norm(pts - self._gates_pos[g], axis=1) < radius
            v_lim = np.where(near, np.minimum(v_lim, limit), v_lim)

        def tangential(v: float, k: float) -> float:
            return float(np.sqrt(max(budget**2 - (v * v * k) ** 2, 0.0)))

        v = v_lim.copy()
        v[0] = min(start_speed, v_lim[0])  # standing start or current speed (replanning)
        for i in range(len(ds)):  # forward: acceleration limit
            v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2.0 * tangential(v[i], kappa[i]) * ds[i]))
        v[-1] = 0.0
        for i in range(len(ds) - 1, -1, -1):  # backward: braking limit
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * tangential(v[i + 1], kappa[i + 1]) * ds[i]))
        dt = 2.0 * ds / np.maximum(v[:-1] + v[1:], 1e-3)
        seg = traj.segment_index(0.5 * (ts[:-1] + ts[1:]))
        durations = np.zeros(n_segments)
        np.add.at(durations, seg, dt)
        return np.maximum(durations, self._min_segment_time)

    def _optimize_times(
        self,
        nodes: list[Node],
        durations: NDArray[np.float64],
        state: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], MinSnapTrajectory]:
        """Iterate: path -> time-optimal speed profile -> new durations -> QP; then make feasible.

        Every iterate is checked for sanity; on a bad solution the last good one is kept.
        """
        base = durations.copy()
        dur = base.copy()
        traj = self._solve(nodes, dur, state)
        start_speed = float(np.linalg.norm(state[0]))
        for _ in range(self._time_iterations):
            target = np.clip(
                self._profile_durations(traj, len(dur), start_speed), 0.5 * base, 3.0 * base
            )
            new_dur = dur**0.5 * target**0.5  # damped update (geometric mean)
            new_traj = self._solve(nodes, new_dur, state)
            if not self._is_sane(new_traj):
                break
            dur, traj = new_dur, new_traj
        for _ in range(10):  # final guarantee for the sampled limits
            worst = float(self._limit_factors(traj).max())
            if worst <= 1.02:
                break
            new_dur = dur * min(worst, 1.5)
            new_traj = self._solve(nodes, new_dur, state)
            if not self._is_sane(new_traj):
                break
            dur, traj = new_dur, new_traj
        return dur, traj

    # ------------------------------------------------------------------------------------------
    # 4. Safety checks on the sampled trajectory
    # ------------------------------------------------------------------------------------------
    def _find_violation(self, traj: MinSnapTrajectory, nodes: list[Node]):
        """Worst violation as (depth, time, point, kind, index), or None."""
        ts = self._sample_times(traj)
        pts = traj(ts)
        seg = np.clip(traj.segment_index(ts), 0, len(nodes) - 2)
        inner = np.array(
            [
                nodes[s].kind in ("pre", "gate") and nodes[s + 1].kind in ("gate", "post")
                for s in range(len(nodes) - 1)
            ]
        )  # straight gate crossing: never re-routed
        found: list[tuple[float, float, NDArray[np.float64], str, int]] = []

        def record(depth_arr: NDArray[np.float64], kind: str, index: int) -> None:
            depth_arr = np.where(inner[seg], 0.0, depth_arr)
            i = int(np.argmax(depth_arr))
            if depth_arr[i] > 0.0:
                found.append((float(depth_arr[i]), float(ts[i]), pts[i].copy(), kind, index))

        if len(self._poles):
            top = float(np.max(self._poles[:, 2])) + self._pole_top_margin
            dist = np.linalg.norm(pts[:, None, :2] - self._poles[None, :, :2], axis=2)
            for k in range(len(self._poles)):
                extra = 0.0 if k in self._known_poles else self._pole_uncertainty
                depth = self._obstacle_clearance + extra - dist[:, k]
                record(np.where(pts[:, 2] < top, depth, 0.0), "pole", k)

        planned = {node.gate_id for node in nodes if node.kind == "gate"}
        for g in range(len(self._gates_pos)):
            outer = self._frame_half + self._frame_margin
            if g not in self._known_gates:
                outer += self._gate_uncertainty
            rel = R.from_quat(self._gates_quat[g]).inv().apply(pts - self._gates_pos[g])
            near_plane = np.abs(rel[:, 0]) < 0.05
            lateral, vertical = np.abs(rel[:, 1]), np.abs(rel[:, 2])
            in_frame = near_plane & (lateral < outer) & (vertical < outer)
            core = (lateral < self._opening_core) & (vertical < self._opening_core)
            planned_pass = (
                core & (np.linalg.norm(rel, axis=1) < 0.30) if g in planned else np.zeros_like(core)
            )
            depth = outer - np.maximum(lateral, vertical)
            record(np.where(in_frame & ~planned_pass, depth, 0.0), "gate", g)

        outside = np.maximum(self._arena_low - pts[:, :2], pts[:, :2] - self._arena_high).max(axis=1)
        record(np.where(outside > 0.0, outside, 0.0), "arena", 0)

        z = pts[:, 2]
        record(np.where(z < self._floor_height, self._floor_height - z, 0.0), "floor", 0)
        record(np.where(z > self._max_flight_height, z - self._max_flight_height, 0.0),
               "ceiling", 0)

        if not found:
            return None
        return max(found, key=lambda item: item[0])

    def _insert_detour(self, nodes: list[Node], traj: MinSnapTrajectory, hit) -> list[Node]:
        _, t, point, kind, index = hit
        seg = int(np.clip(traj.segment_index(np.array([t]))[0], 0, len(nodes) - 2))
        detour = point.copy()
        if kind == "pole":
            away = point[:2] - self._poles[index, :2]
            norm = float(np.linalg.norm(away))
            if norm < 1e-6:
                direction = nodes[seg + 1].pos - nodes[seg].pos
                away = np.array([-direction[1], direction[0]])
                norm = max(float(np.linalg.norm(away)), 1e-6)
            extra = 0.0 if index in self._known_poles else self._pole_uncertainty
            detour[:2] = self._poles[index, :2] + away / norm * (
                self._obstacle_clearance + extra + self._detour_margin
            )
        elif kind == "gate":
            rot = R.from_quat(self._gates_quat[index])
            rel = rot.inv().apply(point - self._gates_pos[index])
            target = self._frame_half + self._frame_margin + 0.12
            if index not in self._known_gates:
                target += self._gate_uncertainty
            options = []
            for new_lateral, new_vertical in (
                (rel[1], target), (target, rel[2]), (-target, rel[2])
            ):
                candidate = self._gates_pos[index] + rot.apply([rel[0], new_lateral, new_vertical])
                if self._floor_height < candidate[2] <= self._max_flight_height:
                    options.append(candidate)
            if options:
                detour = min(options, key=lambda c: float(np.linalg.norm(c - point)))
        elif kind == "arena":
            detour[:2] = np.clip(point[:2], self._arena_low + 0.10, self._arena_high - 0.10)
        elif kind == "floor":
            detour[2] = self._floor_height + 0.10
        elif kind == "ceiling":
            detour[2] = self._max_flight_height - 0.10
        if (
            float(np.linalg.norm(detour - nodes[seg].pos)) < 0.08
            or float(np.linalg.norm(detour - nodes[seg + 1].pos)) < 0.08
        ):
            return nodes  # would create a degenerate tiny segment: leave the plan as it is
        new_nodes = list(nodes)
        new_nodes.insert(seg + 1, Node(detour, "free"))
        return new_nodes

    # ------------------------------------------------------------------------------------------
    # Planning entry point
    # ------------------------------------------------------------------------------------------
    def _make_plan(
        self,
        obs: dict[str, NDArray[np.floating]],
        pos: NDArray[np.float64],
        vel: NDArray[np.float64],
        acc: NDArray[np.float64],
        replan: bool,
    ) -> Plan:
        nodes = self._build_nodes(obs, pos, replan)
        if len(nodes) < 2:
            nodes.append(Node(pos + np.array([0.0, 0.0, 1e-3]), "finish"))
        state = (vel, acc)
        self._start_speed = float(np.linalg.norm(vel))
        if replan and self._start_speed > 0.3 and len(nodes) >= 2:
            # Keep flying along the current velocity for a moment before turning: otherwise the
            # QP has to invent a violent loop when the next gate is not straight ahead.
            lead = pos + vel / self._start_speed * min(0.5, 0.25 * self._start_speed)
            if float(np.linalg.norm(lead - nodes[1].pos)) > 0.15:
                nodes.insert(1, Node(lead, "free"))

        durations = self._initial_durations(nodes)
        traj = self._solve(nodes, durations, state)
        for _ in range(self._max_refine):  # cheap pass with initial durations
            if not self._is_sane(traj):
                break
            hit = self._find_violation(traj, nodes)
            if hit is None:
                break
            new_nodes = self._insert_detour(nodes, traj, hit)
            if len(new_nodes) == len(nodes):
                break
            new_durations = self._initial_durations(new_nodes)
            new_traj = self._solve(new_nodes, new_durations, state)
            if not self._is_sane(new_traj):
                break  # this detour makes the QP explode: keep the last sane plan
            nodes, durations, traj = new_nodes, new_durations, new_traj

        durations, traj = self._optimize_times(nodes, self._initial_durations(nodes), state)
        for _ in range(3):  # re-check after the time optimisation changed the geometry
            if not self._is_sane(traj):
                break
            hit = self._find_violation(traj, nodes)
            if hit is None:
                break
            new_nodes = self._insert_detour(nodes, traj, hit)
            if len(new_nodes) == len(nodes):
                break
            new_durations, new_traj = self._optimize_times(
                new_nodes, self._initial_durations(new_nodes), state
            )
            if not self._is_sane(new_traj):
                break
            nodes, durations, traj = new_nodes, new_durations, new_traj

        # Residual violations (a detour could not be inserted or made the QP explode): slower
        # flight means lower gate speeds and tighter arcs, so stretch all durations and re-check.
        for _ in range(self._stretch_retries):
            if not self._is_sane(traj) or self._find_violation(traj, nodes) is None:
                break
            stretched = durations * 1.25
            new_traj = self._solve(nodes, stretched, state)
            if not self._is_sane(new_traj):
                break
            durations, traj = stretched, new_traj

        if not self._is_sane(traj):
            self.last_failure = "insane trajectory"
            return None
        if float(self._limit_factors(traj).max()) > 1.15:
            self.last_failure = "limits exceeded"
            return None
        gate_times = {
            node.pass_index: float(traj.knots[j])
            for j, node in enumerate(nodes)
            if node.kind == "gate"
        }
        return Plan(traj, nodes, gate_times)

    def _fallback_plan(
        self,
        obs: dict[str, NDArray[np.floating]],
        pos: NDArray[np.float64],
        vel: NDArray[np.float64],
        acc: NDArray[np.float64],
        replan: bool,
    ) -> Plan:
        """Conservative plan (no detours, uniformly stretched) if the optimised one failed."""
        nodes = self._build_nodes(obs, pos, replan)
        if len(nodes) < 2:
            nodes.append(Node(pos + np.array([0.0, 0.0, 1e-3]), "finish"))
        self._start_speed = float(np.linalg.norm(vel))
        dur = self._initial_durations(nodes)
        for _ in range(25):
            traj = self._solve(nodes, dur, (vel, acc))
            if self._is_sane(traj) and float(self._limit_factors(traj).max()) <= 1.05:
                break
            dur = dur * 1.25
        gate_times = {
            node.pass_index: float(traj.knots[j])
            for j, node in enumerate(nodes)
            if node.kind == "gate"
        }
        return Plan(traj, nodes, gate_times)

    # ------------------------------------------------------------------------------------------
    # 5. Online part
    # ------------------------------------------------------------------------------------------
    def _replan(self, obs: dict[str, NDArray[np.floating]], from_reference: bool) -> None:
        measured = np.asarray(obs["pos"], dtype=np.float64)
        if from_reference:
            pos, vel, acc = (x.copy() for x in self._last_ref)
        else:
            velocity = np.asarray(obs["vel"], dtype=np.float64) if "vel" in obs else np.zeros(3)
            pos, vel, acc = measured, velocity, np.zeros(3)
        self._gates_pos, self._gates_quat, self._poles = self._effective(obs)
        plan = self._make_plan(obs, pos, vel, acc, replan=True)
        if plan is None and float(np.linalg.norm(acc)) > 0.0:
            # A pinned start acceleration can make a turn infeasible: accept a small jump.
            plan = self._make_plan(obs, pos, vel, np.zeros(3), replan=True)
        self._last_replan_tick = self._tick
        if plan is None:  # optimisation failed: keep flying the old plan, try again later
            self.failed_replans += 1
            return
        self._plan = plan
        self._t_ref = 0.0
        self._plan_start_tick = self._tick
        self.replan_count += 1

    def _maybe_replan(self, obs: dict[str, NDArray[np.floating]], tracking_error: float) -> None:
        if self._tick - self._last_replan_tick < int(self._replan_cooldown * self._freq):
            return
        gates_pos, gates_quat, poles = self._effective(obs)
        shift = max(
            float(np.max(np.abs(gates_pos - self._gates_pos))),
            float(np.max(np.abs(poles - self._poles))) if len(poles) else 0.0,
        )
        rotated = float(np.max(np.abs(gates_quat - self._gates_quat))) > 0.02
        if shift >= self._replan_threshold or rotated:
            self.last_trigger = f"objects moved (shift {shift:.2f}, rotated {rotated})"
            self._replan(obs, from_reference=tracking_error < self._ref_replan_error)
            return

        pass_index = int(obs["n_gates_passed"])
        planned_time = self._plan.gate_times.get(pass_index)
        if planned_time is not None and self._t_ref > planned_time + self._miss_tolerance:
            self.last_trigger = f"gate {pass_index} not counted"
            self._replan(obs, from_reference=False)  # gate not counted: try again
            return
        if tracking_error > self._derail_distance:
            self.last_trigger = f"derailed ({tracking_error:.2f} m)"
            self._replan(obs, from_reference=False)

    @staticmethod
    def _initial_yaw(obs: dict[str, NDArray[np.floating]]) -> float:
        try:
            return float(R.from_quat(np.asarray(obs["quat"], dtype=np.float64)).as_euler("zyx")[0])
        except (KeyError, ValueError):
            return 0.0

    def _update_yaw(self, velocity: NDArray[np.float64]) -> None:
        """Continuous, rate-limited yaw along the travel direction plus yaw-rate feed-forward."""
        dt = 1.0 / self._freq
        if np.linalg.norm(velocity[:2]) > 0.05:
            target = float(np.arctan2(velocity[1], velocity[0]))
            error = (target - self._desired_yaw + np.pi) % (2.0 * np.pi) - np.pi
            step = float(np.clip(error, -self._max_yaw_rate * dt, self._max_yaw_rate * dt))
            self._desired_yaw += step
            self._yaw_rate = step / dt
        else:
            self._yaw_rate = 0.0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.float32]:
        """Return the desired 16-dimensional full-state command."""
        del info
        self._update_known(obs)
        position = np.asarray(obs["pos"], dtype=np.float64)
        tracking_error = float(np.linalg.norm(position - self._last_ref[0]))
        self._maybe_replan(obs, tracking_error)

        # Reference clock: runs in real time, slows down while the drone lags behind.
        span = max(self._err_max - self._err_deadband, 1e-6)
        target = float(
            np.clip(1.0 - (tracking_error - self._err_deadband) / span, self._min_time_scale, 1.0)
        )
        scale = float(
            np.clip(
                target, self._time_scale - self._scale_fall, self._time_scale + self._scale_rise
            )
        )
        scale_rate = (scale - self._time_scale) * self._freq
        self._time_scale = scale
        traj = self._plan.traj
        self._t_ref = min(self._t_ref + scale / self._freq, traj.t_total)

        if self._t_ref >= traj.t_total - 1e-6:  # hold the end point
            ref_pos = traj(traj.t_total)
            ref_vel = np.zeros(3)
            ref_acc = np.zeros(3)
        else:
            base_vel = traj(self._t_ref, 1)
            ref_pos = traj(self._t_ref)
            ref_vel = base_vel * scale
            ref_acc = traj(self._t_ref, 2) * scale**2 + base_vel * scale_rate
        self._last_ref = (ref_pos.copy(), ref_vel.copy(), ref_acc.copy())

        self._update_yaw(ref_vel)
        quat = R.from_euler("z", self._desired_yaw).as_quat()
        return np.concatenate(
            (ref_pos, ref_vel, ref_acc, quat, np.array([0.0, 0.0, self._yaw_rate])),
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
        """The environment's gate counter is the only source of race progress."""
        del action, reward, terminated, truncated, info
        self._tick += 1
        self._n_passed = int(obs["n_gates_passed"])
        self._finished = self._n_passed >= len(obs["gate_sequence"])
        return self._finished

    def episode_callback(self):
        """Restore the initial plan when the environment reuses this instance."""
        plan, gates_pos, gates_quat, poles = self._initial_plan
        self._plan = plan
        self._gates_pos, self._gates_quat, self._poles = gates_pos, gates_quat, poles
        self._tick = 0
        self._finished = False
        self._t_ref = 0.0
        self._time_scale = 1.0
        self._yaw_rate = 0.0
        self._plan_start_tick = 0
        self._last_replan_tick = -int(self._replan_cooldown * self._freq)

    def render_callback(self, sim: Sim):
        """Planned path in green, current setpoint in red."""
        traj = self._plan.traj
        draw_line(sim, traj(np.linspace(0.0, traj.t_total, 200)), rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, traj(self._t_ref).reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.025)
