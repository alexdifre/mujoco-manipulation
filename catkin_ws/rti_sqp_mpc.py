#!/usr/bin/env python3
"""
Nonlinear MPC model and trajectory utilities for the acados RTI solver.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def rotation_axis_jacobian(rot, Jr, axis_index=2):
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    axis = rot[:, axis_index]
    return -skew(axis) @ Jr


@dataclass
class Trajectory:
    x: np.ndarray
    u: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64)
        self.u = np.asarray(self.u, dtype=np.float64)
        if self.x.ndim != 2 or self.u.ndim != 2:
            raise ValueError("x and u must be rank-2 arrays")
        if self.x.shape[0] != self.u.shape[0] + 1:
            raise ValueError("x must contain horizon+1 states")

    @property
    def horizon(self):
        return self.u.shape[0]

    @property
    def nx(self):
        return self.x.shape[1]

    @property
    def nu(self):
        return self.u.shape[1]

    @property
    def dim(self):
        return self.x.size + self.u.size

    def copy(self):
        return Trajectory(self.x.copy(), self.u.copy())

    def stack(self):
        return np.concatenate([self.x.reshape(-1), self.u.reshape(-1)])

    @classmethod
    def from_vector(cls, y, horizon, nx, nu):
        y = np.asarray(y, dtype=np.float64)
        x_count = (horizon + 1) * nx
        expected = x_count + horizon * nu
        if y.shape != (expected,):
            raise ValueError(f"expected vector shape {(expected,)}, got {y.shape}")
        x = y[:x_count].reshape(horizon + 1, nx)
        u = y[x_count:].reshape(horizon, nu)
        return cls(x, u)


class ProblemModel:
    horizon: int
    nx: int
    nu: int

    def make_initial_trajectory(self, measured_state):
        raise NotImplementedError

    def shift_trajectory(self, trajectory, measured_state):
        raise NotImplementedError

    def cost(self, y):
        raise NotImplementedError

    def residuals(self, y):
        return None

    def equality_constraints(self, y):
        raise NotImplementedError

    def inequality_constraints(self, y):
        return np.zeros(0)

    def delta_bounds(self, y):
        return (np.full_like(y, -np.inf), np.full_like(y, np.inf))


class ArmNMPCProblem(ProblemModel):
    """Nonlinear arm NMPC problem using joint-space ArmDynamics."""

    def __init__(self, arm_dynamics, horizon, p_refs,
                 Qp=None, Qpv=None, Qq=None, Qv=None, R=None,
                 Qf=None, Qaxis=None, Qaxisf=None, Qqf=None, Qvf=None,
                 Rd=None,
                 q_nominal=None, q_terminal=None, previous_tau=None,
                 obstacles=None, safety_margin=0.03,
                 collision_model=None, box_active_mask=None,
                 box_contact_allowed_mask=None,
                 terminal_axis=None, terminal_axis_index=2,
                 delta_q_max=None, delta_dq_max=None, delta_tau_max=None):
        self.arm = arm_dynamics
        self.horizon = int(horizon)
        self.n = int(self.arm.n)
        self.nx = 2 * self.n
        self.nu = self.n
        self.Qp = np.asarray(Qp if Qp is not None else [80.0, 80.0, 100.0])
        self.Qpv = np.asarray(Qpv if Qpv is not None else [0.0, 0.0, 0.0])
        self.Qq = np.asarray(Qq if Qq is not None else [2.0] * 6)
        self.Qv = np.asarray(Qv if Qv is not None else [0.10] * 6)
        self.R = np.asarray(R if R is not None else [1e-3] * 6)
        self.Qf = np.asarray(Qf if Qf is not None else [240.0, 240.0, 300.0])
        self.Qaxis = np.asarray(
            Qaxis if Qaxis is not None else [0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.Qaxisf = np.asarray(
            Qaxisf if Qaxisf is not None else [0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.Qqf = np.asarray(Qqf if Qqf is not None else [8.0] * 6)
        self.Qvf = np.asarray(Qvf if Qvf is not None else [0.25] * 6)
        self.Rd = np.asarray(Rd if Rd is not None else [1e-2] * 6)
        self.q_nominal = np.asarray(
            q_nominal if q_nominal is not None else np.zeros(self.n),
            dtype=np.float64,
        )
        self.q_terminal = np.asarray(
            self.q_nominal if q_terminal is None else q_terminal,
            dtype=np.float64,
        )
        self.terminal_axis = np.asarray(
            [0.0, 0.0, -1.0] if terminal_axis is None else terminal_axis,
            dtype=np.float64,
        )
        self.terminal_axis_index = int(terminal_axis_index)
        self.obstacles = list(obstacles or [])
        self.safety_margin = float(safety_margin)
        self.collision_model = collision_model
        self.delta_q_max = np.asarray(
            delta_q_max if delta_q_max is not None else [0.12] * 6,
            dtype=np.float64,
        )
        self.delta_dq_max = np.asarray(
            delta_dq_max if delta_dq_max is not None else [0.6] * 6,
            dtype=np.float64,
        )
        self.delta_tau_max = np.asarray(
            delta_tau_max if delta_tau_max is not None else [40.0] * 6,
            dtype=np.float64,
        )
        self.set_box_active_mask(box_active_mask)
        self.set_box_contact_allowed_mask(box_contact_allowed_mask)
        self.set_previous_tau(
            np.zeros(self.nu) if previous_tau is None else previous_tau)
        self.set_reference(p_refs)

    def set_box_active_mask(self, box_active_mask=None):
        if box_active_mask is None:
            self.box_active_mask = np.ones(self.horizon + 1, dtype=bool)
            return
        mask = np.asarray(box_active_mask, dtype=bool)
        if mask.shape != (self.horizon + 1,):
            raise ValueError(
                f"box_active_mask must have shape {(self.horizon + 1,)}")
        self.box_active_mask = mask

    def set_box_contact_allowed_mask(self, box_contact_allowed_mask=None):
        if box_contact_allowed_mask is None:
            self.box_contact_allowed_mask = np.zeros(
                self.horizon + 1, dtype=bool)
            return
        mask = np.asarray(box_contact_allowed_mask, dtype=bool)
        if mask.shape != (self.horizon + 1,):
            raise ValueError(
                "box_contact_allowed_mask must have shape "
                f"{(self.horizon + 1,)}")
        self.box_contact_allowed_mask = mask

    def set_reference(self, p_refs, v_refs=None):
        refs = np.asarray(p_refs, dtype=np.float64)
        if refs.shape != (self.horizon + 1, 3):
            raise ValueError(f"p_refs must have shape {(self.horizon + 1, 3)}")
        self.p_refs = refs
        if v_refs is None:
            self.v_refs = np.zeros_like(refs)
            return
        velocities = np.asarray(v_refs, dtype=np.float64)
        if velocities.shape != (self.horizon + 1, 3):
            raise ValueError(
                f"v_refs must have shape {(self.horizon + 1, 3)}")
        self.v_refs = velocities

    def set_previous_tau(self, previous_tau):
        tau = np.asarray(previous_tau, dtype=np.float64)
        if tau.shape != (self.nu,):
            raise ValueError(f"previous_tau must have shape {(self.nu,)}")
        self.previous_tau = tau.copy()

    def _traj(self, y):
        return Trajectory.from_vector(y, self.horizon, self.nx, self.nu)

    def make_initial_trajectory(self, measured_state):
        x = np.empty((self.horizon + 1, self.nx), dtype=np.float64)
        u = np.zeros((self.horizon, self.nu), dtype=np.float64)
        x[0] = np.asarray(measured_state, dtype=np.float64)
        for k in range(self.horizon):
            u[k] = self.arm._clip_tau(self.arm.bias_for_state(x[k]))
            x[k + 1] = self.arm.step_dynamics(x[k], u[k])
        return Trajectory(x, u)

    def shift_trajectory(self, trajectory, measured_state):
        shifted_x = np.empty_like(trajectory.x)
        shifted_u = np.empty_like(trajectory.u)
        shifted_x[:-1] = trajectory.x[1:]
        shifted_u[:-1] = trajectory.u[1:]
        shifted_x[0] = np.asarray(measured_state, dtype=np.float64)
        shifted_u[-1] = self.arm._clip_tau(
            self.arm.bias_for_state(shifted_x[-2]))
        shifted_x[-1] = self.arm.step_dynamics(shifted_x[-2],
                                               shifted_u[-1])
        return Trajectory(shifted_x, shifted_u)

    def _ee_pos(self, x):
        pos, _ = self.arm.forward_kinematics(x[:6])
        return pos

    def _ee_pos_jacobian(self, x):
        pos, _, Jp, _ = self.arm.forward_kinematics_jacobian(x[:6])
        return pos, Jp

    def _ee_velocity(self, x):
        _, Jp = self._ee_pos_jacobian(x)
        return Jp @ x[self.n:]

    def residuals(self, y):
        traj = self._traj(y)
        residuals = []
        sqrt_Qp = np.sqrt(self.Qp)
        sqrt_Qpv = np.sqrt(self.Qpv)
        sqrt_Qq = np.sqrt(self.Qq)
        sqrt_Qv = np.sqrt(self.Qv)
        sqrt_R = np.sqrt(self.R)
        sqrt_Rd = np.sqrt(self.Rd)
        sqrt_Qf = np.sqrt(self.Qf)
        sqrt_Qaxis = np.sqrt(self.Qaxis)
        sqrt_Qaxisf = np.sqrt(self.Qaxisf)
        sqrt_Qqf = np.sqrt(self.Qqf)
        sqrt_Qvf = np.sqrt(self.Qvf)
        previous_tau = self.previous_tau

        for k in range(self.horizon):
            pos, rot = self.arm.forward_kinematics(traj.x[k, :self.n])
            residuals.append(sqrt_Qp * (pos - self.p_refs[k]))
            residuals.append(sqrt_Qaxis * (
                rot[:, self.terminal_axis_index] - self.terminal_axis))
            residuals.append(sqrt_Qpv * (self._ee_velocity(traj.x[k])
                                         - self.v_refs[k]))
            residuals.append(sqrt_Qq * (traj.x[k, :self.n]
                                        - self.q_nominal))
            residuals.append(sqrt_Qv * traj.x[k, 6:])
            residuals.append(sqrt_R * traj.u[k])
            residuals.append(sqrt_Rd * (traj.u[k] - previous_tau))
            previous_tau = traj.u[k]

        pos, rot = self.arm.forward_kinematics(traj.x[-1, :self.n])
        residuals.append(sqrt_Qf * (pos - self.p_refs[-1]))
        residuals.append(sqrt_Qaxisf * (
            rot[:, self.terminal_axis_index] - self.terminal_axis))
        residuals.append(sqrt_Qpv * (self._ee_velocity(traj.x[-1])
                                     - self.v_refs[-1]))
        residuals.append(sqrt_Qqf * (traj.x[-1, :self.n]
                                     - self.q_terminal))
        residuals.append(sqrt_Qvf * traj.x[-1, 6:])
        return np.concatenate(residuals)

    def residuals_with_jacobian(self, y):
        traj = self._traj(y)
        y_size = y.size
        nrows = self.horizon * (3 + 3 + 3 + self.n + self.n
                                + self.nu + self.nu) \
            + 3 + 3 + 3 + self.n + self.n
        jac = sp.lil_matrix((nrows, y_size), dtype=np.float64)
        residuals = []
        sqrt_Qp = np.sqrt(self.Qp)
        sqrt_Qpv = np.sqrt(self.Qpv)
        sqrt_Qq = np.sqrt(self.Qq)
        sqrt_Qv = np.sqrt(self.Qv)
        sqrt_R = np.sqrt(self.R)
        sqrt_Rd = np.sqrt(self.Rd)
        sqrt_Qf = np.sqrt(self.Qf)
        sqrt_Qaxis = np.sqrt(self.Qaxis)
        sqrt_Qaxisf = np.sqrt(self.Qaxisf)
        sqrt_Qqf = np.sqrt(self.Qqf)
        sqrt_Qvf = np.sqrt(self.Qvf)
        x_stride = self.nx
        u_base = (self.horizon + 1) * self.nx
        row = 0

        for k in range(self.horizon):
            x_cols = k * x_stride
            u_cols = u_base + k * self.nu
            pos, rot, Jp, Jr = self.arm.forward_kinematics_jacobian(
                traj.x[k, :self.n])

            residuals.append(sqrt_Qp * (pos - self.p_refs[k]))
            jac[row:row + 3, x_cols:x_cols + self.n] = \
                sqrt_Qp[:, None] * Jp
            row += 3

            stage_axis = rot[:, self.terminal_axis_index]
            residuals.append(sqrt_Qaxis * (
                stage_axis - self.terminal_axis))
            jac[row:row + 3, x_cols:x_cols + self.n] = \
                sqrt_Qaxis[:, None] * rotation_axis_jacobian(
                    rot,
                    Jr,
                    axis_index=self.terminal_axis_index,
                )
            row += 3

            residuals.append(sqrt_Qpv * (Jp @ traj.x[k, self.n:]
                                         - self.v_refs[k]))
            jac[row:row + 3,
                x_cols + self.n:x_cols + self.nx] = sqrt_Qpv[:, None] * Jp
            row += 3

            residuals.append(sqrt_Qq * (traj.x[k, :self.n]
                                        - self.q_nominal))
            jac[row:row + self.n,
                x_cols:x_cols + self.n] = np.diag(sqrt_Qq)
            row += self.n

            residuals.append(sqrt_Qv * traj.x[k, self.n:])
            jac[row:row + self.n,
                x_cols + self.n:x_cols + self.nx] = np.diag(sqrt_Qv)
            row += self.n

            residuals.append(sqrt_R * traj.u[k])
            jac[row:row + self.nu,
                u_cols:u_cols + self.nu] = np.diag(sqrt_R)
            row += self.nu

            if k == 0:
                delta_tau = traj.u[k] - self.previous_tau
            else:
                delta_tau = traj.u[k] - traj.u[k - 1]
            residuals.append(sqrt_Rd * delta_tau)
            jac[row:row + self.nu,
                u_cols:u_cols + self.nu] = np.diag(sqrt_Rd)
            if k > 0:
                prev_u_cols = u_base + (k - 1) * self.nu
                jac[row:row + self.nu,
                    prev_u_cols:prev_u_cols + self.nu] = -np.diag(sqrt_Rd)
            row += self.nu

        x_cols = self.horizon * x_stride
        pos, rot, Jp, Jr = self.arm.forward_kinematics_jacobian(
            traj.x[-1, :self.n])
        residuals.append(sqrt_Qf * (pos - self.p_refs[-1]))
        jac[row:row + 3, x_cols:x_cols + self.n] = sqrt_Qf[:, None] * Jp
        row += 3

        terminal_axis = rot[:, self.terminal_axis_index]
        residuals.append(sqrt_Qaxisf * (
            terminal_axis - self.terminal_axis))
        jac[row:row + 3, x_cols:x_cols + self.n] = \
            sqrt_Qaxisf[:, None] * rotation_axis_jacobian(
                rot,
                Jr,
                axis_index=self.terminal_axis_index,
            )
        row += 3

        residuals.append(sqrt_Qpv * (Jp @ traj.x[-1, self.n:]
                                     - self.v_refs[-1]))
        jac[row:row + 3,
            x_cols + self.n:x_cols + self.nx] = sqrt_Qpv[:, None] * Jp
        row += 3

        residuals.append(sqrt_Qqf * (traj.x[-1, :self.n]
                                     - self.q_terminal))
        jac[row:row + self.n, x_cols:x_cols + self.n] = np.diag(sqrt_Qqf)
        row += self.n

        residuals.append(sqrt_Qvf * traj.x[-1, self.n:])
        jac[row:row + self.n,
            x_cols + self.n:x_cols + self.nx] = np.diag(sqrt_Qvf)
        row += self.n

        if row != nrows:
            raise RuntimeError("internal residual Jacobian row mismatch")
        return np.concatenate(residuals), jac.tocsc()

    def cost(self, y):
        r = self.residuals(y)
        return 0.5 * float(r @ r)

    def equality_constraints(self, y):
        traj = self._traj(y)
        constraints = [traj.x[0] - self.current_state]
        for k in range(self.horizon):
            x_next = self.arm.step_dynamics(traj.x[k], traj.u[k])
            constraints.append(traj.x[k + 1] - x_next)
        return np.concatenate(constraints)

    def _local_dynamics_linearization(self, x, u):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.nx,):
            raise ValueError(f"x must have shape {(self.nx,)}")

        with self.arm._preserve_state():
            self.arm.set_state(x)
            M = self.arm.compute_mass_matrix()

        try:
            Minv = np.linalg.solve(M, np.eye(self.n))
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M)

        dt = float(self.arm.dt)
        A = np.zeros((self.nx, self.nx), dtype=np.float64)
        B = np.zeros((self.nx, self.nu), dtype=np.float64)
        I = np.eye(self.n)

        A[:self.n, :self.n] = I
        A[:self.n, self.n:] = dt * I
        A[self.n:, self.n:] = I
        B[:self.n, :] = dt * dt * Minv
        B[self.n:, :] = dt * Minv

        return A, B

    def equality_constraints_with_jacobian(self, y):
        traj = self._traj(y)
        constraints = []
        nrows = (self.horizon + 1) * self.nx
        jac = sp.lil_matrix((nrows, y.size), dtype=np.float64)
        x_stride = self.nx
        u_base = (self.horizon + 1) * self.nx

        constraints.append(traj.x[0] - self.current_state)
        jac[0:self.nx, 0:self.nx] = sp.eye(self.nx, format="csc")

        for k in range(self.horizon):
            row = (k + 1) * self.nx
            x_cols = k * x_stride
            x_next_cols = (k + 1) * x_stride
            u_cols = u_base + k * self.nu

            x_next = self.arm.step_dynamics(traj.x[k], traj.u[k])
            constraints.append(traj.x[k + 1] - x_next)
            A, B = self._local_dynamics_linearization(
                traj.x[k],
                traj.u[k],
            )

            jac[row:row + self.nx,
                x_cols:x_cols + self.nx] = -A
            jac[row:row + self.nx,
                x_next_cols:x_next_cols + self.nx] = sp.eye(
                    self.nx,
                    format="csc",
                )
            jac[row:row + self.nx,
                u_cols:u_cols + self.nu] = -B

        return np.concatenate(constraints), jac.tocsc()

    def inequality_constraints(self, y):
        traj = self._traj(y)
        values = []
        if self.collision_model is not None:
            values.append(self.collision_model.residuals_for_trajectory(
                traj.x[:, :6],
                box_active_mask=self.box_active_mask,
                box_contact_allowed_mask=self.box_contact_allowed_mask,
            ))

        for k in range(self.horizon + 1):
            for obstacle in self.obstacles:
                pos = self._ee_pos(traj.x[k])
                center = np.asarray(obstacle["center"], dtype=np.float64)
                radius = float(obstacle["radius"]) + self.safety_margin
                values.append(float(np.linalg.norm(pos[:2] - center[:2])
                                    - radius))
        if not values:
            return np.zeros(0)
        return np.concatenate([
            np.atleast_1d(v).astype(np.float64) for v in values
        ])

    def inequality_constraints_with_jacobian(self, y):
        traj = self._traj(y)
        values = []
        rows = []
        cols = []
        data = []
        y_size = y.size

        def append_q_row(k, jac_q):
            row = len(values) - 1
            base = k * self.nx
            jac_q = np.asarray(jac_q, dtype=np.float64).reshape(self.n)
            for j, value in enumerate(jac_q):
                if value != 0.0:
                    rows.append(row)
                    cols.append(base + j)
                    data.append(float(value))

        if self.collision_model is not None:
            for k in range(self.horizon + 1):
                vals_k, jac_q_k = self.collision_model.residuals_and_jacobian(
                    traj.x[k, :self.n],
                    include_box=bool(self.box_active_mask[k]),
                    allow_box_contact_geometry=bool(
                        self.box_contact_allowed_mask[k]),
                )
                for value, jac_q in zip(vals_k, jac_q_k):
                    values.append(float(value))
                    append_q_row(k, jac_q)

        for k in range(self.horizon + 1):
            for obstacle in self.obstacles:
                pos, Jp = self._ee_pos_jacobian(traj.x[k])
                center = np.asarray(obstacle["center"], dtype=np.float64)
                radius = float(obstacle["radius"]) + self.safety_margin
                delta_xy = pos[:2] - center[:2]
                norm_xy = float(np.linalg.norm(delta_xy))
                values.append(norm_xy - radius)
                if norm_xy > 1e-12:
                    grad_pos = np.array([
                        delta_xy[0] / norm_xy,
                        delta_xy[1] / norm_xy,
                        0.0,
                    ], dtype=np.float64)
                else:
                    grad_pos = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                append_q_row(k, grad_pos @ Jp)

        if not values:
            return np.zeros(0), sp.csc_matrix((0, y_size))

        jac = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(len(values), y_size),
            dtype=np.float64,
        ).tocsc()
        return np.asarray(values, dtype=np.float64), jac

    def delta_bounds(self, y):
        traj = self._traj(y)
        lower = np.full_like(y, -np.inf)
        upper = np.full_like(y, np.inf)
        trust_low_x = -np.concatenate([self.delta_q_max, self.delta_dq_max])
        trust_high_x = np.concatenate([self.delta_q_max, self.delta_dq_max])
        trust_low_u = -self.delta_tau_max
        trust_high_u = self.delta_tau_max

        def bounded_trust_interval(abs_low, abs_high, trust_low, trust_high):
            lo = np.maximum(abs_low, trust_low)
            hi = np.minimum(abs_high, trust_high)
            empty = lo > hi
            if np.any(empty):
                # If the warm start is already outside a hard bound by more
                # than the trust radius, keep the local SQP feasible and let
                # repeated RTI corrections move it back gradually.
                lo = lo.copy()
                hi = hi.copy()
                lo[empty] = trust_low[empty]
                hi[empty] = trust_high[empty]
            return lo, hi

        cursor = 0
        for k in range(self.horizon + 1):
            abs_low = np.concatenate([self.arm.q_min, self.arm.dq_min])
            abs_high = np.concatenate([self.arm.q_max, self.arm.dq_max])
            lo, hi = bounded_trust_interval(
                abs_low - traj.x[k],
                abs_high - traj.x[k],
                trust_low_x,
                trust_high_x,
            )
            lower[cursor:cursor + self.nx] = lo
            upper[cursor:cursor + self.nx] = hi
            cursor += self.nx
        for k in range(self.horizon):
            lo, hi = bounded_trust_interval(
                self.arm.tau_min - traj.u[k],
                self.arm.tau_max - traj.u[k],
                trust_low_u,
                trust_high_u,
            )
            lower[cursor:cursor + self.nu] = lo
            upper[cursor:cursor + self.nu] = hi
            cursor += self.nu
        return lower, upper

    def prepare_step(self, measured_state):
        self.current_state = np.asarray(measured_state, dtype=np.float64)


@dataclass
class RTIDiagnostics:
    mpc_step: int
    cost_before: float
    equality_residual_norm_before: float
    inequality_violation_before: float
    qp_status: str
    qp_iterations: int
    qp_objective: float
    delta_norm: float
    alpha: float
    cost_after: float
    equality_residual_norm_after: float
    inequality_violation_after: float
    applied_control: np.ndarray
    sqp_steps: int
    qp_solve_attempts: int
    fallback_used: bool
