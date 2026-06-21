#!/usr/bin/env python3
"""
Real-Time Iteration / pathfollowing SQP utilities for nonlinear MPC.

The solver performs exactly one SQP correction per MPC call:
  1. shift the previous trajectory,
  2. linearize the nonlinear problem at the shifted warm start,
  3. solve one local QP in trajectory increments,
  4. retract the updated trajectory,
  5. apply only the first control.

This module is intentionally a solver layer. Robot-specific dynamics and
forward kinematics are supplied by ProblemModel implementations such as
ArmNMPCProblem below.
"""
from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp


def wrap_to_pi(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def so3_exp(omega):
    theta = float(np.linalg.norm(omega))
    K = skew(omega)
    if theta < 1e-12:
        return np.eye(3) + K
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + A * K + B * (K @ K)


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


class EuclideanManifold:
    def retract(self, point, tangent_increment):
        return np.asarray(point) + np.asarray(tangent_increment)


class TorusManifold:
    def retract(self, point, tangent_increment):
        return wrap_to_pi(np.asarray(point) + np.asarray(tangent_increment))


class UnitVectorManifold:
    def retract(self, point, tangent_increment):
        y = np.asarray(point, dtype=np.float64) + np.asarray(tangent_increment)
        norm = float(np.linalg.norm(y))
        if norm < 1e-12:
            raise ValueError("cannot retract a near-zero unit vector")
        return y / norm


class SO3Manifold:
    def retract(self, point, tangent_increment):
        R = np.asarray(point, dtype=np.float64).reshape(3, 3)
        dR = so3_exp(np.asarray(tangent_increment, dtype=np.float64))
        return R @ dR


class ArmStateManifold:
    """State retraction for x = [q, dq].

    Joint angles are optionally wrapped on T^6. If the robot model uses hard
    position limits, keep wrap_joints=False and rely on QP bounds.
    """

    def __init__(self, n_joints=6, wrap_joints=False):
        self.n = int(n_joints)
        self.wrap_joints = bool(wrap_joints)

    def retract(self, point, tangent_increment):
        x = np.asarray(point, dtype=np.float64)
        dx = np.asarray(tangent_increment, dtype=np.float64)
        y = x + dx
        if self.wrap_joints:
            y[:self.n] = wrap_to_pi(y[:self.n])
        return y


class TrajectoryManifold:
    def __init__(self, horizon, nx, nu, state_manifold=None,
                 control_manifold=None):
        self.horizon = int(horizon)
        self.nx = int(nx)
        self.nu = int(nu)
        self.state_manifold = state_manifold or EuclideanManifold()
        self.control_manifold = control_manifold or EuclideanManifold()

    def retract_vector(self, y, delta_y):
        traj = Trajectory.from_vector(y, self.horizon, self.nx, self.nu)
        delta = Trajectory.from_vector(delta_y, self.horizon, self.nx, self.nu)
        x_new = np.vstack([
            self.state_manifold.retract(traj.x[k], delta.x[k])
            for k in range(self.horizon + 1)
        ])
        u_new = np.vstack([
            self.control_manifold.retract(traj.u[k], delta.u[k])
            for k in range(self.horizon)
        ])
        return Trajectory(x_new, u_new).stack()


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
class DerivativeData:
    cost_value: float
    equality: np.ndarray
    inequality: np.ndarray
    grad_cost: np.ndarray
    hessian: sp.csc_matrix
    jac_eq: sp.csc_matrix
    jac_ineq: sp.csc_matrix


class AnalyticDerivativeProvider:
    """Derivative provider that requires problem-supplied Jacobian hooks."""

    def __init__(self, problem, hessian_mode="gauss_newton",
                 diagonal_hessian=1.0):
        self.problem = problem
        self.hessian_mode = hessian_mode
        self.diagonal_hessian = float(diagonal_hessian)

    def _required_hook(self, name):
        hook = getattr(self.problem, name, None)
        if hook is None:
            raise NotImplementedError(
                f"{type(self.problem).__name__} must implement {name}"
            )
        return hook

    def evaluate(self, y, lambda_bar=None, mu_bar=None):
        cost_value = float(self.problem.cost(y))
        equality, jac_eq = self._required_hook(
            "equality_constraints_with_jacobian")(y)
        inequality, jac_ineq = self._required_hook(
            "inequality_constraints_with_jacobian")(y)
        if self.hessian_mode != "gauss_newton":
            raise NotImplementedError(
                "only gauss_newton Hessians are supported here"
            )

        residuals, jac_res = self._required_hook("residuals_with_jacobian")(y)
        grad_cost = np.asarray(jac_res.T @ residuals).reshape(-1)
        hessian = jac_res.T @ jac_res

        return DerivativeData(cost_value, equality, inequality, grad_cost,
                              hessian.tocsc(), jac_eq, jac_ineq)


class HybridDerivativeProvider(AnalyticDerivativeProvider):
    """Preferred provider name for the current analytic derivative hooks."""


@dataclass
class QPData:
    H: sp.csc_matrix
    g: np.ndarray
    A: sp.csc_matrix
    lower: np.ndarray
    upper: np.ndarray
    n_eq: int
    n_ineq: int
    n_bounds: int


class QPBuilder:
    def __init__(self, regularization=1e-6):
        self.regularization = float(regularization)

    def build(self, derivative_data, delta_lower, delta_upper,
              regularization=None):
        rho = self.regularization if regularization is None else regularization
        n = derivative_data.grad_cost.size
        H = derivative_data.hessian + rho * sp.eye(n, format="csc")
        g = derivative_data.grad_cost

        blocks = []
        lower = []
        upper = []

        if derivative_data.jac_eq.shape[0]:
            blocks.append(derivative_data.jac_eq)
            rhs = -derivative_data.equality
            lower.append(rhs)
            upper.append(rhs)

        if derivative_data.jac_ineq.shape[0]:
            blocks.append(derivative_data.jac_ineq)
            lower.append(-derivative_data.inequality)
            upper.append(np.full(derivative_data.inequality.shape, np.inf))

        blocks.append(sp.eye(n, format="csc"))
        lower.append(np.asarray(delta_lower, dtype=np.float64))
        upper.append(np.asarray(delta_upper, dtype=np.float64))

        A = sp.vstack(blocks, format="csc")
        l = np.concatenate(lower)
        u = np.concatenate(upper)
        return QPData(H.tocsc(), g, A, l, u,
                      derivative_data.jac_eq.shape[0],
                      derivative_data.jac_ineq.shape[0],
                      n)


@dataclass
class QPResult:
    delta_y: np.ndarray
    dual: np.ndarray
    status: str
    status_val: int
    iterations: int
    objective: float
    success: bool


class OSQPSolver:
    def __init__(self, eps_abs=1e-5, eps_rel=1e-5, max_iter=2000):
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.max_iter = int(max_iter)

    def solve(self, qp_data, primal_warm_start=None, dual_warm_start=None):
        solver = osqp.OSQP()
        solver.setup(qp_data.H, qp_data.g, qp_data.A,
                     qp_data.lower, qp_data.upper,
                     verbose=False,
                     warm_starting=True,
                     eps_abs=self.eps_abs,
                     eps_rel=self.eps_rel,
                     max_iter=self.max_iter,
                     polishing=False)
        if primal_warm_start is not None:
            solver.warm_start(x=primal_warm_start, y=dual_warm_start)
        result = solver.solve()
        success = result.info.status_val in (1, 2)
        delta = result.x if result.x is not None else np.zeros_like(qp_data.g)
        dual = result.y if result.y is not None else np.zeros(qp_data.A.shape[0])
        return QPResult(delta, dual, result.info.status,
                        int(result.info.status_val),
                        int(result.info.iter),
                        float(result.info.obj_val),
                        success)


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


class RTISolver:
    """One-step pathfollowing SQP / RTI solver."""

    def __init__(self, problem, derivative_provider=None, qp_builder=None,
                 qp_solver=None, manifold=None, alpha=1.0,
                 fallback_regularization=(1e-4, 1e-2, 1.0),
        debug=False):
        self.problem = problem
        self.derivatives = derivative_provider or \
            HybridDerivativeProvider(problem)
        self.qp_builder = qp_builder or QPBuilder()
        self.qp_solver = qp_solver or OSQPSolver()
        self.manifold = manifold or TrajectoryManifold(
            problem.horizon, problem.nx, problem.nu)
        self.alpha = float(alpha)
        self.fallback_regularization = tuple(fallback_regularization)
        self.debug = bool(debug)
        self.previous_trajectory = None
        self.previous_lambda = None
        self.previous_mu = None
        self.previous_delta = None
        self.mpc_step = 0

    def _warm_start(self, measured_state):
        if self.previous_trajectory is None:
            return self.problem.make_initial_trajectory(measured_state)
        return self.problem.shift_trajectory(self.previous_trajectory,
                                             measured_state)

    def _split_duals(self, qp_data, dual):
        n_eq = qp_data.n_eq
        n_ineq = qp_data.n_ineq
        lam = dual[:n_eq].copy()
        mu = dual[n_eq:n_eq + n_ineq].copy()
        return lam, mu

    def step(self, measured_state):
        self.problem.prepare_step(measured_state)
        warm = self._warm_start(measured_state)
        y_bar = warm.stack()

        deriv = self.derivatives.evaluate(
            y_bar, self.previous_lambda, self.previous_mu)
        delta_lower, delta_upper = self.problem.delta_bounds(y_bar)

        regularizations = (self.qp_builder.regularization,) + \
            self.fallback_regularization
        qp_result = None
        qp_data = None
        attempts = 0
        for rho in regularizations:
            attempts += 1
            qp_data = self.qp_builder.build(deriv, delta_lower, delta_upper,
                                            regularization=rho)
            qp_result = self.qp_solver.solve(
                qp_data,
                primal_warm_start=self.previous_delta,
                dual_warm_start=None,
            )
            if qp_result.success:
                break

        fallback_used = not qp_result.success
        if fallback_used:
            delta = np.zeros_like(y_bar)
            y_new = y_bar.copy()
            traj_new = warm
            if hasattr(self.problem, "previous_tau"):
                applied = self.problem.previous_tau.copy()
            else:
                applied = traj_new.u[0].copy()
            lam = np.zeros(deriv.equality.size)
            mu = np.zeros(deriv.inequality.size)
        else:
            delta = self.alpha * qp_result.delta_y
            y_new = self.manifold.retract_vector(y_bar, delta)
            traj_new = Trajectory.from_vector(
                y_new, self.problem.horizon, self.problem.nx, self.problem.nu)
            applied = traj_new.u[0].copy()
            lam, mu = self._split_duals(qp_data, qp_result.dual)

        cost_after = float(self.problem.cost(y_new))
        eq_after = self.problem.equality_constraints(y_new)
        ineq_after = self.problem.inequality_constraints(y_new)

        if fallback_used:
            self.previous_trajectory = None
            self.previous_lambda = None
            self.previous_mu = None
            self.previous_delta = None
        else:
            self.previous_trajectory = traj_new
            self.previous_lambda = lam
            self.previous_mu = mu
            self.previous_delta = qp_result.delta_y.copy()

        diag = RTIDiagnostics(
            mpc_step=self.mpc_step,
            cost_before=deriv.cost_value,
            equality_residual_norm_before=float(np.linalg.norm(deriv.equality)),
            inequality_violation_before=float(
                np.min(np.minimum(deriv.inequality, 0.0))
                if deriv.inequality.size else 0.0),
            qp_status=qp_result.status,
            qp_iterations=qp_result.iterations,
            qp_objective=qp_result.objective,
            delta_norm=float(np.linalg.norm(delta)),
            alpha=self.alpha,
            cost_after=cost_after,
            equality_residual_norm_after=float(np.linalg.norm(eq_after)),
            inequality_violation_after=float(
                np.min(np.minimum(ineq_after, 0.0)) if ineq_after.size else 0.0),
            applied_control=applied.copy(),
            sqp_steps=1,
            qp_solve_attempts=attempts,
            fallback_used=fallback_used,
        )
        self.mpc_step += 1

        if self.debug:
            print(
                f"RTI step {diag.mpc_step}: cost {diag.cost_before:.3e} -> "
                f"{diag.cost_after:.3e}, |C|={diag.equality_residual_norm_before:.3e}, "
                f"ineq={diag.inequality_violation_before:.3e}, "
                f"QP={diag.qp_status}, |dy|={diag.delta_norm:.3e}"
            )

        if hasattr(self.problem, "set_previous_tau"):
            self.problem.set_previous_tau(applied)

        return applied, traj_new, diag
