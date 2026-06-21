#!/usr/bin/env python3
"""
Joint-space rigid-body arm dynamics for MPC.

State:
    x = [q, dq] in R^12

Control:
    u = tau in R^6, direct joint torques on the arm joints.

The dynamics are not hand-coded. MuJoCo provides the dense mass matrix and
bias forces at the requested state:
    M(q) ddq + h(q, dq) = tau
"""
from contextlib import contextmanager

import numpy as np
import mujoco


def _as_id(model, obj_type, value, what):
    if isinstance(value, str):
        obj_id = mujoco.mj_name2id(model, obj_type, value)
        if obj_id < 0:
            raise ValueError(f"unknown {what} name {value!r}")
        return int(obj_id)
    return int(value)


def _wrap_to_pi(q):
    return (q + np.pi) % (2.0 * np.pi) - np.pi


class ArmDynamics:
    """Torque-controlled fixed-base revolute manipulator dynamics.

    Parameters
    ----------
    model, data:
        MuJoCo model/data pair.
    joint_ids:
        Six arm joint ids or names. Only these joints define q and dq.
    actuator_ids:
        Six arm actuator ids or names. Used for torque limit extraction and
        optional control application. These should correspond to arm motors,
        not gripper actuators.
    ee_site_name:
        End-effector MuJoCo site name. If no site with this name exists, the
        same name is treated as a body name. This keeps the class compatible
        with the current UR10e model, whose tool frame is a body named ee_tcp.
    dt:
        Integration timestep for step_dynamics.
    velocity_limits:
        Optional six-vector of joint velocity limits. MuJoCo XML commonly
        stores position and torque limits but not velocity limits.
    """

    def __init__(self, model, data, joint_ids, actuator_ids,
                 ee_site_name, dt, velocity_limits=None,
                 wrap_unlimited_revolute=False, clip_torques=True):
        self.model = model
        self.data = data
        self.dt = float(dt)
        self.wrap_unlimited_revolute = bool(wrap_unlimited_revolute)
        self.clip_torques = bool(clip_torques)

        self.joint_ids = np.array([
            _as_id(model, mujoco.mjtObj.mjOBJ_JOINT, jid, "joint")
            for jid in joint_ids
        ], dtype=np.int32)
        self.actuator_ids = np.array([
            _as_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid, "actuator")
            for aid in actuator_ids
        ], dtype=np.int32)

        if len(self.joint_ids) != 6:
            raise ValueError("ArmDynamics expects exactly six arm joints")
        if len(self.actuator_ids) != 6:
            raise ValueError("ArmDynamics expects exactly six arm actuators")

        self.n = 6
        self.qpos_addr = model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.dof_addr = model.jnt_dofadr[self.joint_ids].astype(np.int32)

        for jid in self.joint_ids:
            jtype = model.jnt_type[jid]
            if jtype != mujoco.mjtJoint.mjJNT_HINGE:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                         int(jid))
                raise ValueError(f"joint {name!r} is not a revolute hinge")

        if len(set(self.qpos_addr.tolist())) != self.n:
            raise ValueError("arm qpos addresses must be unique")
        if len(set(self.dof_addr.tolist())) != self.n:
            raise ValueError("arm dof addresses must be unique")

        self.ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        self.ee_body_id = -1
        if self.ee_site_id < 0:
            self.ee_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, ee_site_name)
            if self.ee_body_id < 0:
                raise ValueError(
                    f"unknown end-effector site/body {ee_site_name!r}")

        self.q_min, self.q_max = self._read_joint_position_limits()
        self.dq_min, self.dq_max = self._read_velocity_limits(velocity_limits)
        self.tau_min, self.tau_max = self._read_torque_limits()

    @classmethod
    def from_robot(cls, robot, n_arm_joints=None, ee_frame_name=None, dt=None,
                   velocity_limits=None, **kwargs):
        """Build ArmDynamics from this repository's Robot wrapper."""
        n = int(n_arm_joints if n_arm_joints is not None else robot.n)
        if n != 6:
            raise ValueError(
                "from_robot is intended for the 6-DoF arm MPC model")

        model = robot.model
        data = robot.data
        joint_ids = list(range(n))
        actuator_ids = list(range(n))
        frame = ee_frame_name if ee_frame_name is not None else robot._ee_body
        timestep = dt if dt is not None else model.opt.timestep
        vlim = velocity_limits
        if vlim is None and robot.vel_limits is not None:
            vlim = robot.vel_limits
        return cls(model, data, joint_ids, actuator_ids, frame, timestep,
                   velocity_limits=vlim, **kwargs)

    @contextmanager
    def _preserve_state(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        ctrl = self.data.ctrl.copy()
        qfrc_applied = self.data.qfrc_applied.copy()
        try:
            yield
        finally:
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            self.data.ctrl[:] = ctrl
            self.data.qfrc_applied[:] = qfrc_applied
            mujoco.mj_forward(self.model, self.data)

    def _read_joint_position_limits(self):
        q_min = np.full(self.n, -np.inf)
        q_max = np.full(self.n, np.inf)
        for i, jid in enumerate(self.joint_ids):
            if self.model.jnt_limited[jid]:
                q_min[i], q_max[i] = self.model.jnt_range[jid]
        return q_min, q_max

    def _read_velocity_limits(self, velocity_limits):
        if velocity_limits is None:
            vmax = np.full(self.n, np.inf)
        else:
            vmax = np.asarray(velocity_limits, dtype=np.float64)
            if vmax.shape != (self.n,):
                raise ValueError("velocity_limits must have shape (6,)")
            vmax = np.abs(vmax)
        return -vmax, vmax

    def _read_torque_limits(self):
        tau_min = np.full(self.n, -np.inf)
        tau_max = np.full(self.n, np.inf)
        for i, aid in enumerate(self.actuator_ids):
            if self.model.actuator_ctrllimited[aid]:
                tau_min[i], tau_max[i] = self.model.actuator_ctrlrange[aid]
            elif self.model.actuator_forcelimited[aid]:
                tau_min[i], tau_max[i] = self.model.actuator_forcerange[aid]
        return tau_min, tau_max

    def get_state(self):
        q = self.data.qpos[self.qpos_addr].copy()
        dq = self.data.qvel[self.dof_addr].copy()
        return np.concatenate([q, dq])

    def set_state(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (2 * self.n,):
            raise ValueError("state must have shape (12,)")
        self.data.qpos[self.qpos_addr] = x[:self.n]
        self.data.qvel[self.dof_addr] = x[self.n:]
        mujoco.mj_forward(self.model, self.data)

    def compute_mass_matrix(self):
        full = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        mujoco.mj_fullM(self.model, full, self.data.qM)
        return full[np.ix_(self.dof_addr, self.dof_addr)]

    def compute_bias_forces(self):
        return self.data.qfrc_bias[self.dof_addr].copy()

    def _clip_tau(self, tau):
        tau = np.asarray(tau, dtype=np.float64)
        if tau.shape != (self.n,):
            raise ValueError("tau must have shape (6,)")
        if self.clip_torques:
            tau = np.minimum(np.maximum(tau, self.tau_min), self.tau_max)
        return tau

    def forward_dynamics(self, x, tau):
        """Return ddq from M(q) ddq = tau - qfrc_bias."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (2 * self.n,):
            raise ValueError("state must have shape (12,)")
        tau = self._clip_tau(tau)

        with self._preserve_state():
            self.set_state(x)
            M = self.compute_mass_matrix()
            h = self.compute_bias_forces()
            return np.linalg.solve(M, tau - h)

    def bias_for_state(self, x):
        """Return h(q,dq) for an arbitrary arm state without perturbing data."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (2 * self.n,):
            raise ValueError("state must have shape (12,)")
        with self._preserve_state():
            self.set_state(x)
            return self.compute_bias_forces()

    def step_dynamics(self, x, tau):
        """Semi-implicit Euler transition x_next = f_arm(x, tau)."""
        x = np.asarray(x, dtype=np.float64)
        q = x[:self.n].copy()
        dq = x[self.n:].copy()

        ddq = self.forward_dynamics(x, tau)
        dq_next = dq + self.dt * ddq
        dq_next = np.minimum(np.maximum(dq_next, self.dq_min), self.dq_max)

        q_next = q + self.dt * dq_next

        if self.wrap_unlimited_revolute:
            unlimited = ~np.isfinite(self.q_min) & ~np.isfinite(self.q_max)
            q_next[unlimited] = _wrap_to_pi(q_next[unlimited])

        q_clipped = np.minimum(np.maximum(q_next, self.q_min), self.q_max)
        hit_limit = np.abs(q_clipped - q_next) > 1e-12
        if np.any(hit_limit):
            outward_low = hit_limit & (q_next < self.q_min) & (dq_next < 0.0)
            outward_high = hit_limit & (q_next > self.q_max) & (dq_next > 0.0)
            dq_next[outward_low | outward_high] = 0.0
        q_next = q_clipped

        return np.concatenate([q_next, dq_next])

    def forward_kinematics(self, q):
        """Return end-effector position and rotation matrix for q."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.n,):
            raise ValueError("q must have shape (6,)")

        with self._preserve_state():
            self.data.qpos[self.qpos_addr] = q
            self.data.qvel[self.dof_addr] = 0.0
            mujoco.mj_forward(self.model, self.data)
            if self.ee_site_id >= 0:
                pos = self.data.site_xpos[self.ee_site_id].copy()
                rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
            else:
                pos = self.data.xpos[self.ee_body_id].copy()
                rot = self.data.xmat[self.ee_body_id].reshape(3, 3).copy()
        return pos, rot

    def forward_kinematics_jacobian(self, q):
        """Return end-effector pose and MuJoCo point Jacobians for q.

        Jp maps arm joint velocities to end-effector linear velocity, and Jr
        maps them to angular velocity. Only the six arm DoFs are returned.
        """
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.n,):
            raise ValueError("q must have shape (6,)")

        with self._preserve_state():
            self.data.qpos[self.qpos_addr] = q
            self.data.qvel[self.dof_addr] = 0.0
            mujoco.mj_forward(self.model, self.data)
            if self.ee_site_id >= 0:
                pos = self.data.site_xpos[self.ee_site_id].copy()
                rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
                body_id = int(self.model.site_bodyid[self.ee_site_id])
            else:
                pos = self.data.xpos[self.ee_body_id].copy()
                rot = self.data.xmat[self.ee_body_id].reshape(3, 3).copy()
                body_id = self.ee_body_id

            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jac(self.model, self.data, jacp, jacr, pos, body_id)
            Jp = jacp[:, self.dof_addr].copy()
            Jr = jacr[:, self.dof_addr].copy()
        return pos, rot, Jp, Jr

    def apply_torque_control(self, tau):
        """Write arm torque controls to the six arm actuators only."""
        tau = self._clip_tau(tau)
        self.data.ctrl[self.actuator_ids] = tau
        return tau
