#!/usr/bin/env python3
import numpy as np
import mujoco

from robot_config import get_config


class Robot:
    """Complete MuJoCo robot interface. External code only interacts with this
    class and never touches model/data directly."""

    def __init__(self, robot="ur10e", model=None):
        cfg = get_config(robot)
        self._robot_name  = robot
        self._ee_body     = cfg["ee_body"]
        self.n            = cfg["n_arm_joints"]
        self._home        = np.array(cfg.get("home_qpos", [0.0] * self.n))
        v = cfg.get("vel_limits");  self._vel_limits = np.array(v) if v else None
        self._gripper_ids   = cfg.get("gripper_ctrl_ids", [])
        self._gripper_open  = np.array(cfg.get("gripper_open",  []))
        self._gripper_close = np.array(cfg.get("gripper_close", []))

        self._model = model if model is not None else mujoco.MjModel.from_xml_path(cfg["xml"])
        self._data  = mujoco.MjData(self._model)
        self._ee_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._ee_body)

        # Cache AirSkin sensor addresses and collision geom IDs
        self._airskin_adrs = {}   # pad_idx → sensor_adr (kept for compatibility)
        self._airskin_geoms = {}  # pad_idx → geom_id of airskin_i_vhacd
        for i in range(20):
            sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"airskin_{i}")
            if sid >= 0:
                self._airskin_adrs[i] = self._model.sensor_adr[sid]
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"airskin_{i}_vhacd")
            if gid >= 0:
                self._airskin_geoms[i] = gid

        self.reset()

    # ── Simulation control ────────────────────────────────────

    def reset(self):
        """Return to home position with zero velocity."""
        self._data.qpos[:self.n]    = self._home
        self._data.qvel[:]          = 0.0
        self._data.qfrc_applied[:]  = 0.0   # clear any torques from previous run
        mujoco.mj_forward(self._model, self._data)

    def step(self, tau=None):
        """Advance one simulation step. tau: joint torques (n,), None = gravity comp only."""
        self._data.qfrc_applied[:] = 0.0
        if tau is not None:
            self._data.qfrc_applied[:self.n] = tau
        else:
            self._data.qfrc_applied[:self.n] = self.bias_torque
        mujoco.mj_step(self._model, self._data)

    # ── Simulation properties ─────────────────────────────────

    @property
    def joint_limits(self):
        """Returns (lower, upper), each of shape (n,)."""
        lo = self._model.jnt_range[:self.n, 0].copy()
        hi = self._model.jnt_range[:self.n, 1].copy()
        return lo, hi

    @property
    def vel_limits(self):
        """Max joint velocities (n,) in rad/s. None if not specified in config."""
        return self._vel_limits

    # ── Joint state ───────────────────────────────────────────

    @property
    def joint_pos(self):
        """Arm joint angles (n,) in rad."""
        return self._data.qpos[:self.n].copy()

    @property
    def joint_vel(self):
        """Arm joint velocities (n,) in rad/s."""
        return self._data.qvel[:self.n].copy()

    @property
    def joint_torque(self):
        """Applied joint torques (n,) in Nm."""
        return self._data.qfrc_applied[:self.n].copy()


    # ── End-effector state ────────────────────────────────────

    @property
    def ee_pos(self):
        """EE position (3,) in world frame, meters."""
        return self._data.xpos[self._ee_id].copy()

    @property
    def ee_rot(self):
        """EE rotation matrix (3,3) in world frame."""
        return self._data.xmat[self._ee_id].reshape(3, 3).copy()

    @property
    def ee_quat(self):
        """EE quaternion [w, x, y, z] in world frame."""
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self._data.xmat[self._ee_id])
        return q

    @property
    def ee_pose(self):
        """EE SE3 homogeneous transform (4,4) in world frame."""
        T = np.eye(4)
        T[:3, :3] = self.ee_rot
        T[:3,  3] = self.ee_pos
        return T

    @property
    def ee_vel(self):
        """EE velocity [vx,vy,vz, wx,wy,wz] computed as J @ dq."""
        return self.jacobian @ self.joint_vel

    # ── Kinematics ────────────────────────────────────────────

    @property
    def jacobian(self):
        """Geometric Jacobian (6, n): [linear; angular]."""
        jac_pos = np.zeros((3, self._model.nv))
        jac_rot = np.zeros((3, self._model.nv))
        mujoco.mj_jacBody(self._model, self._data, jac_pos, jac_rot, self._ee_id)
        return np.vstack([jac_pos, jac_rot])[:, :self.n]

    # ── Dynamics ──────────────────────────────────────────────

    @property
    def mass_matrix(self):
        """Joint-space mass matrix M(q), shape (n, n)."""
        M = np.zeros((self._model.nv, self._model.nv))
        mujoco.mj_fullM(self._model, M, self._data.qM)
        M = M[:self.n, :self.n]
        return M

    @property
    def bias_torque(self):
        """Coriolis + gravity torques C(q,dq)dq + g(q), shape (n,)."""
        bias = self._data.qfrc_bias[:self.n].copy()
        return bias

    def inv_dynamics(self, ddq_des):
        """Inverse dynamics: τ = M(q)·ddq_des + C(q,dq)dq + g(q)."""
        tau = self.mass_matrix @ ddq_des + self.bias_torque
        return tau

    @property
    def os_mass_matrix(self):
        """Operational-space mass matrix Lambda = (J M⁻¹ Jᵀ)⁻¹, shape (6, 6).
        Damped inverse (1e-4 I) prevents singularity near degenerate configurations."""
        M = self.mass_matrix
        J = self.jacobian
        Lambda_inv = J @ np.linalg.inv(M) @ J.T
        Lambda = np.linalg.inv(Lambda_inv + 1e-4 * np.eye(6))
        return Lambda

    @property
    def os_bias_force(self):
        """Task-space bias term h = J M⁻¹ C, shape (6,).
        Used in controller as: tau = Jᵀ Lambda (ddx_cmd + h).
        """
        M        = self.mass_matrix
        C        = self.bias_torque
        J        = self.jacobian
        return J @ np.linalg.inv(M) @ C

    @property
    def null_space_projector(self):
        """Dynamically-consistent null space projector N^T = I - Jᵀ Λ J M⁻¹, shape (n, n).
        Uses exact Lambda (pinv, no regularization) so that J M⁻¹ N^T = 0 holds exactly
        and null space torques don't leak into task space."""
        M_inv  = np.linalg.inv(self.mass_matrix)
        J      = self.jacobian
        Lambda = np.linalg.pinv(J @ M_inv @ J.T)   # exact, no regularization
        return np.eye(self.n) - J.T @ Lambda @ J @ M_inv

    @property
    def os_gravity_force(self):
        """Operational-space gravity force g = J M⁻¹ g(q), shape (6,)."""
        M = self.mass_matrix
        g = self._data.qfrc_bias[:self.n].copy()
        J = self.jacobian
        g_os = J @ np.linalg.inv(M) @ g
        return g_os

    # ── Gripper ───────────────────────────────────────────────

    def set_gripper(self, values):
        """Set gripper ctrl directly. values: array matching gripper_ctrl_ids."""
        for idx, v in zip(self._gripper_ids, values):
            self._data.ctrl[idx] = v

    def open_gripper(self):
        self.set_gripper(self._gripper_open)

    def close_gripper(self):
        self.set_gripper(self._gripper_close)

    def gripper(self):
        """Return current gripper ctrl values."""
        return np.array([self._data.ctrl[i] for i in self._gripper_ids])

    # ── AirSkin ───────────────────────────────────────────────

    @property
    def airskin(self):
        """Normal contact force (N) on each AirSkin pad.
        Returns dict {pad_index: force_N}.
        Matches by specific airskin_i_vhacd geom, not body, to avoid
        attributing one contact to all pads on the same link."""
        
        m, d = self._model, self._data
        result = {i: 0.0 for i in self._airskin_geoms}
        f6 = np.zeros(6)
        for ci in range(d.ncon):
            c = d.contact[ci]
            mujoco.mj_contactForce(m, d, ci, f6)
            fn = abs(f6[0])
            if fn == 0.0:
                continue
            for i, gid in self._airskin_geoms.items():
                if c.geom1 == gid or c.geom2 == gid:
                    result[i] += fn
        return result

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data
