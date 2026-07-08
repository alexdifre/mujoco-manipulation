#!/usr/bin/env python3
"""SE(3) end-effector manipulation model for grasp/transport/release tasks."""
from dataclasses import dataclass

import mujoco
import numpy as np


def skew(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def make_pose(pos, rot=None):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    if rot is not None:
        pose[:3, :3] = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    return pose


def pose_with_position(pose, pos):
    out = np.asarray(pose, dtype=np.float64).reshape(4, 4).copy()
    out[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return out


def rotmat_to_quat(rot):
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=np.float64).reshape(9))
    return quat


def so3_log(rot):
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    cos_theta = 0.5 * (np.trace(rot) - 1.0)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-10:
        return 0.5 * np.array(
            [
                rot[2, 1] - rot[1, 2],
                rot[0, 2] - rot[2, 0],
                rot[1, 0] - rot[0, 1],
            ],
            dtype=np.float64,
        )
    return (
        theta
        / (2.0 * np.sin(theta))
        * np.array(
            [
                rot[2, 1] - rot[1, 2],
                rot[0, 2] - rot[2, 0],
                rot[1, 0] - rot[0, 1],
            ],
            dtype=np.float64,
        )
    )


def se3_log(pose):
    """Return [translation_error, rotation_error] for a homogeneous transform."""
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    rot = pose[:3, :3]
    trans = pose[:3, 3]
    omega = so3_log(rot)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-10:
        v = trans
    else:
        omega_hat = skew(omega)
        half_theta = 0.5 * theta
        cot_half = 1.0 / np.tan(half_theta)
        v_inv = (
            np.eye(3)
            - 0.5 * omega_hat
            + (1.0 - theta * cot_half / 2.0)
            / (theta * theta)
            * (omega_hat @ omega_hat)
        )
        v = v_inv @ trans
    return np.concatenate([v, omega])


def pose_error(desired_pose, actual_pose):
    desired = np.asarray(desired_pose, dtype=np.float64).reshape(4, 4)
    actual = np.asarray(actual_pose, dtype=np.float64).reshape(4, 4)
    return se3_log(np.linalg.inv(desired) @ actual)


@dataclass
class ManipulationMetrics:
    grasp_pose_error: float
    release_pose_error: float
    lift_margin: float
    grasp_quality: float
    gamma: float


@dataclass
class EndEffectorManipulationDynamics:
    """Kinematic object dynamics induced by a rigid end-effector grasp.

    The grasp map follows the notation from the experiment text:
      X_grasp = X_b^(0) X_g
      X_b(tau) = F_e(q(tau)) X_g^{-1}
    where X_g maps box-frame coordinates into the desired end-effector frame.
    """

    initial_box_pose: np.ndarray
    desired_box_pose: np.ndarray
    desired_grasp_pose: np.ndarray
    lift_height: float
    eps_grasp: float
    eps_release: float
    gamma_max: float
    friction_quality_eps: float

    def __post_init__(self):
        self.initial_box_pose = np.asarray(
            self.initial_box_pose,
            dtype=np.float64,
        ).reshape(4, 4)
        self.desired_box_pose = np.asarray(
            self.desired_box_pose,
            dtype=np.float64,
        ).reshape(4, 4)
        self.desired_grasp_pose = np.asarray(
            self.desired_grasp_pose,
            dtype=np.float64,
        ).reshape(4, 4)
        self.box_to_ee_grasp = np.linalg.inv(
            self.initial_box_pose,
        ) @ self.desired_grasp_pose
        self.ee_to_box_grasp = np.linalg.inv(self.box_to_ee_grasp)

    def grasp_error(self, ee_pose):
        return pose_error(self.desired_grasp_pose, ee_pose)

    def release_error(self, box_pose):
        return pose_error(self.desired_box_pose, box_pose)

    def ee_pose_for_box_pose(self, box_pose):
        return np.asarray(box_pose, dtype=np.float64).reshape(4, 4) @ self.box_to_ee_grasp

    def ee_position_for_box_position(self, box_pos):
        return self.ee_pose_for_box_pose(
            pose_with_position(self.initial_box_pose, box_pos),
        )[:3, 3]

    def attach(self, ee_pose, box_pose):
        self.ee_to_box_grasp = (
            np.linalg.inv(np.asarray(ee_pose, dtype=np.float64).reshape(4, 4))
            @ np.asarray(box_pose, dtype=np.float64).reshape(4, 4)
        )
        self.box_to_ee_grasp = np.linalg.inv(self.ee_to_box_grasp)
        return self.ee_to_box_grasp.copy()

    def attached_box_pose(self, ee_pose):
        return np.asarray(ee_pose, dtype=np.float64).reshape(4, 4) @ self.ee_to_box_grasp

    def lift_margin(self, box_pose):
        return float(np.asarray(box_pose, dtype=np.float64).reshape(4, 4)[2, 3] - self.lift_height)

    def gamma_for_stage(self, stage, stage_hold=0, close_ramp_steps=1, released=False):
        if stage in {"open", "approach", "descend"}:
            return self.gamma_max
        if stage == "close":
            frac = float(stage_hold) / max(int(close_ramp_steps), 1)
            return self.gamma_max * (1.0 - np.clip(frac, 0.0, 1.0))
        if released or stage in {"release", "retreat"}:
            return self.gamma_max
        return 0.0

    def contact_quality(self, left_contact, right_contact, finger_mid_error, finger_aperture, max_aperture):
        centering = 1.0 - float(finger_mid_error) / max(self.eps_grasp, 1e-9)
        closure = 1.0 - float(finger_aperture) / max(float(max_aperture), 1e-9)
        if left_contact and right_contact:
            contact_score = 1.0
        elif left_contact or right_contact:
            contact_score = 0.5
        else:
            contact_score = 0.0
        return min(contact_score, centering, closure) - self.friction_quality_eps

    def metrics(
        self,
        ee_pose,
        box_pose,
        stage,
        stage_hold,
        close_ramp_steps,
        released,
        left_contact,
        right_contact,
        finger_mid_error,
        finger_aperture,
        max_aperture,
    ):
        gamma = self.gamma_for_stage(stage, stage_hold, close_ramp_steps, released)
        return ManipulationMetrics(
            grasp_pose_error=float(np.linalg.norm(self.grasp_error(ee_pose))),
            release_pose_error=float(np.linalg.norm(self.release_error(box_pose))),
            lift_margin=self.lift_margin(box_pose),
            grasp_quality=self.contact_quality(
                left_contact,
                right_contact,
                finger_mid_error,
                finger_aperture,
                max_aperture,
            ),
            gamma=float(gamma),
        )
