#!/usr/bin/env python3
from contextlib import contextmanager
import unittest

import numpy as np

from grasp_phase_mpc import ContactSample
from unified_grasp_mpc import UnifiedGraspMPC


class FakeArm:
    def __init__(self):
        self.q_min = -10.0 * np.ones(6)
        self.q_max = 10.0 * np.ones(6)
        self.dq_min = -10.0 * np.ones(6)
        self.dq_max = 10.0 * np.ones(6)
        self.tau_min = -100.0 * np.ones(6)
        self.tau_max = 100.0 * np.ones(6)
        self.state = np.zeros(12)

    def get_state(self):
        return self.state.copy()

    def _clip_tau(self, tau):
        return np.clip(np.asarray(tau), self.tau_min, self.tau_max)

    @contextmanager
    def _preserve_state(self):
        yield

    def set_state(self, state):
        self.state = np.asarray(state).copy()

    def compute_mass_matrix(self):
        return np.eye(6)

    def compute_bias_forces(self):
        return np.zeros(6)

    def forward_kinematics_jacobian(self, _q):
        Jp = np.zeros((3, 6))
        Jr = np.zeros((3, 6))
        Jp[:, :3] = np.eye(3)
        Jr[:, 3:] = np.eye(3)
        return np.zeros(3), np.eye(3), Jp, Jr


def symmetric_contacts(force=12.0):
    return [
        ContactSample("left", [-0.03, 0.0, 0.0], [1.0, 0.0, 0.0], force, [0, 0]),
        ContactSample("right", [0.03, 0.0, 0.0], [-1.0, 0.0, 0.0], force, [0, 0]),
    ]


class UnifiedGraspMPCTests(unittest.TestCase):
    def setUp(self):
        self.controller = UnifiedGraspMPC(
            FakeArm(), dt=0.04, horizon=1, max_iterations=30
        )

    def test_predictive_state_contains_arm_gripper_and_object(self):
        state = self.controller.state_vector(
            np.zeros(6), np.zeros(6), 0.4, 0.1,
            np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
        )
        self.assertEqual(state.shape, (26,))
        self.assertEqual(self.controller.N_CONTROL, 15)
        self.assertAlmostEqual(state[12], 0.4)
        self.assertAlmostEqual(state[13], 0.1)

    def test_preload_ocp_satisfies_wrench_and_friction_constraints(self):
        output = self.controller.step(
            phase="preload",
            nominal_tau=np.zeros(6),
            close_fraction=0.5,
            desired_gripper_rate=0.0,
            target_normal_force=8.0,
            contacts=symmetric_contacts(),
            finger_positions=(np.array([-0.03, 0.0, 0.0]),
                              np.array([0.03, 0.0, 0.0])),
            object_position=np.array([0.0, 0.0, 0.03]),
            object_rotation=np.eye(3),
            object_linear_velocity=np.zeros(3),
            object_angular_velocity=np.zeros(3),
            object_mass=0.10125,
            object_inertia=np.array([6e-5, 6e-5, 6e-5]),
            ee_target=np.zeros(3),
            object_target=np.array([0.0, 0.0, 0.03]),
            q_target=np.zeros(6),
        )
        self.assertTrue(output.optimizer_success, output.status)
        self.assertFalse(output.fallback_used)
        self.assertLessEqual(output.equality_residual, 2e-3)
        self.assertGreaterEqual(output.min_inequality_margin, -2e-4)
        self.assertLessEqual(output.max_predicted_slip, 0.0251)
        self.assertGreater(output.planned_normal_by_side["left"], 0.0)
        self.assertGreater(output.planned_normal_by_side["right"], 0.0)


if __name__ == "__main__":
    unittest.main()
