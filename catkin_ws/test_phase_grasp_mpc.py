#!/usr/bin/env python3
import unittest

import numpy as np

from grasp_phase_mpc import ContactSample, PhaseGraspMPC


def symmetric_contacts(normal_force=12.0, slip_speed=0.0):
    return [
        ContactSample(
            "left",
            [-0.03, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            normal_force,
            [0.0, 0.0],
            slip_speed,
        ),
        ContactSample(
            "right",
            [0.03, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            normal_force,
            [0.0, 0.0],
            slip_speed,
        ),
    ]


class PhaseGraspMPCTests(unittest.TestCase):
    def test_required_gravity_wrench_is_feasible_for_symmetric_grasp(self):
        controller = PhaseGraspMPC(dt=0.04)
        result = controller.required_wrench_feasibility(
            symmetric_contacts(),
            object_center=np.zeros(3),
            object_mass=0.10125,
        )
        self.assertTrue(result.feasible)
        self.assertLess(result.force_residual, 1e-8)
        self.assertLess(result.torque_residual, 1e-8)
        self.assertGreater(result.normal_by_side["left"], 0.0)
        self.assertGreater(result.normal_by_side["right"], 0.0)

    def test_bilateral_contact_is_required(self):
        controller = PhaseGraspMPC(dt=0.04)
        result = controller.required_wrench_feasibility(
            symmetric_contacts()[:1],
            object_center=np.zeros(3),
            object_mass=0.10125,
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.status, "bilateral_contact_required")

    def test_contact_phase_closes_without_exceeding_command_bounds(self):
        controller = PhaseGraspMPC(dt=0.04, contact_close_rate=0.23)
        output = controller.step(
            "contact",
            close_fraction=0.0,
            contacts=[],
            object_center=np.zeros(3),
            object_mass=0.10125,
        )
        self.assertGreater(output.command_fraction, 0.0)
        self.assertLessEqual(output.command_fraction, 1.0)
        self.assertGreater(output.closure_rate, 0.0)

    def test_lift_gate_requires_consecutive_stable_mpc_samples(self):
        controller = PhaseGraspMPC(
            dt=0.04,
            preload_force=8.0,
            stable_steps_required=3,
        )
        contacts = symmetric_contacts(normal_force=12.0)
        first = controller.step(
            "preload", 0.5, contacts, np.zeros(3), 0.10125
        )
        second = controller.step(
            "preload", 0.5, contacts, np.zeros(3), 0.10125
        )
        third = controller.step(
            "preload", 0.5, contacts, np.zeros(3), 0.10125
        )
        self.assertFalse(first.lift_ready)
        self.assertFalse(second.lift_ready)
        self.assertTrue(third.lift_ready)

    def test_excessive_slip_resets_lift_gate(self):
        controller = PhaseGraspMPC(
            dt=0.04,
            preload_force=8.0,
            stable_steps_required=1,
            max_slip_speed=0.01,
        )
        output = controller.step(
            "preload",
            0.5,
            symmetric_contacts(normal_force=12.0, slip_speed=0.02),
            np.zeros(3),
            0.10125,
        )
        self.assertFalse(output.lift_ready)
        self.assertEqual(output.stable_steps, 0)


if __name__ == "__main__":
    unittest.main()
