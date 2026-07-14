#!/usr/bin/env python3
"""Minimal phase-scheduled MPC for a parallel-jaw physical grasp.

The arm torque NMPC remains in :mod:`rti_sqp_mpc`.  This module implements the
phase supervisor and conservative lift gate used by the unified grasp OCP:

* a finite-horizon optimisation of the normalized gripper closing velocity;
* an online local force/compliance prediction for preload control;
* required-wrench feasibility with unilateral friction-pyramid forces;
* a lift gate based on preload, friction margin, and tangential slip.

Contact modes are scheduled by the caller (contact, preload, lift, hold).
This intentionally avoids contact-implicit complementarity while keeping the
physical grasp constraints inside the controller decision.
"""
from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass
class ContactSample:
    """Measured finger/object contact expressed in the world frame."""

    side: str
    position: np.ndarray
    normal: np.ndarray
    normal_force: float
    tangential_force: np.ndarray
    slip_speed: float = 0.0

    def __post_init__(self):
        if self.side not in {"left", "right"}:
            raise ValueError("contact side must be 'left' or 'right'")
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        normal = np.asarray(self.normal, dtype=np.float64).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            raise ValueError("contact normal must be nonzero")
        self.normal = normal / normal_norm
        self.normal_force = abs(float(self.normal_force))
        self.tangential_force = np.asarray(
            self.tangential_force, dtype=np.float64
        ).reshape(-1)
        self.slip_speed = abs(float(self.slip_speed))


@dataclass
class WrenchFeasibility:
    feasible: bool
    required_wrench: np.ndarray
    achieved_wrench: np.ndarray
    normal_by_side: dict = field(default_factory=dict)
    force_residual: float = np.inf
    torque_residual: float = np.inf
    status: str = "not_solved"


@dataclass
class GraspMPCOutput:
    phase: str
    command_fraction: float
    closure_rate: float
    target_normal_force: float
    predicted_normal_force: float
    actual_normal_by_side: dict
    wrench: WrenchFeasibility
    min_friction_margin: float
    max_slip_speed: float
    bilateral_contact: bool
    stable_steps: int
    lift_ready: bool
    optimizer_success: bool


def _tangent_basis(normal):
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ seed)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(normal, seed)
    t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2) + 1e-12
    return t1, t2


class PhaseGraspMPC:
    """Finite-horizon gripper/preload MPC with a prescribed contact mode."""

    ACTIVE_PHASES = {"contact", "preload", "lift", "hold"}

    def __init__(
        self,
        dt,
        horizon=10,
        friction_coefficient=1.0,
        friction_margin=0.20,
        torsional_friction_coefficient=0.005,
        normal_force_max=40.0,
        preload_force=8.0,
        preload_force_buffer=2.0,
        contact_close_rate=0.23,
        close_rate_max=0.35,
        hold_rate_max=0.12,
        initial_force_gain=60.0,
        max_slip_speed=0.025,
        min_friction_force_margin=0.20,
        lift_safety_factor=1.25,
        stable_steps_required=5,
        gravity=9.81,
        cone_rays=8,
    ):
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.mu = max(float(friction_coefficient) - float(friction_margin), 0.05)
        self.mu_torsion = max(float(torsional_friction_coefficient), 0.0)
        self.normal_force_max = float(normal_force_max)
        self.preload_force = float(preload_force)
        self.preload_force_buffer = float(preload_force_buffer)
        self.contact_close_rate = float(contact_close_rate)
        self.close_rate_max = float(close_rate_max)
        self.hold_rate_max = float(hold_rate_max)
        self.force_gain = float(initial_force_gain)
        self.max_slip_speed = float(max_slip_speed)
        self.min_friction_force_margin = float(min_friction_force_margin)
        self.lift_safety_factor = float(lift_safety_factor)
        self.stable_steps_required = int(stable_steps_required)
        self.gravity = float(gravity)
        self.cone_rays = max(int(cone_rays), 4)
        self.previous_rate = 0.0
        self.previous_fraction = None
        self.previous_mean_force = None
        self.stable_steps = 0

    def reset(self):
        self.previous_rate = 0.0
        self.previous_fraction = None
        self.previous_mean_force = None
        self.stable_steps = 0

    @staticmethod
    def _normal_by_side(contacts):
        values = {"left": 0.0, "right": 0.0}
        for contact in contacts:
            values[contact.side] += contact.normal_force
        return values

    def _update_force_gain(self, close_fraction, contacts):
        normal_by_side = self._normal_by_side(contacts)
        bilateral = normal_by_side["left"] > 0.0 and normal_by_side["right"] > 0.0
        mean_force = 0.5 * (normal_by_side["left"] + normal_by_side["right"])
        if (
            bilateral
            and self.previous_fraction is not None
            and self.previous_mean_force is not None
        ):
            delta_fraction = float(close_fraction) - self.previous_fraction
            delta_force = mean_force - self.previous_mean_force
            if delta_fraction > 1e-4 and delta_force > 0.0:
                observed_gain = np.clip(delta_force / delta_fraction, 10.0, 200.0)
                self.force_gain = 0.8 * self.force_gain + 0.2 * float(observed_gain)
        self.previous_fraction = float(close_fraction)
        self.previous_mean_force = float(mean_force) if bilateral else None

    def required_wrench_feasibility(
        self,
        contacts,
        object_center,
        object_mass,
        desired_vertical_accel=0.0,
    ):
        """Find friction-pyramid forces that support gravity plus lift demand.

        The LP uses nonnegative coefficients on full 360-degree cone rays.
        Small signed wrench slacks keep the solve well posed for noisy MuJoCo
        contact locations; feasibility is accepted only below tight force and
        moment residual thresholds.
        """
        contacts = list(contacts)
        sides = {contact.side for contact in contacts}
        required_force = float(object_mass) * (
            self.gravity + max(float(desired_vertical_accel), 0.0)
        ) * self.lift_safety_factor
        required = np.array([0.0, 0.0, required_force, 0.0, 0.0, 0.0])
        if not {"left", "right"}.issubset(sides):
            return WrenchFeasibility(
                False,
                required,
                np.zeros(6),
                {"left": 0.0, "right": 0.0},
                status="bilateral_contact_required",
            )

        object_center = np.asarray(object_center, dtype=np.float64).reshape(3)
        rays = []
        ray_contact_indices = []
        for contact_index, contact in enumerate(contacts):
            t1, t2 = _tangent_basis(contact.normal)
            lever = contact.position - object_center
            for angle in np.linspace(0.0, 2.0 * np.pi, self.cone_rays, endpoint=False):
                direction = contact.normal + self.mu * (
                    np.cos(angle) * t1 + np.sin(angle) * t2
                )
                rays.append(np.concatenate([direction, np.cross(lever, direction)]))
                ray_contact_indices.append(contact_index)

        grasp_map = np.asarray(rays, dtype=np.float64).T
        n_rays = grasp_map.shape[1]
        # Variables: cone coefficients, positive wrench slack, negative slack.
        A_eq = np.hstack([grasp_map, np.eye(6), -np.eye(6)])
        b_eq = required
        c = np.concatenate([
            1e-3 * np.ones(n_rays),
            np.array([200.0, 200.0, 200.0, 2000.0, 2000.0, 2000.0]),
            np.array([200.0, 200.0, 200.0, 2000.0, 2000.0, 2000.0]),
        ])
        A_ub = np.zeros((len(contacts), n_rays + 12))
        for ray_index, contact_index in enumerate(ray_contact_indices):
            A_ub[contact_index, ray_index] = 1.0
        b_ub = self.normal_force_max * np.ones(len(contacts))
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=[(0.0, None)] * (n_rays + 12),
            method="highs",
        )
        if not result.success:
            return WrenchFeasibility(
                False,
                required,
                np.zeros(6),
                {"left": 0.0, "right": 0.0},
                status=str(result.message),
            )

        coefficients = result.x[:n_rays]
        achieved = grasp_map @ coefficients
        residual = achieved - required
        normal_by_side = {"left": 0.0, "right": 0.0}
        for coefficient, contact_index in zip(coefficients, ray_contact_indices):
            normal_by_side[contacts[contact_index].side] += float(coefficient)
        force_residual = float(np.linalg.norm(residual[:3]))
        torque_residual = float(np.linalg.norm(residual[3:]))
        # The cube uses condim=6, so its soft contacts can also transmit a
        # bounded torsional moment.  The first 10 mNm cover contact-location
        # noise; the remaining capacity is the MuJoCo torsional coefficient
        # times the measured compressive preload.
        torsional_capacity = self.mu_torsion * sum(
            contact.normal_force for contact in contacts
        )
        torque_tolerance = 0.010 + torsional_capacity
        feasible = force_residual <= 0.15 and torque_residual <= torque_tolerance
        return WrenchFeasibility(
            feasible,
            required,
            achieved,
            normal_by_side,
            force_residual,
            torque_residual,
            "optimal" if feasible else "wrench_residual_too_large",
        )

    def _predict(self, close_fraction, forces, rates):
        fraction = float(close_fraction)
        left = float(forces["left"])
        right = float(forces["right"])
        fractions = []
        predicted = []
        for rate in rates:
            next_fraction = fraction + self.dt * float(rate)
            delta = next_fraction - fraction
            left += self.force_gain * delta
            right += self.force_gain * delta
            fraction = next_fraction
            fractions.append(fraction)
            predicted.append((left, right))
        return np.asarray(fractions), np.asarray(predicted)

    def _optimise_rate(self, phase, close_fraction, normal_by_side, target_force, bilateral):
        if phase == "contact":
            if bilateral:
                desired_rate = 0.0
            elif normal_by_side["left"] > 0.0 or normal_by_side["right"] > 0.0:
                desired_rate = 0.25 * self.contact_close_rate
            else:
                desired_rate = self.contact_close_rate
            low, high = 0.0, self.close_rate_max
        elif phase == "preload":
            desired_rate = np.clip(
                0.025 * (target_force - min(normal_by_side.values())),
                0.0,
                self.close_rate_max,
            )
            low, high = 0.0, self.close_rate_max
        else:
            mean_force = 0.5 * sum(normal_by_side.values())
            desired_rate = np.clip(
                0.015 * (target_force - mean_force),
                -self.hold_rate_max,
                self.hold_rate_max,
            )
            low, high = -self.hold_rate_max, self.hold_rate_max

        initial = np.full(self.horizon, desired_rate, dtype=np.float64)

        def objective(rates):
            fractions, predicted = self._predict(close_fraction, normal_by_side, rates)
            previous = self.previous_rate
            value = 0.0
            for k, rate in enumerate(rates):
                value += 0.20 * float(rate * rate)
                value += 0.80 * float((rate - previous) ** 2)
                previous = rate
                if phase == "contact":
                    value += 8.0 * float((rate - desired_rate) ** 2)
                else:
                    force_error = predicted[k] - target_force
                    value += 0.025 * float(force_error @ force_error)
                    value += 0.010 * float((predicted[k, 0] - predicted[k, 1]) ** 2)
            if phase != "contact":
                terminal_error = predicted[-1] - target_force
                value += 0.15 * float(terminal_error @ terminal_error)
            # Keep the prediction away from the aperture limits.
            value += 1e-3 * float(np.sum((fractions - 0.5) ** 2))
            return value

        def path_constraints(rates):
            fractions, predicted = self._predict(close_fraction, normal_by_side, rates)
            return np.concatenate([
                fractions,
                1.0 - fractions,
                predicted.reshape(-1),
                self.normal_force_max - predicted.reshape(-1),
            ])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step",
                category=RuntimeWarning,
            )
            result = minimize(
                objective,
                initial,
                method="SLSQP",
                bounds=[(low, high)] * self.horizon,
                constraints=[{"type": "ineq", "fun": path_constraints}],
                options={"maxiter": 80, "ftol": 1e-7, "disp": False},
            )
        rates = result.x if result.success else initial
        fractions, predicted = self._predict(close_fraction, normal_by_side, rates)
        return (
            float(rates[0]),
            float(np.clip(fractions[0], 0.0, 1.0)),
            float(np.mean(predicted[0])),
            bool(result.success),
        )

    def step(
        self,
        phase,
        close_fraction,
        contacts,
        object_center,
        object_mass,
        desired_vertical_accel=0.0,
    ):
        if phase not in self.ACTIVE_PHASES:
            raise ValueError(f"unsupported grasp MPC phase {phase!r}")
        contacts = list(contacts)
        close_fraction = float(np.clip(close_fraction, 0.0, 1.0))
        normal_by_side = self._normal_by_side(contacts)
        bilateral = normal_by_side["left"] > 1e-6 and normal_by_side["right"] > 1e-6
        self._update_force_gain(close_fraction, contacts)

        wrench = self.required_wrench_feasibility(
            contacts,
            object_center,
            object_mass,
            desired_vertical_accel=desired_vertical_accel,
        )
        required_side_force = max(wrench.normal_by_side.values(), default=0.0)
        minimum_preload = max(
            self.preload_force,
            self.lift_safety_factor * required_side_force,
        )
        target_force = min(
            minimum_preload + self.preload_force_buffer,
            0.9 * self.normal_force_max,
        )
        closure_rate, command_fraction, predicted_force, optimizer_success = (
            self._optimise_rate(
                phase,
                close_fraction,
                normal_by_side,
                target_force,
                bilateral,
            )
        )
        self.previous_rate = closure_rate

        friction_margins = [
            self.mu * contact.normal_force
            - float(np.linalg.norm(contact.tangential_force))
            for contact in contacts
        ]
        min_friction_margin = min(friction_margins, default=-np.inf)
        max_slip = max((contact.slip_speed for contact in contacts), default=np.inf)
        actual_preload = (
            bilateral
            and normal_by_side["left"] >= minimum_preload
            and normal_by_side["right"] >= minimum_preload
        )
        stable = (
            phase in {"preload", "lift", "hold"}
            and wrench.feasible
            and actual_preload
            and min_friction_margin >= self.min_friction_force_margin
            and max_slip <= self.max_slip_speed
        )
        self.stable_steps = self.stable_steps + 1 if stable else 0
        lift_ready = self.stable_steps >= self.stable_steps_required

        return GraspMPCOutput(
            phase=phase,
            command_fraction=command_fraction,
            closure_rate=closure_rate,
            target_normal_force=target_force,
            predicted_normal_force=predicted_force,
            actual_normal_by_side=normal_by_side,
            wrench=wrench,
            min_friction_margin=float(min_friction_margin),
            max_slip_speed=float(max_slip),
            bilateral_contact=bilateral,
            stable_steps=self.stable_steps,
            lift_ready=lift_ready,
            optimizer_success=optimizer_success,
        )
