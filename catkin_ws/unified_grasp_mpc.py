#!/usr/bin/env python3
"""Small prescribed-contact MPC for arm, gripper, object, and contact wrench.

This controller deliberately uses a local, frozen linearisation so it remains
small enough for the physical-grasp regression test.  Its optimisation still
contains all variables needed by the grasp OCP:

state   [q, dq, gamma, dgamma, p_object, v_object, theta_object, omega_object]
control [tau_arm, dgamma, lambda_left, lambda_right]

Each contact wrench is [normal force, two tangential forces, torsional moment].
The contact mode is prescribed by the outer state machine; this is not a
contact-implicit or complementarity-based solver.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import minimize


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


@dataclass
class UnifiedGraspMPCOutput:
    phase: str
    tau: np.ndarray
    command_fraction: float
    gripper_rate: float
    planned_normal_by_side: dict
    planned_wrench_by_side: dict
    predicted_object_position: np.ndarray
    predicted_object_velocity: np.ndarray
    max_predicted_slip: float
    equality_residual: float
    min_inequality_margin: float
    objective: float
    optimizer_success: bool
    fallback_used: bool
    status: str


class UnifiedGraspMPC:
    """Finite-horizon local OCP with a prescribed two-finger contact mode."""

    ACTIVE_PHASES = {"contact", "preload", "lift", "hold"}
    N_ARM = 6
    N_CONTACT = 4
    N_CONTROL = N_ARM + 1 + 2 * N_CONTACT

    def __init__(
        self,
        arm,
        dt,
        horizon=3,
        friction_coefficient=0.8,
        torsional_friction_coefficient=0.005,
        normal_force_max=40.0,
        close_rate_max=0.35,
        hold_rate_max=0.12,
        max_slip_speed=0.025,
        force_gain=60.0,
        gravity=9.81,
        table_z=0.0,
        object_half_height=0.03,
        max_iterations=45,
    ):
        self.arm = arm
        self.dt = float(dt)
        self.horizon = max(int(horizon), 1)
        self.mu = max(float(friction_coefficient), 0.05)
        self.mu_torsion = max(float(torsional_friction_coefficient), 0.0)
        self.normal_force_max = float(normal_force_max)
        self.close_rate_max = float(close_rate_max)
        self.hold_rate_max = float(hold_rate_max)
        self.max_slip_speed = float(max_slip_speed)
        self.force_gain = float(force_gain)
        self.gravity = float(gravity)
        self.table_z = float(table_z)
        self.object_half_height = float(object_half_height)
        self.max_iterations = int(max_iterations)
        self.previous_u = None
        self.previous_phase = None
        self.previous_tau = np.zeros(self.N_ARM)
        self.previous_gripper_rate = 0.0

    def reset(self):
        self.previous_u = None
        self.previous_phase = None
        self.previous_tau = np.zeros(self.N_ARM)
        self.previous_gripper_rate = 0.0

    @staticmethod
    def state_vector(q, dq, gamma, dgamma, p, v, theta, omega):
        """Pack the documented 26-dimensional predictive state."""
        return np.concatenate([
            np.asarray(q).reshape(6), np.asarray(dq).reshape(6),
            [float(gamma), float(dgamma)],
            np.asarray(p).reshape(3), np.asarray(v).reshape(3),
            np.asarray(theta).reshape(3), np.asarray(omega).reshape(3),
        ]).astype(np.float64)

    @staticmethod
    def _strongest_contacts(contacts):
        selected = {}
        for contact in contacts:
            if (contact.side not in selected or
                    contact.normal_force > selected[contact.side].normal_force):
                selected[contact.side] = contact
        return selected

    def _contact_frames(self, contacts, object_position, finger_positions):
        selected = self._strongest_contacts(contacts)
        frames = {}
        left, right = [np.asarray(p, dtype=np.float64).reshape(3)
                       for p in finger_positions]
        axis = right - left
        axis /= np.linalg.norm(axis) + 1e-12
        fallback = {
            "left": (left, axis),
            "right": (right, -axis),
        }
        for side in ("left", "right"):
            if side in selected:
                position = selected[side].position.copy()
                normal = selected[side].normal.copy()
                measured = float(selected[side].normal_force)
            else:
                position, normal = fallback[side]
                position = position.copy()
                normal = normal.copy()
                measured = 0.0
            t1, t2 = _tangent_basis(normal)
            frames[side] = {
                "position": position,
                "lever": position - object_position,
                "normal": normal,
                "t1": t1,
                "t2": t2,
                "measured": measured,
            }
        return frames, set(selected)

    @staticmethod
    def _world_wrench(local_wrench, frame):
        fn, ft1, ft2, torsion = local_wrench
        force = fn * frame["normal"] + ft1 * frame["t1"] + ft2 * frame["t2"]
        moment = np.cross(frame["lever"], force) + torsion * frame["normal"]
        return force, moment

    def _unpack_controls(self, flat):
        return np.asarray(flat, dtype=np.float64).reshape(
            self.horizon, self.N_CONTROL)

    def _rollout(self, flat, context):
        flat_array = np.asarray(flat, dtype=np.float64)
        cached_x = context.get("_cache_x")
        if cached_x is not None and np.array_equal(flat_array, cached_x):
            return context["_cache_rollout"]
        controls = self._unpack_controls(flat)
        q = context["q"].copy()
        dq = context["dq"].copy()
        gamma = float(context["gamma"])
        dgamma = float(context["dgamma"])
        p = context["p"].copy()
        v = context["v"].copy()
        theta = np.zeros(3)
        omega = context["omega"].copy()
        frames = context["frames"]
        M = context["M"]
        bias = context["bias"]
        Jp = context["Jp"]
        Jr = context["Jr"]
        mass = context["mass"]
        inertia_world = context["inertia_world"]
        gravity_force = np.array([0.0, 0.0, -mass * self.gravity])

        states = [self.state_vector(q, dq, gamma, dgamma, p, v, theta, omega)]
        ee_positions = [context["ee"]]
        slips = []
        forces = []
        moments = []

        for control in controls:
            tau = control[:6]
            rate = float(control[6])
            local_left = control[7:11]
            local_right = control[11:15]
            force_left, moment_left = self._world_wrench(local_left, frames["left"])
            force_right, moment_right = self._world_wrench(local_right, frames["right"])
            total_force = force_left + force_right
            total_moment = moment_left + moment_right

            # Frozen local arm dynamics with the object reaction applied at the
            # gripper point.  This couples planned lambda back into tau_arm.
            ddq = np.linalg.solve(
                M,
                tau - bias - Jp.T @ total_force - Jr.T @ total_moment,
            )
            dq = dq + self.dt * ddq
            q = q + self.dt * dq
            # The rate command is the simple gripper input; dgamma is retained
            # explicitly in the predictive state as requested by the OCP.
            dgamma = rate
            gamma = gamma + self.dt * dgamma

            if context["phase"] == "contact":
                # Before preload the support reaction belongs to the table, not
                # to a finger contact decision.  Keep the supported object at
                # rest while the gripper searches for bilateral contact.
                v = np.zeros(3)
                omega = np.zeros(3)
            else:
                acceleration = (total_force + gravity_force) / mass
                angular_acceleration = np.linalg.solve(
                    inertia_world,
                    total_moment - np.cross(omega, inertia_world @ omega),
                )
                v = v + self.dt * acceleration
                p = p + self.dt * v
                omega = omega + self.dt * angular_acceleration
                theta = theta + self.dt * omega

            ee = context["ee"] + Jp @ (q - context["q"])
            ee_velocity = Jp @ dq
            stage_slips = []
            for side in ("left", "right"):
                frame = frames[side]
                object_contact_velocity = v + np.cross(omega, frame["lever"])
                relative = ee_velocity - object_contact_velocity
                tangential = relative - frame["normal"] * float(
                    relative @ frame["normal"]
                )
                stage_slips.append(float(np.linalg.norm(tangential)))

            states.append(self.state_vector(q, dq, gamma, dgamma, p, v, theta, omega))
            ee_positions.append(ee)
            slips.append(stage_slips)
            forces.append((force_left, force_right))
            moments.append((moment_left, moment_right))

        result = {
            "controls": controls,
            "states": np.asarray(states),
            "ee": np.asarray(ee_positions),
            "slips": np.asarray(slips),
            "forces": forces,
            "moments": moments,
        }
        context["_cache_x"] = flat_array.copy()
        context["_cache_rollout"] = result
        return result

    def _desired_acceleration(self, phase, position, velocity, target):
        if phase == "preload":
            return np.clip(-2.0 * velocity, -0.15, 0.15)
        gain = 18.0 if phase == "lift" else 12.0
        damping = 7.0 if phase == "lift" else 6.0
        limit = 0.45 if phase == "lift" else 0.25
        return np.clip(gain * (target - position) - damping * velocity, -limit, limit)

    def _equality_constraints(self, flat, context):
        if context["phase"] == "contact":
            return np.zeros(0)
        rollout = self._rollout(flat, context)
        residuals = []
        mass = context["mass"]
        inertia = context["inertia_world"]
        gravity = np.array([0.0, 0.0, -mass * self.gravity])
        for k, (force_pair, moment_pair) in enumerate(zip(
                rollout["forces"], rollout["moments"])):
            state = rollout["states"][k]
            position = state[14:17]
            velocity = state[17:20]
            omega = state[23:26]
            desired_a = self._desired_acceleration(
                context["phase"], position, velocity, context["object_target"]
            )
            desired_alpha = -5.0 * omega
            residuals.extend(force_pair[0] + force_pair[1] + gravity - mass * desired_a)
            residuals.extend(moment_pair[0] + moment_pair[1] - inertia @ desired_alpha)
        return np.asarray(residuals, dtype=np.float64)

    def _inequality_constraints(self, flat, context):
        rollout = self._rollout(flat, context)
        margins = []
        q_min = self.arm.q_min
        q_max = self.arm.q_max
        dq_min = self.arm.dq_min
        dq_max = self.arm.dq_max
        bilateral = context["bilateral"]

        for k, control in enumerate(rollout["controls"]):
            state_next = rollout["states"][k + 1]
            q = state_next[:6]
            dq = state_next[6:12]
            gamma = state_next[12]
            p = state_next[14:17]
            for value, lower, upper in zip(q, q_min, q_max):
                if np.isfinite(lower):
                    margins.append(value - lower)
                if np.isfinite(upper):
                    margins.append(upper - value)
            margins.extend(dq - dq_min)
            margins.extend(dq_max - dq)
            margins.extend([gamma, 1.0 - gamma])
            # During contact the table supports the cube.  In the other modes
            # this prevents a predicted trajectory from entering the table.
            margins.append(p[2] - self.object_half_height - self.table_z)

            for offset, side in ((7, "left"), (11, "right")):
                fn, ft1, ft2, torsion = control[offset:offset + 4]
                margins.extend([
                    fn,
                    self.normal_force_max - fn,
                    self.mu * fn - abs(ft1) - abs(ft2),
                    self.mu_torsion * fn - abs(torsion),
                ])
                # A measured-force plus local-compliance envelope prevents the
                # optimiser from inventing contact force without closing.
                predicted_capacity = (
                    context["frames"][side]["measured"] + 2.0
                    + self.force_gain * max(gamma - context["gamma"], 0.0)
                )
                margins.append(predicted_capacity - fn)
                if bilateral and context["phase"] != "contact":
                    margins.append(self.max_slip_speed - rollout["slips"][k, 0 if side == "left" else 1])
        return np.asarray(margins, dtype=np.float64)

    def _objective(self, flat, context):
        rollout = self._rollout(flat, context)
        value = 0.0
        previous_tau = self.previous_tau
        for k, control in enumerate(rollout["controls"]):
            tau = control[:6]
            rate = control[6]
            left = control[7:11]
            right = control[11:15]
            state = rollout["states"][k + 1]
            q = state[:6]
            dq = state[6:12]
            p = state[14:17]
            v = state[17:20]
            theta = state[20:23]
            omega = state[23:26]
            ee_error = rollout["ee"][k + 1] - context["ee_target"]
            value += 100.0 * float(ee_error @ ee_error)
            value += 0.80 * float((q - context["q_target"]) @ (q - context["q_target"]))
            value += 0.025 * float(dq @ dq)
            value += 0.020 * float((tau - context["nominal_tau"]) @ (tau - context["nominal_tau"]))
            value += 0.015 * float((tau - previous_tau) @ (tau - previous_tau))
            value += 6.0 * float((rate - context["desired_rate"]) ** 2)
            if context["bilateral"] and context["phase"] != "contact":
                force_target = context["force_target"]
                value += 0.04 * float((left[0] - force_target) ** 2)
                value += 0.04 * float((right[0] - force_target) ** 2)
                value += 0.05 * float((left[0] - right[0]) ** 2)
                value += 300.0 * float(np.sum(rollout["slips"][k] ** 2))
            position_error = p - context["object_target"]
            value += 160.0 * float(position_error @ position_error)
            value += 7.0 * float(v @ v) + 2.0 * float(theta @ theta)
            value += 0.5 * float(omega @ omega)
            previous_tau = tau
        terminal_position_error = rollout["states"][-1, 14:17] - context["object_target"]
        value += 300.0 * float(terminal_position_error @ terminal_position_error)
        return float(value)

    def _bounds(self, phase, bilateral, nominal_tau):
        tau_min = np.where(np.isfinite(self.arm.tau_min), self.arm.tau_min, -1e4)
        tau_max = np.where(np.isfinite(self.arm.tau_max), self.arm.tau_max, 1e4)
        # This is a local SQP correction around the nonlinear acados arm MPC,
        # so keep the torque decision inside its trust region.
        tau_min = np.maximum(tau_min, nominal_tau - 5.0)
        tau_max = np.minimum(tau_max, nominal_tau + 5.0)
        rate_limit = self.close_rate_max if phase in {"contact", "preload"} else self.hold_rate_max
        rate_lower = 0.0 if phase in {"contact", "preload"} else -rate_limit
        bounds = []
        for _ in range(self.horizon):
            bounds.extend(zip(tau_min, tau_max))
            bounds.append((rate_lower, rate_limit))
            if phase == "contact" or not bilateral:
                bounds.extend([(0.0, 0.0)] * 8)
            else:
                for _side in range(2):
                    bounds.extend([
                        (0.0, self.normal_force_max),
                        (-self.mu * self.normal_force_max, self.mu * self.normal_force_max),
                        (-self.mu * self.normal_force_max, self.mu * self.normal_force_max),
                        (-self.mu_torsion * self.normal_force_max,
                         self.mu_torsion * self.normal_force_max),
                    ])
        return bounds

    def _initial_guess(self, context):
        if self.previous_u is not None and self.previous_phase == context["phase"]:
            shifted = np.vstack([self.previous_u[1:], self.previous_u[-1]])
            return shifted.reshape(-1)
        controls = np.zeros((self.horizon, self.N_CONTROL))
        controls[:, :6] = context["nominal_tau"]
        controls[:, 6] = context["desired_rate"]
        if context["bilateral"] and context["phase"] != "contact":
            half_weight = np.array([0.0, 0.0, 0.5 * context["mass"] * self.gravity])
            for k in range(self.horizon):
                for offset, side in ((7, "left"), (11, "right")):
                    frame = context["frames"][side]
                    required_normal = max(
                        context["force_target"], float(half_weight @ frame["normal"])
                    )
                    controls[k, offset] = np.clip(required_normal, 0.0, self.normal_force_max)
                    controls[k, offset + 1] = float(half_weight @ frame["t1"])
                    controls[k, offset + 2] = float(half_weight @ frame["t2"])
        return controls.reshape(-1)

    def step(
        self,
        phase,
        nominal_tau,
        close_fraction,
        desired_gripper_rate,
        target_normal_force,
        contacts,
        finger_positions,
        object_position,
        object_rotation,
        object_linear_velocity,
        object_angular_velocity,
        object_mass,
        object_inertia,
        ee_target,
        object_target,
        q_target,
    ):
        if phase not in self.ACTIVE_PHASES:
            raise ValueError(f"unsupported unified grasp phase {phase!r}")
        nominal_tau = self.arm._clip_tau(np.asarray(nominal_tau, dtype=np.float64))
        q = self.arm.get_state()[:6]
        dq = self.arm.get_state()[6:]
        ee, _, Jp, Jr = self.arm.forward_kinematics_jacobian(q)
        with self.arm._preserve_state():
            self.arm.set_state(np.concatenate([q, dq]))
            mass_matrix = self.arm.compute_mass_matrix()
            bias = self.arm.compute_bias_forces()
        object_position = np.asarray(object_position, dtype=np.float64).reshape(3)
        frames, contact_sides = self._contact_frames(
            contacts, object_position, finger_positions
        )
        bilateral = {"left", "right"}.issubset(contact_sides)
        rotation = np.asarray(object_rotation, dtype=np.float64).reshape(3, 3)
        inertia_world = rotation @ np.diag(np.asarray(object_inertia).reshape(3)) @ rotation.T
        context = {
            "phase": phase,
            "q": q,
            "dq": dq,
            "gamma": float(close_fraction),
            "dgamma": self.previous_gripper_rate,
            "p": object_position,
            "v": np.asarray(object_linear_velocity, dtype=np.float64).reshape(3),
            "omega": np.asarray(object_angular_velocity, dtype=np.float64).reshape(3),
            "mass": float(object_mass),
            "inertia_world": inertia_world,
            "M": mass_matrix,
            "bias": bias,
            "ee": ee,
            "Jp": Jp,
            "Jr": Jr,
            "frames": frames,
            "bilateral": bilateral,
            "nominal_tau": nominal_tau,
            "desired_rate": float(desired_gripper_rate),
            "force_target": float(target_normal_force),
            "ee_target": np.asarray(ee_target, dtype=np.float64).reshape(3),
            "object_target": np.asarray(object_target, dtype=np.float64).reshape(3),
            "q_target": np.asarray(q_target, dtype=np.float64).reshape(6),
        }
        x0 = self._initial_guess(context)
        constraints = [{
            "type": "ineq",
            "fun": lambda flat: self._inequality_constraints(flat, context),
        }]
        if phase != "contact" and bilateral:
            constraints.append({
                "type": "eq",
                "fun": lambda flat: self._equality_constraints(flat, context),
            })
        elif phase != "contact":
            # Wrench equilibrium cannot be solved without both fingers.
            constraints.append({"type": "eq", "fun": lambda _flat: np.ones(6)})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = minimize(
                lambda flat: self._objective(flat, context),
                x0,
                method="SLSQP",
                bounds=self._bounds(phase, bilateral, nominal_tau),
                constraints=constraints,
                options={"maxiter": self.max_iterations, "ftol": 1e-5, "disp": False},
            )

        candidate = result.x if np.size(result.x) == np.size(x0) else x0
        rollout = self._rollout(candidate, context)
        equality = self._equality_constraints(candidate, context)
        inequalities = self._inequality_constraints(candidate, context)
        equality_residual = float(np.max(np.abs(equality))) if equality.size else 0.0
        min_margin = float(np.min(inequalities)) if inequalities.size else np.inf
        mode_valid = bilateral if phase != "contact" else True
        # SLSQP may hit its iteration limit after already finding a feasible
        # point.  For control, feasibility is the meaningful success test.
        success = bool(
            mode_valid
            and np.all(np.isfinite(candidate))
            and equality_residual <= 2e-3
            and min_margin >= -2e-4
        )
        first = rollout["controls"][0]
        if success:
            tau = self.arm._clip_tau(first[:6])
            rate = float(first[6])
            self.previous_u = rollout["controls"].copy()
            self.previous_phase = phase
            self.previous_tau = tau.copy()
            self.previous_gripper_rate = rate
            status = str(result.message)
        else:
            tau = nominal_tau.copy()
            rate = float(desired_gripper_rate)
            self.previous_u = None
            self.previous_phase = None
            self.previous_tau = tau.copy()
            self.previous_gripper_rate = rate
            status = f"fallback: {result.message}"
        command_fraction = float(np.clip(close_fraction + self.dt * rate, 0.0, 1.0))
        first_left = first[7:11].copy()
        first_right = first[11:15].copy()
        return UnifiedGraspMPCOutput(
            phase=phase,
            tau=tau,
            command_fraction=command_fraction,
            gripper_rate=rate,
            planned_normal_by_side={
                "left": float(first_left[0]), "right": float(first_right[0])
            },
            planned_wrench_by_side={"left": first_left, "right": first_right},
            predicted_object_position=rollout["states"][1, 14:17].copy(),
            predicted_object_velocity=rollout["states"][1, 17:20].copy(),
            max_predicted_slip=float(np.max(rollout["slips"][0])),
            equality_residual=equality_residual,
            min_inequality_margin=min_margin,
            objective=float(result.fun) if np.isfinite(result.fun) else np.inf,
            optimizer_success=success,
            fallback_used=not success,
            status=status,
        )
