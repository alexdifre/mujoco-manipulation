#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from acados_rti_solver import AcadosRTIConfig, AcadosRTISolver
from arm_dynamics import ArmDynamics
from environment import environment
from mpc_path_utils import (
    apply_optional_tau_limit,
    build_parabolic_waypoints,
    limit_torque_slew,
    reference_horizon_distances,
    sample_waypoint_path,
    update_path_progress,
    waypoint_arclengths,
)
from manipulation_dynamics import (
    EndEffectorManipulationDynamics,
    make_pose,
    pose_with_position,
    rotmat_to_quat,
)
from rti_sqp_mpc import ArmNMPCProblem, OSQPSolver, QPBuilder, RTISolver


@dataclass
class GraspMetric:
    step: int
    stage: str
    ee_x: float
    ee_y: float
    ee_z: float
    cube_x: float
    cube_y: float
    cube_z: float
    target_x: float
    target_y: float
    target_z: float
    ee_error: float
    cube_lift: float
    left_contact: bool
    right_contact: bool
    grasp_contact: bool
    mpc_status: str
    mpc_fallback: bool
    mpc_ineq: float
    tau_norm: float
    gripper_mean: float
    left_finger_x: float
    left_finger_y: float
    left_finger_z: float
    right_finger_x: float
    right_finger_y: float
    right_finger_z: float
    finger_mid_x: float
    finger_mid_y: float
    finger_mid_z: float
    finger_aperture: float
    finger_mid_error: float
    enclosure_axis_error: float
    enclosure_perp_error: float
    enclosure_vertical_error: float
    cube_enclosed: bool
    grasp_latched: bool
    grasp_pose_error: float
    release_pose_error: float
    lift_margin: float
    grasp_quality: float
    gamma: float
    manipulation_ineq: float
    cube_table_clearance: float
    cube_table_penetration: float


def target_offset(args):
    return np.array([args.target_x_offset, args.target_y_offset, args.target_z_offset], dtype=np.float64)


def cube_top_target(env, clearance, args):
    spec = env._object_defs["cube"]
    half_z = float(spec.get("size", [0.03, 0.03, 0.03])[2])
    pos = env.get_object_pos("cube")
    target = np.array([pos[0], pos[1], pos[2] + half_z + float(clearance)], dtype=np.float64)
    return target + target_offset(args)


def cube_center_target(env, z_offset, args):
    pos = env.get_object_pos("cube")
    target = np.array([pos[0], pos[1], pos[2] + float(z_offset)], dtype=np.float64)
    return target + target_offset(args)


def delivery_cube_pos(initial_cube, args):
    return np.asarray(initial_cube, dtype=np.float64) + np.array(
        [args.delivery_x_offset, args.delivery_y_offset, args.delivery_z_offset],
        dtype=np.float64,
    )


def cube_lift_box_pos(initial_cube, lift_z_offset):
    pos = np.asarray(initial_cube, dtype=np.float64).copy()
    pos[2] += float(lift_z_offset)
    return pos


def cube_min_center_z(env, args):
    return float(args.table_z + env.object_half_height("cube") + args.table_clearance)


def supported_cube_pose(pose, env, args):
    out = np.asarray(pose, dtype=np.float64).reshape(4, 4).copy()
    out[2, 3] = max(float(out[2, 3]), cube_min_center_z(env, args))
    return out


def supported_cube_pos(pos, env, args):
    out = np.asarray(pos, dtype=np.float64).copy()
    out[2] = max(float(out[2]), cube_min_center_z(env, args))
    return out


def manipulation_constraint_margin(stage, metric, args):
    margins = []
    if stage in {"close", "lift", "transport"}:
        margins.append(float(args.grasp_pose_tol - metric.grasp_pose_error))
    if stage in {"lift", "transport"}:
        margins.append(float(metric.lift_margin))
    if stage in {"close", "lift", "transport"}:
        margins.append(float(metric.grasp_quality))
    if stage in {"release", "retreat"}:
        margins.append(float(args.release_tol - metric.release_pose_error))
    return min(margins) if margins else 0.0


def gripper_contacts(env):
    model = env.robot.model
    data = env.robot.data
    cube_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    left_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_outer_knuckle", "left_inner_knuckle", "left_inner_finger")
    }
    right_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("right_outer_knuckle", "right_inner_knuckle", "right_inner_finger")
    }
    left = False
    right = False
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == cube_gid:
            other = int(model.geom_bodyid[c.geom2])
        elif c.geom2 == cube_gid:
            other = int(model.geom_bodyid[c.geom1])
        else:
            continue
        left = left or other in left_bodies
        right = right or other in right_bodies
    return left, right


def gripper_geometry(env):
    model = env.robot.model
    data = env.robot.data
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_inner_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_inner_finger")
    left = data.xpos[left_id].copy()
    right = data.xpos[right_id].copy()
    mid = 0.5 * (left + right)
    aperture = float(np.linalg.norm(left - right))
    return left, right, mid, aperture


def cube_enclosure_quality(env, left_finger, right_finger, cube_pos, args):
    left = np.asarray(left_finger, dtype=np.float64)
    right = np.asarray(right_finger, dtype=np.float64)
    cube = np.asarray(cube_pos, dtype=np.float64)
    delta = right - left
    aperture = float(np.linalg.norm(delta))
    if aperture < 1e-9:
        return False, float("inf"), float("inf")
    axis = delta / aperture
    mid = 0.5 * (left + right)
    rel = cube - mid
    axis_error = float(abs(rel @ axis))
    perp = rel - (rel @ axis) * axis
    perp_error = float(np.linalg.norm(perp))
    vertical_error = float(abs(cube[2] - mid[2]))
    cube_radius = float(np.linalg.norm(env.object_half_extents("cube")))
    enclosed = (
        axis_error <= args.enclosure_axis_tol
        and perp_error <= args.enclosure_perp_tol
        and vertical_error <= args.enclosure_vertical_tol
        and aperture <= 2.0 * cube_radius + args.enclosure_aperture_slack
    )
    return bool(enclosed), axis_error, perp_error, vertical_error


def set_gripper_fraction(robot, fraction):
    fraction = float(np.clip(fraction, 0.0, 1.0))
    command = robot._gripper_open + fraction * (robot._gripper_close - robot._gripper_open)
    robot.set_gripper(command)


def set_robot_joint_state(robot, q):
    robot.data.qpos[:robot.n] = np.asarray(q, dtype=np.float64)
    robot.data.qvel[:] = 0.0
    robot.data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(robot.model, robot.data)


def make_solver(args, env):
    arm = ArmDynamics.from_robot(env.robot, dt=args.mpc_dt)
    refs = np.repeat(env.robot.ee_pos[None, :], args.horizon + 1, axis=0)
    problem = ArmNMPCProblem(
        arm,
        args.horizon,
        refs,
        Qp=[args.ee_pos_weight, args.ee_pos_weight, args.ee_z_weight],
        Qpv=[0.0, 0.0, 0.0],
        Qq=[args.q_weight] * 6,
        Qv=[args.qv_weight] * 6,
        Qf=[args.ee_terminal_weight, args.ee_terminal_weight, args.ee_terminal_z_weight],
        Qaxis=[args.ee_upright_weight, args.ee_upright_weight, 0.0],
        Qaxisf=[args.ee_terminal_upright_weight, args.ee_terminal_upright_weight, 0.0],
        Qqf=[args.qf_weight] * 6,
        Qvf=[args.qvf_weight] * 6,
        Rd=[args.delta_tau_cost] * 6,
        q_nominal=env.robot._home,
        q_terminal=env.robot._home,
        previous_tau=np.zeros(6),
        collision_model=None,
        terminal_axis=[0.0, 0.0, -1.0],
        terminal_axis_index=2,
        delta_q_max=[args.delta_q_max] * 6,
        delta_dq_max=[args.delta_dq_max] * 6,
        delta_tau_max=[args.delta_tau_max] * 6,
    )
    export_dir = Path(args.acados_export_dir)
    if not export_dir.is_absolute():
        export_dir = Path(__file__).resolve().parents[1] / export_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    if args.solver in {"auto", "acados"}:
        try:
            solver = AcadosRTISolver(
                problem,
                config=AcadosRTIConfig(
                    code_export_directory=str(export_dir),
                    qp_solver=args.acados_qp_solver,
                    qp_solver_iter_max=args.acados_qp_solver_iter_max,
                    nlp_solver_type=args.acados_nlp_solver_type,
                    regularization=args.regularization,
                    verbose=args.acados_verbose,
                ),
                debug=args.debug,
            )
            return arm, problem, solver, "acados SQP_RTI"
        except Exception:
            if args.solver == "acados":
                raise
            print("acados unavailable; falling back to local RTI/OSQP solver")

    solver = RTISolver(
        problem,
        qp_builder=QPBuilder(regularization=args.regularization),
        qp_solver=OSQPSolver(max_iter=args.osqp_max_iter),
        debug=args.debug,
    )
    return arm, problem, solver, "local RTI/OSQP"


def solve_ik_position(arm, q0, target, iterations=160, damping=1e-3, step=0.45, tol=0.01):
    q = np.asarray(q0, dtype=np.float64).copy()
    target = np.asarray(target, dtype=np.float64)
    for _ in range(iterations):
        pos, _, Jp, _ = arm.forward_kinematics_jacobian(q)
        err = target - pos
        if np.linalg.norm(err) < float(tol):
            break
        A = Jp @ Jp.T + damping * np.eye(3)
        dq = Jp.T @ np.linalg.solve(A, err)
        q = q + step * dq
        q = np.minimum(np.maximum(q, arm.q_min), arm.q_max)
    return q


def set_reference_to_target(problem, ee_pos, target, args):
    if args.direct_reference:
        problem.set_reference(np.repeat(np.asarray(target)[None, :], args.horizon + 1, axis=0))
        return
    waypoints = build_parabolic_waypoints(ee_pos, target, num_waypoints=args.num_waypoints)
    lengths = waypoint_arclengths(waypoints)
    distances = reference_horizon_distances(
        0.0,
        args.horizon,
        args.mpc_dt,
        args.ref_speed,
        total_distance=float(lengths[-1]),
    )
    refs = sample_waypoint_path(waypoints, lengths, distances)
    problem.set_reference(refs)


def move_target_towards(current, target, max_distance):
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    max_distance = float(max_distance)
    if max_distance <= 0.0:
        return target.copy()
    delta = target - current
    distance = float(np.linalg.norm(delta))
    if distance <= max_distance:
        return target.copy()
    return current + delta * (max_distance / distance)


def stage_target_speed(stage, args):
    if stage == "approach":
        return args.approach_target_speed
    if stage == "descend":
        return args.descend_target_speed
    if stage == "transport":
        return args.transport_target_speed
    if stage == "release":
        return args.release_target_speed
    if stage == "lift":
        return args.lift_target_speed
    if stage == "retreat":
        return args.retreat_target_speed
    return args.target_speed


def run(args):
    env = environment("ur10e")
    env.reset()
    robot = env.robot
    robot.open_gripper()
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
        viewer.opt.sitegroup[:] = 0
        viewer.sync()

    initial_cube = env.get_object_pos("cube").copy()
    min_cube_z = cube_min_center_z(env, args)
    if initial_cube[2] < min_cube_z:
        env.set_object_pose("cube", pos=supported_cube_pos(initial_cube, env, args))
        initial_cube = env.get_object_pos("cube").copy()
    approach_target = cube_top_target(env, args.approach_clearance, args)
    grasp_target = cube_center_target(env, args.grasp_z_offset, args)
    delivery_pos = delivery_cube_pos(initial_cube, args)
    transport_cube_pos = delivery_pos.copy()
    transport_cube_pos[2] = max(
        transport_cube_pos[2] + args.transport_clearance,
        initial_cube[2] + args.lift_z_offset,
        args.lift_height,
    )
    lift_box_pos = cube_lift_box_pos(initial_cube, args.lift_z_offset)
    lift_target = cube_center_target(env, args.lift_z_offset, args)
    retreat_target = np.array(
        [
            delivery_pos[0] + args.retreat_x_offset,
            delivery_pos[1] + args.retreat_y_offset,
            args.home_height,
        ],
        dtype=np.float64,
    )

    if args.start_above_cube:
        start_arm = ArmDynamics.from_robot(robot, dt=args.mpc_dt)
        q_start = solve_ik_position(
            start_arm,
            robot.joint_pos,
            approach_target,
            iterations=500,
            tol=args.start_ik_tol,
        )
        set_robot_joint_state(robot, q_start)

    arm, problem, solver, solver_name = make_solver(args, env)
    base_Qqf = problem.Qqf.copy()
    q_approach = solve_ik_position(
        arm,
        robot.joint_pos,
        approach_target,
        iterations=500,
        tol=args.start_ik_tol,
    )
    q_grasp = solve_ik_position(arm, q_approach, grasp_target)
    _, grasp_rot = arm.forward_kinematics(q_grasp)
    initial_cube_pose = env.get_object_pose("cube")
    delivery_pose = pose_with_position(initial_cube_pose, delivery_pos)
    manipulation = EndEffectorManipulationDynamics(
        initial_box_pose=initial_cube_pose,
        desired_box_pose=delivery_pose,
        desired_grasp_pose=make_pose(grasp_target, grasp_rot),
        lift_height=(
            args.lift_height
            if args.task_mode == "deliver"
            else initial_cube[2] + args.success_lift
        ),
        eps_grasp=args.grasp_pose_tol,
        eps_release=args.release_tol,
        gamma_max=args.gamma_max,
        friction_quality_eps=args.force_closure_eps,
    )
    lift_target = manipulation.ee_position_for_box_position(lift_box_pos)
    q_lift = solve_ik_position(arm, q_grasp, lift_target)
    dt = robot.model.opt.timestep
    mpc_every = max(int(round(args.mpc_dt / dt)), 1)
    current_tau = arm._clip_tau(arm.bias_for_state(arm.get_state()))
    current_tau = apply_optional_tau_limit(current_tau, args.apply_tau_limit)
    target_tau = current_tau.copy()
    problem.set_previous_tau(current_tau)

    stage = "open" if args.start_above_cube else "approach"
    stage_hold = 0
    target = approach_target.copy()
    reference_target = robot.ee_pos.copy()
    grasp_latched = False
    bilateral_contact_hold = 0
    released = False
    ee_to_cube = None
    metrics = []
    max_ineq = 0.0
    control_cost = 0.0
    last_diag = None
    path_progress = 0.0
    dummy_path = np.vstack([robot.ee_pos.copy(), target.copy()])
    dummy_s = waypoint_arclengths(dummy_path)

    for step in range(args.max_steps):
        cube = env.get_object_pos("cube")
        left_contact, right_contact = gripper_contacts(env)
        grasp_contact = left_contact and right_contact
        left_finger, right_finger, finger_mid, finger_aperture = gripper_geometry(env)
        finger_mid_error = float(np.linalg.norm(finger_mid - cube))
        (
            cube_enclosed,
            enclosure_axis_error,
            enclosure_perp_error,
            enclosure_vertical_error,
        ) = cube_enclosure_quality(
            env,
            left_finger,
            right_finger,
            cube,
            args,
        )
        cube_lift = float(cube[2] - initial_cube[2])
        if stage == "close" and grasp_contact:
            bilateral_contact_hold += 1
        else:
            bilateral_contact_hold = 0
        latch_contact_ready = (
            bilateral_contact_hold >= args.bilateral_contact_steps
            if args.require_bilateral_contact
            else grasp_contact
        )
        latch_enclosure_ready = args.allow_enclosure_latch and cube_enclosed
        if (
            stage == "close"
            and not grasp_latched
            and (latch_contact_ready or latch_enclosure_ready)
            and finger_aperture <= args.latch_aperture_threshold
        ):
            grasp_latched = True
            ee_to_cube = manipulation.attach(robot.ee_pose, env.get_object_pose("cube"))
            lift_target = manipulation.ee_position_for_box_position(lift_box_pos)

        approach_ready = (
            np.linalg.norm(robot.ee_pos - approach_target) <= args.reach_tol
            and np.linalg.norm(reference_target - approach_target) <= args.reach_tol
        )
        grasp_ready = (
            np.linalg.norm(robot.ee_pos - grasp_target) <= args.grasp_tol
            and np.linalg.norm(reference_target - grasp_target) <= args.grasp_tol
        )
        if stage == "open" and stage_hold >= args.open_steps:
            stage = "descend"
            stage_hold = 0
        elif stage == "approach" and approach_ready:
            stage = "descend"
            stage_hold = 0
        elif stage == "descend" and grasp_ready:
            stage = "close"
            stage_hold = 0
        elif stage == "close" and (
            grasp_latched
            and (
                finger_aperture <= args.grasp_aperture_threshold
                or stage_hold >= args.close_steps
            )
        ):
            stage = "lift" if args.task_mode == "lift" else "transport"
            stage_hold = 0
        elif stage == "lift" and cube_lift >= args.success_lift:
            break
        elif stage == "transport" and (
            grasp_latched
            and np.linalg.norm(cube - transport_cube_pos) <= args.transport_tol
            and cube[2] >= args.lift_height
        ):
            stage = "release"
            stage_hold = 0
        elif stage == "release" and released and stage_hold >= args.release_hold_steps:
            stage = "retreat"
            stage_hold = 0
        elif stage == "retreat" and (
            released
            and np.linalg.norm(robot.ee_pos - retreat_target) <= args.retreat_tol
            and robot.ee_pos[2] >= args.home_height - args.retreat_height_tol
        ):
            break

        if stage in {"open", "approach"}:
            problem.Qqf = base_Qqf
            robot.open_gripper()
            if args.track_cube_target:
                approach_target = cube_top_target(env, args.approach_clearance, args)
            target = approach_target
            problem.q_terminal = q_approach
        elif stage == "descend":
            problem.Qqf = base_Qqf
            robot.open_gripper()
            if args.track_cube_target:
                grasp_target = cube_center_target(env, args.grasp_z_offset, args)
            target = grasp_target
            problem.q_terminal = q_grasp
        elif stage == "close":
            problem.Qqf = base_Qqf
            set_gripper_fraction(robot, stage_hold / max(args.close_ramp_steps, 1))
            if args.track_cube_target:
                grasp_target = cube_center_target(env, args.grasp_z_offset, args)
            target = grasp_target
            problem.q_terminal = q_grasp
        elif stage == "transport":
            problem.Qqf = base_Qqf
            robot.close_gripper()
            target = (
                manipulation.ee_position_for_box_position(transport_cube_pos)
                if ee_to_cube is not None
                else lift_target
            )
            problem.q_terminal = q_lift
        elif stage == "lift":
            problem.Qqf = base_Qqf
            robot.close_gripper()
            if args.track_cube_target:
                lift_box_pos = cube_lift_box_pos(initial_cube, args.lift_z_offset)
                lift_target = manipulation.ee_position_for_box_position(lift_box_pos)
            target = lift_target
            problem.q_terminal = q_lift
        elif stage == "release":
            problem.Qqf = base_Qqf
            release_err = float(np.linalg.norm(cube - delivery_pos))
            if release_err <= args.release_tol:
                released = True
            if released:
                robot.open_gripper()
            else:
                robot.close_gripper()
            target = (
                manipulation.ee_position_for_box_position(delivery_pos)
                if ee_to_cube is not None
                else lift_target
            )
            problem.q_terminal = q_lift
        else:
            problem.Qqf = np.full(6, args.retreat_qf_weight, dtype=np.float64)
            robot.open_gripper()
            target = retreat_target
            problem.q_terminal = robot._home

        if step % mpc_every == 0:
            active_target_speed = stage_target_speed(stage, args)
            reference_target = move_target_towards(
                reference_target,
                target,
                active_target_speed * args.mpc_dt,
            )
            set_reference_to_target(problem, robot.ee_pos, reference_target, args)
            problem.set_previous_tau(current_tau)
            mpc_tau, _, diag = solver.step(arm.get_state())
            last_diag = diag
            max_ineq = max(max_ineq, diag.inequality_violation_after)
            desired_tau = arm._clip_tau(mpc_tau if not diag.fallback_used else current_tau)
            if args.motion_scale < 1.0:
                bias_tau = arm._clip_tau(arm.bias_for_state(arm.get_state()))
                desired_tau = bias_tau + float(args.motion_scale) * (desired_tau - bias_tau)
                desired_tau = arm._clip_tau(desired_tau)
            desired_tau = apply_optional_tau_limit(desired_tau, args.apply_tau_limit)
            target_tau = limit_torque_slew(
                desired_tau,
                current_tau,
                args.tau_slew_rate * args.mpc_dt,
            )

        current_tau = limit_torque_slew(target_tau, current_tau, args.tau_slew_rate * dt)
        control_cost += float(current_tau @ current_tau) * dt
        env.step(current_tau)
        if released:
            release_pos = supported_cube_pos(delivery_pos, env, args)
            env.set_object_pose(
                "cube",
                pos=release_pos,
                quat=rotmat_to_quat(delivery_pose[:3, :3]),
                min_center_z=min_cube_z,
            )
        elif grasp_latched and ee_to_cube is not None:
            cube_pose = supported_cube_pose(
                manipulation.attached_box_pose(robot.ee_pose),
                env,
                args,
            )
            env.set_object_pose(
                "cube",
                pos=cube_pose[:3, 3],
                quat=rotmat_to_quat(cube_pose[:3, :3]),
                min_center_z=min_cube_z,
            )
        else:
            free_cube_pos = env.get_object_pos("cube")
            if free_cube_pos[2] < min_cube_z:
                env.set_object_pose(
                    "cube",
                    pos=free_cube_pos,
                    quat=env.get_object_quat("cube"),
                    min_center_z=min_cube_z,
                )
        if viewer is not None:
            viewer.sync()
            time.sleep(dt)
        stage_hold += 1

        cube = env.get_object_pos("cube")
        left_contact, right_contact = gripper_contacts(env)
        grasp_contact = left_contact and right_contact
        cube_lift = float(cube[2] - initial_cube[2])
        post_target = target.copy()
        err = float(np.linalg.norm(robot.ee_pos - post_target))
        left_finger, right_finger, finger_mid, finger_aperture = gripper_geometry(env)
        finger_mid_error = float(np.linalg.norm(finger_mid - cube))
        (
            cube_enclosed,
            enclosure_axis_error,
            enclosure_perp_error,
            enclosure_vertical_error,
        ) = cube_enclosure_quality(
            env,
            left_finger,
            right_finger,
            cube,
            args,
        )
        table_clearance = env.object_table_clearance("cube", table_z=args.table_z)
        table_penetration = max(0.0, -table_clearance)
        manip_metric = manipulation.metrics(
            ee_pose=robot.ee_pose,
            box_pose=env.get_object_pose("cube"),
            stage=stage,
            stage_hold=stage_hold,
            close_ramp_steps=args.close_ramp_steps,
            released=released,
            left_contact=left_contact,
            right_contact=right_contact,
            finger_mid_error=finger_mid_error,
            finger_aperture=finger_aperture,
            max_aperture=max(args.latch_aperture_threshold, args.grasp_aperture_threshold),
        )
        manipulation_ineq = min(
            manipulation_constraint_margin(stage, manip_metric, args),
            table_clearance,
        )
        path_progress, _, _, _ = update_path_progress(
            robot.ee_pos,
            dummy_path,
            dummy_s,
            path_progress,
            dt,
            0.0,
            args.max_path_lead,
            args.waypoint_tracking_tol,
        )
        metrics.append(
            GraspMetric(
                step=step,
                stage=stage,
                ee_x=float(robot.ee_pos[0]),
                ee_y=float(robot.ee_pos[1]),
                ee_z=float(robot.ee_pos[2]),
                cube_x=float(cube[0]),
                cube_y=float(cube[1]),
                cube_z=float(cube[2]),
                target_x=float(post_target[0]),
                target_y=float(post_target[1]),
                target_z=float(post_target[2]),
                ee_error=err,
                cube_lift=cube_lift,
                left_contact=left_contact,
                right_contact=right_contact,
                grasp_contact=grasp_contact,
                mpc_status="" if last_diag is None else last_diag.qp_status,
                mpc_fallback=False if last_diag is None else bool(last_diag.fallback_used),
                mpc_ineq=0.0 if last_diag is None else float(last_diag.inequality_violation_after),
                tau_norm=float(np.linalg.norm(current_tau)),
                gripper_mean=float(np.mean(robot.gripper())) if len(robot._gripper_ids) else 0.0,
                left_finger_x=float(left_finger[0]),
                left_finger_y=float(left_finger[1]),
                left_finger_z=float(left_finger[2]),
                right_finger_x=float(right_finger[0]),
                right_finger_y=float(right_finger[1]),
                right_finger_z=float(right_finger[2]),
                finger_mid_x=float(finger_mid[0]),
                finger_mid_y=float(finger_mid[1]),
                finger_mid_z=float(finger_mid[2]),
                finger_aperture=finger_aperture,
                finger_mid_error=finger_mid_error,
                enclosure_axis_error=enclosure_axis_error,
                enclosure_perp_error=enclosure_perp_error,
                enclosure_vertical_error=enclosure_vertical_error,
                cube_enclosed=cube_enclosed,
                grasp_latched=grasp_latched,
                grasp_pose_error=manip_metric.grasp_pose_error,
                release_pose_error=manip_metric.release_pose_error,
                lift_margin=manip_metric.lift_margin,
                grasp_quality=manip_metric.grasp_quality,
                gamma=manip_metric.gamma,
                manipulation_ineq=manipulation_ineq,
                cube_table_clearance=table_clearance,
                cube_table_penetration=table_penetration,
            )
        )

    final_cube = env.get_object_pos("cube").copy()
    if viewer is not None:
        viewer.close()
    final_lift = float(final_cube[2] - initial_cube[2])
    delivery_error = float(np.linalg.norm(final_cube - delivery_pos))
    retreat_error = float(np.linalg.norm(robot.ee_pos - retreat_target))
    active_lift_margins = [
        m.lift_margin for m in metrics if m.stage in {"lift", "transport"}
    ]
    physical_bilateral_contact_any = any(
        m.left_contact and m.right_contact for m in metrics
    )
    two_finger_enclosure_any = any(m.cube_enclosed for m in metrics)
    valid_two_finger_grasp_any = physical_bilateral_contact_any or two_finger_enclosure_any
    max_table_penetration = max(
        (m.cube_table_penetration for m in metrics),
        default=0.0,
    )
    if args.task_mode == "lift":
        success = bool(
            metrics
            and final_lift >= args.success_lift
            and any(m.grasp_latched for m in metrics)
            and valid_two_finger_grasp_any
            and max_table_penetration <= args.table_penetration_tol
        )
    else:
        success = bool(
            metrics
            and released
            and delivery_error <= args.release_tol
            and retreat_error <= args.retreat_tol
            and robot.ee_pos[2] >= args.home_height - args.retreat_height_tol
            and any(m.grasp_latched for m in metrics)
            and valid_two_finger_grasp_any
            and max_table_penetration <= args.table_penetration_tol
        )
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "environment": "main MuJoCo environment",
        "robot": "ur10e",
        "solver": solver_name,
        "steps": len(metrics),
        "initial_cube_pos": initial_cube.tolist(),
        "final_cube_pos": final_cube.tolist(),
        "approach_target": approach_target.tolist(),
        "grasp_target": grasp_target.tolist(),
        "lift_target": lift_target.tolist(),
        "transport_cube_target": transport_cube_pos.tolist(),
        "delivery_cube_target": delivery_pos.tolist(),
        "retreat_target": retreat_target.tolist(),
        "desired_grasp_pose": manipulation.desired_grasp_pose.tolist(),
        "box_to_ee_grasp": manipulation.box_to_ee_grasp.tolist(),
        "manipulation_lift_height_m": manipulation.lift_height,
        "final_cube_lift_m": final_lift,
        "max_cube_lift_m": max((m.cube_lift for m in metrics), default=0.0),
        "final_delivery_error_m": delivery_error,
        "final_retreat_error_m": retreat_error,
        "min_ee_error_m": min((m.ee_error for m in metrics), default=float("inf")),
        "min_grasp_pose_error": min((m.grasp_pose_error for m in metrics), default=float("inf")),
        "final_release_pose_error": metrics[-1].release_pose_error if metrics else float("inf"),
        "min_active_lift_margin": min(active_lift_margins, default=0.0),
        "max_active_lift_margin": max(active_lift_margins, default=0.0),
        "max_grasp_quality": max((m.grasp_quality for m in metrics), default=-float("inf")),
        "grasp_quality_success_any": any(m.grasp_quality >= 0.0 for m in metrics),
        "min_cube_table_clearance_m": min(
            (m.cube_table_clearance for m in metrics),
            default=0.0,
        ),
        "max_cube_table_penetration_m": max_table_penetration,
        "min_manipulation_constraint_margin": min(
            (m.manipulation_ineq for m in metrics),
            default=0.0,
        ),
        "grasp_contact_any": any(m.grasp_contact for m in metrics),
        "physical_bilateral_contact_any": physical_bilateral_contact_any,
        "two_finger_enclosure_any": two_finger_enclosure_any,
        "valid_two_finger_grasp_any": valid_two_finger_grasp_any,
        "grasp_latched_any": any(m.grasp_latched for m in metrics),
        "released": released,
        "left_contact_any": any(m.left_contact for m in metrics),
        "right_contact_any": any(m.right_contact for m in metrics),
        "max_ineq_violation": max_ineq,
        "control_cost": control_cost,
        "success": success,
    }, metrics


def write_outputs(report, metrics, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"ur10e_acados_grasp_metrics_{stamp}.csv"
    json_path = out_dir / f"ur10e_acados_grasp_report_{stamp}.json"
    md_path = out_dir / f"ur10e_acados_grasp_report_{stamp}.md"
    plot_path = out_dir / f"ur10e_acados_grasp_plot_{stamp}.png"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(GraspMetric.__dataclass_fields__.keys()))
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.__dict__)
    if metrics:
        import matplotlib.pyplot as plt

        steps = [m.step for m in metrics]
        fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        axes[0].plot(steps, [m.ee_error for m in metrics], label="ee error")
        axes[0].set_ylabel("m")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(steps, [m.cube_lift for m in metrics], label="cube lift", color="tab:green")
        axes[1].axhline(report["manipulation_lift_height_m"] - report["initial_cube_pos"][2],
                        color="tab:gray", linestyle="--", linewidth=1)
        axes[1].set_ylabel("m")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(steps, [m.grasp_quality for m in metrics], label="grasp quality", color="tab:purple")
        axes[2].axhline(0.0, color="tab:gray", linestyle="--", linewidth=1)
        axes[2].set_ylabel("quality")
        axes[2].grid(True, alpha=0.3)
        axes[3].plot(steps, [m.tau_norm for m in metrics], label="tau norm", color="tab:red")
        axes[3].set_ylabel("Nm")
        axes[3].set_xlabel("simulation step")
        axes[3].grid(True, alpha=0.3)
        for ax in axes:
            ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        report["plot_png"] = plot_path.name
    else:
        report["plot_png"] = None
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# UR10e acados grasp report",
                "",
                f"- success: {report['success']}",
                f"- environment: {report['environment']}",
                f"- robot: {report['robot']}",
                f"- solver: {report['solver']}",
                f"- steps: {report['steps']}",
                f"- final cube lift m: {report['final_cube_lift_m']:.4f}",
                f"- max cube lift m: {report['max_cube_lift_m']:.4f}",
                f"- final delivery error m: {report['final_delivery_error_m']:.4f}",
                f"- final retreat error m: {report['final_retreat_error_m']:.4f}",
                f"- min ee error m: {report['min_ee_error_m']:.4f}",
                f"- min grasp pose error: {report['min_grasp_pose_error']:.4f}",
                f"- final release pose error: {report['final_release_pose_error']:.4f}",
                f"- min active lift margin: {report['min_active_lift_margin']:.4f}",
                f"- max active lift margin: {report['max_active_lift_margin']:.4f}",
                f"- max grasp quality: {report['max_grasp_quality']:.4f}",
                f"- grasp quality success any: {report['grasp_quality_success_any']}",
                f"- min cube table clearance m: {report['min_cube_table_clearance_m']:.4f}",
                f"- max cube table penetration m: {report['max_cube_table_penetration_m']:.6f}",
                f"- min manipulation constraint margin: {report['min_manipulation_constraint_margin']:.4f}",
                f"- control cost: {report['control_cost']:.4f}",
                f"- grasp contact any: {report['grasp_contact_any']}",
                f"- physical bilateral contact any: {report['physical_bilateral_contact_any']}",
                f"- two finger enclosure any: {report['two_finger_enclosure_any']}",
                f"- valid two finger grasp any: {report['valid_two_finger_grasp_any']}",
                f"- grasp latched any: {report['grasp_latched_any']}",
                f"- released: {report['released']}",
                f"- left contact any: {report['left_contact_any']}",
                f"- right contact any: {report['right_contact_any']}",
                f"- metrics csv: {csv_path.name}",
                f"- plot png: {report['plot_png']}",
                f"- raw json: {json_path.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description="UR10e main-environment cube grasp with acados MPC.")
    parser.add_argument("--task-mode", choices=["lift", "deliver"], default="lift")
    parser.add_argument("--start-above-cube", action="store_true")
    parser.add_argument("--start-ik-tol", type=float, default=1e-4)
    parser.add_argument("--open-steps", type=int, default=160)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--mpc-dt", type=float, default=0.04)
    parser.add_argument("--num-waypoints", type=int, default=30)
    parser.add_argument("--ref-speed", type=float, default=0.08)
    parser.add_argument("--target-speed", type=float, default=0.0)
    parser.add_argument("--approach-target-speed", type=float, default=0.0)
    parser.add_argument("--descend-target-speed", type=float, default=0.0)
    parser.add_argument("--transport-target-speed", type=float, default=0.0)
    parser.add_argument("--release-target-speed", type=float, default=0.0)
    parser.add_argument("--lift-target-speed", type=float, default=0.0)
    parser.add_argument("--retreat-target-speed", type=float, default=0.0)
    parser.add_argument("--direct-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--track-cube-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--approach-clearance", type=float, default=0.02)
    parser.add_argument("--grasp-z-offset", type=float, default=0.0)
    parser.add_argument("--lift-z-offset", type=float, default=0.18)
    parser.add_argument("--target-x-offset", type=float, default=0.0)
    parser.add_argument("--target-y-offset", type=float, default=0.0)
    parser.add_argument("--target-z-offset", type=float, default=0.0)
    parser.add_argument("--delivery-x-offset", type=float, default=0.12)
    parser.add_argument("--delivery-y-offset", type=float, default=-0.08)
    parser.add_argument("--delivery-z-offset", type=float, default=0.0)
    parser.add_argument("--transport-clearance", type=float, default=0.08)
    parser.add_argument("--lift-height", type=float, default=0.62)
    parser.add_argument("--home-height", type=float, default=0.78)
    parser.add_argument("--retreat-x-offset", type=float, default=0.0)
    parser.add_argument("--retreat-y-offset", type=float, default=-0.16)
    parser.add_argument("--success-lift", type=float, default=0.10)
    parser.add_argument("--transport-tol", type=float, default=0.035)
    parser.add_argument("--release-tol", type=float, default=0.035)
    parser.add_argument("--retreat-tol", type=float, default=0.06)
    parser.add_argument("--retreat-height-tol", type=float, default=0.04)
    parser.add_argument("--retreat-qf-weight", type=float, default=0.2)
    parser.add_argument("--release-hold-steps", type=int, default=120)
    parser.add_argument("--reach-tol", type=float, default=0.04)
    parser.add_argument("--grasp-tol", type=float, default=0.035)
    parser.add_argument("--close-steps", type=int, default=250)
    parser.add_argument("--close-ramp-steps", type=int, default=220)
    parser.add_argument("--grasp-aperture-threshold", type=float, default=0.075)
    parser.add_argument("--latch-aperture-threshold", type=float, default=0.12)
    parser.add_argument("--grasp-latch-distance", type=float, default=0.045)
    parser.add_argument("--require-bilateral-contact", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bilateral-contact-steps", type=int, default=3)
    parser.add_argument("--allow-enclosure-latch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enclosure-axis-tol", type=float, default=0.018)
    parser.add_argument("--enclosure-perp-tol", type=float, default=0.075)
    parser.add_argument("--enclosure-vertical-tol", type=float, default=0.040)
    parser.add_argument("--enclosure-aperture-slack", type=float, default=0.070)
    parser.add_argument("--grasp-pose-tol", type=float, default=0.12)
    parser.add_argument("--gamma-max", type=float, default=1.0)
    parser.add_argument("--force-closure-eps", type=float, default=0.05)
    parser.add_argument("--table-z", type=float, default=0.0)
    parser.add_argument("--table-clearance", type=float, default=0.001)
    parser.add_argument("--table-penetration-tol", type=float, default=1e-6)
    parser.add_argument("--ee-pos-weight", type=float, default=220.0)
    parser.add_argument("--ee-z-weight", type=float, default=260.0)
    parser.add_argument("--ee-terminal-weight", type=float, default=450.0)
    parser.add_argument("--ee-terminal-z-weight", type=float, default=520.0)
    parser.add_argument("--ee-upright-weight", type=float, default=8.0)
    parser.add_argument("--ee-terminal-upright-weight", type=float, default=20.0)
    parser.add_argument("--q-weight", type=float, default=0.6)
    parser.add_argument("--qv-weight", type=float, default=0.04)
    parser.add_argument("--qf-weight", type=float, default=1.5)
    parser.add_argument("--qvf-weight", type=float, default=0.08)
    parser.add_argument("--delta-tau-cost", type=float, default=0.05)
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--delta-q-max", type=float, default=0.08)
    parser.add_argument("--delta-dq-max", type=float, default=0.35)
    parser.add_argument("--delta-tau-max", type=float, default=22.0)
    parser.add_argument("--tau-slew-rate", type=float, default=600.0)
    parser.add_argument("--apply-tau-limit", type=float, default=0.0)
    parser.add_argument("--max-path-lead", type=float, default=0.08)
    parser.add_argument("--waypoint-tracking-tol", type=float, default=0.08)
    parser.add_argument("--solver", choices=["auto", "acados", "osqp"], default="auto")
    parser.add_argument("--osqp-max-iter", type=int, default=2000)
    parser.add_argument("--acados-export-dir", default=str(Path("acados_generated") / "ur10e_rti_grasp"))
    parser.add_argument("--acados-qp-solver", default="PARTIAL_CONDENSING_HPIPM")
    parser.add_argument("--acados-qp-solver-iter-max", type=int, default=200)
    parser.add_argument("--acados-nlp-solver-type", default="SQP_RTI")
    parser.add_argument("--regularization", type=float, default=1e-8)
    parser.add_argument("--acados-verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments") / "ur10e_acados_grasp")
    parser.add_argument("--viewer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report, metrics = run(args)
    csv_path, json_path, md_path = write_outputs(report, metrics, args.out_dir)
    print(json.dumps(report, indent=2))
    print(f"metrics_csv={csv_path}")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
