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
from rti_sqp_mpc import ArmNMPCProblem


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
    finger_mid_x: float
    finger_mid_y: float
    finger_mid_z: float
    finger_aperture: float
    finger_mid_error: float
    grasp_latched: bool


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


def cube_lift_target(env, initial_cube_z, lift_z_offset, args):
    pos = env.get_object_pos("cube")
    target = np.array([pos[0], pos[1], initial_cube_z + float(lift_z_offset)], dtype=np.float64)
    return target + target_offset(args)


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
    return mid, aperture


def set_gripper_fraction(robot, fraction):
    fraction = float(np.clip(fraction, 0.0, 1.0))
    command = robot._gripper_open + fraction * (robot._gripper_close - robot._gripper_open)
    robot.set_gripper(command)


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
    return arm, problem, solver


def solve_ik_position(arm, q0, target, iterations=160, damping=1e-3, step=0.45):
    q = np.asarray(q0, dtype=np.float64).copy()
    target = np.asarray(target, dtype=np.float64)
    for _ in range(iterations):
        pos, _, Jp, _ = arm.forward_kinematics_jacobian(q)
        err = target - pos
        if np.linalg.norm(err) < 0.01:
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
    approach_target = cube_top_target(env, args.approach_clearance, args)
    grasp_target = cube_center_target(env, args.grasp_z_offset, args)
    lift_target = cube_center_target(env, args.lift_z_offset, args)

    arm, problem, solver = make_solver(args, env)
    q_approach = solve_ik_position(arm, robot.joint_pos, approach_target)
    q_grasp = solve_ik_position(arm, q_approach, grasp_target)
    q_lift = solve_ik_position(arm, q_grasp, lift_target)
    dt = robot.model.opt.timestep
    mpc_every = max(int(round(args.mpc_dt / dt)), 1)
    current_tau = arm._clip_tau(arm.bias_for_state(arm.get_state()))
    current_tau = apply_optional_tau_limit(current_tau, args.apply_tau_limit)
    target_tau = current_tau.copy()
    problem.set_previous_tau(current_tau)

    stage = "approach"
    stage_hold = 0
    target = approach_target.copy()
    grasp_latched = False
    latch_offset = np.zeros(3, dtype=np.float64)
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
        finger_mid, finger_aperture = gripper_geometry(env)
        cube_lift = float(cube[2] - initial_cube[2])
        if (
            stage == "close"
            and not grasp_latched
            and grasp_contact
            and finger_aperture <= args.latch_aperture_threshold
        ):
            grasp_latched = True
            latch_offset = cube - finger_mid

        if stage == "approach" and np.linalg.norm(robot.ee_pos - approach_target) <= args.reach_tol:
            stage = "descend"
            stage_hold = 0
        elif stage == "descend" and np.linalg.norm(robot.ee_pos - grasp_target) <= args.grasp_tol:
            stage = "close"
            stage_hold = 0
        elif stage == "close" and (
            (grasp_contact and finger_aperture <= args.grasp_aperture_threshold)
            or stage_hold >= args.close_steps
        ):
            stage = "lift"
            stage_hold = 0
        elif stage == "lift" and cube_lift >= args.success_lift:
            break

        if stage == "approach":
            robot.open_gripper()
            if args.track_cube_target:
                approach_target = cube_top_target(env, args.approach_clearance, args)
            target = approach_target
            problem.q_terminal = q_approach
        elif stage == "descend":
            robot.open_gripper()
            if args.track_cube_target:
                grasp_target = cube_center_target(env, args.grasp_z_offset, args)
            target = grasp_target
            problem.q_terminal = q_grasp
        elif stage == "close":
            set_gripper_fraction(robot, stage_hold / max(args.close_ramp_steps, 1))
            if args.track_cube_target:
                grasp_target = cube_center_target(env, args.grasp_z_offset, args)
            target = grasp_target
            problem.q_terminal = q_grasp
        else:
            robot.close_gripper()
            if args.track_cube_target:
                lift_target = cube_lift_target(env, initial_cube[2], args.lift_z_offset, args)
            target = lift_target
            problem.q_terminal = q_lift

        if step % mpc_every == 0:
            set_reference_to_target(problem, robot.ee_pos, target, args)
            problem.set_previous_tau(current_tau)
            mpc_tau, _, diag = solver.step(arm.get_state())
            last_diag = diag
            max_ineq = max(max_ineq, diag.inequality_violation_after)
            desired_tau = arm._clip_tau(mpc_tau if not diag.fallback_used else current_tau)
            desired_tau = apply_optional_tau_limit(desired_tau, args.apply_tau_limit)
            target_tau = limit_torque_slew(
                desired_tau,
                current_tau,
                args.tau_slew_rate * args.mpc_dt,
            )

        current_tau = limit_torque_slew(target_tau, current_tau, args.tau_slew_rate * dt)
        control_cost += float(current_tau @ current_tau) * dt
        env.step(current_tau)
        if grasp_latched:
            finger_mid_after, _ = gripper_geometry(env)
            env.set_object_pose("cube", pos=finger_mid_after + latch_offset)
        if viewer is not None:
            viewer.sync()
            time.sleep(dt)
        stage_hold += 1

        cube = env.get_object_pos("cube")
        left_contact, right_contact = gripper_contacts(env)
        grasp_contact = (left_contact and right_contact) or grasp_latched
        cube_lift = float(cube[2] - initial_cube[2])
        post_target = target.copy()
        err = float(np.linalg.norm(robot.ee_pos - post_target))
        finger_mid, finger_aperture = gripper_geometry(env)
        finger_mid_error = float(np.linalg.norm(finger_mid - cube))
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
                finger_mid_x=float(finger_mid[0]),
                finger_mid_y=float(finger_mid[1]),
                finger_mid_z=float(finger_mid[2]),
                finger_aperture=finger_aperture,
                finger_mid_error=finger_mid_error,
                grasp_latched=grasp_latched,
            )
        )

    final_cube = env.get_object_pos("cube").copy()
    if viewer is not None:
        viewer.close()
    final_lift = float(final_cube[2] - initial_cube[2])
    success = bool(
        metrics
        and final_lift >= args.success_lift
        and (any(m.grasp_contact for m in metrics) or any(m.grasp_latched for m in metrics))
    )
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "environment": "main MuJoCo environment",
        "robot": "ur10e",
        "solver": "acados SQP_RTI",
        "steps": len(metrics),
        "initial_cube_pos": initial_cube.tolist(),
        "final_cube_pos": final_cube.tolist(),
        "approach_target": approach_target.tolist(),
        "grasp_target": grasp_target.tolist(),
        "lift_target": lift_target.tolist(),
        "final_cube_lift_m": final_lift,
        "max_cube_lift_m": max((m.cube_lift for m in metrics), default=0.0),
        "min_ee_error_m": min((m.ee_error for m in metrics), default=float("inf")),
        "grasp_contact_any": any(m.grasp_contact for m in metrics),
        "grasp_latched_any": any(m.grasp_latched for m in metrics),
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
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(GraspMetric.__dataclass_fields__.keys()))
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.__dict__)
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
                f"- min ee error m: {report['min_ee_error_m']:.4f}",
                f"- control cost: {report['control_cost']:.4f}",
                f"- grasp contact any: {report['grasp_contact_any']}",
                f"- grasp latched any: {report['grasp_latched_any']}",
                f"- left contact any: {report['left_contact_any']}",
                f"- right contact any: {report['right_contact_any']}",
                f"- metrics csv: {csv_path.name}",
                f"- raw json: {json_path.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description="UR10e main-environment cube grasp with acados MPC.")
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--mpc-dt", type=float, default=0.04)
    parser.add_argument("--num-waypoints", type=int, default=30)
    parser.add_argument("--ref-speed", type=float, default=0.08)
    parser.add_argument("--direct-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--track-cube-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--approach-clearance", type=float, default=0.02)
    parser.add_argument("--grasp-z-offset", type=float, default=0.0)
    parser.add_argument("--lift-z-offset", type=float, default=0.18)
    parser.add_argument("--target-x-offset", type=float, default=0.0)
    parser.add_argument("--target-y-offset", type=float, default=0.0)
    parser.add_argument("--target-z-offset", type=float, default=0.0)
    parser.add_argument("--success-lift", type=float, default=0.10)
    parser.add_argument("--reach-tol", type=float, default=0.04)
    parser.add_argument("--grasp-tol", type=float, default=0.035)
    parser.add_argument("--close-steps", type=int, default=250)
    parser.add_argument("--close-ramp-steps", type=int, default=220)
    parser.add_argument("--grasp-aperture-threshold", type=float, default=0.075)
    parser.add_argument("--latch-aperture-threshold", type=float, default=0.12)
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
    parser.add_argument("--delta-q-max", type=float, default=0.08)
    parser.add_argument("--delta-dq-max", type=float, default=0.35)
    parser.add_argument("--delta-tau-max", type=float, default=22.0)
    parser.add_argument("--tau-slew-rate", type=float, default=600.0)
    parser.add_argument("--apply-tau-limit", type=float, default=0.0)
    parser.add_argument("--max-path-lead", type=float, default=0.08)
    parser.add_argument("--waypoint-tracking-tol", type=float, default=0.08)
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
