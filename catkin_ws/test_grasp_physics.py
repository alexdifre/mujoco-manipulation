#!/usr/bin/env python3
"""
Physical grasp test for the UR10e + 2F-85 pipeline (no kinematic attach).

Runs approach -> descend -> close -> lift -> hold with the same MPC stack
as run_ur10e_acados_grasp.py and prints compact per-stage diagnostics
(contacts, enclosure, force closure, friction cone).

Usage:
  cd catkin_ws && python3 test_grasp_physics.py [--viewer]
"""
import argparse
import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import mujoco
import numpy as np

mj = mujoco

sys.path.insert(0, os.path.dirname(__file__))

from run_ur10e_acados_grasp import (
    apply_optional_tau_limit,
    cube_center_target,
    cube_enclosure_quality,
    cube_lift_box_pos,
    cube_min_center_z,
    cube_top_target,
    delivery_cube_pos,
    gripper_geometry,
    limit_torque_slew,
    make_pose,
    make_solver,
    move_target_towards,
    parse_args as main_parse_args,
    pose_with_position,
    set_gripper_fraction,
    set_reference_to_target,
    solve_ik_position,
    stage_target_speed,
    supported_cube_pos,
)
from arm_dynamics import ArmDynamics
from environment import environment
from grasp_phase_mpc import ContactSample, PhaseGraspMPC
from unified_grasp_mpc import UnifiedGraspMPC
from manipulation_dynamics import EndEffectorManipulationDynamics, make_pose, pose_with_position


# ── 6-DOF IK solver ──────────────────────────────────────────────────────────


def solve_ik_finger_mid(arm, q0, target_pos, target_rot,
                        iterations=2000, damping=1e-4, step=0.5,
                        pos_tol=0.002, rot_tol=0.03):
    """Solve IK directly for finger_mid position + EE orientation.

    Two-phase approach:
    1. Position-only IK (ignore orientation) — get finger_mid to target
    2. Full 6-DOF IK — refine with orientation
    """
    q = np.asarray(q0, dtype=np.float64).copy()
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_rot = np.asarray(target_rot, dtype=np.float64)

    model = arm.model
    data = arm.data
    left_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left_inner_finger")
    right_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_inner_finger")

    def compute_finger_mid_and_ee_rot(q_test):
        with arm._preserve_state():
            data.qpos[arm.qpos_addr] = q_test
            data.qvel[arm.dof_addr] = 0.0
            mj.mj_forward(model, data)
            lp = data.xpos[left_body_id].copy()
            rp = data.xpos[right_body_id].copy()
            fm = 0.5 * (lp + rp)
            if arm.ee_site_id >= 0:
                ee_rot = data.site_xmat[arm.ee_site_id].reshape(3, 3).copy()
                ee_body_id = int(model.site_bodyid[arm.ee_site_id])
            else:
                ee_rot = data.xmat[arm.ee_body_id].reshape(3, 3).copy()
                ee_body_id = arm.ee_body_id
        return fm, ee_rot, lp, rp, ee_body_id

    # Phase 1: Position-only IK (3-DOF, ignore orientation)
    for it in range(iterations // 2):
        fm, ee_rot, lp, rp, ee_body_id = compute_finger_mid_and_ee_rot(q)
        err_pos = target_pos - fm
        if np.linalg.norm(err_pos) < pos_tol:
            break

        # Compute position Jacobian only
        with arm._preserve_state():
            data.qpos[arm.qpos_addr] = q
            data.qvel[arm.dof_addr] = 0.0
            mj.mj_forward(model, data)
            jacp_left = np.zeros((3, model.nv), dtype=np.float64)
            jacp_right = np.zeros((3, model.nv), dtype=np.float64)
            mj.mj_jac(model, data, jacp_left, None, lp, left_body_id)
            mj.mj_jac(model, data, jacp_right, None, rp, right_body_id)

        Jp_mid = 0.5 * (jacp_left[:, arm.dof_addr] + jacp_right[:, arm.dof_addr])
        # Damped least squares for position only (3x6 -> 3x3)
        A = Jp_mid @ Jp_mid.T + damping * np.eye(3)
        dq = Jp_mid.T @ np.linalg.solve(A, err_pos)
        q = q + step * dq
        q = np.minimum(np.maximum(q, arm.q_min), arm.q_max)

    # Phase 2: Full 6-DOF IK (position + orientation)
    for it in range(iterations // 2):
        fm, ee_rot, lp, rp, ee_body_id = compute_finger_mid_and_ee_rot(q)
        err_pos = target_pos - fm

        # Orientation error
        R_err = target_rot @ ee_rot.T
        trace = np.clip(R_err[0, 0] + R_err[1, 1] + R_err[2, 2], -1, 3)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        if abs(angle) < 1e-8:
            err_rot = np.zeros(3)
        else:
            axis = np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ]) / (2 * np.sin(angle))
            err_rot = angle * axis

        err = np.concatenate([err_pos, err_rot])
        pos_ok = np.linalg.norm(err_pos) < pos_tol
        rot_ok = np.linalg.norm(err_rot) < rot_tol
        if pos_ok and rot_ok:
            break

        # Compute full Jacobian
        with arm._preserve_state():
            data.qpos[arm.qpos_addr] = q
            data.qvel[arm.dof_addr] = 0.0
            mj.mj_forward(model, data)
            jacp_left = np.zeros((3, model.nv), dtype=np.float64)
            jacp_right = np.zeros((3, model.nv), dtype=np.float64)
            jacr_ee = np.zeros((3, model.nv), dtype=np.float64)
            mj.mj_jac(model, data, jacp_left, None, lp, left_body_id)
            mj.mj_jac(model, data, jacp_right, None, rp, right_body_id)
            mj.mj_jac(model, data, None, jacr_ee, data.xpos[ee_body_id], ee_body_id)

        Jp_mid = 0.5 * (jacp_left[:, arm.dof_addr] + jacp_right[:, arm.dof_addr])
        Jr_ee = jacr_ee[:, arm.dof_addr]
        J = np.vstack([Jp_mid, Jr_ee])
        A = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(A, err)
        q = q + step * dq
        q = np.minimum(np.maximum(q, arm.q_min), arm.q_max)

    fm_final, _, _, _, _ = compute_finger_mid_and_ee_rot(q)
    final_pos_err = np.linalg.norm(fm_final - target_pos)
    return q, final_pos_err < pos_tol, it + 1


def solve_ik_pose(arm, q0, target_pos, target_rot,
                  iterations=500, damping=1e-3, step=0.4,
                  pos_tol=0.005, rot_tol=0.05,
                  q_bias=None, bias_weight=0.0):
    """Solve IK for both position and orientation using damped least squares.

    target_pos: (3,) desired EE position
    target_rot: (3,3) desired EE rotation matrix
    """
    q = np.asarray(q0, dtype=np.float64).copy()
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_rot = np.asarray(target_rot, dtype=np.float64)
    q_bias = None if q_bias is None else np.asarray(q_bias, dtype=np.float64)

    for it in range(iterations):
        pos, rot, Jp, Jr = arm.forward_kinematics_jacobian(q)

        # Position error
        err_pos = target_pos - pos

        # Orientation error (axis-angle from R_err = target @ current.T)
        R_err = target_rot @ rot.T
        # Extract axis-angle from rotation matrix
        trace = np.clip(R_err[0, 0] + R_err[1, 1] + R_err[2, 2], -1, 3)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        if abs(angle) < 1e-8:
            err_rot = np.zeros(3)
        else:
            axis = np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ]) / (2 * np.sin(angle))
            err_rot = angle * axis

        err = np.concatenate([err_pos, err_rot])
        pos_ok = np.linalg.norm(err_pos) < pos_tol
        rot_ok = np.linalg.norm(err_rot) < rot_tol
        if pos_ok and rot_ok:
            break

        # Stack Jacobians: [Jp; Jr] → 6×6
        J = np.vstack([Jp, Jr])
        A = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(A, err)

        if q_bias is not None and bias_weight > 0.0:
            dq = dq + bias_weight * (q_bias - q)

        q = q + step * dq
        q = np.minimum(np.maximum(q, arm.q_min), arm.q_max)

    return q, pos_ok and rot_ok, it + 1


def compute_grasp_orientation(cube_pos, approach_from="front"):
    """Compute desired gripper orientation for a side-grasp.

    Finger spread axis (local x) → horizontal, perpendicular to approach dir
    Approach axis (local z) → horizontal, along approach direction
    """
    # We want the fingers to approach from the +y direction (front of robot)
    # and spread along the x-axis (left-right)
    # 
    # Desired frame:
    #   x_local = [1, 0, 0]  (finger spread, horizontal left-right)
    #   y_local = [0, 0, -1] (perpendicular to spread, pointing down)
    #   z_local = [0, 1, 0]  (approach direction, from front)
    #
    # But we need to figure out the actual EE frame convention.
    # Let's use: z_local = approach direction, x_local = finger spread
    if approach_from == "front":
        z_local = np.array([0.0, 1.0, 0.0])
        x_local = np.array([1.0, 0.0, 0.0])
    elif approach_from == "side":
        z_local = np.array([1.0, 0.0, 0.0])
        x_local = np.array([0.0, 1.0, 0.0])
    else:
        z_local = np.array([0.0, 1.0, 0.0])
        x_local = np.array([1.0, 0.0, 0.0])

    y_local = np.cross(z_local, x_local)
    y_local /= np.linalg.norm(y_local)
    x_local = np.cross(y_local, z_local)
    x_local /= np.linalg.norm(x_local)

    R = np.column_stack([x_local, y_local, z_local])
    return R


# ── Contact analysis ─────────────────────────────────────────────────────────

def verify_gripper_collision_geometry(env):
    """Print the collision geom actually used by each gripper part.

    The knuckle vhacd files are fat slabs (inner: 69x68x25 mm vs the real
    13x25x91 mm bar) that invisibly block the grasp region. After the model
    fix the knuckles should collide via the thin visual-mesh hulls instead.
    """
    model, data = env.robot.model, env.robot.data
    ee_z = env.robot.ee_pos[2]
    print("\n──── Gripper collision geometry (z relative to TCP) ────")
    for part in ("left_inner_finger", "right_inner_finger",
                 "left_inner_knuckle", "right_inner_knuckle",
                 "left_outer_knuckle", "right_outer_knuckle"):
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, part)
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != bid:
                continue
            if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
                continue
            mid_ = int(model.geom_dataid[gid])
            mesh_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_MESH, mid_) or "?"
            vs, vn = model.mesh_vertadr[mid_], model.mesh_vertnum[mid_]
            v = model.mesh_vert[vs:vs + vn]
            R = data.geom_xmat[gid].reshape(3, 3)
            w = v @ R.T + data.geom_xpos[gid]
            ext = v.max(axis=0) - v.min(axis=0)
            flag = ("  <-- FAT VHACD SLAB (blocks grasp!)"
                    if "knuckle" in part and mesh_name.endswith("_vhacd") else "")
            print(f"  {part:20s} mesh={mesh_name:24s} "
                  f"size=[{ext[0]:.3f} {ext[1]:.3f} {ext[2]:.3f}] "
                  f"z_rel_tcp=[{w[:, 2].min() - ee_z:+.4f},{w[:, 2].max() - ee_z:+.4f}]{flag}")
    print()


def get_contacts_with_cube(env):
    model = env.robot.model
    data = env.robot.data
    cube_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "cube_geom")
    cube_body = int(model.geom_bodyid[cube_gid])
    contacts = []
    f6 = np.zeros(6)
    for ci in range(data.ncon):
        c = data.contact[ci]
        if c.geom1 != cube_gid and c.geom2 != cube_gid:
            continue
        mj.mj_contactForce(model, data, ci, f6)
        if c.geom1 == cube_gid:
            other_gid, sign = c.geom2, -1.0
        else:
            other_gid, sign = c.geom1, 1.0
        normal = sign * c.frame[:3].copy()
        other_body = int(model.geom_bodyid[other_gid])
        jac_other = np.zeros((3, model.nv), dtype=np.float64)
        jac_cube = np.zeros((3, model.nv), dtype=np.float64)
        mj.mj_jac(model, data, jac_other, None, c.pos, other_body)
        mj.mj_jac(model, data, jac_cube, None, c.pos, cube_body)
        relative_velocity = (jac_other - jac_cube) @ data.qvel
        tangential_velocity = relative_velocity - normal * float(relative_velocity @ normal)
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, other_body)
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, other_gid)
        contacts.append({
            "index": ci, "f6": f6.copy(), "fn": float(f6[0]),
            "ft": f6[1:3].copy(), "pos": c.pos.copy(), "normal": normal,
            "other_body": other_body,
            "other_body_name": body_name,
            "other_geom_name": geom_name,
            "slip_speed": float(np.linalg.norm(tangential_velocity)),
        })
    return contacts


def classify_finger(body_id, left_bodies, right_bodies):
    if body_id in left_bodies:
        return "left"
    if body_id in right_bodies:
        return "right"
    return "other"


def gripper_close_fraction(robot):
    """Normalized position-servo command: 0=open, 1=closed."""
    command = robot.gripper()
    span = robot._gripper_close - robot._gripper_open
    valid = np.abs(span) > 1e-12
    if not np.any(valid):
        return 0.0
    fraction = (command[valid] - robot._gripper_open[valid]) / span[valid]
    return float(np.clip(np.mean(fraction), 0.0, 1.0))


def finger_contact_samples(env, left_bodies, right_bodies):
    samples = []
    for contact in get_contacts_with_cube(env):
        side = classify_finger(contact["other_body"], left_bodies, right_bodies)
        if side not in {"left", "right"}:
            continue
        samples.append(ContactSample(
            side=side,
            position=contact["pos"],
            normal=contact["normal"],
            normal_force=contact["fn"],
            tangential_force=contact["ft"],
            slip_speed=contact["slip_speed"],
        ))
    return samples


def print_diagnostics(env, stage_label, left_bodies, right_bodies, grasp_decision=None):
    """Compact grasp-quality snapshot; returns the metrics dict."""
    MU = 1.0  # cube sliding friction — keep in sync with environment.py
    cube = env.get_object_pos("cube")
    contacts = get_contacts_with_cube(env)

    left_contact = False
    right_contact = False
    finger_contacts = []
    other_names = []
    for c in contacts:
        side = classify_finger(c["other_body"], left_bodies, right_bodies)
        if side == "left":
            left_contact = True
        elif side == "right":
            right_contact = True
        if side in ("left", "right"):
            finger_contacts.append(c)
        else:
            other_names.append(c["other_body_name"] or "?")
    bilateral = left_contact and right_contact

    # Enclosure — same pad-center geometry as the latch logic
    lp, rp, mid, aperture = gripper_geometry(env)
    axis = (rp - lp) / (aperture + 1e-12)
    rel = cube - mid
    axis_err = float(abs(rel @ axis))
    perp_err = float(np.linalg.norm(rel - (rel @ axis) * axis))
    vert_err = float(abs(cube[2] - mid[2]))
    cube_r = float(np.linalg.norm(env.object_half_extents("cube")))
    # vert tol 0.08: finger_mid uses the pad body origins (pad TOPS, at TCP
    # height), which sit ~60 mm above the cube center in this grasp pose.
    enclosed = (axis_err <= 0.018 and perp_err <= 0.075
                and vert_err <= 0.080
                and aperture <= 2.0 * cube_r + 0.070)

    # Force closure from friction-cone rays of the finger contacts
    rays = []
    for c in finger_contacts:
        if c["fn"] <= 0.0:
            continue
        n = c["normal"]
        p = c["pos"] - cube
        t1 = np.cross(n, [0, 0, 1]) if abs(n[2]) < 0.9 else np.cross(n, [1, 0, 0])
        t1 /= np.linalg.norm(t1) + 1e-12
        t2 = np.cross(n, t1)
        t2 /= np.linalg.norm(t2) + 1e-12
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            d = n + MU * (np.cos(angle) * t1 + np.sin(angle) * t2)
            d /= np.linalg.norm(d) + 1e-12
            rays.append(np.concatenate([d, np.cross(p, d)]))
    G = np.array(rays, dtype=np.float64).T if rays else np.zeros((6, 0))
    sigma_min = float(np.linalg.svd(G, compute_uv=False)[-1]) if G.shape[1] >= 6 else 0.0
    force_closure = sigma_min >= 0.05
    wrench_feasible = bool(
        grasp_decision is not None and grasp_decision.wrench.feasible
    )
    wrench_evaluated = grasp_decision is not None
    wrench_label = "PASS" if wrench_feasible else ("FAIL" if wrench_evaluated else "N/A")

    all_in_cone = all(
        np.linalg.norm(c["ft"]) / abs(c["fn"]) <= MU
        for c in finger_contacts if abs(c["fn"]) > 1e-6
    ) if finger_contacts else False

    print(f"\n── {stage_label} ──")
    print(f"  cube=[{cube[0]:+.3f} {cube[1]:+.3f} {cube[2]:+.3f}]  "
          f"pad_mid=[{mid[0]:+.3f} {mid[1]:+.3f} {mid[2]:+.3f}]  aperture={aperture:.3f}")
    for c in finger_contacts:
        mu_req = np.linalg.norm(c["ft"]) / max(abs(c["fn"]), 1e-9)
        print(f"    {c['other_body_name']:22s} Fn={abs(c['fn']):6.2f} N  mu_req={mu_req:5.2f}  "
              f"n=[{c['normal'][0]:+.2f} {c['normal'][1]:+.2f} {c['normal'][2]:+.2f}]")
    if other_names:
        print(f"    other contacts: {', '.join(sorted(set(other_names)))}")
    print(f"  bilateral {'PASS' if bilateral else 'FAIL'} | "
          f"enclosure {'PASS' if enclosed else 'FAIL'} "
          f"(axis {axis_err:.3f}, perp {perp_err:.3f}, vert {vert_err:.3f}) | "
          f"force-closure {'PASS' if force_closure else 'FAIL'} (sigma_min {sigma_min:.3f}) | "
          f"required-wrench {wrench_label} | "
          f"friction-cone {'PASS' if all_in_cone else 'FAIL'}")

    return {
        "bilateral": bilateral, "force_closure": force_closure,
        "wrench_feasible": wrench_feasible,
        "wrench_evaluated": wrench_evaluated,
        "enclosed": enclosed, "friction_cone": all_in_cone,
        "aperture": aperture, "axis_err": axis_err,
        "perp_err": perp_err, "vert_err": vert_err,
        "n_contacts": len(contacts), "n_finger_contacts": len(finger_contacts),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_test(args):
    env = environment("ur10e")
    env.reset()
    robot = env.robot
    robot.open_gripper()
    verify_gripper_collision_geometry(env)

    initial_cube = env.get_object_pos("cube").copy()
    min_cube_z = cube_min_center_z(env, args)
    if initial_cube[2] < min_cube_z:
        env.set_object_pose("cube", pos=supported_cube_pos(initial_cube, env, args))
        initial_cube = env.get_object_pos("cube").copy()

    approach_target = cube_top_target(env, args.approach_clearance + 0.03, args)
    grasp_target = cube_center_target(env, args.grasp_z_offset, args)
    print(f"cube=[{initial_cube[0]:+.3f} {initial_cube[1]:+.3f} {initial_cube[2]:+.3f}]  "
          f"approach_z={approach_target[2]:.3f}  grasp_z={grasp_target[2]:.3f} "
          f"(= cube center {args.grasp_z_offset:+.3f})")
    delivery_pos = delivery_cube_pos(initial_cube, args)

    model = robot.model
    left_bodies = {
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)
        for n in ("left_outer_knuckle", "left_inner_knuckle", "left_inner_finger")
    }
    right_bodies = {
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)
        for n in ("right_outer_knuckle", "right_inner_knuckle", "right_inner_finger")
    }
    cube_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "cube")
    cube_mass = float(model.body_mass[cube_body_id])
    _, cube_inertia = env.get_object_dynamics("cube")

    arm, problem, solver, solver_name = make_solver(args, env)
    grasp_mpc = PhaseGraspMPC(
        dt=args.mpc_dt,
        horizon=args.grasp_mpc_horizon,
        friction_coefficient=args.grasp_friction_coefficient,
        friction_margin=args.grasp_friction_margin,
        torsional_friction_coefficient=args.grasp_torsional_friction,
        normal_force_max=args.grasp_normal_force_max,
        preload_force=args.grasp_preload_force,
        contact_close_rate=args.grasp_contact_close_rate,
        close_rate_max=args.grasp_close_rate_max,
        max_slip_speed=args.grasp_slip_speed_max,
        lift_safety_factor=args.grasp_lift_safety_factor,
        stable_steps_required=args.grasp_stable_mpc_steps,
    )
    unified_grasp_mpc = UnifiedGraspMPC(
        arm=arm,
        dt=args.mpc_dt,
        horizon=args.unified_grasp_horizon,
        friction_coefficient=(
            args.grasp_friction_coefficient - args.grasp_friction_margin
        ),
        torsional_friction_coefficient=args.grasp_torsional_friction,
        normal_force_max=args.grasp_normal_force_max,
        close_rate_max=args.grasp_close_rate_max,
        max_slip_speed=args.grasp_slip_speed_max,
        table_z=args.table_z,
        object_half_height=env.object_half_height("cube"),
        max_iterations=args.unified_grasp_max_iterations,
    )

    # Position-only IK for the stage targets, then sanity-check that the pad
    # midpoint at q_grasp lands on the cube center.
    q_approach = solve_ik_position(arm, robot.joint_pos, approach_target, iterations=500, tol=0.01)
    q_grasp = solve_ik_position(arm, q_approach, grasp_target, iterations=500, tol=0.01)
    with arm._preserve_state():
        arm.data.qpos[arm.qpos_addr] = q_grasp
        arm.data.qvel[arm.dof_addr] = 0.0
        mj.mj_forward(arm.model, arm.data)
        lid = mj.mj_name2id(arm.model, mj.mjtObj.mjOBJ_BODY, "left_inner_finger")
        rid = mj.mj_name2id(arm.model, mj.mjtObj.mjOBJ_BODY, "right_inner_finger")
        fm = 0.5 * (arm.data.xpos[lid] + arm.data.xpos[rid])
    print(f"solver={solver_name}  finger_mid@q_grasp=[{fm[0]:+.3f} {fm[1]:+.3f} {fm[2]:+.3f}]  "
          f"(body origins = pad TOPS, ~30 mm above the pad faces)")

    grasp_target_adjusted = grasp_target

    # ── Set up manipulation dynamics ──
    base_Qqf = problem.Qqf.copy()
    _, grasp_rot = arm.forward_kinematics(q_grasp)
    initial_cube_pose = env.get_object_pose("cube")
    delivery_pose = pose_with_position(initial_cube_pose, delivery_pos)
    manipulation = EndEffectorManipulationDynamics(
        initial_box_pose=initial_cube_pose,
        desired_box_pose=delivery_pose,
        desired_grasp_pose=make_pose(grasp_target, grasp_rot),
        lift_height=initial_cube[2] + args.success_lift,
        eps_grasp=args.grasp_pose_tol,
        eps_release=args.release_tol,
        gamma_max=args.gamma_max,
        friction_quality_eps=args.force_closure_eps,
    )
    lift_box_pos = cube_lift_box_pos(initial_cube, args.lift_z_offset)
    lift_target = manipulation.ee_position_for_box_position(lift_box_pos)
    q_lift = solve_ik_position(arm, q_grasp, lift_target, iterations=500, tol=0.01)

    # ── Viewer ──
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
        viewer.opt.sitegroup[:] = 0
        viewer.sync()

    dt = robot.model.opt.timestep
    mpc_every = max(int(round(args.mpc_dt / dt)), 1)
    current_tau = arm._clip_tau(arm.bias_for_state(arm.get_state()))
    current_tau = apply_optional_tau_limit(current_tau, args.apply_tau_limit)
    target_tau = current_tau.copy()
    problem.set_previous_tau(current_tau)

    stage = "approach"
    stage_hold = 0
    target = approach_target.copy()
    reference_target = robot.ee_pos.copy()
    grasp_latched = False
    bilateral_contact_hold = 0
    ee_to_cube = None
    last_diag = None
    grasp_decision = None
    unified_decision = None
    hold_completed = False
    hold_cube_target = None
    diagnostics = {}

    for step in range(args.max_steps):
        cube = env.get_object_pos("cube")
        contact_samples = finger_contact_samples(env, left_bodies, right_bodies)
        left_contact = any(contact.side == "left" for contact in contact_samples)
        right_contact = any(contact.side == "right" for contact in contact_samples)
        grasp_contact = left_contact and right_contact
        left_finger, right_finger, finger_mid, finger_aperture = gripper_geometry(env)
        cube_lift = float(cube[2] - initial_cube[2])

        if stage == "contact" and grasp_contact:
            bilateral_contact_hold += 1
        else:
            bilateral_contact_hold = 0

        contact_ready = (
            bilateral_contact_hold >= args.bilateral_contact_steps
            if args.require_bilateral_contact else grasp_contact
        )
        cube_enclosed_flag = cube_enclosure_quality(env, left_finger, right_finger, cube, args)[0]
        enclosure_ready = cube_enclosed_flag if args.require_enclosure_for_latch else True

        approach_ready = (
            np.linalg.norm(robot.ee_pos - approach_target) <= args.reach_tol
            and np.linalg.norm(reference_target - approach_target) <= args.reach_tol
        )
        grasp_ready = (
            np.linalg.norm(robot.ee_pos - grasp_target_adjusted) <= args.grasp_tol
            and np.linalg.norm(reference_target - grasp_target_adjusted) <= args.grasp_tol
        )

        # Trajectory trace (watch for dip-below-then-rise before the grasp)
        if step % 100 == 0 and stage in ("approach", "descend", "contact", "preload"):
            print(f"  [{stage:8s} {step:4d}] ee_z={robot.ee_pos[2]:+.3f} -> tgt_z={target[2]:+.3f}  "
                  f"xy_err={np.linalg.norm(robot.ee_pos[:2] - target[:2]):.3f}  "
                  f"aperture={finger_aperture:.3f}")

        if stage == "approach" and approach_ready:
            stage = "descend"
            stage_hold = 0
            print(f">>> approach -> descend  (step {step})")
        elif stage == "descend" and grasp_ready:
            stage = "contact"
            stage_hold = 0
            print(f">>> descend -> contact MPC  (step {step})")
            diagnostics["at_close_start"] = print_diagnostics(
                env, f"CLOSE START (step {step})", left_bodies, right_bodies)
        elif (stage == "contact" and contact_ready and enclosure_ready
              and finger_aperture <= args.latch_aperture_threshold):
            stage = "preload"
            stage_hold = 0
            print(f">>> contact -> preload MPC  (step {step})")
        elif (
            stage == "preload"
            and grasp_decision is not None
            and grasp_decision.lift_ready
            and unified_decision is not None
            and unified_decision.optimizer_success
        ):
            grasp_latched = True
            ee_to_cube = manipulation.attach(robot.ee_pose, env.get_object_pose("cube"))
            lift_target = manipulation.ee_position_for_box_position(lift_box_pos)
            diagnostics["at_latch"] = print_diagnostics(
                env, f"GRASP MPC FEASIBLE (step {step})", left_bodies, right_bodies,
                grasp_decision=grasp_decision)
            stage = "lift"
            stage_hold = 0
            print(
                f">>> preload -> lift MPC  (step {step}, "
                f"Fn=({grasp_decision.actual_normal_by_side['left']:.1f},"
                f"{grasp_decision.actual_normal_by_side['right']:.1f}) N, "
                f"slip={grasp_decision.max_slip_speed:.4f} m/s)"
            )
            diagnostics["at_lift_start"] = print_diagnostics(
                env, f"LIFT START (step {step})", left_bodies, right_bodies,
                grasp_decision=grasp_decision)
        elif stage == "lift" and cube_lift >= args.success_lift:
            print(f">>> lift success  (step {step}, lift={cube_lift:.4f} m) — holding")
            hold_cube_target = cube.copy()
            lift_done_contacts = finger_contact_samples(env, left_bodies, right_bodies)
            grasp_decision = grasp_mpc.step(
                phase="lift",
                close_fraction=gripper_close_fraction(robot),
                contacts=lift_done_contacts,
                object_center=cube,
                object_mass=cube_mass,
            )
            diagnostics["at_lift_done"] = print_diagnostics(
                env, f"LIFT DONE (step {step})", left_bodies, right_bodies,
                grasp_decision=grasp_decision)
            stage = "hold"
            stage_hold = 0
            continue
        elif stage == "hold":
            if stage_hold % 200 == 0:
                cz = env.get_object_pos("cube")[2]
                print(f"  [hold {stage_hold:3d}] cube_z={cz:.4f}  above_table={'YES' if cz > 0.05 else 'NO'}")
            if stage_hold >= 500:  # ~10 s of pure physical holding
                cz = env.get_object_pos("cube")[2]
                print(f">>> hold complete — cube_z={cz:.4f} m, held physically: {'YES' if cz > 0.05 else 'NO'}")
                final_contacts = finger_contact_samples(env, left_bodies, right_bodies)
                grasp_decision = grasp_mpc.step(
                    phase="hold",
                    close_fraction=gripper_close_fraction(robot),
                    contacts=final_contacts,
                    object_center=env.get_object_pos("cube"),
                    object_mass=cube_mass,
                )
                diagnostics["at_hold_done"] = print_diagnostics(
                    env, f"HOLD DONE (step {step})", left_bodies, right_bodies,
                    grasp_decision=grasp_decision)
                hold_completed = cz > 0.05
                break

        if stage in {"approach"}:
            problem.Qqf = base_Qqf
            robot.open_gripper()
            target = approach_target
            problem.q_terminal = q_approach
        elif stage == "descend":
            problem.Qqf = base_Qqf
            robot.open_gripper()
            target = grasp_target_adjusted.copy()
            # Safety: prevent EE from going too low during descent
            if target[2] < 0.04:
                target[2] = 0.04
            problem.q_terminal = q_grasp
        elif stage in {"contact", "preload"}:
            problem.Qqf = base_Qqf
            target = grasp_target_adjusted
            problem.q_terminal = q_grasp
        elif stage == "lift":
            problem.Qqf = base_Qqf
            lift_safe = (
                grasp_decision is not None
                and grasp_decision.bilateral_contact
                and grasp_decision.wrench.feasible
                and grasp_decision.min_friction_margin >= 0.0
                and grasp_decision.max_slip_speed <= 2.0 * args.grasp_slip_speed_max
            )
            target = lift_target if lift_safe else robot.ee_pos.copy()
            problem.q_terminal = q_lift
        elif stage == "hold":
            problem.Qqf = base_Qqf
            target = robot.ee_pos.copy()  # hold current position
            problem.q_terminal = robot.joint_pos.copy()

        if step % mpc_every == 0:
            if stage in PhaseGraspMPC.ACTIVE_PHASES:
                grasp_decision = grasp_mpc.step(
                    phase=stage,
                    close_fraction=gripper_close_fraction(robot),
                    contacts=contact_samples,
                    object_center=cube,
                    object_mass=cube_mass,
                    desired_vertical_accel=0.15 if stage == "lift" else 0.0,
                )
                if args.debug or (stage in {"contact", "preload"} and step % 100 == 0):
                    print(
                        f"    grasp MPC: phase={stage} close={grasp_decision.command_fraction:.3f} "
                        f"rate={grasp_decision.closure_rate:+.3f}/s "
                        f"Fn=({grasp_decision.actual_normal_by_side['left']:.1f},"
                        f"{grasp_decision.actual_normal_by_side['right']:.1f})/"
                        f"{grasp_decision.target_normal_force:.1f} N "
                        f"wrench={'OK' if grasp_decision.wrench.feasible else 'NO'} "
                        f"res=({grasp_decision.wrench.force_residual:.3f} N,"
                        f"{grasp_decision.wrench.torque_residual:.4f} Nm) "
                        f"margin={grasp_decision.min_friction_margin:.2f} N "
                        f"slip={grasp_decision.max_slip_speed:.4f} m/s "
                        f"stable={grasp_decision.stable_steps}/{args.grasp_stable_mpc_steps}"
                    )
            active_speed = stage_target_speed(stage, args)
            reference_target = move_target_towards(reference_target, target, active_speed * args.mpc_dt)
            set_reference_to_target(problem, robot.ee_pos, reference_target, args)
            problem.set_previous_tau(current_tau)
            mpc_tau, _, diag = solver.step(arm.get_state())
            last_diag = diag
            nominal_tau = arm._clip_tau(
                mpc_tau if not diag.fallback_used else current_tau
            )
            desired_tau = nominal_tau

            if stage in UnifiedGraspMPC.ACTIVE_PHASES and grasp_decision is not None:
                object_linear_velocity, object_angular_velocity = env.get_object_twist("cube")
                if stage == "lift" and np.linalg.norm(target - lift_target) < 1e-6:
                    object_target = lift_box_pos
                elif stage == "hold" and hold_cube_target is not None:
                    object_target = hold_cube_target
                else:
                    object_target = initial_cube
                unified_decision = unified_grasp_mpc.step(
                    phase=stage,
                    nominal_tau=nominal_tau,
                    close_fraction=gripper_close_fraction(robot),
                    desired_gripper_rate=grasp_decision.closure_rate,
                    target_normal_force=grasp_decision.target_normal_force,
                    contacts=contact_samples,
                    finger_positions=(left_finger, right_finger),
                    object_position=cube,
                    object_rotation=env.get_object_pose("cube")[:3, :3],
                    object_linear_velocity=object_linear_velocity,
                    object_angular_velocity=object_angular_velocity,
                    object_mass=cube_mass,
                    object_inertia=cube_inertia,
                    ee_target=reference_target,
                    object_target=object_target,
                    q_target=problem.q_terminal,
                )
                desired_tau = unified_decision.tau
                set_gripper_fraction(robot, unified_decision.command_fraction)
                if args.debug or (
                    stage in {"preload", "lift", "hold"}
                    and (step % 100 == 0 or unified_decision.fallback_used)
                ):
                    print(
                        f"    unified OCP: phase={stage} "
                        f"ok={'YES' if unified_decision.optimizer_success else 'NO'} "
                        f"fallback={'YES' if unified_decision.fallback_used else 'NO'} "
                        f"lambda_n=({unified_decision.planned_normal_by_side['left']:.2f},"
                        f"{unified_decision.planned_normal_by_side['right']:.2f}) N "
                        f"eq={unified_decision.equality_residual:.2e} "
                        f"ineq={unified_decision.min_inequality_margin:.2e} "
                        f"slip_pred={unified_decision.max_predicted_slip:.4f} m/s"
                    )
            desired_tau = apply_optional_tau_limit(desired_tau, args.apply_tau_limit)
            target_tau = limit_torque_slew(desired_tau, current_tau, args.tau_slew_rate * args.mpc_dt)

        current_tau = limit_torque_slew(target_tau, current_tau, args.tau_slew_rate * dt)

        env.step(current_tau)

        free_cube_pos = env.get_object_pos("cube")
        if stage not in ("hold",) and free_cube_pos[2] < min_cube_z:
            env.set_object_pose("cube", pos=free_cube_pos, quat=env.get_object_quat("cube"), min_center_z=min_cube_z)

        if viewer:
            viewer.sync()
            time.sleep(dt)
        stage_hold += 1

    # ── Summary ──
    print(f"\n{'=' * 78}")
    print(f"  {'stage':16s} {'bilateral':>9s} {'enclosure':>9s} {'f-closure':>9s} "
          f"{'wrench':>7s} {'f-cone':>7s} {'aperture':>8s} {'contacts':>8s}")
    for label, d in diagnostics.items():
        flags = ["PASS" if d[k] else "FAIL"
                 for k in ("bilateral", "enclosed", "force_closure")]
        flags.append(
            "PASS" if d["wrench_feasible"]
            else ("FAIL" if d["wrench_evaluated"] else "N/A")
        )
        flags.append("PASS" if d["friction_cone"] else "FAIL")
        print(f"  {label:16s} {flags[0]:>9s} {flags[1]:>9s} {flags[2]:>9s} "
              f"{flags[3]:>7s} {flags[4]:>7s} {d['aperture']:8.3f} "
              f"{d['n_finger_contacts']:8d}")
    print("=" * 78)

    if not hold_completed:
        raise RuntimeError(
            "grasp MPC test did not complete a physical lift and hold "
            f"within {args.max_steps} simulation steps"
        )

    if viewer:
        print("\nViewer open — close to exit.")
        while viewer.is_running():
            time.sleep(0.1)
        viewer.close()


def main():
    viewer = "--viewer" in sys.argv[1:]
    sys.argv = [sys.argv[0], "--task-mode", "lift", "--max-steps", "5000",
                "--grasp-tol", "0.05",
                # TCP target = cube center + 6 cm: pads [+0.00,+0.07] rel TCP wrap
                # the cube's upper half without bottoming out on the table.
                "--grasp-z-offset", "0.06",
                # finger_mid is measured at the pad body origins (pad tops),
                # ~60 mm above the cube center here — relax the vertical tol
                # accordingly or the latch never fires.
                "--enclosure-vertical-tol", "0.08",
                # Finite reference speeds so the MPC reference glides instead of
                # teleporting — removes the dive-below-then-rise overshoot.
                "--approach-target-speed", "0.30",
                "--descend-target-speed", "0.15",
                "--lift-target-speed", "0.15"]
    if viewer:
        sys.argv.append("--viewer")
    args = main_parse_args()
    run_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
