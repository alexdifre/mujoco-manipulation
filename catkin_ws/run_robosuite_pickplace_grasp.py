#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class PickPlaceMetric:
    step: int
    stage: str
    eef_x: float
    eef_y: float
    eef_z: float
    object_x: float
    object_y: float
    object_z: float
    target_x: float
    target_y: float
    target_z: float
    eef_error: float
    object_lift: float
    gripper_command: float
    grasp_latched: bool
    action_norm: float
    action_saturation: float
    reward: float


def require_robosuite():
    try:
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "robosuite is not installed. Install it in the robot_sim environment "
            "or into the project .python_packages directory before launching this experiment."
        ) from exc
    return suite, load_composite_controller_config


def make_controller_config(load_composite_controller_config, args):
    config = load_composite_controller_config(controller=args.controller)
    for part in ("right", "left"):
        if part in config.get("body_parts", {}):
            config["body_parts"][part]["type"] = args.arm_controller
    return config


def unwrap_reset(result):
    if isinstance(result, tuple):
        return result[0]
    return result


def unwrap_step(result):
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info
    return result


def find_object_pos(obs, object_type):
    candidates = [
        f"{object_type}_pos",
        f"{object_type.capitalize()}_pos",
        f"{object_type.lower()}_pos",
        f"{object_type.upper()}_pos",
    ]
    for key in candidates:
        if key in obs:
            return np.asarray(obs[key], dtype=np.float64), key

    object_keys = [
        key for key in obs
        if key.endswith("_pos")
        and "eef" not in key.lower()
        and "gripper" not in key.lower()
        and "robot" not in key.lower()
        and "bin" not in key.lower()
    ]
    if not object_keys:
        raise KeyError(f"Could not find an object position key in observation keys: {sorted(obs)}")
    key = object_keys[0]
    return np.asarray(obs[key], dtype=np.float64), key


def eef_pos(obs):
    for key in ("robot0_eef_pos", "eef_pos"):
        if key in obs:
            return np.asarray(obs[key], dtype=np.float64)
    eef_keys = [key for key in obs if key.endswith("eef_pos")]
    if eef_keys:
        return np.asarray(obs[eef_keys[0]], dtype=np.float64)
    raise KeyError(f"Could not find end-effector position in observation keys: {sorted(obs)}")


def forward_offset_from_robot(pos, distance):
    xy = np.asarray(pos[:2], dtype=np.float64)
    norm = float(np.linalg.norm(xy))
    if norm <= 1e-9 or abs(distance) <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    offset = np.zeros(3, dtype=np.float64)
    offset[:2] = xy / norm * float(distance)
    return offset


def make_action(env, eef, target, gripper_command, args):
    action = np.zeros(env.action_dim, dtype=np.float64)
    pos_error = target - eef
    action[:3] = np.clip(args.position_gain * pos_error, -args.max_pos_action, args.max_pos_action)
    if env.action_dim >= 7:
        action[3:6] = 0.0
    action[-1] = float(gripper_command)
    return np.clip(action, -1.0, 1.0)


def check_env_success(env):
    checker = getattr(env, "_check_success", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except TypeError:
        return False


def stage_target(stage, object_pos, args):
    forward = forward_offset_from_robot(object_pos, args.grasp_forward_offset)
    if stage in {"open", "approach"}:
        return object_pos + np.array([0.0, 0.0, args.approach_clearance], dtype=np.float64)
    if stage in {"descend", "close"}:
        return object_pos + forward + np.array([0.0, 0.0, args.grasp_z_offset], dtype=np.float64)
    return object_pos + forward + np.array([0.0, 0.0, args.lift_z_offset], dtype=np.float64)


def write_outputs(args, metrics, report):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"robosuite_pickplace_grasp_metrics_{stamp}.csv"
    json_path = out_dir / f"robosuite_pickplace_grasp_report_{stamp}.json"
    md_path = out_dir / f"robosuite_pickplace_grasp_report_{stamp}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PickPlaceMetric.__dataclass_fields__.keys()))
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.__dict__)

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Robosuite PickPlace Grasp Report",
                "",
                f"- success: `{report['success']}`",
                f"- environment: `{report['environment']}`",
                f"- robot: `{report['robot']}`",
                f"- object_key: `{report['object_key']}`",
                f"- steps: `{report['steps']}`",
                f"- final_object_lift_m: `{report['final_object_lift_m']}`",
                f"- max_object_lift_m: `{report['max_object_lift_m']}`",
                f"- min_eef_error_m: `{report['min_eef_error_m']}`",
                f"- grasp_latched_any: `{report['grasp_latched_any']}`",
                f"- robosuite_success: `{report['robosuite_success']}`",
                f"- control_cost: `{report['control_cost']}`",
                f"- max_action_saturation: `{report['max_action_saturation']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, md_path


def run(args):
    suite, load_composite_controller_config = require_robosuite()
    controller_config = make_controller_config(load_composite_controller_config, args)
    env = suite.make(
        env_name="PickPlace",
        robots=args.robot,
        controller_configs=controller_config,
        has_renderer=args.viewer,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=args.control_freq,
        horizon=args.max_steps,
        ignore_done=True,
        reward_shaping=True,
        single_object_mode=2,
        object_type=args.object_type,
    )

    obs = unwrap_reset(env.reset())
    obj, object_key = find_object_pos(obs, args.object_type)
    initial_object = obj.copy()
    metrics = []
    stage = "open"
    stage_hold = 0
    grasp_latched = False
    control_cost = 0.0
    robosuite_success = False
    reached_lift_success = False
    final_object = initial_object.copy()

    for step in range(args.max_steps):
        eef = eef_pos(obs)
        obj, _ = find_object_pos(obs, args.object_type)
        final_object = obj.copy()
        object_lift = float(obj[2] - initial_object[2])
        target = stage_target(stage, obj if args.track_object else initial_object, args)
        eef_error = float(np.linalg.norm(eef - target))

        if stage == "open" and stage_hold >= args.open_steps:
            stage = "approach"
            stage_hold = 0
        elif stage == "approach" and eef_error <= args.reach_tol:
            stage = "descend"
            stage_hold = 0
        elif stage == "descend" and eef_error <= args.grasp_tol:
            stage = "close"
            stage_hold = 0
        elif stage == "close":
            object_near = np.linalg.norm(eef - obj) <= args.grasp_latch_distance
            if stage_hold >= args.close_steps and object_near:
                grasp_latched = True
                stage = "lift"
                stage_hold = 0
        elif stage == "lift" and object_lift >= args.success_lift:
            reached_lift_success = True
            robosuite_success = check_env_success(env)
            break

        if stage in {"open", "approach", "descend"}:
            gripper_command = args.open_command
        else:
            gripper_command = args.close_command

        target = stage_target(stage, obj if args.track_object else initial_object, args)
        action = make_action(env, eef, target, gripper_command, args)
        obs, reward, done, _ = unwrap_step(env.step(action))
        if args.viewer:
            env.render()

        action_norm = float(np.linalg.norm(action))
        action_saturation = float(np.max(np.maximum(np.abs(action) - 0.98, 0.0)))
        control_cost += action_norm * action_norm / float(args.control_freq)
        metrics.append(
            PickPlaceMetric(
                step=step,
                stage=stage,
                eef_x=float(eef[0]),
                eef_y=float(eef[1]),
                eef_z=float(eef[2]),
                object_x=float(obj[0]),
                object_y=float(obj[1]),
                object_z=float(obj[2]),
                target_x=float(target[0]),
                target_y=float(target[1]),
                target_z=float(target[2]),
                eef_error=float(np.linalg.norm(eef - target)),
                object_lift=object_lift,
                gripper_command=float(gripper_command),
                grasp_latched=grasp_latched,
                action_norm=action_norm,
                action_saturation=action_saturation,
                reward=float(reward),
            )
        )
        stage_hold += 1
        if done and not args.ignore_done:
            break

    env.close()
    final_obj = final_object
    object_lifts = [metric.object_lift for metric in metrics] or [0.0]
    object_lifts.append(float(final_obj[2] - initial_object[2]))
    eef_errors = [metric.eef_error for metric in metrics] or [float("inf")]
    success = bool(
        (reached_lift_success or max(object_lifts) >= args.success_lift)
        and (grasp_latched or any(m.grasp_latched for m in metrics))
    )
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "environment": "Robosuite PickPlace",
        "robot": args.robot,
        "controller": args.controller,
        "arm_controller": args.arm_controller,
        "object_type": args.object_type,
        "object_key": object_key,
        "steps": len(metrics),
        "initial_object_pos": initial_object.tolist(),
        "final_object_pos": final_obj.tolist(),
        "approach_clearance": args.approach_clearance,
        "grasp_forward_offset": args.grasp_forward_offset,
        "final_object_lift_m": float(final_obj[2] - initial_object[2]),
        "max_object_lift_m": float(max(object_lifts)),
        "min_eef_error_m": float(min(eef_errors)),
        "grasp_latched_any": bool(any(m.grasp_latched for m in metrics)),
        "robosuite_success": robosuite_success,
        "control_cost": float(control_cost),
        "max_action_saturation": float(max(m.action_saturation for m in metrics) if metrics else 0.0),
        "success": success,
    }
    csv_path, json_path, md_path = write_outputs(args, metrics, report)
    return report, csv_path, json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description="Finite-state Robosuite PickPlace grasp/lift experiment.")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--controller", default="BASIC")
    parser.add_argument("--arm-controller", default="OSC_POSE")
    parser.add_argument("--object-type", default="can")
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--open-steps", type=int, default=40)
    parser.add_argument("--close-steps", type=int, default=40)
    parser.add_argument("--approach-clearance", type=float, default=0.12)
    parser.add_argument("--grasp-z-offset", type=float, default=0.015)
    parser.add_argument("--grasp-forward-offset", type=float, default=0.01)
    parser.add_argument("--lift-z-offset", type=float, default=0.22)
    parser.add_argument("--success-lift", type=float, default=0.08)
    parser.add_argument("--reach-tol", type=float, default=0.025)
    parser.add_argument("--grasp-tol", type=float, default=0.020)
    parser.add_argument("--grasp-latch-distance", type=float, default=0.085)
    parser.add_argument("--position-gain", type=float, default=8.0)
    parser.add_argument("--max-pos-action", type=float, default=0.20)
    parser.add_argument("--open-command", type=float, default=-1.0)
    parser.add_argument("--close-command", type=float, default=1.0)
    parser.add_argument("--track-object", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ignore-done", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments") / "robosuite_pickplace_grasp")
    parser.add_argument("--viewer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report, csv_path, json_path, md_path = run(args)
    print(json.dumps(report, indent=2))
    print(f"metrics_csv={csv_path}")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    if not report["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
