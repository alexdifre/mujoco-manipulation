#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET

MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))

# ── Minimal manual config ──────────────────────────────────────────────────────
# Auto-detected from XML: n_arm_joints, gripper_ctrl_ids
# Must be specified manually:
#   xml        - MuJoCo model path (relative to MODELS_ROOT)
#   ee_body    - end-effector body name (design choice)
#   home_qpos  - home joint configuration (design choice)
#   vel_limits - from official datasheet (URDF values are unreliable)
#   gripper_open / gripper_close - what "open" and "closed" mean for this gripper

ROBOT_CONFIGS = {
    "ur10e": {
        "xml":       "ur10e/ur10e.xml",
        "ee_body":   "ee_tcp",
        "home_qpos": [1.57089435, -1.42615965, 1.58303971, -1.69997267, -1.57142825, -0.00000032],
        "vel_limits": [2.094, 2.094, 3.141, 3.141, 3.141, 3.141],  # UR official (rad/s)
        "gripper_open": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "gripper_close": [0.628, 0.872, 0.628, -0.628, -0.872, 0.628],
    },
}


def _parse_xml(xml_path):
    """Auto-detect n_arm_joints and gripper_ctrl_ids from MuJoCo XML actuators.
    Arm joints = <motor> actuators; gripper = <position> actuators.
    """
    tree = ET.parse(xml_path)
    actuator = tree.getroot().find("actuator")
    

    motor_count = 0
    gripper_ids = []
    for i, act in enumerate(actuator):
        if act.tag == "motor":
            motor_count += 1
        elif act.tag == "position":
            gripper_ids.append(i)

    return motor_count, gripper_ids


def get_config(robot_name):
    

    cfg = ROBOT_CONFIGS[robot_name].copy()
    cfg["xml"] = os.path.join(MODELS_ROOT, cfg["xml"])

    n_arm, gripper_ids = _parse_xml(cfg["xml"])
    cfg["n_arm_joints"] = n_arm
    if gripper_ids and "gripper_ctrl_ids" not in cfg:
        cfg["gripper_ctrl_ids"] = gripper_ids

    return cfg
