#!/usr/bin/env python3
import numpy as np


def parabolic_arc_height(p0, p_goal):
    p0 = np.asarray(p0, dtype=np.float64)
    p_goal = np.asarray(p_goal, dtype=np.float64)
    distance = float(np.linalg.norm(p_goal - p0))
    return max(0.08, 0.35 * distance)


def build_parabolic_waypoints(p0, p_goal, num_waypoints=80):
    p0 = np.asarray(p0, dtype=np.float64)
    p_goal = np.asarray(p_goal, dtype=np.float64)
    count = max(2, int(num_waypoints))

    midpoint = 0.5 * (p0 + p_goal)
    p_control = midpoint + np.array(
        [0.0, 0.0, parabolic_arc_height(p0, p_goal)],
        dtype=np.float64,
    )

    waypoints = np.empty((count, 3), dtype=np.float64)
    for i in range(count):
        s = i / float(count - 1)
        waypoints[i] = (
            (1.0 - s) ** 2 * p0
            + 2.0 * (1.0 - s) * s * p_control
            + s ** 2 * p_goal
        )
    return waypoints


def waypoint_arclengths(waypoints):
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) < 2:
        return np.zeros(len(waypoints), dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def sample_waypoint_path(waypoints, arclengths, distances):
    waypoints = np.asarray(waypoints, dtype=np.float64)
    arclengths = np.asarray(arclengths, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    distances = np.clip(distances, 0.0, float(arclengths[-1]))
    samples = np.empty((distances.size, waypoints.shape[1]), dtype=np.float64)
    for axis in range(waypoints.shape[1]):
        samples[:, axis] = np.interp(distances, arclengths, waypoints[:, axis])
    return samples


def reference_horizon_distances(
    start_distance,
    horizon,
    dt,
    speed,
    lookahead=0.0,
    total_distance=None,
):
    dt = float(dt)
    distance = float(start_distance) + float(lookahead)
    distances = np.empty(horizon + 1, dtype=np.float64)
    if total_distance is None:
        step_distance = max(float(speed) * dt, 1e-6)
        return distance + step_distance * np.arange(horizon + 1)

    total_distance = float(total_distance)
    distance = min(distance, total_distance)
    for k in range(horizon + 1):
        distances[k] = distance
        remaining = total_distance - distance
        if remaining <= 0.0:
            continue
        step_distance = max(float(speed) * dt, 1e-6)
        distance = min(total_distance, distance + step_distance)
    return distances


def closest_waypoint_index(ee_pos, waypoints):
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    waypoints = np.asarray(waypoints, dtype=np.float64)
    distances = np.linalg.norm(waypoints - ee_pos[None, :], axis=1)
    return int(np.argmin(distances)), float(np.min(distances))


def update_path_progress(
    ee_pos,
    waypoints,
    arclengths,
    path_progress,
    dt,
    speed,
    max_path_lead,
    tracking_tolerance,
):
    closest_index, distance_to_path = closest_waypoint_index(ee_pos, waypoints)
    closest_progress = float(arclengths[closest_index])
    lead_limit = closest_progress + float(max_path_lead)
    progress = min(max(float(path_progress), closest_progress), lead_limit)
    advance_radius = max(float(max_path_lead), float(tracking_tolerance), 1e-9)
    if distance_to_path <= advance_radius:
        progress += max(float(speed), 0.0) * float(dt)
    progress = min(progress, lead_limit)
    progress = min(progress, float(arclengths[-1]))
    progress_index = int(np.searchsorted(arclengths, progress, side="right") - 1)
    progress_index = int(np.clip(progress_index, 0, len(waypoints) - 1))
    return progress, progress_index, closest_index, distance_to_path


def limit_torque_slew(desired_tau, previous_tau, max_delta_tau):
    desired_tau = np.asarray(desired_tau, dtype=np.float64)
    previous_tau = np.asarray(previous_tau, dtype=np.float64)
    max_delta_tau = float(max_delta_tau)
    if max_delta_tau <= 0.0:
        return desired_tau
    return np.clip(
        desired_tau,
        previous_tau - max_delta_tau,
        previous_tau + max_delta_tau,
    )


def apply_optional_tau_limit(tau, apply_tau_limit):
    tau = np.asarray(tau, dtype=np.float64)
    limit = float(apply_tau_limit)
    if limit <= 0.0:
        return tau
    return np.clip(tau, -limit, limit)
