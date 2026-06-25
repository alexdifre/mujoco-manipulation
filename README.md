# UR10e MPC Grasp

Run the MuJoCo/acados grasp experiment only through:

```powershell
.\launch_ur10e.ps1
```

For a non-GUI verification run:

```powershell
.\launch_ur10e.ps1 -Headless
```

The launcher executes `catkin_ws/run_ur10e_acados_grasp.py`, which performs the
finite-state MPC experiment: approach above the cube, descend, close the
gripper, lift, and verify contact/lift success.

Reports, metrics CSV files, and launcher logs are written to
`experiments/ur10e_acados_grasp`.

## Robosuite PickPlace Grasp

The same finite-state grasp / lift experiment is also available for the
Robosuite `PickPlace` environment:

```powershell
.\launch_robosuite_pickplace.ps1 -Headless
```

This launcher runs `catkin_ws/run_robosuite_pickplace_grasp.py` with a Panda arm,
OSC pose control, a single `can` object, and the same stages: open above the
object, approach, descend, close, lift, and report success/failure.

Robosuite reports and metrics are written to
`experiments/robosuite_pickplace_grasp`.
