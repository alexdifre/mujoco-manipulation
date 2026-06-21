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
