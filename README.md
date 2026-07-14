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

## Physical grasp MPC test

The phase-scheduled supervisor detects contact and requests preload. During
contact, preload, lift, and hold, a prescribed-contact receding-horizon OCP
jointly optimizes arm torque, gripper closing velocity, and the two contact
wrenches. Its predictive state contains arm position/velocity, normalized
gripper opening, and cube pose/twist. Unilateral force, friction-pyramid,
torsional friction, force limits, low slip, table clearance, and object wrench
balance are enforced as optimization constraints. Lift is authorized only when
both the supervisor and this unified OCP report a stable feasible grasp.
Run the physical (non-welded) grasp test with:

```powershell
& "C:\conda-forge\envs\mlc-stack\python.exe" catkin_ws\test_grasp_physics.py --viewer
```

Omit `--viewer` for the deterministic headless verification used by the tests.
