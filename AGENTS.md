You are working on an MPC control experiment for a UR10e/robotic arm in MuJoCo.

Goal:
Implement and test an experiment where the arm moves to a cube, aligns the gripper, closes the gripper, lifts the cube, and reports success/failure.

Rules:
- Work autonomously inside this repository.
- Do not ask for confirmation unless a destructive operation is needed.
- Prefer small commits/diffs.
- Always run tests or simulation checks after changes.
- Save logs, plots, and a short report of each experiment.
- Do not modify files outside the project folder.
- Do not use network access unless explicitly required.

Tasks:
1. Inspect the codebase.
2. Identify the robot model, cube body, gripper actuator, MPC controller, and simulation entry point.
3. Implement a finite-state experiment:
   - approach above cube
   - descend
   - align gripper
   - close gripper
   - lift cube
   - verify cube height/contact
4. Add metrics:
   - end-effector error
   - cube position
   - grasp success
   - control cost
   - constraint violations
5. Run the simulation and debug until the experiment succeeds.
6. Produce a final report with what changed and how to reproduce.