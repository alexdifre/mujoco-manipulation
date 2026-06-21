You are working on an MPC control experiment for a UR10e/robotic arm in MuJoCo.

Goal:
Implement and test an experiment where the arm moves to a cube, aligns the gripper, closes the gripper, lifts the cube, and reports success/failure.

Rules:
- Work autonomously inside this repository.
- Do not ask for confirmation unless a destructive operation is needed.
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
5. Run the simulation and study/analize the metrics just gathered
6. observe if it's failure or success
IF FAILURE:
7. Ripeti il ciclo dal punto 2.
If SUCCESS:
7. Termina e finisci la risposta con un breve riassunto di cosa è successo
8. PUSHA su github tutte le modifiche