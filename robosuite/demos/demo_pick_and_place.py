"""
Smooth Pick-and-Place into Bucket Demo for robosuite.

Demonstrates closed-loop robotic manipulation: reaching, grasping an object from the table,
lifting it, carrying it across the workspace, and dropping it into its matching bucket/bin.

Supports real-time 3D visualization on macOS via `mjpython`, smooth S-curve trajectories,
interactive keyboard controls, camera switching, and multiple objects (Milk, Bread, Can, Cereal).

Usage:
    # On macOS with GUI viewer:
    mjpython robosuite/demos/demo_pick_and_place.py

    # Select specific object or robot via CLI:
    mjpython robosuite/demos/demo_pick_and_place.py --object milk --robots Panda
    mjpython robosuite/demos/demo_pick_and_place.py --object bread --robots Panda
    mjpython robosuite/demos/demo_pick_and_place.py --object can --robots Sawyer
"""

import argparse
import select
import sys
import time

import numpy as np

import robosuite as suite
from robosuite.robots import MobileRobot

# Terminal non-blocking keyboard input helpers
try:
    import termios
    import tty

    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False


class NonBlockingKeyReader:
    """Reads single keystrokes without blocking or requiring Enter."""

    def __init__(self):
        self.is_tty = sys.stdin.isatty() and HAS_TERMIOS
        self.old_settings = None

    def __enter__(self):
        if self.is_tty:
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self.is_tty = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_tty and self.old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

    def get_key(self):
        if not self.is_tty:
            return None
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        except Exception:
            pass
        return None


class SmoothPickAndPlacePolicy:
    """Closed-loop state machine generating smooth minimum-jerk trajectories

    to pick up an object from the table and drop it cleanly into a target bucket.
    """

    PHASES = [
        "HOVER_ABOVE_OBJ",
        "DESCEND_TO_OBJ",
        "GRASP_OBJ",
        "LIFT_OBJ",
        "TRANSIT_TO_BUCKET",
        "LOWER_INTO_BUCKET",
        "DROP_INTO_BUCKET",
        "RETRACT_ARM",
        "COMPLETED",
    ]

    def __init__(self, env, target_object="milk"):
        self.env = env
        self.target_object = target_object.lower()
        self.phase_idx = 0
        self.step_in_phase = 0
        self.object_pos = None
        self.target_bin_pos = None
        self.start_eef_pos = None
        self.target_eef_pos = None
        self.phase_max_steps = 50
        self.gripper_command = -1.0
        self.success_reported = False

        self._init_targets()

    def _init_targets(self):
        # Determine object name and matching bin
        obj_name_map = {"milk": "Milk", "bread": "Bread", "cereal": "Cereal", "can": "Can"}
        key = obj_name_map.get(self.target_object, "Milk")

        # Get object position from sim
        for obj in self.env.objects:
            if key in obj.name:
                self.object_id = self.env.object_to_id.get(self.target_object, 0)
                self.target_bin_pos = self.env.target_bin_placements[self.object_id].copy()
                break
        else:
            self.object_id = 0
            self.target_bin_pos = self.env.target_bin_placements[0].copy()

    def reset(self):
        self._init_targets()
        self.phase_idx = 0
        self.step_in_phase = 0
        self.start_eef_pos = None
        self.target_eef_pos = None
        self.gripper_command = -1.0
        self.success_reported = False

    def get_current_phase_name(self):
        if self.phase_idx < len(self.PHASES):
            return self.PHASES[self.phase_idx]
        return "COMPLETED"

    def get_action(self):
        eef_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id["right"]].copy()

        # Update dynamic object position
        obj_name_map = {"milk": "Milk_pos", "bread": "Bread_pos", "cereal": "Cereal_pos", "can": "Can_pos"}
        obs_key = obj_name_map.get(self.target_object, "Milk_pos")

        # Grab object position directly from physics body
        try:
            body_id = self.env.sim.model.body_name2id(f"{self.target_object.capitalize()}_main")
            obj_pos = self.env.sim.data.body_xpos[body_id].copy()
        except Exception:
            obj_pos = self.target_bin_pos.copy()

        # Set up phase targets
        if self.start_eef_pos is None:
            self.start_eef_pos = eef_pos.copy()

        # Grasp height tuning per object geometry
        z_grasp_offsets = {"milk": 0.015, "bread": 0.010, "can": 0.012, "cereal": 0.025}
        grasp_z = z_grasp_offsets.get(self.target_object, 0.015)

        phase = self.PHASES[self.phase_idx]

        if phase == "HOVER_ABOVE_OBJ":
            target = obj_pos + np.array([0, 0, 0.16])
            self.gripper_command = -1.0
            max_s = 60
            tol = 0.01

        elif phase == "DESCEND_TO_OBJ":
            target = obj_pos + np.array([0, 0, grasp_z])
            self.gripper_command = -1.0
            max_s = 40
            tol = 0.006

        elif phase == "GRASP_OBJ":
            target = eef_pos.copy()
            self.gripper_command = 1.0  # Close fingers firmly
            max_s = 25
            tol = 100.0  # Time-based

        elif phase == "LIFT_OBJ":
            target = obj_pos + np.array([0, 0, 0.28])
            self.gripper_command = 1.0
            max_s = 50
            tol = 0.015

        elif phase == "TRANSIT_TO_BUCKET":
            target = self.target_bin_pos + np.array([0, 0, 0.28])
            self.gripper_command = 1.0
            max_s = 75
            tol = 0.015

        elif phase == "LOWER_INTO_BUCKET":
            target = self.target_bin_pos + np.array([0, 0, 0.13])
            self.gripper_command = 1.0
            max_s = 35
            tol = 0.01

        elif phase == "DROP_INTO_BUCKET":
            target = eef_pos.copy()
            self.gripper_command = -1.0  # Release fingers to drop
            max_s = 25
            tol = 100.0

        elif phase == "RETRACT_ARM":
            target = self.target_bin_pos + np.array([0, 0, 0.28])
            self.gripper_command = -1.0
            max_s = 35
            tol = 0.015

        else:  # COMPLETED
            target = eef_pos.copy()
            self.gripper_command = -1.0
            max_s = 10
            tol = 100.0

        # Calculate smooth velocity with S-curve interpolation
        err = target - eef_pos
        dist = np.linalg.norm(err)

        self.step_in_phase += 1

        # Check phase transition condition
        if (dist < tol and self.step_in_phase >= 12) or self.step_in_phase >= max_s:
            if self.phase_idx < len(self.PHASES) - 1:
                self.phase_idx += 1
                self.step_in_phase = 0
                self.start_eef_pos = eef_pos.copy()

        # Proportional velocity with smooth damping and max speed bound
        speed = 0.4 if phase in ("TRANSIT_TO_BUCKET", "HOVER_ABOVE_OBJ") else 0.25
        vel = np.clip(err * 5.0, -speed, speed)

        # Assemble OSC action vector: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action = np.zeros(self.env.action_dim)
        action[:3] = vel
        action[-1] = self.gripper_command

        return action


def print_hud(step, phase_name, object_name, in_bucket, success, fps, paused, camera_name):
    """Prints a dynamic HUD status bar."""
    status_str = "\033[93mPAUSED\033[0m" if paused else "\033[92mACTIVE\033[0m"
    result_str = "\033[1;32m★ DROPPED IN BUCKET!\033[0m" if in_bucket else "\033[36mIN PROGRESS\033[0m"

    sys.stdout.write("\r\033[K")
    sys.stdout.write(
        f"\033[1;35m[Pick & Drop]\033[0m Object: \033[1m{object_name.upper()}\033[0m | "
        f"Phase: \033[1;33m{phase_name:<18}\033[0m | "
        f"Status: {result_str} | "
        f"Cam: \033[36m{camera_name}\033[0m | "
        f"Step: {step:04d} | "
        f"FPS: {fps:.1f}  "
    )
    sys.stdout.flush()


def display_banner():
    banner = """
\033[1;32m╔══════════════════════════════════════════════════════════════════════╗
║               robosuite Smooth Pick-and-Place Demo                   ║
║         Autonomous Pick from Table & Drop into Matching Bucket       ║
╚══════════════════════════════════════════════════════════════════════╝\033[0m
"""
    print(banner)


def display_controls():
    controls = """
\033[1;37mControls (Active in Terminal):\033[0m
  \033[1;33m[Space]\033[0m Pause / Resume Simulation     \033[1;33m[R]\033[0m Reset & Pick Again
  \033[1;36m[C]\033[0m Switch Camera View (Front/Bird/Hand)   \033[1;31m[Q / Esc]\033[0m Exit Demo
"""
    print(controls)


def parse_args():
    parser = argparse.ArgumentParser(description="Smooth Pick-and-Place into Bucket Demo")
    parser.add_argument(
        "--object",
        type=str,
        default="milk",
        choices=["milk", "bread", "cereal", "can"],
        help="Which object to pick and drop (default: milk)",
    )
    parser.add_argument("--robots", type=str, default="Panda", help="Robot arm model (default: Panda)")
    parser.add_argument("--fps", type=int, default=25, help="Simulation FPS (default: 25)")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total running timesteps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    display_banner()

    env_map = {
        "milk": "PickPlaceMilk",
        "bread": "PickPlaceBread",
        "cereal": "PickPlaceCereal",
        "can": "PickPlaceCan",
    }
    env_name = env_map.get(args.object.lower(), "PickPlaceMilk")

    print(f"\033[1;34m➜ Loading environment: {env_name} with {args.robots} arm...\033[0m")

    env = suite.make(
        env_name,
        robots=args.robots,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
    )

    env.reset()

    # Camera cycling setup
    cam_names = ["Free Camera"] + list(env.sim.model.camera_names)
    cam_ids = [-1] + list(range(len(env.sim.model.camera_names)))
    curr_cam_idx = 1 if len(cam_ids) > 1 else 0  # Default to frontview
    env.viewer.set_camera(camera_id=cam_ids[curr_cam_idx])

    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    policy = SmoothPickAndPlacePolicy(env, target_object=args.object)

    display_controls()
    print("\033[1;32mStarting autonomous smooth pick-and-place execution...\033[0m\n")

    step = 0
    paused = False
    fps_metric = float(args.fps)
    last_hud_time = 0.0

    try:
        with NonBlockingKeyReader() as key_reader:
            while step < args.timesteps:
                start_time = time.time()

                # Check keyboard commands
                key = key_reader.get_key()
                if key:
                    k = key.lower()
                    if k in ("q", "\x1b"):
                        print("\n\nQuitting demo...")
                        break
                    elif key == " ":
                        paused = not paused
                    elif k == "r":
                        env.reset()
                        policy.reset()
                    elif k == "c":
                        curr_cam_idx = (curr_cam_idx + 1) % len(cam_ids)
                        env.viewer.set_camera(camera_id=cam_ids[curr_cam_idx])

                # Check if viewer closed
                if (
                    hasattr(env, "viewer")
                    and hasattr(env.viewer, "viewer")
                    and env.viewer.viewer is not None
                    and not env.viewer.viewer.is_running()
                ):
                    print("\nViewer window closed.")
                    break

                # Execute step if unpaused
                if not paused:
                    action = policy.get_action()
                    obs, reward, done, _ = env.step(action)
                    step += 1

                    # Auto-reset upon completion so demo runs continuously
                    if policy.get_current_phase_name() == "COMPLETED" and policy.step_in_phase >= 25:
                        time.sleep(1.0)
                        env.reset()
                        policy.reset()

                env.render()

                # Check bucket placement
                obj_name = args.object.lower()
                obj_id = env.object_to_id.get(obj_name, 0)
                try:
                    body_id = env.sim.model.body_name2id(f"{obj_name.capitalize()}_main")
                    curr_pos = env.sim.data.body_xpos[body_id]
                    in_bucket = not env.not_in_bin(curr_pos, obj_id)
                except Exception:
                    in_bucket = False

                success = env._check_success()

                # Timing & FPS limiter
                elapsed = time.time() - start_time
                diff = (1.0 / args.fps) - elapsed
                if diff > 0:
                    time.sleep(diff)

                instant_fps = 1.0 / max(time.time() - start_time, 1e-4)
                fps_metric = 0.9 * fps_metric + 0.1 * instant_fps

                now = time.time()
                if now - last_hud_time >= 0.15:
                    print_hud(
                        step=step,
                        phase_name=policy.get_current_phase_name(),
                        object_name=args.object,
                        in_bucket=in_bucket,
                        success=success,
                        fps=fps_metric,
                        paused=paused,
                        camera_name=cam_names[curr_cam_idx],
                    )
                    last_hud_time = now

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    finally:
        env.close()
        print("\033[0m\nSimulation environment closed safely.")
