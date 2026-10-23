"""
Enhanced Interactive Action Demo for robosuite.

Supports multiple smooth, realistic robotic motion modes as well as original random actions,
with real-time interactive keyboard controls (switch modes, pause, reset, switch cameras)
and a rich terminal UI/UX.

Usage:
    # On macOS with GUI viewer:
    mjpython robosuite/demos/demo_random_action.py

    # Or pass direct command line arguments:
    mjpython robosuite/demos/demo_random_action.py --environment Lift --robots Panda --mode smooth
    mjpython robosuite/demos/demo_random_action.py --environment Lift --robots Baxter --mode wave
"""

import argparse
import select
import sys
import time

import numpy as np

import robosuite as suite
from robosuite.robots import MobileRobot
from robosuite.utils.input_utils import choose_environment, choose_multi_arm_config, choose_robots

# Terminal non-blocking keyboard input helpers
try:
    import termios
    import tty

    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False


class NonBlockingKeyReader:
    """Reads single keystrokes from stdin without requiring the user to press Enter."""

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
        """Returns single key string if available, else None."""
        if not self.is_tty:
            return None
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                return ch
        except Exception:
            pass
        return None


class MultiModeActionController:
    """Generates continuous, smooth, or stylized actions across single-arm, bimanual,

    and multi-robot setups with seamless transitions between modes.
    """

    MODES = [
        ("smooth", "Smooth Random Walk (Fluid Exploration)"),
        ("wave", "Harmonic Sinusoidal (Synchronized Wave)"),
        ("reach", "Pick & Lift Cycle (Reach, Grasp, Ascend)"),
        ("circle", "Cartesian Circular Orbit (3D Loop)"),
        ("jitter", "Unfiltered Noise (Original Raw Random)"),
        ("hold", "Stationary Hold (Zero Action)"),
    ]

    def __init__(self, env, initial_mode="smooth", control_freq=20):
        self.env = env
        self.mode = initial_mode if initial_mode in [m[0] for m in self.MODES] else "smooth"
        self.action_dim = env.action_dim
        self.low, self.high = env.action_spec
        self.dt = 1.0 / max(1, control_freq)
        self.time = 0.0
        self.step_idx = 0

        # State for smooth random walk (Ornstein-Uhlenbeck process with damping)
        self.ou_state = np.zeros(self.action_dim)
        self.theta = 1.8  # Mean reversion rate
        self.sigma = 0.5  # Volatility

        # Previous action for smooth cross-mode blending
        self.prev_action = np.zeros(self.action_dim)
        self.blend_alpha = 0.85

        # Unique phase offsets and frequencies per dimension for harmonic wave
        self.freqs = np.linspace(0.25, 0.55, self.action_dim)
        self.phases = np.random.uniform(0, 2 * np.pi, self.action_dim)

    def set_mode(self, new_mode):
        if new_mode in [m[0] for m in self.MODES]:
            self.mode = new_mode

    def cycle_mode(self):
        modes = [m[0] for m in self.MODES]
        curr_idx = modes.index(self.mode)
        next_idx = (curr_idx + 1) % len(modes)
        self.mode = modes[next_idx]
        return self.get_mode_display()

    def get_mode_display(self):
        for key, name in self.MODES:
            if key == self.mode:
                return name
        return self.mode.upper()

    def reset(self):
        self.ou_state = np.zeros(self.action_dim)
        self.prev_action = np.zeros(self.action_dim)
        self.time = 0.0
        self.step_idx = 0

    def get_action(self):
        self.time += self.dt
        self.step_idx += 1

        # Check if we can generate actions semantically per-robot
        robot_actions = []
        for robot_idx, robot in enumerate(self.env.robots):
            act = self._get_robot_action(robot, robot_idx)
            robot_actions.append(act)

        if robot_actions:
            raw_action = np.concatenate(robot_actions)
        else:
            raw_action = self._get_fallback_action()

        # Ensure bounds are strictly respected
        raw_action = np.clip(raw_action, self.low, self.high)

        # Smooth blend with previous action to prevent discontinuous jumps
        if self.mode != "jitter":
            blended = self.blend_alpha * self.prev_action + (1.0 - self.blend_alpha) * raw_action
        else:
            blended = raw_action

        self.prev_action = blended.copy()
        return blended

    def _get_robot_action(self, robot, robot_offset=0):
        """Generates appropriate action per robot controller part."""
        if not hasattr(robot, "composite_controller") or not hasattr(
            robot.composite_controller, "part_controllers"
        ):
            return np.zeros(robot.action_dim)

        part_controllers = robot.composite_controller.part_controllers
        split_indexes = getattr(robot.composite_controller, "_action_split_indexes", {})
        action_dict = {}

        t = self.time
        for part_name, ctrl in part_controllers.items():
            if part_name in split_indexes:
                start_idx, end_idx = split_indexes[part_name]
                ctrl_dim = end_idx - start_idx
            else:
                ctrl_dim = getattr(ctrl, "control_dim", 1)

            if ctrl_dim <= 0:
                continue

            if "gripper" in part_name:
                # Gripper control logic
                if self.mode == "reach":
                    # Gripper synchronizes with pick-and-lift phases
                    cycle_time = 4.0
                    phase = (t % cycle_time) / cycle_time
                    gripper_val = 1.0 if 0.4 <= phase < 0.85 else -1.0
                elif self.mode in ("smooth", "wave", "circle"):
                    # Gentle periodic opening and closing
                    gripper_val = np.sin(0.4 * np.pi * t + robot_offset * np.pi)
                elif self.mode == "jitter":
                    gripper_val = np.random.uniform(-1.0, 1.0)
                else:
                    gripper_val = 0.0
                action_dict[part_name] = np.full(ctrl_dim, gripper_val)

            else:
                # Arm / body part control
                part_act = np.zeros(ctrl_dim)

                if self.mode == "smooth":
                    # Filtered random walk
                    noise = np.random.randn(ctrl_dim)
                    part_act = 0.35 * np.sin(0.8 * t + np.arange(ctrl_dim)) + 0.15 * noise

                elif self.mode == "wave":
                    # Harmonic sinusoidal motion (anti-symmetric for bimanual left/right arms)
                    sign = -1.0 if "left" in part_name else 1.0
                    part_act = 0.3 * np.sin(1.2 * t * np.linspace(1, 1.5, ctrl_dim) + sign * np.pi / 4)

                elif self.mode == "circle":
                    # Circular orbit in XY plane with gentle vertical breathing
                    if ctrl_dim >= 3:
                        omega = 1.2
                        sign = -1.0 if "left" in part_name else 1.0
                        part_act[0] = 0.3 * np.cos(omega * t)
                        part_act[1] = 0.3 * np.sin(omega * t) * sign
                        part_act[2] = 0.12 * np.sin(2.0 * omega * t)

                elif self.mode == "reach":
                    # Pick-and-lift cyclic motion: descend -> hold/grasp -> ascend -> return
                    cycle_time = 4.0
                    phase = (t % cycle_time) / cycle_time
                    if ctrl_dim >= 3:
                        if phase < 0.35:  # Descend towards object/table
                            part_act[2] = -0.35
                        elif phase < 0.55:  # Pause & grasp
                            part_act[2] = 0.0
                        elif phase < 0.85:  # Ascend / lift
                            part_act[2] = 0.35
                        else:  # Reset height
                            part_act[2] = 0.0

                elif self.mode == "jitter":
                    # Raw noise scaled to reasonable bounds
                    part_act = 0.4 * np.random.randn(ctrl_dim)

                elif self.mode == "hold":
                    part_act = np.zeros(ctrl_dim)

                action_dict[part_name] = part_act

        try:
            return robot.composite_controller.create_action_vector(action_dict)
        except Exception:
            return np.zeros(robot.action_dim)

    def _get_fallback_action(self):
        """Fallback action generator when composite part controller is not available."""
        t = self.time
        d = self.action_dim
        if self.mode == "smooth":
            noise = np.random.randn(d)
            self.ou_state += -self.theta * self.ou_state * self.dt + self.sigma * np.sqrt(self.dt) * noise
            return np.clip(self.ou_state, -0.6, 0.6)
        elif self.mode == "wave":
            return 0.35 * np.sin(2 * np.pi * self.freqs * t + self.phases)
        elif self.mode == "circle":
            act = np.zeros(d)
            act[0] = 0.3 * np.cos(1.2 * t)
            if d > 1:
                act[1] = 0.3 * np.sin(1.2 * t)
            if d > 2:
                act[2] = 0.1 * np.sin(2.4 * t)
            return act
        elif self.mode == "reach":
            act = np.zeros(d)
            cycle = (t % 4.0) / 4.0
            if d > 2:
                act[2] = -0.3 if cycle < 0.4 else (0.3 if cycle < 0.8 else 0.0)
            if d > 6:
                act[6] = 1.0 if 0.4 <= cycle < 0.85 else -1.0
            return act
        elif self.mode == "jitter":
            return np.clip(0.4 * np.random.randn(d), -1.0, 1.0)
        else:
            return np.zeros(d)


def print_hud(env_name, robot_name, mode_name, step, max_steps, fps, paused, camera_name):
    """Renders a clean, compact HUD status in the terminal."""
    status_str = "\033[93mPAUSED\033[0m" if paused else "\033[92mRUNNING\033[0m"
    sys.stdout.write("\r\033[K")  # Clear line
    sys.stdout.write(
        f"\033[1;36m[robosuite]\033[0m {env_name} | \033[1m{robot_name}\033[0m | "
        f"Mode: \033[1;32m{mode_name}\033[0m | "
        f"Cam: \033[33m{camera_name}\033[0m | "
        f"Step: {step}/{max_steps} | "
        f"{status_str} | "
        f"FPS: {fps:.1f}  "
    )
    sys.stdout.flush()


def display_welcome_banner():
    banner = f"""
\033[1;34m╔══════════════════════════════════════════════════════════════════════╗
║                   robosuite Interactive Action Demo                  ║
║                  Version {suite.__version__:<10}  |  Platform: macOS / Unix           ║
╚══════════════════════════════════════════════════════════════════════╝\033[0m
"""
    print(banner)


def display_controls_help():
    controls = """
\033[1;37mKeyboard Controls (Active in Terminal):\033[0m
  \033[1;32m[M]\033[0m Cycle Action Mode      \033[1;32m[1-6]\033[0m Jump to Mode (1:Smooth, 2:Wave, 3:Reach, etc.)
  \033[1;33m[Space]\033[0m Pause / Resume     \033[1;33m[R]\033[0m Reset Simulation
  \033[1;36m[C]\033[0m Switch Camera View     \033[1;31m[Q / Esc]\033[0m Clean Exit
"""
    print(controls)


def parse_args():
    parser = argparse.ArgumentParser(description="robosuite Interactive Smooth & Multi-Action Demo")
    parser.add_argument("--environment", type=str, default=None, help="Environment name (e.g. Lift, Stack, Door)")
    parser.add_argument("--robots", nargs="+", type=str, default=None, help="Robot(s) to use (e.g. Panda, Baxter)")
    parser.add_argument(
        "--mode",
        type=str,
        default="smooth",
        choices=["smooth", "wave", "reach", "circle", "jitter", "hold"],
        help="Initial action mode (default: smooth)",
    )
    parser.add_argument("--timesteps", type=int, default=10000, help="Maximum simulation timesteps")
    parser.add_argument("--fps", type=int, default=25, help="Target frames per second (default: 25)")
    parser.add_argument("--config", type=str, default=None, help="Environment configuration (for multi-arm)")
    parser.add_argument("--non-interactive", action="store_true", help="Bypass menu prompts and use defaults")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    display_welcome_banner()

    options = {}

    # Environment Selection
    if args.environment:
        options["env_name"] = args.environment
    elif args.non_interactive:
        options["env_name"] = "Lift"
    else:
        options["env_name"] = choose_environment()

    # Robot Selection
    if args.robots:
        options["robots"] = args.robots if len(args.robots) > 1 else args.robots[0]
        if args.config:
            options["env_configuration"] = args.config
    elif args.non_interactive:
        options["robots"] = "Panda"
    else:
        if "TwoArm" in options["env_name"]:
            options["env_configuration"] = choose_multi_arm_config()
            if options["env_configuration"] == "single-robot":
                options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True, exclude_single_arm=True)
            else:
                options["robots"] = []
                for i in range(2):
                    print(f"Please choose Robot {i+1}...\n")
                    options["robots"].append(choose_robots(exclude_bimanual=False, use_humanoids=True))
        elif "Humanoid" in options["env_name"]:
            options["robots"] = choose_robots(use_humanoids=True)
        else:
            options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True)

    print(f"\n\033[1;32m✓ Initializing {options['env_name']} with {options['robots']}...\033[0m")

    # Initialize environment
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    env.reset()

    # Camera cycling setup
    cam_names = ["Free Camera"] + list(env.sim.model.camera_names)
    cam_ids = [-1] + list(range(len(env.sim.model.camera_names)))
    curr_cam_idx = 1 if len(cam_ids) > 1 else 0  # default to first fixed camera (frontview)
    env.viewer.set_camera(camera_id=cam_ids[curr_cam_idx])

    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    action_ctrl = MultiModeActionController(env, initial_mode=args.mode, control_freq=20)

    display_controls_help()
    print("Starting simulation loop...\n")

    paused = False
    step = 0
    fps_metric = float(args.fps)
    last_hud_time = 0.0

    mode_keys = {"1": "smooth", "2": "wave", "3": "reach", "4": "circle", "5": "jitter", "6": "hold"}

    try:
        with NonBlockingKeyReader() as key_reader:
            while step < args.timesteps:
                start_time = time.time()

                # Process keyboard inputs
                key = key_reader.get_key()
                if key:
                    key_lower = key.lower()
                    if key_lower in ("q", "\x1b"):  # q or Esc
                        print("\n\nQuitting simulation...")
                        break
                    elif key == " ":
                        paused = not paused
                    elif key_lower == "m":
                        action_ctrl.cycle_mode()
                    elif key in mode_keys:
                        action_ctrl.set_mode(mode_keys[key])
                    elif key_lower == "r":
                        env.reset()
                        action_ctrl.reset()
                    elif key_lower == "c":
                        curr_cam_idx = (curr_cam_idx + 1) % len(cam_ids)
                        env.viewer.set_camera(camera_id=cam_ids[curr_cam_idx])

                # Check if MuJoCo GUI window was closed
                if (
                    hasattr(env, "viewer")
                    and hasattr(env.viewer, "viewer")
                    and env.viewer.viewer is not None
                    and not env.viewer.viewer.is_running()
                ):
                    print("\nViewer window closed.")
                    break

                # Step simulation if not paused
                if not paused:
                    action = action_ctrl.get_action()
                    obs, reward, done, _ = env.step(action)
                    step += 1

                env.render()

                # Frame rate limiter
                elapsed = time.time() - start_time
                target_period = 1.0 / args.fps
                diff = target_period - elapsed
                if diff > 0:
                    time.sleep(diff)

                # FPS calculation
                actual_loop_time = max(time.time() - start_time, 1e-4)
                instant_fps = 1.0 / actual_loop_time
                fps_metric = 0.9 * fps_metric + 0.1 * instant_fps

                # Update HUD periodically (5 Hz)
                now = time.time()
                if now - last_hud_time >= 0.2:
                    print_hud(
                        env_name=options["env_name"],
                        robot_name=str(options["robots"]),
                        mode_name=action_ctrl.get_mode_display(),
                        step=step,
                        max_steps=args.timesteps,
                        fps=fps_metric,
                        paused=paused,
                        camera_name=cam_names[curr_cam_idx],
                    )
                    last_hud_time = now

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user (Ctrl+C).")
    finally:
        env.close()
        print("\033[0m\nSimulation environment closed safely.")
