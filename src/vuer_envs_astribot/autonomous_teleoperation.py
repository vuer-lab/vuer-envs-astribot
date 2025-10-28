#!/usr/bin/env python3
"""
File: autonomous_teleoperation.py
Brief: Real-time teleoperation that streams autonomous policy actions to robot execution.

Overview
========
This script bridges an autonomous policy with the robot execution logic to enable
real-time policy deployment while preserving safety features.

Architecture
------------
1. PolicyClient provides action stream from autonomous policy
2. AutonomousBridge consumes actions and processes them
3. Real-time processing: position smoothing, safety checks
4. Direct robot execution with configurable parameters
5. Optional data recording for evaluation

Key Features
------------
* Real-time streaming: Low latency from policy to robot
* Safety features: Position jump detection, velocity limiting, emergency stop
* Configurable control: Support for both arms or single arm
* Graceful degradation: Falls back to safe stop if policy fails

Usage Examples
--------------
# Basic autonomous control (both arms)
python3 autonomous_teleoperation.py --robot_ip 192.168.200.111

# Left arm only with custom safety limits
python3 autonomous_teleoperation.py --robot_ip 192.168.200.111 --trajectory_mode left_only --max_jump 0.2

# With time delay (slower, safer)
python3 autonomous_teleoperation.py --robot_ip 192.168.200.111 --time_factor 2.0

# Without gripper control
python3 autonomous_teleoperation.py --robot_ip 192.168.200.111 --disable-gripper

# Dry-run (no robot execution, only policy testing)
python3 autonomous_teleoperation.py --dry-run
"""

import sys
import time
import queue
import threading
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, Dict, Any

from utils.hand_pose_utils import rotation_matrix_to_quaternion

# Import robot control
try:
    from core.astribot_api.astribot_client import Astribot
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("Warning: Astribot API not available. Robot execution disabled.")


class PolicyClient:
    """Simple policy-agnostic bridge for executing actions from external environment.

    This class manages the entire robot teleoperation pipeline internally.
    Just instantiate it and call execute_action() to send actions to the robot.

    Example usage from your environment:
        # Initialize (automatically connects to robot and starts bridge)
        client = PolicyClient(
            robot_ip='192.168.200.111',
            trajectory_mode='both',
            freq=250.0
        )

        # In your policy loop:
        action = {
            'left_position': [x, y, z],
            'left_orientation': [qx, qy, qz, qw],
            'left_gripper': 50.0,
            # ... right arm ...
        }
        client.execute_action(action)

        # Cleanup when done
        client.stop()
    """

    def __init__(self,
                 robot_ip: str = "192.168.200.111",
                 trajectory_mode: str = "both",
                 max_jump: float = 0.15,
                 max_velocity: float = 1.0,
                 buffer_size: int = 5,
                 time_factor: float = 1.0,
                 freq: float = 250.0,
                 disable_gripper: bool = False,
                 dry_run: bool = False,
                 enable_keyboard_stop: bool = False):
        """Initialize policy client and start robot bridge.

        Args:
            robot_ip: Robot IP address (kept for compatibility, not used by current API)
            trajectory_mode: Which arm(s) to control ('both', 'left_only', 'right_only')
            max_jump: Maximum position change per frame (meters)
            max_velocity: Maximum velocity (m/s)
            buffer_size: Smoothing buffer size (frames)
            time_factor: Time scaling factor (>1 = slower)
            freq: Robot control frequency (Hz)
            disable_gripper: Disable gripper control
            dry_run: No robot execution, only testing
            enable_keyboard_stop: Enable keyboard emergency stop
        """
        print("[PolicyClient] Initializing...")

        # Store configuration
        self.config = {
            'robot_ip': robot_ip,
            'trajectory_mode': trajectory_mode,
            'max_jump': max_jump,
            'max_velocity': max_velocity,
            'buffer_size': buffer_size,
            'time_factor': time_factor,
            'freq': freq,
            'disable_gripper': disable_gripper,
            'dry_run': dry_run,
            'enable_keyboard_stop': enable_keyboard_stop
        }

        # Action queue for communication with bridge
        self.action_queue = queue.Queue(maxsize=10)
        self.running = False

        # Create and start the bridge in a background thread
        self.bridge = None
        self.bridge_thread = None
        self._start_bridge()

        print("[PolicyClient] Ready to receive actions!")

    def _start_bridge(self):
        """Initialize and start the autonomous bridge in background thread."""
        # Create mock args object from config
        class Args:
            pass

        args = Args()
        for key, value in self.config.items():
            setattr(args, key, value)

        # Create bridge
        self.bridge = AutonomousBridge(self, args)

        # Start bridge in background thread
        self.running = True
        self.bridge_thread = threading.Thread(target=self._run_bridge, daemon=False)
        self.bridge_thread.start()

        # Wait a bit for bridge to initialize
        time.sleep(0.5)

    def _run_bridge(self):
        """Run the bridge (called in background thread)."""
        try:
            self.bridge.start()
        except Exception as e:
            print(f"[PolicyClient] Bridge error: {e}")
            import traceback
            traceback.print_exc()

    def execute_action(self, action: Dict[str, Any], block: bool = True, timeout: float = 1.0) -> bool:
        """Execute an action on the robot.

        This method should be called from your external environment/policy.

        Args:
            action: Action dict with format:
                {
                    'left_position': np.array([x, y, z]) or list,  # Optional
                    'left_orientation': np.array([qx, qy, qz, qw]) or list,  # Optional (quaternion)
                    'left_gripper': float,  # 0-100, Optional
                    'right_position': np.array([x, y, z]) or list,  # Optional
                    'right_orientation': np.array([qx, qy, qz, qw]) or list,  # Optional (quaternion)
                    'right_gripper': float,  # 0-100, Optional
                }
            block: If True, wait for space in queue. If False, return immediately if queue full.
            timeout: Timeout in seconds if block=True

        Returns:
            True if action was queued successfully, False otherwise
        """
        if not self.running:
            print("[PolicyClient] Warning: Client not running, action ignored")
            return False

        try:
            self.action_queue.put(action, block=block, timeout=timeout if block else None)
            return True
        except queue.Full:
            print("[PolicyClient] Warning: Action queue full, action dropped")
            return False

    def get_action(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get next action from queue (used internally by bridge).

        Args:
            timeout: Timeout in seconds for waiting for action

        Returns:
            Action dict or None if no action available
        """
        try:
            action = self.action_queue.get(timeout=timeout)
            return action
        except queue.Empty:
            return None

    def stop(self):
        """Stop the policy client and cleanup."""
        print("\n[PolicyClient] Stopping...")
        self.running = False

        # Clear action queue
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except queue.Empty:
                break

        # Stop bridge
        if self.bridge:
            self.bridge.running = False

        # Wait for bridge thread to finish
        if self.bridge_thread and self.bridge_thread.is_alive():
            self.bridge_thread.join(timeout=5.0)

        print("[PolicyClient] Stopped")

    def get_robot_state(self) -> Optional[Dict[str, Any]]:
        """Get current robot state for observation (optional, implement if needed).

        Returns:
            Robot state dict or None if not available
        """
        # TODO: Implement if you need to get robot state for your policy
        # This could return joint positions, end-effector poses, etc.
        if self.bridge and self.bridge.robot:
            # Could query robot state here
            pass
        return None

    def is_ready(self) -> bool:
        """Check if the client is ready to receive actions.

        Returns:
            True if client is running and bridge is ready
        """
        return self.running and self.bridge is not None and self.bridge.robot_ready


def parse_autonomous_arguments():
    """Parse command-line arguments for autonomous teleoperation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous policy teleoperation: Policy → Astribot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Robot connection settings
    parser.add_argument("--robot_ip", type=str, default="192.168.200.111",
                        help="Robot IP address (not used in current Astribot API, but kept for compatibility)")

    # Policy settings
    parser.add_argument("--policy_config", type=str, default=None,
                        help="Path to policy configuration file (JSON)")

    # Trajectory settings
    parser.add_argument("--trajectory_mode", type=str, default="both",
                        choices=["both", "left_only", "right_only"],
                        help="Which arm(s) to control")

    # Safety settings
    parser.add_argument("--max_jump", type=float, default=0.15,
                        help="Maximum position change per frame (meters)")
    parser.add_argument("--max_velocity", type=float, default=1.0,
                        help="Maximum velocity (m/s)")
    parser.add_argument("--buffer_size", type=int, default=5,
                        help="Smoothing buffer size (frames)")

    # Timing settings
    parser.add_argument("--time_factor", type=float, default=1.0,
                        help="Time scaling factor (>1 = slower)")
    parser.add_argument("--freq", type=float, default=250.0,
                        help="Robot control frequency (Hz)")

    # Control settings
    parser.add_argument("--disable-gripper", action="store_true",
                        help="Disable gripper control (position/orientation only)")

    # Execution settings
    parser.add_argument("--dry-run", action="store_true",
                        help="Policy only, no robot execution")

    # Emergency stop
    parser.add_argument("--enable-keyboard-stop", action="store_true",
                        help="Enable keyboard-based emergency stop (press 's' to stop)")

    return parser.parse_args()


class AutonomousBridge:
    """Bridge between autonomous policy and robot execution."""

    def __init__(self, policy_client: PolicyClient, args):
        """Initialize autonomous bridge.

        Args:
            policy_client: PolicyClient instance
            args: Parsed command-line arguments
        """
        self.policy_client = policy_client
        self.args = args

        # Trajectory mode
        self.trajectory_mode = args.trajectory_mode

        # Safety parameters
        self.max_jump = args.max_jump
        self.max_velocity = args.max_velocity
        self.time_factor = args.time_factor

        # Smoothing buffer
        self.buffer_size = args.buffer_size
        self.left_buffer = deque(maxlen=self.buffer_size)
        self.right_buffer = deque(maxlen=self.buffer_size)

        # Last known positions (for jump detection)
        self.last_left_pos = None
        self.last_right_pos = None
        self.last_timestamp = None

        # Robot control
        self.robot = None
        self.robot_ready = False

        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = time.time()

        # Control flags
        self.running = False
        self.emergency_stop = False

    def is_safe_motion(self, current_pos: np.ndarray, last_pos: Optional[np.ndarray],
                       dt: float) -> Tuple[bool, str]:
        """Check if motion is within safety limits.

        Returns:
            (is_safe, reason) tuple
        """
        if last_pos is None:
            return True, "OK"

        # Check position jump
        delta = np.linalg.norm(current_pos - last_pos)
        if delta > self.max_jump:
            return False, f"Position jump too large: {delta:.3f}m > {self.max_jump}m"

        # Check velocity
        if dt > 0:
            velocity = delta / dt
            if velocity > self.max_velocity:
                return False, f"Velocity too high: {velocity:.2f}m/s > {self.max_velocity}m/s"

        return True, "OK"

    def smooth_position(self, buffer: deque, new_pos: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing to position."""
        buffer.append(new_pos)
        if len(buffer) == 0:
            return new_pos
        return np.mean([p for p in buffer], axis=0)

    def process_action(self, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming action data with safety checks.

        Args:
            action: Action dict from policy

        Returns:
            Processed waypoint dict or None if unsafe
        """
        # Timestamp
        current_time = time.time()
        dt = 0.0
        if self.last_timestamp is not None:
            dt = current_time - self.last_timestamp
        self.last_timestamp = current_time

        result = {
            'timestamp': current_time,
            'left_waypoint': None,
            'right_waypoint': None,
            'left_gripper': 50.0,
            'right_gripper': 50.0,
            'safe': True,
            'reason': 'OK'
        }

        # Process left arm action
        if 'left_position' in action and action['left_position'] is not None:
            left_pos = np.array(action['left_position'])

            # Safety check
            is_safe, reason = self.is_safe_motion(left_pos, self.last_left_pos, dt)
            if not is_safe:
                print(f"[Safety] Left arm: {reason}")
                result['safe'] = False
                result['reason'] = reason
                self.dropped_frames += 1
                return None

            # Smooth position
            left_pos_smooth = self.smooth_position(self.left_buffer, left_pos)

            # Get orientation (quaternion)
            if 'left_orientation' in action and action['left_orientation'] is not None:
                left_quat = np.array(action['left_orientation'])
            else:
                left_quat = np.array([0, 0, 0, 1])  # Identity

            # Build waypoint [x, y, z, qx, qy, qz, qw]
            result['left_waypoint'] = np.concatenate([left_pos_smooth, left_quat])

            # Get gripper value
            if 'left_gripper' in action:
                result['left_gripper'] = float(action['left_gripper'])

            self.last_left_pos = left_pos

        # Process right arm action
        if 'right_position' in action and action['right_position'] is not None:
            right_pos = np.array(action['right_position'])

            # Safety check
            is_safe, reason = self.is_safe_motion(right_pos, self.last_right_pos, dt)
            if not is_safe:
                print(f"[Safety] Right arm: {reason}")
                result['safe'] = False
                result['reason'] = reason
                self.dropped_frames += 1
                return None

            # Smooth position
            right_pos_smooth = self.smooth_position(self.right_buffer, right_pos)

            # Get orientation (quaternion)
            if 'right_orientation' in action and action['right_orientation'] is not None:
                right_quat = np.array(action['right_orientation'])
            else:
                right_quat = np.array([0, 0, 0, 1])  # Identity

            # Build waypoint [x, y, z, qx, qy, qz, qw]
            result['right_waypoint'] = np.concatenate([right_pos_smooth, right_quat])

            # Get gripper value
            if 'right_gripper' in action:
                result['right_gripper'] = float(action['right_gripper'])

            self.last_right_pos = right_pos

        return result

    def execute_waypoint(self, waypoint_data: Dict[str, Any]):
        """Execute single waypoint on robot.

        Args:
            waypoint_data: Processed waypoint dict from process_action
        """
        if not self.robot_ready or self.args.dry_run:
            return

        # Build command based on trajectory mode
        waypoints = []
        names = []

        if self.trajectory_mode == 'both':
            if waypoint_data['left_waypoint'] is not None:
                waypoints.append(waypoint_data['left_waypoint'].tolist())
                names.append(self.robot.arm_left_name)
            if waypoint_data['right_waypoint'] is not None:
                waypoints.append(waypoint_data['right_waypoint'].tolist())
                names.append(self.robot.arm_right_name)

            # Add grippers if enabled
            if not self.args.disable_gripper:
                if waypoint_data['left_waypoint'] is not None:
                    waypoints.append([waypoint_data['left_gripper']])
                    names.append(self.robot.effector_left_name)
                if waypoint_data['right_waypoint'] is not None:
                    waypoints.append([waypoint_data['right_gripper']])
                    names.append(self.robot.effector_right_name)

        elif self.trajectory_mode == 'left_only':
            if waypoint_data['left_waypoint'] is not None:
                waypoints.append(waypoint_data['left_waypoint'].tolist())
                names.append(self.robot.arm_left_name)
                if not self.args.disable_gripper:
                    waypoints.append([waypoint_data['left_gripper']])
                    names.append(self.robot.effector_left_name)

        elif self.trajectory_mode == 'right_only':
            if waypoint_data['right_waypoint'] is not None:
                waypoints.append(waypoint_data['right_waypoint'].tolist())
                names.append(self.robot.arm_right_name)
                if not self.args.disable_gripper:
                    waypoints.append([waypoint_data['right_gripper']])
                    names.append(self.robot.effector_right_name)

        # Execute on robot (single waypoint, minimal lag)
        if waypoints and names:
            try:
                # Use minimal time for responsive following
                dt = 1.0 / self.args.freq
                self.robot.move_cartesian_waypoints(
                    names,
                    [waypoints],  # Single waypoint
                    [dt],         # Minimal time for real-time response
                    use_wbc=True
                )
            except Exception as e:
                print(f"[Robot] Execution error: {e}")

    def autonomous_loop(self):
        """Main autonomous control loop - gets actions from policy and executes on robot."""
        print("[Autonomous] Starting control loop...")
        print("\n" + ">" * 80)
        print("AUTONOMOUS POLICY CONTROL ACTIVE")
        if self.args.enable_keyboard_stop:
            print("Press 's' + Enter to trigger emergency stop")
        else:
            print("Press Ctrl+C to stop")
        print(">" * 80 + "\n")

        # Start keyboard listener if enabled
        if self.args.enable_keyboard_stop:
            def keyboard_listener():
                while self.running:
                    try:
                        user_input = input()
                        if user_input.lower() == 's':
                            print("\n[Emergency Stop] Keyboard stop triggered!")
                            self.emergency_stop = True
                            break
                    except:
                        break

            keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
            keyboard_thread.start()

        while self.running and not self.emergency_stop:
            try:
                # Get action from policy
                action = self.policy_client.get_action(timeout=1.0)

                if action is None:
                    print("[Autonomous] No action from policy (timeout)")
                    continue

                # Process action (safety checks, smoothing)
                waypoint_data = self.process_action(action)

                if waypoint_data is None:
                    continue

                # Execute on robot
                self.execute_waypoint(waypoint_data)

                # Statistics
                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    drop_rate = self.dropped_frames / self.frame_count * 100 if self.frame_count > 0 else 0
                    print(f"[Stats] Frames: {self.frame_count} | FPS: {fps:.1f} | Dropped: {drop_rate:.1f}%")

            except KeyboardInterrupt:
                print("\n[Autonomous] Interrupted by user")
                break
            except Exception as e:
                print(f"[Autonomous] Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    def start(self):
        """Initialize robot and start autonomous control."""
        print("\n" + "=" * 80)
        print("AUTONOMOUS POLICY TELEOPERATION")
        print("=" * 80)
        print(f"Mode: {self.trajectory_mode}")
        print(f"Safety: max_jump={self.max_jump}m, max_velocity={self.max_velocity}m/s")
        print(f"Smoothing: buffer_size={self.buffer_size} frames")
        print(f"Timing: freq={self.args.freq}Hz, time_factor={self.time_factor}x")
        print(f"Gripper: {'disabled' if self.args.disable_gripper else 'enabled'}")
        print(f"Dry-run: {'YES (no robot execution)' if self.args.dry_run else 'NO'}")

        # Initialize robot
        if not self.args.dry_run:
            if not ROBOT_AVAILABLE:
                print("\n[ERROR] Astribot API not available. Use --dry-run to test policy without robot.")
                return False

            try:
                print("\n[Robot] Connecting to Astribot...")
                self.robot = Astribot(freq=self.args.freq)
                print("[Robot] Moving to home position...")
                self.robot.move_to_home()
                self.robot_ready = True
                print("[Robot] Ready!")
            except Exception as e:
                print(f"[ERROR] Failed to initialize robot: {e}")
                return False
        else:
            print("\n[Dry-run] Skipping robot initialization")

        # Start control loop
        self.running = True

        print("\n[Autonomous] Starting control...")
        print("Press Ctrl+C to stop")
        print("-" * 80)

        try:
            self.autonomous_loop()
        except KeyboardInterrupt:
            print("\n[Autonomous] Stopping...")
        finally:
            self.stop()

        return True

    def stop(self):
        """Stop autonomous control and cleanup."""
        print("\n[Autonomous] Stopping...")
        self.running = False

        # Stop policy
        self.policy_client.stop()

        # Return robot to home
        if self.robot_ready and self.robot is not None:
            try:
                print("[Robot] Returning to home position...")
                self.robot.move_to_home()
            except Exception as e:
                print(f"[Robot] Error returning home: {e}")

        # Statistics
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        drop_rate = self.dropped_frames / self.frame_count * 100 if self.frame_count > 0 else 0

        print("\n" + "=" * 80)
        print("AUTONOMOUS TELEOPERATION SUMMARY")
        print("=" * 80)
        print(f"Total frames: {self.frame_count}")
        print(f"Dropped frames: {self.dropped_frames} ({drop_rate:.1f}%)")
        print(f"Average FPS: {fps:.1f}")
        print(f"Duration: {elapsed:.1f}s")
        print("=" * 80)


def main():
    """Main entry point for autonomous teleoperation (for CLI usage)."""
    args = parse_autonomous_arguments()

    print("\n" + "=" * 80)
    print("AUTONOMOUS POLICY TELEOPERATION SYSTEM")
    print("=" * 80)
    print(f"Trajectory mode: {args.trajectory_mode}")

    # Create policy client with CLI arguments
    # This automatically starts the bridge in the background
    policy_client = PolicyClient(
        robot_ip=args.robot_ip,
        trajectory_mode=args.trajectory_mode,
        max_jump=args.max_jump,
        max_velocity=args.max_velocity,
        buffer_size=args.buffer_size,
        time_factor=args.time_factor,
        freq=args.freq,
        disable_gripper=args.disable_gripper,
        dry_run=args.dry_run,
        enable_keyboard_stop=args.enable_keyboard_stop
    )

    # Wait for bridge thread to finish (it runs until Ctrl+C or error)
    try:
        if policy_client.bridge_thread:
            policy_client.bridge_thread.join()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
        policy_client.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())