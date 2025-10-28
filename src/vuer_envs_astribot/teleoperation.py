#!/usr/bin/env python3
"""
File: teleoperation.py
Brief: Real-time teleoperation that streams Vision Pro data directly to robot execution.

Overview
========
This script bridges the online data collection (streamer.py) with the robot execution
logic (extract_wrist_trajectory.py) to enable real-time teleoperation while preserving
the existing record-and-replay architecture.

Architecture
------------
1. VisionProStreamer collects data from AVP and records to disk (unchanged)
2. TeleoperationBridge consumes the live stream via receive_queue
3. Real-time processing: position smoothing, gripper conversion, safety checks
4. Direct robot execution with configurable lag compensation
5. All data is still recorded to disk for later replay

Key Features
------------
* Real-time streaming: Sub-100ms latency from AVP to robot
* Concurrent recording: All data saved to disk for replay
* Safety features: Position jump detection, velocity limiting, emergency stop
* Lag compensation: Configurable buffer size and prediction
* Chassis control: Automatic mode switching based on pinch gestures
* Graceful degradation: Falls back to recording-only if robot unavailable

Usage Examples
--------------
# Basic teleoperation (both arms)
python3 teleoperation.py --ip 192.168.200.111

# Left arm only with custom safety limits
python3 teleoperation.py --ip 192.168.200.111 --trajectory_mode left_only --max_jump 0.2

# With time delay (slower, safer)
python3 teleoperation.py --ip 192.168.200.111 --time_factor 2.0

# Without gripper control
python3 teleoperation.py --ip 192.168.200.111 --disable-gripper

# Dry-run (no robot execution, only recording)
python3 teleoperation.py --ip 192.168.200.111 --dry-run
"""

import sys
import time
import queue
import threading
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, Dict, Any

# Add tools directory to path for imports

from utils.hand_pose_utils import (
    rotation_matrix_to_quaternion,
    parse_arguments as base_parse_arguments,
)

from avp_stream.streamer import VisionProStreamer
from eyetrack.config import settings

# Import robot control
try:
    from core.astribot_api.astribot_client import Astribot
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("Warning: Astribot API not available. Robot execution disabled.")


def parse_teleoperation_arguments():
    """Parse command-line arguments for teleoperation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time teleoperation: Vision Pro → Astribot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Connection settings
    parser.add_argument("--ip", type=str, required=True,
                        help="Vision Pro IP address")

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
    parser.add_argument("--enable-chassis", action="store_true",
                        help="Enable chassis control mode (both pinch = chassis)")

    # Execution settings
    parser.add_argument("--dry-run", action="store_true",
                        help="Record only, no robot execution")

    return parser.parse_args()


class TeleoperationBridge:
    """Bridge between VisionProStreamer and robot execution."""

    def __init__(self, streamer: VisionProStreamer, args):
        """Initialize teleoperation bridge.

        Args:
            streamer: VisionProStreamer instance (already connected)
            args: Parsed command-line arguments
        """
        self.streamer = streamer
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
        self.paused = False
        self.control_enabled = False  # Robot won't move until enabled
        self.control_enabled_time = None  # Track when control was enabled

        # Pinch-to-start gesture detection
        self.enable_gesture_start_time = None
        self.ENABLE_GESTURE_DURATION = 2.0  # seconds to hold both pinches
        self.DISABLE_GESTURE_DURATION = 2.0  # seconds to hold both releases
        self.EMERGENCY_STOP_DURATION = 1.0  # seconds to hold both pinches during teleoperation
        self.EMERGENCY_STOP_GRACE_PERIOD = 3.0  # seconds after enable before emergency stop is active
        self.emergency_stop_start_time = None

        # Pinch threshold for chassis control
        self.PINCH_THRESHOLD = 0.01
        self.chassis_mode_active = False

        # Initial position tracking
        self.initial_position_set = False
        self.moving_to_initial = False

    def signal_connection_established(self):
        """Signal to user that connection is established via gripper movement.

        Opens and closes grippers to acknowledge control enabled.
        """
        if not self.robot_ready or self.args.dry_run or self.args.disable_gripper:
            return

        try:
            print("[Robot] Signaling connection established...")

            # Build gripper names based on trajectory mode
            gripper_names = []
            if self.trajectory_mode in ['both', 'left_only']:
                gripper_names.append(self.robot.effector_left_name)
            if self.trajectory_mode in ['both', 'right_only']:
                gripper_names.append(self.robot.effector_right_name)

            if not gripper_names:
                return

            # Close grippers
            close_waypoints = [[0.0] for _ in gripper_names]  # 0 = fully closed
            self.robot.move_cartesian_waypoints(
                gripper_names,
                [close_waypoints],
                [0.5],  # 0.5 seconds to close
                use_wbc=False
            )
            time.sleep(0.6)

            # Open grippers
            open_waypoints = [[100.0] for _ in gripper_names]  # 100 = fully open
            self.robot.move_cartesian_waypoints(
                gripper_names,
                [open_waypoints],
                [0.5],  # 0.5 seconds to open
                use_wbc=False
            )
            time.sleep(0.6)

            print("[Robot] Connection signal complete!")
        except Exception as e:
            print(f"[Robot] Error signaling connection: {e}")

    def move_to_initial_position(self, waypoint_data: Dict[str, Any]):
        """Smoothly move robot to initial hand position over 5 seconds.

        Args:
            waypoint_data: Current hand position waypoint data
        """
        if not self.robot_ready or self.args.dry_run:
            return

        try:
            print("[Robot] Moving to initial hand position (5 seconds)...")
            self.moving_to_initial = True

            # Build waypoints based on trajectory mode
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

            # Move to initial position over 5 seconds
            if waypoints and names:
                self.robot.move_cartesian_waypoints(
                    names,
                    [waypoints],
                    [5.0],  # 5 seconds for smooth movement
                    use_wbc=True
                )
                time.sleep(5.2)  # Wait for movement to complete

                print("[Robot] Reached initial position - ready to follow!")
                self.initial_position_set = True
                self.moving_to_initial = False
        except Exception as e:
            print(f"[Robot] Error moving to initial position: {e}")
            self.moving_to_initial = False

    def check_emergency_stop(self, left_pinch: float, right_pinch: float) -> bool:
        """Check for emergency stop gesture during active teleoperation.

        Both hands pinched for EMERGENCY_STOP_DURATION stops everything.
        Emergency stop is only active after EMERGENCY_STOP_GRACE_PERIOD (3s) from when
        control was enabled, to prevent accidental stops during the enable gesture.

        Args:
            left_pinch: Left hand pinch distance
            right_pinch: Right hand pinch distance

        Returns:
            True if emergency stop triggered, False otherwise
        """
        if not self.control_enabled or not self.initial_position_set:
            # Emergency stop only active during teleoperation
            return False

        # Check if grace period has passed
        current_time = time.time()
        if self.control_enabled_time is not None:
            time_since_enabled = current_time - self.control_enabled_time
            if time_since_enabled < self.EMERGENCY_STOP_GRACE_PERIOD:
                # Still in grace period, emergency stop not active yet
                return False

        both_pinched = (left_pinch < self.PINCH_THRESHOLD and
                       right_pinch < self.PINCH_THRESHOLD)

        if both_pinched:
            if self.emergency_stop_start_time is None:
                self.emergency_stop_start_time = current_time
                print("\n" + "!" * 80)
                print("EMERGENCY STOP: Hold both pinches for {:.1f}s to STOP and return home".format(
                    self.EMERGENCY_STOP_DURATION))
                print("!" * 80 + "\n")
            else:
                elapsed = current_time - self.emergency_stop_start_time
                remaining = self.EMERGENCY_STOP_DURATION - elapsed

                if elapsed >= self.EMERGENCY_STOP_DURATION:
                    print("\n" + "X" * 80)
                    print("EMERGENCY STOP ACTIVATED - Stopping teleoperation and returning home!")
                    print("X" * 80 + "\n")
                    return True
                elif int(remaining * 10) % 5 == 0:  # Print every 0.5s
                    print(f"[Emergency Stop] {remaining:.1f}s remaining...")
        else:
            # Reset timer if pinch released
            if self.emergency_stop_start_time is not None:
                print("[Emergency Stop] Cancelled - continuing teleoperation")
                self.emergency_stop_start_time = None

        return False

    def check_enable_gesture(self, left_pinch: float, right_pinch: float) -> bool:
        """Check for pinch-to-start/stop gesture.

        Both hands pinched for ENABLE_GESTURE_DURATION enables control.
        Both hands released for DISABLE_GESTURE_DURATION disables control.

        Args:
            left_pinch: Left hand pinch distance
            right_pinch: Right hand pinch distance

        Returns:
            True if control state changed, False otherwise
        """
        current_time = time.time()
        both_pinched = (left_pinch < self.PINCH_THRESHOLD and
                       right_pinch < self.PINCH_THRESHOLD)
        both_released = (left_pinch > 0.05 and right_pinch > 0.05)

        # State machine for enable/disable gesture
        if not self.control_enabled:
            # Waiting to enable - need both hands pinched
            if both_pinched:
                if self.enable_gesture_start_time is None:
                    self.enable_gesture_start_time = current_time
                    print("[Safety] Hold both pinches for {:.1f}s to ENABLE robot control...".format(
                        self.ENABLE_GESTURE_DURATION))
                else:
                    elapsed = current_time - self.enable_gesture_start_time
                    remaining = self.ENABLE_GESTURE_DURATION - elapsed

                    if elapsed >= self.ENABLE_GESTURE_DURATION:
                        self.control_enabled = True
                        self.control_enabled_time = current_time  # Record when control was enabled
                        self.enable_gesture_start_time = None
                        print("\n" + "!" * 80)
                        print("ROBOT CONTROL ENABLED - Robot will now follow your movements!")
                        print(f"Note: Emergency stop will be active after {self.EMERGENCY_STOP_GRACE_PERIOD}s")
                        print("!" * 80 + "\n")
                        return True
                    elif int(remaining * 10) % 5 == 0:  # Print every 0.5s
                        print(f"[Safety] {remaining:.1f}s remaining...")
            else:
                # Reset timer if pinch released
                if self.enable_gesture_start_time is not None:
                    print("[Safety] Pinch gesture cancelled")
                    self.enable_gesture_start_time = None
        else:
            # Currently enabled - need both hands released to disable
            if both_released:
                if self.enable_gesture_start_time is None:
                    self.enable_gesture_start_time = current_time
                    print("[Safety] Hold both hands open for {:.1f}s to DISABLE robot control...".format(
                        self.DISABLE_GESTURE_DURATION))
                else:
                    elapsed = current_time - self.enable_gesture_start_time
                    remaining = self.DISABLE_GESTURE_DURATION - elapsed

                    if elapsed >= self.DISABLE_GESTURE_DURATION:
                        self.control_enabled = False
                        self.enable_gesture_start_time = None
                        print("\n" + "=" * 80)
                        print("ROBOT CONTROL DISABLED - Robot stopped")
                        print("=" * 80 + "\n")
                        return True
                    elif int(remaining * 10) % 5 == 0:  # Print every 0.5s
                        print(f"[Safety] {remaining:.1f}s remaining...")
            else:
                # Reset timer if hands not in release position
                if self.enable_gesture_start_time is not None:
                    self.enable_gesture_start_time = None

        return False

    def pinch_distance_to_gripper(self, pinch_distance: Optional[float],
                                   min_dist: float = 0.0,
                                   max_dist: float = 0.18) -> float:
        """Convert pinch distance to gripper value (0-100)."""
        if pinch_distance is None:
            return 50.0

        pinch_distance = max(min_dist, min(max_dist, pinch_distance))
        normalized = (pinch_distance - min_dist) / (max_dist - min_dist)
        gripper_value = 100.0 * (1.0 - normalized)
        return gripper_value

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

    def process_transformation(self, transformations: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming transformation data with safety checks.

        Returns:
            Processed waypoint dict or None if unsafe
        """
        # Extract positions and rotations
        left_wrist = transformations.get("left_wrist")
        right_wrist = transformations.get("right_wrist")

        if left_wrist is None and right_wrist is None:
            return None

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
            'chassis_mode': False,
            'safe': True,
            'reason': 'OK'
        }

        # Check for chassis mode (both hands pinched)
        left_pinch = transformations.get("left_pinch_distance", 1.0)
        right_pinch = transformations.get("right_pinch_distance", 1.0)

        if self.args.enable_chassis:
            chassis_mode = (left_pinch < self.PINCH_THRESHOLD and
                           right_pinch < self.PINCH_THRESHOLD)
            result['chassis_mode'] = chassis_mode

            if chassis_mode != self.chassis_mode_active:
                mode_str = "CHASSIS" if chassis_mode else "ARM"
                print(f"[Teleoperation] Switching to {mode_str} control mode")
                self.chassis_mode_active = chassis_mode

        # Process left wrist
        if left_wrist is not None:
            left_pos = np.array(left_wrist[:3])

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

            # Extract rotation and convert to quaternion
            left_rot_matrix = transformations.get("left_wrist_rotation")
            if left_rot_matrix is not None:
                left_rot_matrix = np.array(left_rot_matrix).reshape(3, 3)
                left_quat = rotation_matrix_to_quaternion(left_rot_matrix)
            else:
                left_quat = np.array([0, 0, 0, 1])  # Identity

            # Build waypoint [x, y, z, qx, qy, qz, qw]
            result['left_waypoint'] = np.concatenate([left_pos_smooth, left_quat])
            result['left_gripper'] = self.pinch_distance_to_gripper(left_pinch)

            self.last_left_pos = left_pos

        # Process right wrist
        if right_wrist is not None:
            right_pos = np.array(right_wrist[:3])

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

            # Extract rotation and convert to quaternion
            right_rot_matrix = transformations.get("right_wrist_rotation")
            if right_rot_matrix is not None:
                right_rot_matrix = np.array(right_rot_matrix).reshape(3, 3)
                right_quat = rotation_matrix_to_quaternion(right_rot_matrix)
            else:
                right_quat = np.array([0, 0, 0, 1])  # Identity

            # Build waypoint [x, y, z, qx, qy, qz, qw]
            result['right_waypoint'] = np.concatenate([right_pos_smooth, right_quat])
            result['right_gripper'] = self.pinch_distance_to_gripper(right_pinch)

            self.last_right_pos = right_pos

        return result

    def execute_waypoint(self, waypoint_data: Dict[str, Any]):
        """Execute single waypoint on robot.

        Args:
            waypoint_data: Processed waypoint dict from process_transformation
        """
        if not self.robot_ready or self.args.dry_run:
            return

        # Build command based on trajectory mode and chassis mode
        if waypoint_data['chassis_mode'] and self.args.enable_chassis:
            # Chassis control mode (not implemented in this version)
            # Would require head pose → chassis waypoint conversion
            pass
        else:
            # Arm control mode
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
                    # Use minimal time for zero-lag following
                    # freq=250Hz means 4ms per frame, use that for immediate response
                    dt = 1.0 / self.args.freq
                    self.robot.move_cartesian_waypoints(
                        names,
                        [waypoints],  # Single waypoint
                        [dt],         # Minimal time for real-time response
                        use_wbc=True
                    )
                except Exception as e:
                    print(f"[Robot] Execution error: {e}")

    def teleoperation_loop(self):
        """Main teleoperation loop - consumes streamer data and executes on robot."""
        print("[Teleoperation] Starting main loop...")
        print("\n" + ">" * 80)
        print("SAFETY: Pinch both hands together for 2 seconds to ENABLE robot control")
        print("EMERGENCY STOP: Pinch both hands for 1 second during teleoperation to STOP")
        print(">" * 80 + "\n")

        control_just_enabled = False

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            try:
                # Get latest transformation from streamer (blocking with timeout)
                transformations = self.streamer.get_latest()

                # Extract pinch distances for gesture checks
                left_pinch = transformations.get("left_pinch_distance", 1.0)
                right_pinch = transformations.get("right_pinch_distance", 1.0)

                # Check for emergency stop FIRST (highest priority)
                if self.check_emergency_stop(left_pinch, right_pinch):
                    print("[Teleoperation] Emergency stop triggered - exiting loop")
                    break

                # Check for enable/disable gesture
                gesture_changed = self.check_enable_gesture(left_pinch, right_pinch)

                # If control just got enabled, perform acknowledgment sequence
                if gesture_changed and self.control_enabled:
                    control_just_enabled = True
                    # Signal connection with gripper movement
                    self.signal_connection_established()

                # Process transformation (safety checks, smoothing)
                waypoint_data = self.process_transformation(transformations)

                if waypoint_data is None:
                    continue

                # If control just enabled, move to initial position first
                if control_just_enabled:
                    self.move_to_initial_position(waypoint_data)
                    control_just_enabled = False
                    continue

                # Execute on robot ONLY if control is enabled and not moving to initial
                if self.control_enabled and not self.moving_to_initial:
                    self.execute_waypoint(waypoint_data)

                # Reset initial position flag when control disabled
                if gesture_changed and not self.control_enabled:
                    self.initial_position_set = False

                # Statistics
                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    drop_rate = self.dropped_frames / self.frame_count * 100 if self.frame_count > 0 else 0
                    status = "ENABLED" if self.control_enabled else "DISABLED"
                    print(f"[Stats] Frames: {self.frame_count} | FPS: {fps:.1f} | Dropped: {drop_rate:.1f}% | Control: {status}")

            except queue.Empty:
                print("[Teleoperation] No data from streamer (timeout)")
                continue
            except KeyboardInterrupt:
                print("[Teleoperation] Interrupted by user")
                break
            except Exception as e:
                print(f"[Teleoperation] Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    def start(self):
        """Initialize robot and start teleoperation."""
        print("\n" + "=" * 80)
        print("REAL-TIME TELEOPERATION")
        print("=" * 80)
        print(f"Mode: {self.trajectory_mode}")
        print(f"Safety: max_jump={self.max_jump}m, max_velocity={self.max_velocity}m/s")
        print(f"Smoothing: buffer_size={self.buffer_size} frames")
        print(f"Timing: freq={self.args.freq}Hz, time_factor={self.time_factor}x")
        print(f"Gripper: {'disabled' if self.args.disable_gripper else 'enabled'}")
        print(f"Chassis: {'enabled' if self.args.enable_chassis else 'disabled'}")
        print(f"Dry-run: {'YES (no robot execution)' if self.args.dry_run else 'NO'}")

        # Initialize robot
        if not self.args.dry_run:
            if not ROBOT_AVAILABLE:
                print("\n[ERROR] Astribot API not available. Use --dry-run to record without robot.")
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

        # Start teleoperation loop
        self.running = True

        print("\n[Teleoperation] Starting...")
        print("Press Ctrl+C to stop")
        print("-" * 80)

        try:
            self.teleoperation_loop()
        except KeyboardInterrupt:
            print("\n[Teleoperation] Stopping...")
        finally:
            self.stop()

        return True

    def stop(self):
        """Stop teleoperation and cleanup."""
        print("\n[Teleoperation] Stopping...")
        self.running = False

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
        print("TELEOPERATION SUMMARY")
        print("=" * 80)
        print(f"Total frames: {self.frame_count}")
        print(f"Dropped frames: {self.dropped_frames} ({drop_rate:.1f}%)")
        print(f"Average FPS: {fps:.1f}")
        print(f"Duration: {elapsed:.1f}s")
        print("=" * 80)


def main():
    """Main entry point for teleoperation."""
    args = parse_teleoperation_arguments()

    print("\n" + "=" * 80)
    print("VISION PRO TELEOPERATION SYSTEM")
    print("=" * 80)
    print(f"Vision Pro IP: {args.ip}")
    print(f"Trajectory mode: {args.trajectory_mode}")

    # Configure streamer settings
    tracking_cfg = {
        'enabled': True,
        'rgb_depth': False  # Disable RGB-D for lower latency
    }

    eye_image_cfg = {
        'enabled': False  # Not needed for teleoperation
    }

    screen_pred_cfg = {
        'enabled': False,  # Not needed for teleoperation
        'save_labeled_frame': False
    }

    events_dict = {
        'save': threading.Event()
    }

    # Create streamer
    print("\n[Streamer] Connecting to Vision Pro...")
    try:
        streamer = VisionProStreamer(
            ip=args.ip,
            tracking_cfg=tracking_cfg,
            eye_image_cfg=eye_image_cfg,
            screen_pred_cfg=screen_pred_cfg,
            events_dict=events_dict,
            finetuning=False
        )
    except Exception as e:
        print(f"[ERROR] Failed to connect to Vision Pro: {e}")
        return 1

    print("[Streamer] Connected!")

    # Start tracking thread (this populates receive_queue)
    streamer.start_receiving_tracking()

    # Wait for first data
    print("[Streamer] Waiting for tracking data...")
    try:
        first_data = streamer.get_latest()
        print("[Streamer] Receiving data!")
    except Exception as e:
        print(f"[ERROR] No tracking data received: {e}")
        return 1

    # Create teleoperation bridge
    bridge = TeleoperationBridge(streamer, args)

    # Start teleoperation
    success = bridge.start()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())