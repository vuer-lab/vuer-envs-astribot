#!/usr/bin/env python3
"""
File: robot_executor.py
Brief: Receive waypoints over network and execute on robot with gesture-based control.

This script runs on the robot machine. It:
1. Listens for incoming waypoint data from data_streamer.py
2. Implements pinch-to-start, emergency stop gestures
3. Executes waypoints on robot with minimal latency
4. Provides visual/tactile feedback via gripper

Usage:
    # On robot machine:
    python3 robot_executor.py --port 50051 --freq 250.0
"""

import sys
import time
import socket
import struct
import msgpack
import threading
import numpy as np
from typing import Optional, Dict, Any
from collections import deque

# Import robot control
try:
    from core.astribot_api.astribot_client import Astribot
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("Warning: Astribot API not available. Use --dry-run for testing.")


class RobotExecutor:
    """Executes waypoints on robot with gesture-based safety control."""

    def __init__(self, args):
        """Initialize robot executor.

        Args:
            args: Command-line arguments
        """
        self.args = args
        self.port = args.port
        self.freq = args.freq

        # Robot
        self.robot = None
        self.robot_ready = False

        # Network
        self.server_socket = None
        self.client_socket = None
        self.running = False

        # Control state
        self.control_enabled = False
        self.initial_position_set = False
        self.moving_to_initial = False
        self.control_enabled_time = None  # Track when control was enabled

        # Latest waypoint buffer (for separate receive thread)
        self.latest_waypoint = None
        self.waypoint_lock = threading.Lock()
        self.receive_thread = None
        self.waypoints_received = 0
        self.waypoints_skipped = 0

        # Gesture detection
        self.enable_gesture_start_time = None
        self.emergency_stop_start_time = None
        self.PINCH_THRESHOLD = 0.01
        self.ENABLE_GESTURE_DURATION = 2.0
        self.EMERGENCY_STOP_DURATION = 1.0
        self.EMERGENCY_STOP_GRACE_PERIOD = 3.0  # seconds after enable before emergency stop is active

        # Statistics
        self.frame_count = 0
        self.executed_count = 0
        self.start_time = time.time()

    def signal_connection_established(self):
        """Signal connection via gripper close-open."""
        if not self.robot_ready or self.args.dry_run:
            return

        try:
            print("[Robot] Signaling connection established...")

            gripper_names = [self.robot.effector_left_name, self.robot.effector_right_name]

            # Close grippers
            close_waypoints = [[0.0], [0.0]]
            self.robot.move_cartesian_waypoints(
                gripper_names, [close_waypoints], [0.5], use_wbc=False
            )
            time.sleep(0.6)

            # Open grippers
            open_waypoints = [[100.0], [100.0]]
            self.robot.move_cartesian_waypoints(
                gripper_names, [open_waypoints], [0.5], use_wbc=False
            )
            time.sleep(0.6)

            print("[Robot] Connection signal complete!")
        except Exception as e:
            print(f"[Robot] Error signaling connection: {e}")

    def move_to_initial_position(self, waypoint_data: Dict[str, Any]):
        """Move robot to initial hand position over 5 seconds."""
        if not self.robot_ready or self.args.dry_run:
            return

        try:
            print("[Robot] Moving to initial hand position (5 seconds)...")
            self.moving_to_initial = True

            waypoints = []
            names = []

            if waypoint_data['left_waypoint'] is not None:
                waypoints.append(waypoint_data['left_waypoint'])
                names.append(self.robot.arm_left_name)
            if waypoint_data['right_waypoint'] is not None:
                waypoints.append(waypoint_data['right_waypoint'])
                names.append(self.robot.arm_right_name)

            # Add grippers
            if waypoint_data['left_waypoint'] is not None:
                waypoints.append([waypoint_data['left_gripper']])
                names.append(self.robot.effector_left_name)
            if waypoint_data['right_waypoint'] is not None:
                waypoints.append([waypoint_data['right_gripper']])
                names.append(self.robot.effector_right_name)

            if waypoints and names:
                self.robot.move_cartesian_waypoints(names, [waypoints], [3.0], use_wbc=True)
                time.sleep(5.2)

                print("[Robot] Reached initial position - ready to follow!")
                self.initial_position_set = True
                self.moving_to_initial = False

        except Exception as e:
            print(f"[Robot] Error moving to initial position: {e}")
            self.moving_to_initial = False

    def check_emergency_stop(self, left_pinch: float, right_pinch: float) -> bool:
        """Check for emergency stop gesture (both pinch for 1s during teleoperation).

        Emergency stop is only active after EMERGENCY_STOP_GRACE_PERIOD (3s) from when
        control was enabled, to prevent accidental stops during the enable gesture.
        """
        if not self.control_enabled or not self.initial_position_set:
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
                print(f"EMERGENCY STOP: Hold both pinches for {self.EMERGENCY_STOP_DURATION:.1f}s to STOP")
                print("!" * 80 + "\n")
            else:
                elapsed = current_time - self.emergency_stop_start_time

                if elapsed >= self.EMERGENCY_STOP_DURATION:
                    print("\n" + "X" * 80)
                    print("EMERGENCY STOP ACTIVATED - Returning to home!")
                    print("X" * 80 + "\n")
                    return True
                elif int(elapsed * 10) % 5 == 0:
                    remaining = self.EMERGENCY_STOP_DURATION - elapsed
                    print(f"[Emergency Stop] {remaining:.1f}s remaining...")
        else:
            if self.emergency_stop_start_time is not None:
                print("[Emergency Stop] Cancelled")
                self.emergency_stop_start_time = None

        return False

    def check_enable_gesture(self, left_pinch: float, right_pinch: float) -> bool:
        """Check for enable gesture (both pinch for 2s)."""
        if self.control_enabled:
            return False

        current_time = time.time()
        both_pinched = (left_pinch < self.PINCH_THRESHOLD and
                       right_pinch < self.PINCH_THRESHOLD)

        if both_pinched:
            if self.enable_gesture_start_time is None:
                self.enable_gesture_start_time = current_time
                print(f"[Safety] Hold both pinches for {self.ENABLE_GESTURE_DURATION:.1f}s to ENABLE...")
            else:
                elapsed = current_time - self.enable_gesture_start_time

                if elapsed >= self.ENABLE_GESTURE_DURATION:
                    self.control_enabled = True
                    self.control_enabled_time = current_time  # Record when control was enabled
                    self.enable_gesture_start_time = None
                    print("\n" + "!" * 80)
                    print("ROBOT CONTROL ENABLED - Robot will follow your movements!")
                    print(f"Note: Emergency stop will be active after {self.EMERGENCY_STOP_GRACE_PERIOD}s")
                    print("!" * 80 + "\n")
                    return True
                elif int(elapsed * 10) % 5 == 0:
                    remaining = self.ENABLE_GESTURE_DURATION - elapsed
                    print(f"[Safety] {remaining:.1f}s remaining...")
        else:
            if self.enable_gesture_start_time is not None:
                print("[Safety] Enable gesture cancelled")
                self.enable_gesture_start_time = None

        return False

    def execute_waypoint(self, waypoint_data: Dict[str, Any]):
        """Execute waypoint on robot."""
        if not self.robot_ready or self.args.dry_run:
            return

        try:
            waypoints = []
            names = []

            # Arms
            if waypoint_data['left_waypoint'] is not None:
                waypoints.append(waypoint_data['left_waypoint'])
                names.append(self.robot.arm_left_name)
            if waypoint_data['right_waypoint'] is not None:
                waypoints.append(waypoint_data['right_waypoint'])
                names.append(self.robot.arm_right_name)

            # Grippers
            if waypoint_data['left_waypoint'] is not None:
                waypoints.append([waypoint_data['left_gripper']])
                names.append(self.robot.effector_left_name)
            if waypoint_data['right_waypoint'] is not None:
                waypoints.append([waypoint_data['right_gripper']])
                names.append(self.robot.effector_right_name)

            if waypoints and names:
                tic = time.time()
                self.robot.set_cartesian_pose(names, waypoints, use_wbc=True)
                toc = time.time()
                print("Total Time:", toc - tic)
                self.executed_count += 1

        except Exception as e:
            print(f"[Robot] Execution error: {e}")

    def receive_waypoint(self) -> Optional[Dict[str, Any]]:
        """Receive one waypoint from network."""
        try:
            # Receive length prefix (4 bytes)
            length_data = self.client_socket.recv(4)
            if not length_data:
                return None

            length = struct.unpack('!I', length_data)[0]

            # Receive payload
            data = b''
            while len(data) < length:
                chunk = self.client_socket.recv(length - len(data))
                if not chunk:
                    return None
                data += chunk

            # Deserialize
            waypoint_data = msgpack.unpackb(data, raw=False)

            # Convert lists back to numpy arrays
            if waypoint_data['left_waypoint'] is not None:
                waypoint_data['left_waypoint'] = np.array(waypoint_data['left_waypoint'])
            if waypoint_data['right_waypoint'] is not None:
                waypoint_data['right_waypoint'] = np.array(waypoint_data['right_waypoint'])

            return waypoint_data

        except Exception as e:
            print(f"[Network] Error receiving waypoint: {e}")
            return None

    def receive_loop(self):
        """Background thread that continuously receives waypoints.

        Only keeps the latest waypoint in buffer. This ensures the execution
        loop always processes fresh data without accumulating lag.
        """
        print("[Receiver] Starting receive thread...")

        while self.running:
            try:
                waypoint = self.receive_waypoint()
                if waypoint is None:
                    print("[Receiver] Connection lost")
                    self.running = False
                    break

                # Update latest waypoint (thread-safe)
                with self.waypoint_lock:
                    if self.latest_waypoint is not None:
                        self.waypoints_skipped += 1
                    self.latest_waypoint = waypoint
                    self.waypoints_received += 1

            except Exception as e:
                print(f"[Receiver] Error: {e}")
                import traceback
                traceback.print_exc()
                self.running = False
                break

        print("[Receiver] Receive thread stopped")

    def execution_loop(self):
        """Main execution loop - processes latest waypoint from receive thread.

        This loop continuously feeds the latest waypoint to the robot.
        move_cartesian_waypoints() blocks naturally, and when it finishes,
        we immediately grab the latest waypoint and execute it.
        """
        print("[Executor] Starting execution loop...")
        print("\n" + ">" * 80)
        print("SAFETY: Pinch both hands for 2s to ENABLE robot control")
        print("EMERGENCY STOP: Pinch both hands for 1s during teleoperation to STOP")
        print(">" * 80 + "\n")

        control_just_enabled = False
        last_stats_time = time.time()

        while self.running:
            try:
                # Get latest waypoint from receive thread (thread-safe)
                with self.waypoint_lock:
                    waypoint_data = self.latest_waypoint
                    # Reset skip counter since we're consuming this waypoint
                    self.waypoints_skipped = 0

                # Wait for first waypoint
                if waypoint_data is None:
                    time.sleep(0.001)  # Short sleep to avoid busy-wait
                    continue

                # Extract pinch distances
                left_pinch = waypoint_data.get('left_pinch', 1.0)
                right_pinch = waypoint_data.get('right_pinch', 1.0)

                # Check emergency stop FIRST
                if self.check_emergency_stop(left_pinch, right_pinch):
                    print("[Executor] Emergency stop triggered - exiting")
                    break

                # Check enable gesture
                gesture_changed = self.check_enable_gesture(left_pinch, right_pinch)

                if gesture_changed and self.control_enabled:
                    control_just_enabled = True
                    self.signal_connection_established()

                # Move to initial position
                if control_just_enabled:
                    self.move_to_initial_position(waypoint_data)
                    control_just_enabled = False
                    continue

                # Execute waypoint if enabled (THIS BLOCKS - which is good!)
                if self.control_enabled and not self.moving_to_initial:
                    self.execute_waypoint(waypoint_data)
                    self.frame_count += 1
                else:
                    # If not enabled, sleep briefly to avoid busy loop
                    time.sleep(0.01)

                # Statistics (print every 1 second instead of every 100 frames)
                current_time = time.time()
                if current_time - last_stats_time >= 1.0:
                    elapsed = current_time - self.start_time
                    fps = self.executed_count / elapsed if elapsed > 0 else 0
                    status = "ENABLED" if self.control_enabled else "DISABLED"
                    current_timestamp = waypoint_data.get('timestamp', current_time)
                    latency = (current_time - current_timestamp) * 1000  # ms

                    # Calculate how many waypoints arrived while we were executing the last one
                    with self.waypoint_lock:
                        skipped_this_period = self.waypoints_skipped

                    print(f"[Stats] Executed: {self.executed_count} | Exec FPS: {fps:.1f} | Control: {status}")
                    print(f"[Stats] Total Received: {self.waypoints_received} | Skipped last cycle: {skipped_this_period} | Latency: {latency:.0f}ms")
                    last_stats_time = current_time

            except KeyboardInterrupt:
                print("[Executor] Interrupted by user")
                break
            except Exception as e:
                print(f"[Executor] Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    def start(self):
        """Start robot executor."""
        print("\n" + "=" * 80)
        print("ROBOT EXECUTOR - Network → Robot")
        print("=" * 80)
        print(f"Listening on port: {self.port}")
        print(f"Robot frequency: {self.freq}Hz")
        print(f"Dry-run: {'YES' if self.args.dry_run else 'NO'}")

        # Initialize robot
        if not self.args.dry_run:
            if not ROBOT_AVAILABLE:
                print("\n[ERROR] Astribot API not available. Use --dry-run for testing.")
                return False

            try:
                print("\n[Robot] Connecting to Astribot...")
                self.robot = Astribot()
                print("[Robot] Moving to home position...")
                self.robot.move_to_home()
                self.robot_ready = True
                print("[Robot] Ready!")
            except Exception as e:
                print(f"[ERROR] Failed to initialize robot: {e}")
                return False
        else:
            print("\n[Dry-run] Skipping robot initialization")

        # Start TCP server
        print(f"\n[Network] Starting TCP server on port {self.port}...")
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(1)
            print(f"[Network] Listening for connections...")

            # Wait for data streamer connection
            self.client_socket, client_address = self.server_socket.accept()
            self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[Network] Connected to data streamer at {client_address}")

        except Exception as e:
            print(f"[ERROR] Failed to start server: {e}")
            return False

        # Start receive thread
        self.running = True
        self.receive_thread = threading.Thread(target=self.receive_loop, daemon=True)
        self.receive_thread.start()
        print("[Network] Receive thread started")

        # Start execution loop
        print("\n[Executor] Starting...")
        print("Press Ctrl+C to stop")
        print("-" * 80)

        try:
            self.execution_loop()
        except KeyboardInterrupt:
            print("\n[Executor] Stopping...")
        finally:
            self.stop()

        return True

    def stop(self):
        """Stop executor and cleanup."""
        print("\n[Executor] Stopping...")
        self.running = False

        # Wait for receive thread to finish
        if self.receive_thread is not None and self.receive_thread.is_alive():
            print("[Network] Waiting for receive thread to stop...")
            self.receive_thread.join(timeout=2.0)
            if self.receive_thread.is_alive():
                print("[Network] Receive thread did not stop cleanly")

        # Return robot to home
        if self.robot_ready and self.robot is not None:
            try:
                print("[Robot] Returning to home position...")
                self.robot.move_to_home()
            except Exception as e:
                print(f"[Robot] Error returning home: {e}")

        # Close sockets
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        # Statistics
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        skip_rate = (self.waypoints_skipped / self.waypoints_received * 100) if self.waypoints_received > 0 else 0

        print("\n" + "=" * 80)
        print("EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Total frames: {self.frame_count}")
        print(f"Executed frames: {self.executed_count}")
        print(f"Waypoints received: {self.waypoints_received}")
        print(f"Waypoints skipped: {self.waypoints_skipped} ({skip_rate:.1f}%)")
        print(f"Average FPS: {fps:.1f}")
        print(f"Duration: {elapsed:.1f}s")
        print("=" * 80)


def parse_arguments():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Robot Executor: Network → Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--port", type=int, default=50051,
                        help="TCP port to listen on (default: 50051)")
    parser.add_argument("--freq", type=float, default=250.0,
                        help="Robot control frequency (Hz)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run mode (no robot execution)")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    executor = RobotExecutor(args)
    success = executor.start()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())