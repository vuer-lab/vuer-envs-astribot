import numpy as np
import shutil
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract and execute wrist trajectories from Astribot session data."
    )
    parser.add_argument(
        "--base_path", type=str, default="/Users/yanbinghan/fortyfive/iseeyou/multi-stream-data",
        help="Path to the session base directory (e.g., /home/user/multi-stream-data)"
    )
    parser.add_argument(
        "--trajectory_mode", type=str, default="both",
        choices=["both", "left_only", "right_only"],
        help="Which arm(s) to control: both | left_only | right_only"
    )
    parser.add_argument(
        "--downsample_factor", type=int, default=3,
        help="Frame downsampling factor (e.g., 3 means use every 3rd frame)"
    )
    parser.add_argument(
        "--include_rotations", type=bool, default=True,
        help="Include wrist orientation (True/False)"
    )
    parser.add_argument(
        "--max_jump", type=float, default=0.3,
        help="Maximum allowed position change between waypoints (meters)"
    )
    parser.add_argument(
        "--time_factor", type=float, default=4.0,
        help="Multiplier to make actions slower or faster"
    )
    parser.add_argument(
        "--freq", type=float, default=250.0,
        help="Control frequency of the robot in Hz"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="If set, will execute trajectory on robot without asking for confirmation"
    )
    parser.add_argument(
        "--disable-gripper", action="store_true",
        help="If set, will not control the grippers (only arm position/orientation)"
    )
    return parser.parse_args()


def set_axes_equal(ax):
    """
    Fallback for older Matplotlib: make 3D axes have equal scale by
    expanding limits to a common radius around their midpoints.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_mid = np.mean(z_limits)

    # The plot radius is the half of the max range
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

def quat_angle_deg(q1, q2):
    # ensure unit and account for double cover (q ~ -q)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def rotation_matrix_to_quaternion(rot_matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw].

    Args:
        rot_matrix: 3x3 rotation matrix

    Returns:
        Quaternion as [qx, qy, qz, qw]
    """
    if rot_matrix is None:
        return np.array([0, 0, 0, 1])  # Identity quaternion

    r = Rotation.from_matrix(rot_matrix)
    quat = r.as_quat()  # Returns [qx, qy, qz, qw]
    return quat


def rotation_matrix_gen(x_angle: float, y_angle: float, z_angle: float) -> np.ndarray:
    """
    Generate a 3×3 rotation matrix given rotations around the x, y, and z axes.
    Angles should be provided in degrees.

    Args:
        x_angle: rotation about the X-axis (pitch, in degrees)
        y_angle: rotation about the Y-axis (yaw, in degrees)
        z_angle: rotation about the Z-axis (roll, in degrees)

    Returns:
        np.ndarray: 3×3 composite rotation matrix (Rz * Ry * Rx)
    """

    # Convert degrees to radians
    x_rad = np.radians(x_angle)
    y_rad = np.radians(y_angle)
    z_rad = np.radians(z_angle)

    # Rotation about X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x_rad), -np.sin(x_rad)],
        [0, np.sin(x_rad), np.cos(x_rad)]
    ])

    # Rotation about Y-axis
    Ry = np.array([
        [np.cos(y_rad), 0, np.sin(y_rad)],
        [0, 1, 0],
        [-np.sin(y_rad), 0, np.cos(y_rad)]
    ])

    # Rotation about Z-axis
    Rz = np.array([
        [np.cos(z_rad), -np.sin(z_rad), 0],
        [np.sin(z_rad), np.cos(z_rad), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def delete_small_data_sessions(base_path):
    # 1. Delete directories smaller than 1 MB
    threshold_bytes = 1 * 1024 * 1024  # 1 MB
    for d in base_path.iterdir():
        if d.is_dir():
            total_size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            if total_size < threshold_bytes:
                print(f"Deleting {d} ({total_size / 1024:.1f} KB)")
                shutil.rmtree(d)


def find_latest_session(base_path: Path):
    """Find the most recent valid session directory."""
    delete_small_data_sessions(base_path)
    dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not dirs:
        raise RuntimeError("No valid session directories found.")
    session_path = max(dirs, key=lambda p: p.stat().st_mtime)
    print(f"Latest session: {session_path}")
    return session_path