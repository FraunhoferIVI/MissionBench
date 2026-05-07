import math
from typing import List
from utils import Pose, quaternion_to_eulerian_angle
from pathlib import Path

def get_movement(pose1: Pose, pose2: Pose, threshold: float = 0.5) -> str:
    """Compares two poses and returns the dominant movement direction and distance/degrees."""
    
    # 1. Check for Yaw changes first
    yaw_diff = pose2.yaw_degrees - pose1.yaw_degrees
    yaw_diff = (yaw_diff + 180) % 360 - 180  # Normalize to [-180, 180]
    
    if abs(yaw_diff) > 2.0:  # 2 degrees threshold for rotation
        direction = "turn right" if yaw_diff > 0 else "turn left"
        return f"{direction} by {abs(yaw_diff):.2f} degrees"

    # 2. Calculate global position differences
    dx = pose2.x - pose1.x
    dy = pose2.y - pose1.y
    dz = pose2.z - pose1.z

    # If the drone barely moved, return stationary
    if max(abs(dx), abs(dy), abs(dz)) < threshold:
        return "hovering / stationary"

    # 3. Check for Up/Down movement (AirSim uses NED: -Z is Up, +Z is Down)
    if abs(dz) > max(abs(dx), abs(dy)):
        direction = "move down" if dz > 0 else "move up"
        return f"{direction} by {abs(dz):.2f} m"

    # 4. Project global X/Y deltas onto the drone's local coordinate frame
    yaw_rad = pose1.yaw
    local_x = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)  # Local forward/backward
    local_y = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad) # Local right/left

    # Determine dominant local movement
    if abs(local_x) > abs(local_y):
        direction = "move forward" if local_x > 0 else "move backward"
        return f"{direction} by {abs(local_x):.2f} m"
    else:
        direction = "strafe right" if local_y > 0 else "strafe left"
        return f"{direction} by {abs(local_y):.2f} m"

def parse_airsim_file_and_get_movements(filepath: str, debug: bool=False, img_root: str=None) -> List[str]:
    """Reads the AirSim txt file, creates Pose objects, and calculates movements."""
    poses = []
    movements = []
    img_paths = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        for line in lines[1:]:  # Skip the header row
            if not line.strip():
                continue
                
            parts = line.strip().split('\t')
            img_path = parts[-1]  # Not used in movement calculation, but can be stored if needed
            if img_root is not None:
                img_path = str(Path(img_root) / img_path)
            img_paths.append(img_path)
            # Extract coordinates and quaternions 
            pos_x, pos_y, pos_z = float(parts[2]), float(parts[3]), float(parts[4])
            q_w, q_x, q_y, q_z = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
            
            # Convert quaternion to euler degrees using your provided function
            roll, pitch, yaw = quaternion_to_eulerian_angle(q_w, q_x, q_y, q_z)
            
            # Create Pose object
            current_pose = Pose(pos_x, pos_y, pos_z, roll, pitch, yaw)
            poses.append(current_pose)

        # Compare consecutive poses to find the movement
        for i in range(1, len(poses)):
            if debug:
                print(f"Comparing poses {poses[i-1]} and {poses[i]}")
            action = get_movement(poses[i-1], poses[i])
            movements.append(action)
            
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        
    return img_paths, movements


if __name__ == "__main__":
    file_path = "/home/uam/taehyoung/suman/pose/dataset/Patrol/NHEnv/main_road_mission3/airsim_rec.txt" 
    results = parse_airsim_file_and_get_movements(file_path, debug=True)