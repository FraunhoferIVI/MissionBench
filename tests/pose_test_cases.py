"""Test cases for pose movement detection."""
from utils import Pose


# Test cases for get_movement function
# Format: (pose1, pose2, expected_output)
MOVEMENT_TEST_CASES = [
    # Forward/Backward movement
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0),
        Pose(x=1, y=0, z=0, roll=0, pitch=0, yaw=0),
        "move forward by 1.00 m"
    ),
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0),
        Pose(x=-1, y=0, z=0, roll=0, pitch=0, yaw=0),
        "move backward by 1.00 m"
    ),
    
    # Strafe movement (left/right)
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0),
        Pose(x=0, y=1, z=0, roll=0, pitch=0, yaw=0),
        "strafe right by 1.00 m"
    ),
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0),
        Pose(x=0, y=-1, z=0, roll=0, pitch=0, yaw=0),
        "strafe left by 1.00 m"
    ),
    
    # Up/Down movement
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0),
        Pose(x=0, y=0, z=1, roll=0, pitch=0, yaw=0),
        "move down by 1.00 m"
    ),
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0),
        Pose(x=0, y=0, z=-1, roll=0, pitch=0, yaw=0),
        "move up by 1.00 m"
    ),
    
    # Rotation (yaw changes)
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0),
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=45),
        "turn right by 45.00 degrees"
    ),
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0),
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=-45),
        "turn left by 45.00 degrees"
    ),
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=90),
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=180),
        "turn right by 90.00 degrees"
    ),
    
    # Hovering/Stationary
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0),
        Pose(x=0.1, y=0.1, z=0.1, roll=0, pitch=0, yaw=0),
        "hovering / stationary"
    ),
    (
        Pose(x=5, y=10, z=15, roll=45, pitch=30, yaw=270),
        Pose(x=5, y=10, z=15, roll=45, pitch=30, yaw=270),
        "hovering / stationary"
    ),
    
    # Larger movements
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0),
        Pose(x=5, y=0, z=0, roll=0, pitch=0, yaw=0),
        "move forward by 5.00 m"
    ),
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0),
        Pose(x=0, y=0, z=10, roll=0, pitch=0, yaw=0),
        "move down by 10.00 m"
    ),
    
    # Movement with rotation (rotation takes priority if yaw diff > 2 degrees)
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0),
        Pose(x=1, y=1, z=0, roll=0, pitch=0, yaw=10),
        "turn right by 10.00 degrees"
    ),
    
    # Forward movement with drone rotated 90 degrees
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=90),
        Pose(x=0, y=1, z=0, roll=0, pitch=0, yaw=90),
        "move forward by 1.00 m"
    ),
    
    # Backward movement with drone rotated 45 degrees
    (
        Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=45),
        Pose(x=-0.707, y=-0.707, z=0, roll=0, pitch=0, yaw=45),
        "move backward by 1.00 m"
    ),
]
