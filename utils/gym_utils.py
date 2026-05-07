from PIL import Image
from typing import Optional, List
import math


class Pose:
    def __init__(self, x: float, y: float, z: float, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
        """Simple class to represent a 3D pose, including position and orientation(must be given in degrees).`
        """
        self.x = x
        self.y = y
        self.z = z
        self.roll = math.radians(roll)
        self.pitch = math.radians(pitch)
        self.yaw = math.radians(yaw)

    @property
    def yaw_degrees(self):
        return math.degrees(self.yaw)

    def __repr__(self):
        return (f"Pose(x={self.x}, y={self.y}, z={self.z}, "
                f"roll={self.roll}, pitch={self.pitch}, yaw={self.yaw})")


class Observation:
    def __init__(self, image: Optional[Image.Image], pose: Pose, collisions: List[str], topdown_map: Optional[Image.Image] = None):
        self.image = image
        self.pose = pose
        self.collisions = collisions  # List of object names that the drone collided with
        self.topdown_map = topdown_map

    def __repr__(self):
        return f"Observation(image={type(self.image).__name__}, pose={self.pose}, collisions={self.collisions}, topdown_map={type(self.topdown_map).__name__})"