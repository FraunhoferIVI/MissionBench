import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add the utils directory to the system path
from utils import Pose
from utils import get_movement
from pose_test_cases import MOVEMENT_TEST_CASES


@pytest.mark.parametrize("pose1, pose2, expected_output", MOVEMENT_TEST_CASES)
def test_get_movement(pose1, pose2, expected_output):
    output = get_movement(pose1, pose2)
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

if __name__ == "__main__":
    pose1 = Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0)
    pose2 = Pose(x=1, y=0, z=0, roll=0, pitch=0, yaw=0)
    expected_output = "move forward 1 meter"
    output = get_movement(pose1, pose2)
    print(f"Output: {output}")
