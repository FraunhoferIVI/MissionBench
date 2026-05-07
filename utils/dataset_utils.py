def convert_pose_to_relative_movement_commands(pose1:Pose, pose2: Pose):
    """
    1. airsim_rec.txt , get two poses

    """
    x_y_z_precision = 0.01 # cm
    yaw__precsion = 1 # degree

    action_command = ["move forward", "move backward", "strafe left", "strafe right", "ascend", "descend", "turn left", "turn right"]
    # return move forward 5m
    pass

if __name__ == "__main__":
    # read airsim_rec.txt, get the poses, and convert them to relative movement commands
    pass
