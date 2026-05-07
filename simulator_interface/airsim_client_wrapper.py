# standard imports
import math
import os
import json

import yaml
import cosysairsim as airsim
import numpy as np
import sys
import pathlib
from PIL import Image, ImageDraw, ImageFont
import io
import datetime
import time
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    # Fall back to legacy gym if gymnasium isn't available
    import gym
    from gym import spaces
from typing import Optional, Union, List
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
# local imports
from simulator_interface.airsim_compat import patch_cosysairsim_msgpack
from utils.gym_utils import Pose, Observation
from utils.utils import binvox_read_as_3d_array, get_active_settings_file
from simulator_interface.unreal_process_manager import UnrealProcessManager
import argparse
import shutil

patch_cosysairsim_msgpack(airsim)

class AirSimClientWrapper(gym.Env):
    def __init__(self, unreal_binary_path,
                center_position, startup_wait_sec=15,
                rendering_enabled=False,
                settings_file_path=None, 
                topdown_map_altitude=200,
                collision_check_mode=False):
        super().__init__()
        self.unreal_manager = UnrealProcessManager(
            unreal_binary_path=unreal_binary_path,
            startup_wait_sec=startup_wait_sec,
            settings_file_path=settings_file_path
        )
        self.settings_file_path = settings_file_path
        self.sim_mode = self.unreal_manager.sim_mode
        if self.sim_mode == "Multirotor":
            self.client = airsim.MultirotorClient()
        else:
            self.client = airsim.VehicleClient()
        self.connected = False
        self.rendering_enabled = rendering_enabled
        self.map_image = None
        self.center_position = center_position
        self.step_top_down = False # this flag controls whether to include the top-down map in observations during steps (can be expensive to generate every step)
        # Define gym environment spaces
        self.action_space = spaces.Box(
            low=np.array([-100, -100, -100, -np.pi]),  # x, y, z, yaw limits
            high=np.array([100, 100, 0, np.pi]),       # Note: z=0 is ground level, yaw in radians
            dtype=np.float32
        )

        # Observation space will be defined after connecting
        self.observation_space = None
        self.voxel_grid = None
        self.topdown_map_altitude = topdown_map_altitude
        self.unreal_manager._free_rpc_port()
        self.collision_check_mode = collision_check_mode
        self.last_reset_pose_error = None
        self.last_move_used_teleport = False
    def connect(self):
        self.unreal_manager.reset(rendering=self.rendering_enabled)
        self.client.confirmConnection()
        if self.sim_mode == "Multirotor":
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync().join()
        self.connected = True

    def reset(self, center_position=None):
        """Reset the environment and return initial observation"""
        if center_position is not None:
            self.center_position = center_position

        if not self.connected:
            self.connect()

        # Reset drone to initial position
        if self.sim_mode == "Multirotor":
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            if self.center_position is not None:
                # Put multirotor in active flight state first; otherwise it can
                # snap back/fall after pose set right after reset.
                try:
                    self.client.takeoffAsync().join()
                except Exception:
                    pass

                # Instant respawn at configured start pose (no flying to start).
                self._teleport_to_pose(
                    self.center_position,
                    force_ignore_collision=True
                )
                # Keep controller active at teleported pose.
                try:
                    self.client.hoverAsync().join()
                except Exception:
                    pass
            else:
                self.client.takeoffAsync().join()
        else:
            # For CV mode, just reset pose to origin
            origin_pose = airsim.Pose()
            self.client.simSetVehiclePose(origin_pose, ignore_collision=not self.collision_check_mode)
        # reset pose is exactly the configured center position in Multirotor mode.
        if self.sim_mode == "Multirotor" and self.center_position is not None:
            self._teleport_to_pose(
                self.center_position,
                force_ignore_collision=True
            )
            try:
                self.client.hoverAsync().join()
            except Exception:
                pass
            p = self.get_current_pose()
            dx = p.x - self.center_position.x
            dy = p.y - self.center_position.y
            dz = p.z - self.center_position.z
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            self.last_reset_pose_error = {
                "dx": float(dx),
                "dy": float(dy),
                "dz": float(dz),
                "dist": float(dist),
            }
            print(
                "Reset pose error: "
                f"dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}, "
                f"dist={dist:.3f}m"
            )
        else:
            self.last_reset_pose_error = None

        # Return initial observation captured exactly at the finalized reset pose.
        pause_for_initial_obs = False
        try:
            if self.sim_mode == "Multirotor":
                self.client.simPause(True)
                pause_for_initial_obs = True
            observation = self._get_observation()
        finally:
            if pause_for_initial_obs:
                self.client.simPause(False)

        return observation

    def step(self, action: Union[np.ndarray, Pose], topdown=False):
        """Execute action and return observation, reward, done, info"""
        # Action can be either:
        # 1. np.ndarray with [x, y, z, yaw] where yaw is in radians
        # 2. Pose object with x, y, z, and yaw attributes
        self.step_top_down = topdown
        if isinstance(action, Pose):
            target_pose = action
        else:
            # If action is array [x, y, z, yaw], convert to Pose
            target_pose = Pose(action[0], action[1], action[2], yaw=action[3])

        baseline_collision_ts = self._get_collision_timestamp()
        self.move_to_pose(target_pose)

        collision_state = self._detect_collision_event(
            baseline_timestamp=baseline_collision_ts
        )
        collision_info = self._collision_state_to_names(collision_state)
        observation = self._get_observation(collision_override=collision_info)
        reward = 0.0  # at the moment we dont use any reward
        done = False   # Define your termination condition
        info = {
            "sim_mode": self.sim_mode,
            "collision_checked": self.collision_check_mode,
            "collision_mode": "physics" if self.sim_mode == "Multirotor" else "overlap_or_contact",
            "movement_mode": (
                "teleport_fallback" if self.last_move_used_teleport else "physics"
            ),
            "collision_info": collision_info,
            "collision_state": collision_state,
            "collision_count": len(collision_info),
            "collided": bool(collision_state.get("has_collided", False)),
        }

        return observation, reward, done, info

    def _collision_state_to_names(self, collision_state: dict) -> List[str]:
        """Convert collision state dict to legacy list[str] format."""
        if not collision_state.get("has_collided", False):
            return []
        object_name = collision_state.get("object_name") or "Unknown"
        return [object_name]

    def _get_collision_timestamp(self) -> int:
        """Best-effort collision timestamp retrieval (0 if unavailable)."""
        try:
            collision = self.client.simGetCollisionInfo()
            return int(getattr(collision, "time_stamp", 0) or 0)
        except Exception:
            return 0

    def get_collision_state(self) -> dict:
        """Get detailed collision state from AirSim collision API."""
        try:
            collision = self.client.simGetCollisionInfo()
        except Exception:
            return {
                "has_collided": False,
                "object_name": None,
                "time_stamp": 0,
                "penetration_depth": 0.0,
            }

        object_name = getattr(collision, "object_name", None) or None
        has_collided = bool(getattr(collision, "has_collided", False))
        time_stamp = int(getattr(collision, "time_stamp", 0) or 0)
        penetration_depth = float(getattr(collision, "penetration_depth", 0.0) or 0.0)

        return {
            "has_collided": has_collided,
            "object_name": object_name,
            "time_stamp": time_stamp,
            "penetration_depth": penetration_depth,
        }

    def _detect_collision_event(self, baseline_timestamp: int = 0, window_sec: float = 0.35, poll_dt: float = 0.05) -> dict:
        """Poll collision info for a short window to catch transient contacts."""
        end_t = time.time() + window_sec
        latest = self.get_collision_state()
        if latest.get("has_collided"):
            return latest

        while time.time() < end_t:
            time.sleep(poll_dt)
            current = self.get_collision_state()
            current_ts = int(current.get("time_stamp", 0) or 0)
            if current.get("has_collided"):
                return current
            if baseline_timestamp and current_ts > baseline_timestamp:
                return {
                    **current,
                    "has_collided": True,
                }
            latest = current
        return latest

    def get_collision_info(self) -> List[str]:
        """Get list of objects the drone has collided with."""
        collision_state = self.get_collision_state()
        return self._collision_state_to_names(collision_state)

    def _get_observation(self, collision_override: Optional[List[str]] = None) -> Observation:
        """Get current observation (image + pose + collisions)"""
        current_pose = self.get_current_pose()
        current_image = self.get_image("0", save_dir=None)  # Don't save during environment steps
        current_collisions = (
            collision_override
            if collision_override is not None
            else self.get_collision_info()
        )
        print(f"Current Pose: {current_pose}")
        current_2dmap = None
        return Observation(current_image, current_pose, current_collisions, current_2dmap)

    def get_image(self, camera_name, image_type=airsim.ImageType.Scene, save_dir="./project_logs/imgs",
                  annotation: Optional[str] = None):
        """Return a single image from the specified camera as a PIL image.
        Optionally save the image to save_dir with a unique name if provided."""
        img_bytes = self.client.simGetImage(camera_name, image_type)
        if img_bytes is not None:
            pil_img = Image.open(io.BytesIO(img_bytes))
            

            if save_dir:
                pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                img_path = pathlib.Path(save_dir) / f"cam_{camera_name}_{current_time}.png"
                pil_img.save(img_path)
            return pil_img
        
        return None
    
    def get_cam_info(self, camera_name: str) -> airsim.CameraInfo:
        """Get camera info for the specified camera"""
        try:
            cam_info = self.client.simGetCameraInfo(camera_name)
            return cam_info
        except Exception as e:
            print(f"Error getting camera info for '{camera_name}': {e}")
            return None

    def calculate_camera_intrinsics(self, camera_name: str):
        """Calculate camera intrinsic parameters for the specified camera"""
        cam_info = self.get_cam_info(camera_name)
        if cam_info is None:
            return None
        cameras_settings = self.unreal_manager._get_sim_mode_and_vehicles_from_settings(self.settings_file_path)
        cameras_settings = cameras_settings[1]["MyDrone"]["Cameras"]
        fov_degrees = cam_info.fov
        image_width = cameras_settings[camera_name]["CaptureSettings"][0]["Width"]
        image_height = cameras_settings[camera_name]["CaptureSettings"][0]["Height"]

        hfov_radians = math.radians(fov_degrees)
        hfocal_length = (image_width / 2) / math.tan(hfov_radians / 2)
        
        aspect_ratio = image_width / image_height
        vfocal_length = hfocal_length / aspect_ratio    
        c_x = image_width / 2
        c_y = image_height / 2

        intrinsic_matrix = np.array([
            [hfocal_length, 0, c_x],
            [0, vfocal_length, c_y],
            [0, 0, 1]
        ])

        return intrinsic_matrix
    
    def get_object_pose(self, object_name: str) -> Pose:
        """Get the pose (position and orientation) of an object in the environment"""
        try:
            # Get object pose from AirSim
            object_pose = self.client.simGetObjectPose(object_name)

            # Extract position
            pos = object_pose.position

            # Extract orientation (quaternion) and convert to Euler angles
            q = object_pose.orientation

            # Convert quaternion to Euler angles (roll, pitch, yaw)
            yaw = np.arctan2(2.0*(q.w_val*q.z_val + q.x_val*q.y_val),
                            1.0 - 2.0*(q.y_val**2 + q.z_val**2))
            pitch = np.arcsin(2.0*(q.w_val*q.y_val - q.z_val*q.x_val))
            roll = np.arctan2(2.0*(q.w_val*q.x_val + q.y_val*q.z_val),
                            1.0 - 2.0*(q.x_val**2 + q.y_val**2))

            return Pose(pos.x_val, pos.y_val, pos.z_val, roll, pitch, yaw)

        except Exception as e:
            print(f"Error getting pose for object '{object_name}': {e}")
            return None

    def move_to_pose(self, pose: Pose | airsim.Pose, velocity=50, position_timeout_sec=20.0, yaw_timeout_sec=8.0):
        # Move to position and set orientation
        if isinstance(pose, airsim.Pose):
            yaw_radians = np.arctan2(2.0 * (pose.orientation.w_val * pose.orientation.z_val + pose.orientation.x_val * pose.orientation.y_val),
                                     1.0 - 2.0 * (pose.orientation.y_val**2 + pose.orientation.z_val**2))
            pose = Pose(
                pose.position.x_val,
                pose.position.y_val,
                pose.position.z_val,
                yaw=np.degrees(yaw_radians)
            )
        # print(self.get_cam_info("0"))
        if self.sim_mode == "Multirotor":
            self.last_move_used_teleport = False
            start_t = time.time()
            pos_err = float("inf")

            for _ in range(3):
                self.client.moveToPositionAsync(
                    pose.x,
                    pose.y,
                    pose.z,
                    velocity,
                    timeout_sec=position_timeout_sec,
                ).join()

                # Set orientation using yaw
                if hasattr(pose, 'yaw'):
                    self.client.rotateToYawAsync(
                        np.degrees(pose.yaw),
                        timeout_sec=yaw_timeout_sec,
                    ).join()

                current_pose = self.get_current_pose()
                pos_err = np.sqrt(
                    (current_pose.x - pose.x) ** 2
                    + (current_pose.y - pose.y) ** 2
                    + (current_pose.z - pose.z) ** 2
                )
                if pos_err < 1.0:
                    break
                time.sleep(0.05)

            # Fallback for cases where physics/controller refuses the move.
            if pos_err >= 1.0:
                self.last_move_used_teleport = True
                self._teleport_to_pose(pose, force_ignore_collision=False)
                try:
                    self.client.hoverAsync().join()
                except Exception:
                    pass
                current_pose = self.get_current_pose()
                pos_err = np.sqrt(
                    (current_pose.x - pose.x) ** 2
                    + (current_pose.y - pose.y) ** 2
                    + (current_pose.z - pose.z) ** 2
                )

            end_t = time.time()
            print(
                "move_to_pose finished in "
                f"{end_t - start_t:.2f}s "
                f"(target=({pose.x:.2f}, {pose.y:.2f}, {pose.z:.2f}), "
                f"position_error={pos_err:.2f}m)"
            )
        else:
            # For CV mode, set vehicle pose directly
            airsim_pose = airsim.Pose(
                airsim.Vector3r(pose.x, pose.y, pose.z),
                airsim.euler_to_quaternion(0, 0, pose.yaw)
            )
            self.client.simSetVehiclePose(airsim_pose, ignore_collision=not self.collision_check_mode)

    def _teleport_to_pose(self, pose: Pose | airsim.Pose, force_ignore_collision: bool = True):
        """Instantly place the vehicle at pose using AirSim pose-set API."""
        if isinstance(pose, airsim.Pose):
            airsim_pose = pose
        else:
            airsim_pose = airsim.Pose(
                airsim.Vector3r(pose.x, pose.y, pose.z),
                airsim.euler_to_quaternion(pose.roll, pose.pitch, pose.yaw)
            )

        ignore_collision = (
            True if force_ignore_collision else not self.collision_check_mode
        )

        paused_here = False
        try:
            if self.sim_mode == "Multirotor":
                # Freeze physics while teleporting so controller does not
                # immediately pull the drone away from requested pose.
                self.client.simPause(True)
                paused_here = True

            # In Multirotor, one call can occasionally be ignored right after
            # reset. Retry and verify with a short settle time.
            for _ in range(3):
                self.client.simSetVehiclePose(
                    airsim_pose,
                    ignore_collision=ignore_collision
                )
                time.sleep(0.05)

                current_pose = self.client.simGetVehiclePose()
                dx = current_pose.position.x_val - airsim_pose.position.x_val
                dy = current_pose.position.y_val - airsim_pose.position.y_val
                dz = current_pose.position.z_val - airsim_pose.position.z_val
                if np.sqrt(dx * dx + dy * dy + dz * dz) < 0.5:
                    break
        finally:
            if paused_here:
                self.client.simPause(False)

    def get_current_pose(self) -> Pose:
        if self.sim_mode == "Multirotor":
            # In Multirotor mode, use flight controller state. Using
            # simGetVehiclePose can report a frame that is offset by camera
            # setup in some environments.
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
        else:
            airsim_pose = self.client.simGetVehiclePose()
            pos = airsim_pose.position
            orientation = airsim_pose.orientation
        # Convert quaternion to Euler angles
        q = orientation
        yaw = np.arctan2(2.0*(q.w_val*q.z_val + q.x_val*q.y_val),
                         1.0 - 2.0*(q.y_val**2 + q.z_val**2))
        pitch = np.arcsin(2.0*(q.w_val*q.y_val - q.z_val*q.x_val))
        roll = np.arctan2(2.0*(q.w_val*q.x_val + q.y_val*q.z_val),
                          1.0 - 2.0*(q.x_val**2 + q.y_val**2))
        return Pose(pos.x_val, pos.y_val, pos.z_val, roll, pitch, yaw)

          
if __name__ == "__main__":
    mission_yaml_path = f"./dataset/Visual_Inspection/NHEnv/grey_car_unittest_mission30"
    config_file = "./configs/config.yaml"
    with open(mission_yaml_path, 'r') as f:
        mission_config = yaml.safe_load(f)

    # Load mission configuration
    start_waypoint = mission_config.get("drone_start_pose", {})
    print(f"Start waypoint: {start_waypoint}")
    # gt_waypoint = {"x": -19.199999809265137, "y": 77.5999984741211, "z": -2.0, "yaw": 270}
    gt_waypoint = mission_config.get("ground_truth").get("waypoint_gt", {})

    image_save_dir = "./project_logs/"
    # completely remove the image save direcctory if it exists
    if os.path.exists(image_save_dir):
        shutil.rmtree(image_save_dir)
        print(f"Removed existing image save directory: {image_save_dir}")
    # create the directory
    pathlib.Path(image_save_dir).mkdir(parents=True, exist_ok=True)
    print(f"Created image save directory: {image_save_dir}")
    # Create argument parser
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test AirSimClientWrapper')
    parser.add_argument('--config', type=str, 
                       default="./configs/config.yaml",
                       help='Path to config.yaml file')
    parser.add_argument('--env-yaml', type=str,
                       default="./configs/environments.yaml",
                       help='Path to environments.yaml file')
    args = parser.parse_args()
    
    config_file = args.config
    env_yaml_path = args.env_yaml
    step_image_save_dir = f"{image_save_dir}/test_imgs"
    
    # Create image_save_dir if it doesn't exist
    pathlib.Path(step_image_save_dir).mkdir(parents=True, exist_ok=True)


    # Get environment name from config and binary path from environments.yaml
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    env_name = config.get("environment")
    collision_check_mode = config.get("CollisionCheckMode", False)
    # Import utility function
    from utils.utils import get_unreal_binary
    unreal_binary = get_unreal_binary(env_name, env_yaml_path)
    
    settings_file_path = get_active_settings_file(config_file)

    rendering_enabled = config.get("rendering") # Set to False if you want to run without rendering
    print("Testing AirSimClientWrapper as Gym Environment...")
    print(f"Using config: {config_file}")
    print(f"Using environment: {env_name}")
    print(f"Unreal binary: {unreal_binary}")
    print(f"Settings file: {settings_file_path}")
    print(f"Image save directory: {image_save_dir}")

    env = None
    # try:
    # Try to create environment
    print("Creating AirSim environment...")
    # # Now move the drone to demonstrate tracking
    # print("\n=== Moving Drone ===")

    start_pose = Pose(x=start_waypoint.get("x"), y=start_waypoint.get("y"), z=start_waypoint.get("z"), yaw=start_waypoint.get("yaw")) #env.get_object_pose("Car_10")
    gt_pose = Pose(x=gt_waypoint.get("x"), y=gt_waypoint.get("y"), z=gt_waypoint.get("z"), yaw=gt_waypoint.get("yaw")) #env.get_object_pose("Car_10")
    # pos_red_car = Pose(x=95, y=-72, z=-9, yaw=0) #env.get_object_pose("Car_10")
    # pos_red_car = Pose(x=60, y=3, z=-3, yaw=180) #env.get_object_pose("Car_10")
    # if pos_red_car is None:
    #     pos_red_car = Pose(x=100, y=0, z=-8, yaw=0)  # Default if not found
    env = AirSimClientWrapper(unreal_binary,
                              center_position=start_pose, 
                              rendering_enabled=rendering_enabled, 
                              settings_file_path=settings_file_path,
                              collision_check_mode=collision_check_mode)
    # basketball_hook_pos = env.get_object_pose("Basketball_Hoop_01_424")
    # print(f"Basketball hoop position: {basketball_hook_pos}")
    # Try to connect and reset
    print("Connecting to AirSim...")
    initial_obs = env.reset()
    print("Connected successfully!")
    print(f"Initial observation: {initial_obs}")
    try:
        initial_obs.image.save("./project_logs/test_imgs/initial_fp_image.png")
    except Exception as e:
        print(f"Error saving start fp image: {e}")
    if initial_obs.topdown_map is not None:
        try:
            initial_obs.topdown_map.save("./project_logs/test_imgs/initial_topdown_map.png")
        except Exception as e:
            print(f"Error saving initial topdown map: {e}")

    

    
    # print(f"Moving to position of red car: {pos_red_car}")

    # save voxel grid
    # print("\nSaving voxel grid...")
    # saving the 2d map image
    # print("\nSaving 2D map image...")
    # env.save_2d_map_image()

    # Take a step
    print("\nTaking action step...")
    action = start_pose
    obs, reward, done, info = env.step(action, topdown=True)
    try:
        obs.image.save("./project_logs/test_imgs/start_fp_image.png")
    except Exception as e:
        print(f"Error saving start fp image: {e}")
    if obs.topdown_map is not None:
        try:
            obs.topdown_map.save("./project_logs/test_imgs/start_topdown_map.png")
        except Exception as e:
            print(f"Error saving start topdown images: {e}")

    # action = gt_pose
    # obs, reward, done, info = env.step(action)
    # # obs.image.save("./project_logs/test_imgs/gt_fp_image.png")
    # # obs.topdown_map.save("./project_logs/test_imgs/gt_topdown_map.png")
    # print(f"Step completed. Reward: {reward}, Done: {done}, Info: {info}")
    # print(f"After step observation: {obs}")


    # # Get and save current image for testing
    # print("\nCapturing image...")
    
    # current_image = env.get_image("3", save_dir=image_save_dir)
    # current_image.save(f"{step_image_save_dir}/topdown_image.png")
    # if current_image:
    #     print(f"Image captured and saved. Size: {current_image.size}")
    # else:
    #     print("Failed to capture image")

    # # Change camera pitch to -75 degrees and capture image
    # print("\nChanging camera pitch to -75 degrees...")
    # camera_name = "0"
    # pitch_degrees = -75
    
    # # Get current camera info
    # cam_info = env.client.simGetCameraInfo(camera_name)
    # current_pose = cam_info.pose
    
    # # Create new camera pose with -75 degree pitch
    # # Pitch is rotation around Y-axis in NED coordinates
    # new_orientation = airsim.euler_to_quaternion(
    #     pitch=np.radians(pitch_degrees),  # Convert to radians
    #     roll=0,
    #     yaw=0
    # )
    
    # new_camera_pose = airsim.Pose(
    #     current_pose.position,
    #     new_orientation
    # )
    
    # # Set the new camera pose
    # env.client.simSetCameraPose(camera_name, new_camera_pose)
    # print(f"Camera pitch set to {pitch_degrees} degrees")
    
    # # Capture image with the new camera angle
    # pitch_image = env.get_image(camera_name, save_dir=None)
    
    # if pitch_image:
    #     # Save with pitch in filename
    #     pitch_image_path = f"{step_image_save_dir}/camera_pitch_{pitch_degrees}_image.png"
    #     pitch_image.save(pitch_image_path)
    #     print(f"Image with pitch {pitch_degrees}° saved to: {pitch_image_path}")
    # else:
    #     print(f"Failed to capture image with pitch {pitch_degrees}°")

    print("Test completed successfully!")

    # except Exception as e:
    #     print(f"Error during environment setup or testing: {e}")
    #     print("Attempting to clean up resources...")

    # finally:
    #     # Always try to clean up resources
    #     if env is not None:
    #         try:
    #             print("Stopping Unreal process...")
    #             env.unreal_manager.stop_process()
    #             print("Resources cleaned up successfully.")
    #         except Exception as cleanup_error:
    #             print(f"Error during cleanup: {cleanup_error}")
    #     else:
    #         print("No environment to clean up.")

    #     print("Exiting gracefully.")