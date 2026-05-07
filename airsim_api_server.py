import time
import argparse
import base64
import io
import signal
from contextlib import contextmanager
from flask import Flask, request, jsonify, Response
from simulator_interface.airsim_client_wrapper import AirSimClientWrapper
from utils.gym_utils import Pose
from utils.utils import get_unreal_binary, get_active_settings_file 
import os
import yaml
import pathlib
import numpy as np

_STEP_HARD_TIMEOUT_S = 90  # kill /step if AirSim hangs longer than this


@contextmanager
def _step_timeout(seconds: int):
    """Raise TimeoutError if the block takes longer than *seconds*."""
    def _handler(signum, frame):
        raise TimeoutError(f"AirSim /step timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

CAM_NAME = "0"

app = Flask(__name__)

# Global variables to be set via command-line arguments
env = None
image_save_dir = None


@app.route("/step", methods=["POST"])
def step() -> Response:
    """Advance the simulator and return pose, collisions, and image output."""
    data = request.json
    x, y, z, yaw, current_idx, image_save_dir, annotation = data["x"], data["y"], data["z"], data["yaw"], data.get("current_idx", 0), data.get("image_save_dir", None), data.get("annotation", None)
    send_image = data.get("send_image", False)
    action = Pose(x, y, z, 0, 0, yaw)
    print(f"Stepping to: {action}, {annotation = }")
    if image_save_dir is None:
        image_save_dir = "/tmp"
    try:
        with _step_timeout(_STEP_HARD_TIMEOUT_S):
            obs, reward, done, info = env.step(action)
            stepped_top_down_map_path = os.path.join(image_save_dir, f"stepped_top_down_map_{current_idx}.png")
            # obs.topdown_map.save(stepped_top_down_map_path)
            time.sleep(0.5)  # Wait for the environment to stabilize
            img = env.get_image(CAM_NAME, save_dir=None, annotation=annotation)
    except TimeoutError as exc:
        print(f"   ✘ /step timed out: {exc}")
        return jsonify({"status": "error", "message": str(exc), "image_path": None, "collision_info": None}), 504
    img_path = os.path.join(image_save_dir, f"airsim_latest_{current_idx}.png")
    image_payload = None
    if img:
        img.save(img_path)
        if send_image:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    collision_info = info.get("collision_info") if isinstance(info, dict) else None
    if collision_info is None and obs is not None:
        collision_info = getattr(obs, "collisions", None)
    if collision_info is None:
        collision_info = env.get_collision_info()
    collided = bool(info.get("collided", False)) if isinstance(info, dict) else bool(collision_info)
    collision_state = info.get("collision_state") if isinstance(info, dict) else None
    movement_mode = info.get("movement_mode") if isinstance(info, dict) else None
    current_pose = env.get_current_pose()
    msg = {
        "stepped_top_down_map_path": stepped_top_down_map_path,
        "image_path": img_path if img else None,
        "image_base64": image_payload if send_image else None,
        "collision_info": collision_info,
        "collided": collided,
        "collision_state": collision_state,
        "movement_mode": movement_mode,
        "current_pose": str(current_pose),  # Keep for backward compatibility
        "pose": {
            "x": current_pose.x,
            "y": current_pose.y,
            "z": current_pose.z,
            "yaw_rad": current_pose.yaw,  # Internal storage (radians)
            "yaw_deg": current_pose.yaw_degrees  # Converted to degrees
        }
    }
    if collided:
        print(
            f"⚠ Collision detected in /step: info={collision_info}, "
            f"state={collision_state}, movement_mode={movement_mode}"
        )
    return jsonify(msg)

@app.route("/reset", methods=["POST"])
def reset() -> Response:
    """Reset simulator and optionally set a new center/start pose.

    Accepted JSON payloads:
      {"center_position": {"x": ..., "y": ..., "z": ..., "yaw": ...}}
      {"x": ..., "y": ..., "z": ..., "yaw": ...}
    """
    data = request.get_json(silent=True) or {}

    center_position = None
    center_payload = data.get("center_position", data)
    if isinstance(center_payload, dict) and all(
        key in center_payload for key in ("x", "y", "z")
    ):
        try:
            center_position = Pose(
                x=float(center_payload["x"]),
                y=float(center_payload["y"]),
                z=float(center_payload["z"]),
                roll=float(center_payload.get("roll", 0.0)),
                pitch=float(center_payload.get("pitch", 0.0)),
                yaw=float(center_payload.get("yaw", 0.0)),
            )
        except (TypeError, ValueError) as e:
            return jsonify({
                "status": "error",
                "message": f"Invalid center_position payload: {e}",
            }), 400

    try:
        obs = env.reset(center_position=center_position)
    except Exception as exc:
        # One-shot recovery path for transient RPC disconnects.
        print(f"   ⚠ /reset failed: {exc}")
        print("   Attempting simulator reconnect and one retry...")
        try:
            env.connected = False
            env.connect()
            obs = env.reset(center_position=center_position)
            print("   ✓ /reset recovered after reconnect")
        except Exception as recovery_exc:
            print(f"   ✘ /reset recovery failed: {recovery_exc}")
            return jsonify({
                "status": "error",
                "message": (
                    "Simulator reset failed after reconnect attempt. "
                    f"initial_error={exc}; recovery_error={recovery_exc}"
                ),
            }), 503

    initial_top_down_map = getattr(env, "topdown_map", None)
    initial_top_down_map_with_drone = getattr(env, "topdown_map_with_drone", None)

    initial_top_down_map_path = os.path.join(
        image_save_dir,
        "initial_top_down_map.png",
    )
    initial_top_down_map_with_drone_path = os.path.join(
        image_save_dir,
        "initial_top_down_map_with_drone.png",
    )

    if initial_top_down_map is not None:
        try:
            initial_top_down_map.save(initial_top_down_map_path)
        except Exception as e:
            print(f"Warning: could not save initial_top_down_map: {e}")
            initial_top_down_map_path = None
    else:
        initial_top_down_map_path = None

    if initial_top_down_map_with_drone is not None:
        try:
            initial_top_down_map_with_drone.save(
                initial_top_down_map_with_drone_path
            )
        except Exception as e:
            print(
                "Warning: could not save initial_top_down_map_with_drone: "
                f"{e}"
            )
            initial_top_down_map_with_drone_path = None
    else:
        initial_top_down_map_with_drone_path = None
    img = getattr(obs, "image", None) if obs is not None else None
    if img is None:
        img = env.get_image(CAM_NAME, save_dir=image_save_dir)
    img_path = os.path.join(image_save_dir, "airsim_initial.png")
    if img:
        img.save(img_path)
    collision_info = env.get_collision_info()
    current_pose = getattr(obs, "pose", None) if obs is not None else None
    if current_pose is None:
        current_pose = env.get_current_pose()
    msg = {
        "initial_top_down_map_path": initial_top_down_map_path,
        "initial_top_down_map_with_drone_path": initial_top_down_map_with_drone_path,
        "image_path": img_path if img else None,
        "collision_info": collision_info,
        "current_pose": str(current_pose),
        "reset_pose_error": getattr(env, "last_reset_pose_error", None),
    }
    print(f"{msg = }")
    return jsonify(msg)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AirSim API Server')
    parser.add_argument('--config', type=str,
                       default="./configs/config.yaml",
                       help='Path to config.yaml file')
    parser.add_argument('--env-yaml', type=str,
                       default="./configs/environments.yaml",
                       help='Path to environments.yaml file')
    parser.add_argument('--image-dir', type=str,
                       default="/tmp/airsim_imgs",
                       help='Directory to save images')
    parser.add_argument('--host', type=str,
                       default="0.0.0.0",
                       help='Host to bind the server to')
    parser.add_argument('--port', type=int,
                       default=5001,
                       help='Port to bind the server to')
    parser.add_argument('--rendering', action='store_true',
                       help='Enable rendering in AirSim')
    args = parser.parse_args()
    
    # Set global variables
    image_save_dir = args.image_dir
    
    # Create image directory if it doesn't exist
    pathlib.Path(image_save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Prefer AIRSIM_ENVIRONMENT env var (set by environment manager for dynamic switching)
    # Fall back to config.yaml or default
    env_name = os.getenv("AIRSIM_ENVIRONMENT") or config.get("environment", "AirSimNH")
    
    # Get unreal binary and settings file
    unreal_binary = get_unreal_binary(env_name, args.env_yaml)
    settings_file_path = get_active_settings_file(args.config)
    
    env_source = "AIRSIM_ENVIRONMENT env var" if os.getenv("AIRSIM_ENVIRONMENT") else "config.yaml"
    print(f"Starting AirSim API Server...")
    print(f"Config: {args.config}")
    print(f"Environment YAML: {args.env_yaml}")
    print(f"Environment: {env_name} (from {env_source})")
    print(f"Unreal Binary: {unreal_binary}")
    print(f"Settings File: {settings_file_path}")
    print(f"Image Directory: {image_save_dir}")
    print(f"Rendering: {args.rendering}")
    print(f"Server: {args.host}:{args.port}")
    start_pose = Pose(x=0, y=0, z=-5, yaw=0)
    # Initialize AirSim environment
    env = AirSimClientWrapper(
        unreal_binary,
        rendering_enabled=args.rendering,
        settings_file_path=settings_file_path,
        center_position=start_pose,
        collision_check_mode=bool(config.get("CollisionCheckMode", False))
    )
    env.reset()
    
    print(f"\n✓ AirSim environment initialized successfully!")
    print(f"✓ Server running on http://{args.host}:{args.port}")
    
    # Run Flask server in single-threaded mode (msgpackrpc requires event loop in main thread)
    app.run(host=args.host, port=args.port, threaded=False)