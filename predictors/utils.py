from PIL import Image
import base64
import io
import imageio.v3 as iio
from pathlib import Path
from typing import Union, Tuple
import os
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
import sys
from pathlib import Path
import os
import pathlib  
import numpy as np

def add_module_path(module_dir_name: str, current_file: str):
    """
    Adds the absolute path of a module directory (sibling to the current file's parent)
    to sys.path for dynamic imports.

    Args:
        module_dir_name (str): Name of the module directory (e.g., "airsim-interface", "prompter")
        current_file (str): __file__ of the calling script
    """
    current_file = pathlib.Path(os.path.abspath(current_file))
    module_path = (current_file.parent.parent.parent / module_dir_name).resolve()
    print(f"Adding module path: {module_path}")
    sys.path.insert(0, str(module_path))

def setup_environment(
    use_langsmith: bool = False, langsmith_project: str = "hlap-demo", env_file_name: str = ".env"
) -> None:
    """
    Load environment variables and configure API keys.

    This function should be called at the start of any script using the planner module.
    It handles:
    - Loading .env file
    - Disabling LangSmith by default (overriding .env settings)
    - Optionally enabling LangSmith with timestamped project names
    - Verifying API keys are loaded

    Required environment variables:
    - OPENAI_API_KEY: For GPT models (gpt-4o, gpt-4o-mini)
    - ANTHROPIC_API_KEY: For Claude models (claude-3-5-sonnet, etc.)
    - GOOGLE_API_KEY: For Gemini models (gemini-1.5-pro, gemini-1.5-flash, etc.)
    - AWS_ACCESS_KEY_ID: For AWS Bedrock models (amazon.*, anthropic.* on Bedrock)
    - AWS_SECRET_ACCESS_KEY: For AWS Bedrock models
    - AWS_DEFAULT_REGION: For AWS Bedrock (e.g., us-east-1)

    Optional environment variables:
    - LANGSMITH_API_KEY: For LangSmith tracing (only if use_langsmith=True)

    Args:
        use_langsmith: Whether to enable LangSmith tracing (default: False)
        langsmith_project: Base name for LangSmith project (timestamped automatically)

    Example:
        >>> from planner.utils import setup_environment
        >>> setup_environment(use_langsmith=True, langsmith_project="my-experiment")
        ✓ Environment configured successfully
        ✓ Langsmith logging enabled
          - Project: my-experiment_20250121_143052
    """
    # Load .env file from project root
    load_dotenv(env_file_name)  # update with your .env file

    # Default: Disable LangSmith tracing
    # This overrides any LANGSMITH_TRACING=true from .env file
    # Only enable if explicitly requested via use_langsmith=True parameter
    os.environ["LANGSMITH_TRACING"] = "false"

    # Verify API keys are loaded
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION")
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not openai_key and not anthropic_key and not google_key and not aws_access_key:
        print("⚠ Warning: No API keys found in environment!")
        print("   Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or AWS credentials in your .env file")
    else:
        print("✓ Environment configured successfully")
        if openai_key:
            print(f"  - OpenAI API key loaded: ***{openai_key[-4:]}")
        if gemini_key:
            print(f"  - Gemini API key loaded: ***{gemini_key[-4:]}")
        if anthropic_key:
            print(f"  - Anthropic API key loaded: ***{anthropic_key[-4:]}")
        if google_key:
            print(f"  - Google API key loaded: ***{google_key[-4:]}")
        if aws_access_key and aws_secret_key:
            print(f"  - AWS Access Key loaded: ***{aws_access_key[-4:]}")
            print(f"  - AWS Region: {aws_region or 'us-east-1 (default)'}")
        if use_langsmith and not langsmith_key:
            print("⚠ Warning: LANGSMITH_API_KEY not found! Langsmith logging disabled.")
            use_langsmith = False
        elif use_langsmith:
            print(f"  - LangSmith API key loaded: ***{langsmith_key[-4:]}")

    # Date time setup for logging
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")
    langsmith_project_name = f"{langsmith_project}_{current_time}"

    # Control LangSmith tracing based on use_langsmith parameter
    if use_langsmith:
        # Import here to avoid circular dependency
        from utils import langsmith_logging

        # Enable tracing and set project name
        langsmith_logging.langsmith(project_name=langsmith_project_name)
        print("✓ Langsmith logging enabled")
        print("  - Project:", langsmith_project_name)
    else:
        # Tracing already disabled above (LANGSMITH_TRACING=false)
        print("✓ Langsmith logging disabled")


def display_results_one_stage(state: Dict, show_raw: bool = False) -> None:
    """
    Display planning results for one stage planner in a formatted way.

    Args:
        state: Final state from graph execution
        show_raw: Whether to show raw VLM response
    """
    # Display raw response if requested
    if show_raw and state.get("raw_response"):
        print("\n" + "-" * 60)
        print("RAW VLM RESPONSE")
        print("-" * 60)
        print(state["raw_response"])
        print("-" * 60)

    # Display parsed actions
    print("\n" + "=" * 60)
    print("HIGH-LEVEL ACTION SEQUENCE")
    print("=" * 60)

    parsed_actions = state.get("parsed_actions", [])
    if not parsed_actions:
        print("No actions were parsed from the response")
    else:
        for action in parsed_actions:
            print(f"  {action['order']}. {action['action']}")

    print("=" * 60)
    print(f"Total actions: {len(parsed_actions)}")

    # Display evaluation results if available
    if state.get("evaluation_result"):
        eval_result = state["evaluation_result"]
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Overall Similarity: {eval_result['overall_similarity']:.2%}")
        print(f"Exact Matches: {eval_result['exact_matches']}/{eval_result['num_predicted']}")
        print(f"Predicted: {eval_result['num_predicted']} actions")
        print(f"Ground Truth: {eval_result['num_ground_truth']} actions")
        print("=" * 60)

    
# Calculate angular deviation with proper wrapping
# Normalize angle difference to [-180, 180] range
def normalize_angle_difference(angle1, angle2):
    """
    Calculate the shortest angular difference between two angles.
    Handles wraparound (e.g., 180° and -180° are the same).
    
    Returns difference in range [-180, 180] degrees.
    """
    diff = angle1 - angle2
    # Normalize to [-180, 180]
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff
    
def format_drone_pose(pose: Optional[Dict[str, float]]) -> str:
    """
    Format drone pose dict into readable string.

    Args:
        pose: Dict with keys: x, y, z, roll, pitch, yaw

    Returns:
        Formatted string describing drone state
    """
    if not pose:
        return "No pose data"
    roll = pose.get("roll", 0.0)
    pitch = pose.get("pitch", 0.0)
    return f"Position: ({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f})m | Orientation: R={roll:.1f}° P={pitch:.1f}° Y={pose['yaw']:.1f}°"

def stitch_images_to_video(image_folder, image_files, output_path=None, fps=10, codec='libx264', quality=8):
    """
    If output_path is provided, save the video to that path and return True/False for success.
    If output_path is None, return video bytes (for Streamlit sidebar).
    """
    # if not image_files:
    #     return False if output_path else None
    # try:
    first_img = Image .open(os.path.join(image_folder, image_files[0]))
    size = first_img.size
    frames = []
    for fname in image_files:
        img = Image .open(os.path.join(image_folder, fname)).convert('RGB').resize(size)
        frames.append(np.array(img))
    if output_path:
        iio.imwrite(output_path, frames, fps=fps, codec=codec, quality=quality)
        return True
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            video_path = tmpfile.name
        iio.imwrite(video_path, frames, fps=fps, codec=codec, quality=quality)
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        os.remove(video_path)
        return video_bytes
        
def format_camera_orientation(camera_orientation: Optional[Dict[str, float]]) -> str:
    """
    Format camera orientation dict into readable string.

    Args:
        camera_orientation: Dict with keys: roll, pitch, yaw

    Returns:
        Formatted string describing drone state
    """
    if not camera_orientation:
        return "No camera orientation data"

    return f"Camera Orientation: R={camera_orientation['roll']:.1f}° P={camera_orientation['pitch']:.1f}° Y={camera_orientation['yaw']:.1f}°"

def create_initial_state_one_stage(
    image: str,
    instruction: str,
    model_name: str = "gpt-4o-mini",
    drone_pose: Optional[Dict[str, float]] = None,
    camera_orientation: Optional[Dict[str, float]] = None,
    ground_truth: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Create initial state dict for one stage graph execution.

    Args:
        image: Path to image file, URL, or PIL Image
        instruction: Mission objective
        model_name: VLM model to use
        drone_pose: Current drone position and orientation
        camera_orientation: Current camera orientation relative to drone body
        ground_truth: Optional expected actions for evaluation
        metadata: Optional metadata for LangSmith tracing

    Returns:
        Initial state dict ready for graph.invoke()
    """
    return {
        "image": image,
        "instruction": instruction,
        "drone_pose": drone_pose,
        "camera_orientation": camera_orientation,
        "model_name": model_name,
        "raw_response": None,
        "parsed_actions": None,
        "ground_truth": ground_truth,
        "evaluation_result": None,
        "metadata": metadata,
    }


def create_initial_state_two_stage(
    tolerances,
    mission: dict,
    image: str,
    instruction: str,
    drone_pose: Dict[str, float],
    camera_orientation: Dict[str, float],
    model_name: str = "gpt-4o-mini",
    low_level_model_name: Optional[str] = None,
    ground_truth: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Create initial state dict for two-stage graph execution.

    Args:
        image: Path to image file, URL, or PIL Image
        instruction: Mission objective
        drone_pose: Current drone position and orientation
        camera_orientation: Current camera orientation relative to drone body
        model_name: VLM model for high-level planning
        low_level_model_name: VLM model for waypoint generation (defaults to model_name)
        ground_truth: Optional expected high-level actions for evaluation
        metadata: Optional metadata for LangSmith tracing

    Returns:
        Initial state dict ready for graph.invoke()
    """
    # in order to consider a mission successful, the final waypoint must be within these tolerances of the goal

    return {
        "mission": mission,
        "tolerances": tolerances,
        "image": image,
        "instruction": instruction,
        "drone_pose": drone_pose,
        "camera_orientation": camera_orientation,
        "model_name": model_name,
        "low_level_model_name": low_level_model_name,
        # Stage 1 outputs (initially None)
        "raw_response": None,
        "parsed_actions": [],
        # Stage 2 tracking (initially None)
        "current_action_index": 1,
        "current_action": None,
        # Stage 2 outputs (initially None)
        "raw_waypoint_response": None,
        "waypoints": None,
        "all_waypoints": None,
        # Evaluation (optional)
        "ground_truth": ground_truth,
        "evaluation_result": None,
        # Metadata (optional)
        "metadata": metadata,
    }

def create_initial_step_by_step(
    tolerances,
    mission: dict,
    image: str,
    instruction: str,
    drone_pose: Dict[str, float],
    camera_orientation: Dict[str, float],
    model_name: str = "gpt-4o-mini",
    low_level_model_name: Optional[str] = None,
    ground_truth: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    scenario_results_dir="data/results/step_by_step",
    img_enhancement_type = {},
    high_level_action_prompt_type = "step_by_step",
    strategy: str = "vanilla_step_by_step",
    num_history_images: int = 1,
    capture_interval: float = 0.0,
) -> Dict:
    """
    Create initial state dict for two-stage graph execution.

    Args:
        image: Path to image file, URL, or PIL Image
        instruction: Mission objective
        drone_pose: Current drone position and orientation
        camera_orientation: Current camera orientation relative to drone body
        model_name: VLM model for high-level planning
        low_level_model_name: VLM model for waypoint generation (defaults to model_name)
        ground_truth: Optional expected high-level actions for evaluation
        metadata: Optional metadata for LangSmith tracing

    Returns:
        Initial state dict ready for graph.invoke()
    """
    # in order to consider a mission successful, the final waypoint must be within these tolerances of the goal

    return {
        "mission": mission,
        "tolerances": tolerances,
        "image": image,
        "instruction": instruction,
        "drone_pose": drone_pose,
        "camera_orientation": camera_orientation,
        "model_name": model_name,
        "low_level_model_name": low_level_model_name,
        "spatial_results": None,
        # Stage 1 outputs (initially None)
        "raw_response": None,
        # history tracking
        "img_history": [image],
        "parsed_actions": [],
        "collision_info": [],
        "collided": False,
        "stop_reason": "",
        # Stage 2 tracking (initially None)
        "current_action_index": 0,
        "current_action": None,
        # Stage 2 outputs (initially None)
        "all_raw_waypoint_response": [],
        "all_parsed_waypoints": [],
        "all_drone_poses": [],
        # Evaluation (optional)
        "ground_truth": ground_truth,
        "evaluation_result": None,
        # Metadata (optional)
        "metadata": metadata,
        "scenario_results_dir": scenario_results_dir,
        "img_enhancement_type": img_enhancement_type,
        "strategy": strategy,
        "high_level_prompt_type": high_level_action_prompt_type,
        "high_level_action_prompt_type": high_level_action_prompt_type,  # Keep for backward compat
        "num_history_images": num_history_images,
        "capture_interval": capture_interval,
        }
    
    
def display_results_two_stage(state: Dict, show_raw: bool = False) -> None:
    """
    Display two-stage planning results in a formatted way.

    Args:
        state: Final state from graph execution
        show_raw: Whether to show raw VLM responses
    """
    # Display raw responses if requested
    if show_raw:
        if state.get("raw_response"):
            print("\n" + "-" * 60)
            print("RAW STAGE 1 RESPONSE (High-Level Actions)")
            print("-" * 60)
            print(state["raw_response"])
            print("-" * 60)

    # Display Stage 1: High-level actions
    print("\n" + "=" * 60)
    print(f"STAGE 1: HIGH-LEVEL ACTION SEQUENCE - {state['model_name']}")
    print("=" * 60)

    parsed_actions = state.get("parsed_actions", [])
    if not parsed_actions:
        print("No high-level actions were generated")
        return
    else:
        for action in parsed_actions:
            print(f"  {action['order']}. {action['action']}")

    print("=" * 60)
    print(f"Total high-level actions: {len(parsed_actions)}")

    # Display Stage 2: Waypoints for each action
    print("\n" + "=" * 60)
    print(f"STAGE 2: LOW-LEVEL WAYPOINTS - {state['low_level_model_name']}")
    print("=" * 60)

    all_waypoints = state.get("all_waypoints", [])
    print(f"{all_waypoints=}")
    if not all_waypoints:
        print("No waypoints were generated")
    else:
        total_waypoints = 0
        for i, (action, waypoints) in enumerate(zip(parsed_actions, all_waypoints), 1):
            print(f"\nAction {i}: {action['action']}")
            print(f"  Waypoints ({len(waypoints)}):")
            print(
                f"    {i}. dx={waypoints['dx']:6.1f}m  dy={waypoints['dy']:6.1f}m  dz={waypoints['dz']:6.1f}m  dyaw={waypoints['dyaw']:6.1f}°"
            )
            total_waypoints += len(waypoints)

        print("\n" + "=" * 60)
        print(f"Total waypoints across all actions: {total_waypoints}")

    # Display evaluation results if available
    if state.get("evaluation_result"):
        eval_result = state["evaluation_result"]
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS (High-Level Actions)")
        print("=" * 60)
        print(f"overall_high_level_reasoning_similarity: {eval_result['overall_high_level_reasoning_similarity']:.2%}")
        print(f"exact_matches_high_level: {eval_result['exact_matches_high_level']}/{eval_result['num_predicted']}")
        print(f"Predicted: {eval_result['num_predicted']} actions")
        print(f"Ground Truth: {eval_result['num_ground_truth']} actions")
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS (Low-Level Actions)")
        print("=" * 60)
        print(f"waypoint_deviation: {eval_result['waypoint_deviation']}")
        print(f"gt_waypoints: {eval_result['gt_waypoints']}")
        print(f"predicted_waypoints: {eval_result['predicted_waypoints']}")
        print("=" * 60)

        print("MISSION RESULTS")
        print("=" * 60)
        print(f"Mission Success: {eval_result['mission_passed']}")
        print("=" * 60)


# =============================================================================
# API HELPER FUNCTIONS
# =============================================================================


def create_mission_state(
    mission_id: str,
    instruction: str,
    image: str,
    drone_pose: Dict[str, float],
    camera_orientation: Dict[str, float],
    parsed_actions: List[Dict[str, str]],
    model_name: str = "gpt-4o-mini",
    low_level_model_name: Optional[str] = None,
    planning_mode: str = "two_stage",
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Create initial mission state for iterative API execution.

    Args:
        mission_id: Unique identifier for this mission
        instruction: Mission objective
        image: Initial observation image
        drone_pose: Initial drone position and orientation
        camera_orientation: Initial camera orientation relative to drone body
        parsed_actions: High-level actions from Stage 1
        model_name: VLM model for high-level planning
        low_level_model_name: VLM model for waypoint generation
        planning_mode: Planning mode ("single_stage" or "two_stage")
        metadata: Optional metadata for tracking

    Returns:
        MissionState dict ready for API storage
    """
    import time

    return {
        # Mission identification
        "mission_id": mission_id,
        "instruction": instruction,
        # Planning configuration
        "planning_mode": planning_mode,
        # Current state
        "image": image,
        "drone_pose": drone_pose,
        "camera_orientation": camera_orientation,
        "model_name": model_name,
        "low_level_model_name": low_level_model_name,
        # Stage 1 results
        "parsed_actions": parsed_actions,
        "total_actions": len(parsed_actions),
        # Execution tracking
        "current_action_index": 0,
        "execution_status": "initialized",
        # Accumulated history (empty at start)
        "completed_actions": [],
        "all_waypoints": [],
        "execution_log": [(time.time(), "info", f"Mission initialized: {instruction}")],
        "drone_trajectory": [drone_pose],
        "errors": [],
        # Metadata
        "metadata": metadata or {},
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def create_action_execution_state(
    image: str, drone_pose: Dict[str, float], camera_orientation: Dict[str, float], current_action: str, model_name: str = "gpt-4o-mini"
) -> Dict:
    """
    Create state for executing a single action.

    Args:
        image: Current observation image
        drone_pose: Current drone position and orientation
        camera_orientation: Current camera orientation relative to drone body
        current_action: High-level action to execute
        model_name: VLM model for waypoint generation

    Returns:
        ActionExecutionState dict ready for graph execution
    """
    return {
        # Inputs
        "image": image,
        "drone_pose": drone_pose,
        "camera_orientation": camera_orientation,
        "current_action": current_action,
        "model_name": model_name,
        # Outputs (initially None)
        "raw_waypoint_response": None,
        "waypoints": None,
        "execution_time": None,
        # Logging (empty at start)
        "messages": [],
    }



def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Load image from file path"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(path)


def resize_image(image: Image.Image, max_size: Tuple[int, int] = (800, 600)) -> Image.Image:
    """Resize image while maintaining aspect ratio"""
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    if format.upper() == "JPEG":
        # Convert RGBA to RGB if necessary for JPEG
        if image.mode == "RGBA":
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1])
            image = rgb_image
        image.save(buffer, format=format, quality=quality)
    else:
        image.save(buffer, format=format)

    return base64.b64encode(buffer.getvalue()).decode()


def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    # Remove data URL prefix if present
    if base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]

    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))