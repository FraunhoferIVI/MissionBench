"""
State Definitions for Planning Workflows

This module defines the state structures used in LangGraph workflows:
- HLAPState: Single-stage high-level action planning
- TwoStageState: Two-stage hierarchical planning (high-level → waypoints)
- MissionState: Persistent mission state for iterative API execution
- ActionExecutionState: Single action execution state for API requests
"""

from typing import Annotated, Dict, List, Optional, TypedDict
from langgraph.graph.message import add_messages


# =============================================================================
# CUSTOM REDUCERS
# =============================================================================


def append_to_list(existing: Optional[List], new: any) -> List:
    """
    Reducer: Append single item to list.

    Used for accumulating single items over time (e.g., completed actions).
    """
    if existing is None:
        existing = []
    if new is not None:
        return existing + [new]
    return existing


def extend_list(existing: Optional[List], new: Optional[List]) -> List:
    """
    Reducer: Extend list with new items.

    Used for accumulating multiple items at once (e.g., error logs).
    """
    if existing is None:
        existing = []
    if new is not None and len(new) > 0:
        return existing + new
    return existing



class StepByStepState(TypedDict):
    """
    State for Two-Stage Hierarchical Planning workflow.

    This state manages the complete two-stage process:
    Stage 1: Generate high-level actions from image + instruction
    Stage 2: Convert each high-level action to low-level waypoints

    The state flows through:
    1. Generate high-level actions (like HLAPState)
    2. Loop through each action:
       - Select next action
       - Generate waypoints for that action
       - Parse waypoints
    3. Accumulate all waypoints
    4. Evaluate (optional)

    Attributes:
        # Stage 1 inputs (same as HLAPState)
        image: Path to image file, URL, or PIL Image object
        instruction: Natural language mission objective
        drone_pose: Current drone position and orientation
        camera_orientation: Current camera orientation relative to drone body
        model_name: VLM model for high-level planning
        low_level_model_name: VLM model for waypoint generation (can be different/faster)

        # Stage 1 outputs
        raw_response: Raw text output from Stage 1 VLM
        parsed_actions: List of high-level actions
            Format: [{"order": 1, "action": "turn right 30 degrees"}, ...]

        # Stage 2 state tracking
        current_action_index: Index of action currently being processed (0-based)
        current_action: Current high-level action being converted to waypoints

        # Stage 2 outputs
        raw_waypoint_response: Raw text output from Stage 2 VLM for current action
        waypoints: Parsed waypoints for current action
            Format: [{"dx": 0.0, "dy": 5.0, "dz": 0.0, "dyaw": 0.0}, ...]
        all_waypoints: Accumulated waypoints for all actions (list of lists)
            Format: [[action1_waypoints], [action2_waypoints], ...]

        # Evaluation (optional)
        ground_truth: Optional list of expected high-level actions
        evaluation_result: Optional evaluation metrics

        # Metadata (optional)
        metadata: Optional metadata for LangSmith tracing
    """

    # Stage 1 inputs
    mission: dict
    tolerances: Dict[str, float]
    image: str
    instruction: str
    drone_pose: Optional[Dict[str, float]]
    camera_orientation: Optional[Dict[str, float]]
    model_name: str
    low_level_model_name: Optional[str]  # Can use faster model for waypoints
    # Spatial reasoning (pix2world integration)
    detected_pixel: Optional[tuple]  # (u, v) for spatial reasoning
    spatial_results: Optional[Dict]  # {"distance_m": float, "hit_point_enu": [x,y,z], ...}
    spatial_reasoning: str
    raw_response_hlap_step_by_step: Optional[str]
    should_continue_response: Optional[str]
    collided: Optional[bool]
    collision_info: Optional[List[str]]
    stop_reason: Optional[str]
    # history tracking
    parsed_actions: Optional[List[Dict[str, str]]]
    img_history: Optional[List[str]]
    
    # Stage 2 tracking
    current_action_index: Optional[int]
    current_action: Optional[str]
    # Stage 2 outputs
    all_raw_waypoint_response: Optional[List[str]]
    all_parsed_waypoints: Optional[List[List[Dict[str, float]]]]
    all_drone_poses: Optional[List[Dict[str, float]]]
    
    # Evaluation (optional)
    ground_truth: Optional[List[str]]
    evaluation_result: Optional[Dict[str, float]]

    # Metadata (optional)
    metadata: Optional[Dict]
    scenario_results_dir: str
    # Verbosity (controls node printing)
    verbose: Optional[bool]
    task_gt_pred: Optional[str]

    # Bounding box predictions (one entry per step)
    predicted_bbs: Optional[List[Optional[List[float]]]]         # [[x1,y1,x2,y2] | None, ...]
    predicted_bb_depths: Optional[List[Optional[float]]]         # [depth_m | None, ...]
    predicted_bb_labels: Optional[List[Optional[str]]]           # [target_object_name | None, ...]
    annotated_bb_image_paths: Optional[List[Optional[str]]]      # paths to BB-annotated step images

    # configs
    strategy: str
    img_enhancement_type: dict[str, str]
    high_level_prompt_type: str
    high_level_action_prompt_type: str
    num_history_images: Optional[int]  # Number of images to pass to VLM (1, 2, 3, etc.)
    capture_interval: Optional[float]  # >0 enables captures every N meters and N*10 degrees