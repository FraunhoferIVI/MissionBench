"""
Node Functions for Planning Workflows

This module contains reusable node functions for LangGraph workflows.
Each node function takes a state dict and returns updates to that state.

Single-Stage Node Functions:
- node_generate_actions: Call VLM to generate action sequence
- node_parse_actions: Parse VLM output into structured format
- node_evaluate: Compare against ground truth (if available)

Two-Stage Node Functions:
- node_select_next_action: Select next high-level action to process
- node_generate_waypoints: Call low-level VLM to generate waypoints
- node_parse_waypoints: Parse waypoint output into structured format
- should_continue_waypoint_generation: Check if more actions to process
"""

import pickle
import os
from pprint import pp
import shutil
import json
import time
import math
import re
import signal
import hashlib
from pathlib import Path
from PIL import Image
from typing import Dict, Optional
import sys
sys.path.insert(0, "/".join(Path(__file__).parts[:-2]))  # Adjust the path to import from predictors and graphs
from predictors.vision import ActionPredictor, WaypointGenerator, PixelPredictor
from graphs.state import StepByStepState
from predictors.utils import  normalize_angle_difference
from graphs.patrol_metric import PatrolMetric
from predictors.base import VisionChain
from langchain_core.messages import HumanMessage
from prompter.prompt_generator import generate_prompt
from utils.image_utils import fetch_image, ImageEnhancementUtils
from simulator_interface.airsim_tools import fetch_image_from_simulator
from utils.utils import _append_step_payload


STRATEGY_TO_HIGH_LEVEL_PROMPT_TYPE = {
    "spf_step_by_step": "step_by_step",
    "vanilla_step_by_step": "step_by_step",
    "step_by_step_1vlm": "step_by_step_1vlm",
    "step_by_step_1vlm_with_bb": "step_by_step_1vlm_with_bb",
}

_MAX_INTERMEDIATE_CAPTURE_STEPS = 120
_ANGULAR_CAPTURE_INTERVAL_MULTIPLIER = 10.0


def _capture_intermediate_step_frames(
    state: StepByStepState,
    current_index: int,
    img_enhancement_type: Optional[dict[str, str]] = None,
) -> Optional[tuple[str, list[str], int]]:
    """Capture intermediate images between previous and current pose.

    Returns:
        Tuple of (final_image_path, collision_info, intermediate_count) or None
        when legacy behavior should be used.
    """
    try:
        interval = float(state.get("capture_interval", 0.0) or 0.0)
    except Exception:
        interval = 0.0

    # Backward compatibility: disabled or invalid interval keeps old behavior.
    if interval <= 0:
        return None

    all_poses = state.get("all_drone_poses") or []
    target_pose = state.get("drone_pose") or {}
    if not isinstance(target_pose, dict):
        return None

    if len(all_poses) >= 2 and isinstance(all_poses[-2], dict):
        prev_pose = all_poses[-2]
    else:
        prev_pose = (state.get("mission") or {}).get("drone_start_pose") or {}

    sx = float(prev_pose.get("x", 0.0))
    sy = float(prev_pose.get("y", 0.0))
    sz = float(prev_pose.get("z", 0.0))
    syaw = float(prev_pose.get("yaw", 0.0))

    tx = float(target_pose.get("x", sx))
    ty = float(target_pose.get("y", sy))
    tz = float(target_pose.get("z", sz))
    tyaw = float(target_pose.get("yaw", syaw))

    distance_m = math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2 + (tz - sz) ** 2)
    yaw_delta_deg = abs(normalize_angle_difference(tyaw, syaw))
    angular_interval_deg = interval * _ANGULAR_CAPTURE_INTERVAL_MULTIPLIER

    distance_steps = int(math.ceil(distance_m / interval)) if distance_m > 0 else 1
    yaw_steps = int(math.ceil(yaw_delta_deg / angular_interval_deg)) if yaw_delta_deg > 0 else 1
    steps = max(distance_steps, yaw_steps)

    # Very high interval naturally collapses to one shot (legacy behavior).
    if steps <= 1:
        return None

    # Safety: avoid huge capture bursts; fallback to legacy behavior.
    if steps > _MAX_INTERMEDIATE_CAPTURE_STEPS:
        return None

    scenario_results_dir = Path(state["scenario_results_dir"])
    imgs_dir = scenario_results_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    enhancer = ImageEnhancementUtils() if img_enhancement_type is not None else None
    latest_img_path = None
    latest_collision_info = []
    saved_intermediate = 0
    prev_saved_hash = None

    for i in range(1, steps + 1):
        alpha = i / steps
        x = sx + (tx - sx) * alpha
        y = sy + (ty - sy) * alpha
        z = sz + (tz - sz) * alpha
        yaw = syaw + (tyaw - syaw) * alpha

        result = fetch_image_from_simulator(x=x, y=y, z=z, yaw=yaw)
        raw_img = result.get("image_path")
        collision_info = result.get("collision_info") or []
        if not collision_info and result.get("collided"):
            collision_info = ["Unknown"]
        if collision_info:
            latest_collision_info = collision_info

        if not raw_img:
            continue

        image_to_copy = raw_img
        if enhancer is not None:
            enhanced = enhancer.enhance_image(image_path=raw_img, enhancement_type=img_enhancement_type)
            if enhanced:
                image_to_copy = enhanced

        if i < steps:
            try:
                with open(image_to_copy, "rb") as f:
                    current_hash = hashlib.md5(f.read()).hexdigest()
            except Exception:
                current_hash = None

            # Skip back-to-back duplicate segment frames to avoid apparent
            # timestep jumps caused by repeated visuals in stitched media.
            if current_hash is not None and current_hash == prev_saved_hash:
                continue

            intermediate_path = imgs_dir / f"airsim_latest_{current_index}_seg_{saved_intermediate + 1:03d}.png"
            shutil.copy(image_to_copy, intermediate_path)
            saved_intermediate += 1
            if current_hash is not None:
                prev_saved_hash = current_hash

        latest_img_path = image_to_copy

    if latest_img_path is None:
        return None

    return latest_img_path, latest_collision_info, saved_intermediate


def _resolve_high_level_prompt_type(state: StepByStepState) -> str:
    """Resolve high-level prompt type with backward compatibility.

    Priority:
    1) Explicit state prompt fields
    2) Strategy-derived mapping
    3) Safe default (step_by_step)
    """
    explicit_prompt_type = state.get("high_level_prompt_type") or state.get("high_level_action_prompt_type")
    if isinstance(explicit_prompt_type, str) and explicit_prompt_type:
        return explicit_prompt_type

    strategy = state.get("strategy")
    if isinstance(strategy, str) and strategy:
        return STRATEGY_TO_HIGH_LEVEL_PROMPT_TYPE.get(strategy.lower(), "step_by_step")

    return "step_by_step"

def bb_and_depth_predictor(state: StepByStepState) -> Dict:
    """_summary_

    Args:
        state (StepByStepState): _description_

    Returns:
        Dict: {"spatial_information": {"oject_name": {"distance": float, "world_coordinates": [x, y, z]}}}
        
    """
    step_start = time.time()
    if state.get("verbose"):
        print(f"\n Generating pixel and depth prediction with model: {state['model_name']}")
    # Initialize predictor
    predictor = PixelPredictor(model_name=state["model_name"], mission=state)
    vlm_start = time.time()
    # Generate actions (prompt is saved directly to disk within predict_actions)
    parsed_response_dict = None
    prediction_error = None
    if False: # "spatial_results" in state["mission"]["ground_truth"]:
        print(f"   📝 Using spatial results from previous step: {state['spatial_results']}")
        parsed_response_dict = state["mission"]["ground_truth"]["spatial_results"]
        try:
            shutil.copyfile(
                os.path.join(Path(state["image"]).parent, "annotated_image.png"),
                os.path.join(state.get("scenario_results_dir"), f"annotated_image_step_{state.get('current_action_index',0)}.png"),
            )
        except Exception as exc:
            prediction_error = f"failed to copy annotated image: {exc}"
    else:
        try:
            print("starting to predict bb")
            parsed_response_dict = predictor.predict_bb_and_depth(
                image=state["image"],
                results_dir=state.get("scenario_results_dir"),
            )
            annotated_img = predictor.draw_bounding_boxes(
                Image.open(state["image"]),
                (parsed_response_dict.get("bb"), parsed_response_dict.get("depth"), parsed_response_dict.get("label")),
            )
            print("finished predicting bb")
            annotated_img.save(
                f"{state.get('scenario_results_dir')}/annotated_image_step_{state.get('current_action_index',0)}.png"
            )
        except Exception as exc:
            prediction_error = f"bb/depth prediction failed: {exc}"
    vlm_elapsed = time.time() - vlm_start
    if state.get("verbose"):
        if isinstance(parsed_response_dict, dict):
            print(f"   ✓ Generated spatial information for {len(parsed_response_dict)} objects")
        else:
            print("   ⚠ No spatial information generated")
    current_index = state["current_action_index"]
    step_elapsed = time.time() - step_start
    with open(os.path.join(state.get("scenario_results_dir"), f"spatial_info_step_{current_index}.json"), "w") as f:
        payload = parsed_response_dict if isinstance(parsed_response_dict, dict) else {"error": prediction_error or "spatial_results_unavailable"}
        json.dump(payload, f, indent=2)
    # from planner.spatial_tools import compute_pixel_distance
    # lets take the center point of the boudning box [x_min, y_min, x_max, y_max] and convert to image x and y
    spatial_results = "spatial_results_unavailable"
    u = v = x = y = None
    depth = None
    label = None
    if isinstance(parsed_response_dict, dict):
        bb = parsed_response_dict.get("bb")
        depth = parsed_response_dict.get("depth")
        label = parsed_response_dict.get("label")
        if isinstance(bb, (list, tuple)) and len(bb) == 4 and depth is not None and label:
            u, v = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2  # pixel coordinates (x, y)
            width, height = state["mission"]["camera"]["width"], state["mission"]["camera"]["height"]
            x = int(u / 1000 * width)
            y = int(v / 1000 * height)
            spatial_results = f"{label} at depth {depth}m"
        elif prediction_error is None:
            prediction_error = "invalid or incomplete bb/depth/label in response"
    # spatial_result = compute_pixel_distance.invoke({
    #     "u": x,
    #     "v": y,
    #     "x": state["drone_pose"]["x"],
    #     "y": state["drone_pose"]["y"],
    #     "z": state["drone_pose"]["z"],  # 50m altitude (NED convention)
    #     "pitch": 0.0,  # Pitched down
    #     "yaw": state["drone_pose"]["yaw"],
    #     "roll": 0.0,
    #     "camera_roll": 0.0,
    #     "camera_pitch": -45.0,
    #     "camera_yaw": 0.0})
    # save the individual parameters and the final spatial result
    
    # with open(os.path.join(state.get("scenario_results_dir"), f"spatial_result_step_{current_index}.json"), "w") as f:
    #     json.dump({
    #         "pixel_coordinates": {"u": u, "v": v},
    #         "image_coordinates": {"x": x, "y": y},
    #         "drone_pose": state["drone_pose"],
    #         "spatial_result": spatial_result
    #     }, f, indent=2)
    with open(os.path.join(state.get("scenario_results_dir"), f"spatial_result_step_{current_index}.json"), "w") as f:
        json.dump({
            "pixel_coordinates": {"u": u, "v": v},
            "image_coordinates": {"x": x, "y": y},
            "drone_pose": state["drone_pose"],
            "spatial_results": spatial_results,
            "error": prediction_error,
        }, f, indent=2)
        
    return {"spatial_results": spatial_results}
   
def node_generate_step_by_step_action(state: StepByStepState) -> Dict:
    """
    Node: Generate high-level actions using VLM.

    This node:
    1. Initializes ActionPredictor with specified model
    2. Formats drone pose and camera orientation into context (if provided)
    3. Calls VLM with image + instruction + pose + camera orientation context
    4. Returns raw VLM response

    Args:
        state: Current workflow state containing image, instruction, drone_pose, camera_orientation, model_name

    Returns:
        Dict with 'raw_response' key containing VLM output
    """
    step_start = time.time()
    if state.get("verbose"):
        print(f"\n Generating step by step action with model: {state['model_name']}")
    highlevel_prompt_type = _resolve_high_level_prompt_type(state)
    if highlevel_prompt_type in ("step_by_step_1vlm", "step_by_step_1vlm_with_bb"):
        system_prompt_type = "step_by_step_1vlm_system"
    else:        
        system_prompt_type = None  # for backward compatibility

    # Initialize predictor
    predictor = ActionPredictor(model_name=state["model_name"], mission=state)
    
    # Determine how many images to pass based on num_history_images setting.
    # Semantics: pass the most recent N images in reverse chronological order (newest first).
    # Example with available history [..., img3, img4]: N=2 -> [img4, img3]
    num_images = max(1, int(state.get("num_history_images", 1)))
    img_history = state.get("img_history", [])
    images_to_pass = list(reversed(img_history[-num_images:])) if img_history else []

    if state.get("verbose"):
        print(f"   🖼️ Passing {len(images_to_pass)} image(s) to VLM (requested window={num_images}, history={len(img_history)})")
    # Generate actions (prompt is saved directly to disk within predict_actions)
    raw_response = predictor.predict_actions(
        image=images_to_pass,  # Pass multiple images instead of just the latest
        mission=state,
        prompt_type=highlevel_prompt_type,
        save_prompt_dir=state.get("scenario_results_dir"),
        step_index=state.get("current_action_index", 0),
        system_prompt_type=system_prompt_type,  # Use same prompt type for system prompt to ensure consistency

    )
    # print("\n📝 Raw action response:", raw_response)
    if state.get("verbose"):
        print(f"   ✓ Generated {len(raw_response.split(chr(10)))} lines of output")
    current_index = state["current_action_index"]
    step_elapsed = time.time() - step_start
    # Persist raw action output for debugging (aggregated by step)
    _append_step_payload(
        state.get("scenario_results_dir"),
        current_index,
        {
            "raw_response_hlap_step_by_step": raw_response,
            "time_total_hlap_node": round(step_elapsed, 2),
            # Image history tracking
            "images_used": [os.path.basename(img) for img in images_to_pass],
            "num_images_requested": num_images,
            "num_images_available": len(img_history),
            "image_history_window_semantics": "newest-first (most recent = first in list)"
        },
    )

    return {"raw_response_hlap_step_by_step": raw_response}

def node_parse_step_by_step_action(state: StepByStepState) -> Dict:
    """
    Node: Parse VLM response into structured action.

    Extracts action from text like:
        Next Action: Fly forward 50 meters

    get structured format:
        [{"order": 1, "action": "Fly forward 50 meters"}]

    Args:
        state: Current workflow state containing raw_response

    Returns:
        Dict with 'parsed_actions' key containing list of action dicts
    """
    step_start = time.time()
    if state.get("verbose"):
        print("\n Parsing step-by-step action...")

    raw_response_hlap_step_by_step = state["raw_response_hlap_step_by_step"]
    # Backward-compatible guard: some providers can return stringified dict payloads
    # like {'type': 'text', 'text': '...'}; extract the embedded text when possible.
    if isinstance(raw_response_hlap_step_by_step, str) and "'text':" in raw_response_hlap_step_by_step:
        text_match = re.search(r"['\"]text['\"]\s*:\s*['\"](.*?)['\"]\s*,\s*['\"]extras['\"]\s*:", raw_response_hlap_step_by_step, re.DOTALL)
        if text_match:
            raw_response_hlap_step_by_step = text_match.group(1)
            # unescape common newlines from repr-like payloads
            raw_response_hlap_step_by_step = raw_response_hlap_step_by_step.replace("\\n", "\n")
    actions = []
    reasoning = None
    task_gt_pred = None
    
    # Try to extract XML-formatted reasoning and action
    import re
    
    reasoning_match = re.search(r'<Reasoning>(.*?)</Reasoning>', raw_response_hlap_step_by_step, re.DOTALL | re.IGNORECASE)
    action_match = re.search(r'<Action>(.*?)</Action>', raw_response_hlap_step_by_step, re.DOTALL | re.IGNORECASE)
    
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        task_gt_match = re.search(r"task_gt\s*:\s*(.+)$", reasoning, re.IGNORECASE)
        if task_gt_match:
            task_gt_pred = task_gt_match.group(1).strip()
        if state.get("verbose"):
            print(f"   💭 Reasoning: {reasoning[:100]}..." if len(reasoning) > 100 else f"   💭 Reasoning: {reasoning}")
    
    if action_match:
        action_text = action_match.group(1).strip()
        # Prefer explicit function-style commands inside <Action> when present
        # Example: move_forward(6.0), move_down(2.0)
        command_matches = re.findall(
            r"\b(move_forward|move_backward|move_up|move_down|strafe_left|strafe_right|turn_left|turn_right)\s*\(\s*([-+]?\d*\.?\d+)\s*\)",
            action_text,
            re.IGNORECASE,
        )
        if command_matches:
            actions = [f"{cmd.lower()}({mag})" for cmd, mag in command_matches]
            if state.get("verbose"):
                print(f"   ✓ Parsed action command(s) from <Action>: {actions}")
        else:
            actions.append(action_text)
            if state.get("verbose"):
                print(f"   ✓ Parsed action: {action_text}")
    else:
        # Prefer explicit function-style commands if present anywhere in the response.
        # Example: move_forward(6.0), move_down(4.0), turn_right(30)
        command_matches = re.findall(
            r"\b(move_forward|move_backward|move_up|move_down|strafe_left|strafe_right|turn_left|turn_right)\s*\(\s*([-+]?\d*\.?\d+)\s*\)",
            raw_response_hlap_step_by_step,
            re.IGNORECASE,
        )
        if command_matches:
            actions = [f"{cmd.lower()}({mag})" for cmd, mag in command_matches]
            if state.get("verbose"):
                print(f"   ✓ Parsed function-style action(s): {actions}")

        # Fallback: Look for "Next Action:" pattern for backward compatibility
        if not actions:
        # Fallback: Look for "Next Action:" pattern for backward compatibility
            lines = raw_response_hlap_step_by_step.split("\n")
        
            for line in lines:
                line = line.strip()
                if "Next Action:" in line:
                    # Extract action text after "Next Action:"
                    action_text = line.replace("Next Action:", "").strip()
                    if action_text:
                        actions.append(action_text)
                        if state.get("verbose"):
                            print(f"   ✓ Parsed action (legacy format): {action_text}")
                        break
        
        # If no "Next Action:" prefix found, try to use the first non-empty line as the action
            if not actions:
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("<"):
                        actions.append(line)
                        if state.get("verbose"):
                            print(f"   ✓ Parsed action (no prefix): {line}")
                        break
    
    if not actions:
        if state.get("verbose"):
            print(f"   ⚠ Warning: No action could be parsed from response")
            print(f"      Response did not contain <Action>...</Action> tags, 'Next Action:' prefix, or valid action text")
            print(f"      Raw response: {raw_response_hlap_step_by_step[:200]}...")

    # Enforce exactly one action per step (backward-compatible safeguard).
    # If multiple actions are parsed, keep only the first one.
    if len(actions) > 1:
        if state.get("verbose"):
            print(f"   ⚠ Multiple actions parsed ({len(actions)}). Enforcing single-action mode, keeping first: {actions[0]}")
        actions = [actions[0]]

    current_index = state["current_action_index"]

    # In single-VLM mode, <Action> can be either concrete motion or "navigation done".
    # If navigation is done, do not enqueue waypoint generation for this step.
    nav_status = "continue"
    if actions and "navigation done" in actions[0].lower():
        nav_status = "done"
    stop_reason = "vlm_signaled_done" if nav_status == "done" else None

    parsed_actions = state.get("parsed_actions", [])
    if nav_status == "continue" and actions:
        parsed_actions = parsed_actions + actions
    step_elapsed = time.time() - step_start
    # Persist parsed action for debugging (aggregated by step)
    _append_step_payload(
        state.get("scenario_results_dir"),
        current_index,
        {
            "current_action": actions,
            "reasoning_hlap_step_by_step": reasoning,
            "nav_status_from_action": nav_status,
            "stop_reason": stop_reason,
            "task_gt_pred": task_gt_pred,
            "time_hlap_parsing_node": round(step_elapsed, 3),
        },
    )

    result = {
        "current_action": actions,
        "parsed_actions": parsed_actions,
        "nav_status": nav_status,
        "task_gt_pred": task_gt_pred,
    }
    if stop_reason:
        result["stop_reason"] = stop_reason
    return result


def node_parse_step_by_step_action_with_bb(state: StepByStepState) -> Dict:
    """Extends ``node_parse_step_by_step_action`` by also extracting a
    ``<BoundingBox>`` prediction from the VLM response.

    The VLM is asked (via ``step_by_step_1vlm_with_bb`` prompt) to output::

        <BoundingBox>[x1,y1,x2,y2,depth,target_object_name]</BoundingBox>
        <Reasoning>... task_gt: value</Reasoning>
        <Action>...</Action>

    Coordinates are 0-1000 normalised. ``-1,-1,-1,-1,-1,'unknown'`` means not visible.
    The box is appended to ``state["predicted_bbs"]`` (one entry per step).
    """
    base_result = node_parse_step_by_step_action(state)

    raw = state.get("raw_response_hlap_step_by_step", "") or ""
    predicted_bb: Optional[list] = None
    parsed_coords: Optional[list] = None
    parsed_depth: Optional[float] = None
    parsed_target_object_name: Optional[str] = None
    matched_bb_payload: Optional[str] = None

    # 1) XML-style tags (supports prompt updates with alias tags)
    tag_patterns = [
        r"<BoundingBox>\s*([^<]+?)\s*</BoundingBox>",
        r"<BBox>\s*([^<]+?)\s*</BBox>",
        r"<Bounding_Box>\s*([^<]+?)\s*</Bounding_Box>",
    ]
    for pattern in tag_patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if not match:
            continue
        matched_bb_payload = match.group(1)
        values = re.findall(r"-?\d+(?:\.\d+)?", matched_bb_payload)
        if len(values) >= 4:
            parsed_coords = [int(round(float(v))) for v in values[:4]]
            if len(values) >= 5:
                parsed_depth = float(values[4])
            break

    # 2) JSON/text-style fields (e.g. "bounding_box": [x1,y1,x2,y2])
    if parsed_coords is None:
        field_patterns = [
            r'"bounding_box"\s*:\s*\[\s*([^\]]+?)\s*\]',
            r'"bbox"\s*:\s*\[\s*([^\]]+?)\s*\]',
            r"bounding_box\s*[:=]\s*\[\s*([^\]]+?)\s*\]",
            r"bbox\s*[:=]\s*\[\s*([^\]]+?)\s*\]",
        ]
        for pattern in field_patterns:
            match = re.search(pattern, raw, re.IGNORECASE)
            if not match:
                continue
            matched_bb_payload = match.group(1)
            values = re.findall(r"-?\d+(?:\.\d+)?", matched_bb_payload)
            if len(values) >= 4:
                parsed_coords = [int(round(float(v))) for v in values[:4]]
                if len(values) >= 5:
                    parsed_depth = float(values[4])
                break

    # Parse target object name from BB payload when provided.
    if matched_bb_payload:
        quoted_name_match = re.search(r"['\"]([^'\"]+)['\"]", matched_bb_payload)
        if quoted_name_match:
            parsed_target_object_name = quoted_name_match.group(1).strip()
        else:
            parts = [part.strip() for part in re.split(r",", matched_bb_payload)]
            if len(parts) >= 6:
                candidate = parts[5].strip(" [](){}'\"\n\t")
                if candidate and not re.fullmatch(r"-?\d+(?:\.\d+)?", candidate):
                    parsed_target_object_name = candidate

    if parsed_coords is not None:
        # -1,-1,-1,-1 => not visible convention
        if all(v < 0 for v in parsed_coords):
            predicted_bb = None
            if parsed_depth is not None and parsed_depth < 0:
                parsed_depth = None
            if parsed_target_object_name and parsed_target_object_name.lower() in {"unknown", "none", "n/a"}:
                parsed_target_object_name = None
        else:
            # Normalize ordering and clamp to prompt coordinate space [0, 1000]
            x1, y1, x2, y2 = parsed_coords
            x_low, x_high = sorted([x1, x2])
            y_low, y_high = sorted([y1, y2])
            predicted_bb = [
                max(0, min(1000, x_low)),
                max(0, min(1000, y_low)),
                max(0, min(1000, x_high)),
                max(0, min(1000, y_high)),
            ]

        if state.get("verbose"):
            print(
                f"   \U0001f3af Predicted BB: {parsed_coords}"
                + (" (not visible)" if predicted_bb is None else f" -> normalized {predicted_bb}")
                + (f", depth={parsed_depth}" if parsed_depth is not None else "")
                + (f", target={parsed_target_object_name}" if parsed_target_object_name else "")
            )
    else:
        if state.get("verbose"):
            print("   \u26a0 No bounding-box output found in VLM response")

    existing_bbs = list(state.get("predicted_bbs") or [])
    existing_bbs.append(predicted_bb)
    existing_depths = list(state.get("predicted_bb_depths") or [])
    existing_depths.append(parsed_depth)
    existing_labels = list(state.get("predicted_bb_labels") or [])
    existing_labels.append(parsed_target_object_name)

    # Draw BB directly on the SAME image that was sent to the VLM
    # (in-place), so history images are annotated without creating
    # redundant side-car files.
    current_index = state.get("current_action_index", 0)
    img_history = state.get("img_history") or []
    current_image = img_history[-1] if img_history else state.get("image", "")
    annotated_path = None
    if predicted_bb and current_image and os.path.exists(current_image):
        target_object: str = parsed_target_object_name or state.get("mission", {}).get("mission", {}).get("metadata", {}).get("target_object", "target")
        depth_to_draw = parsed_depth if parsed_depth is not None else 0.0
        try:
            pil_img = Image.open(current_image)
            predictor = PixelPredictor(model_name=state["model_name"], mission=state)
            annotated_pil = predictor.draw_bounding_boxes(
                pil_img, (predicted_bb, depth_to_draw, target_object)
            )
            annotated_pil.save(current_image)
            annotated_path = current_image
            if state.get("verbose"):
                print(f"   🖼️  BB drawn in-place on VLM image: {annotated_path}")
        except Exception as exc:
            if state.get("verbose"):
                print(f"   ⚠ Failed to draw BB in-place on current image: {exc}")

    _append_step_payload(
        state.get("scenario_results_dir"),
        current_index,
        {
            "predicted_bb": predicted_bb,
            "predicted_bb_raw": parsed_coords,
            "predicted_bb_depth": parsed_depth,
            "predicted_bb_target_object": parsed_target_object_name,
            "annotated_bb_image_path": annotated_path,
        },
    )

    return {
        **base_result,
        "predicted_bbs": existing_bbs,
        "predicted_bb_depths": existing_depths,
        "predicted_bb_labels": existing_labels,
    }


def generate_waypoint_parametric(
    action_text: str,
    yaw_degrees: float,
    verbose: bool = False
) -> Optional[Dict[str, float]]:
    """
    Generate waypoint using parametric/mathematical approach with rotation matrix.
    
    Parses high-level action strings and converts them to (dx, dy, dz, dyaw) based on 
    the drone's current yaw angle using proper coordinate transformations.
    
    Supported action patterns:
    - "move forward <distance>" → forward movement
    - "move backward <distance>" → backward movement
    - "strafe left <distance>" → left lateral movement
    - "strafe right <distance>" → right lateral movement
    - "ascend <distance>" → upward movement (dz negative in NED)
    - "descend <distance>" → downward movement (dz positive in NED)
    - "turn left <angle>" → counter-clockwise rotation
    - "turn right <angle>" → clockwise rotation
    
    Args:
        action_text: High-level action description (e.g., "move forward 10m")
        yaw_degrees: Current drone yaw angle in degrees
        verbose: Print debug information
        
    Returns:
        Dict with dx, dy, dz, dyaw or None if parsing fails
    """
    try:
        action_lower = action_text.lower().strip()
        yaw_rad = math.radians(yaw_degrees)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        
        waypoint = {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0}
        
        # Prefer function-call argument extraction to avoid picking numbering
        # from long reasoning text (e.g., "1.", "2.") when present.
        fn_arg_match = re.search(
            r"\b(?:move_forward|move_backward|move_up|move_down|strafe_left|strafe_right|turn_left|turn_right)\s*\(\s*([-+]?\d*\.?\d+)\s*\)",
            action_lower,
            re.IGNORECASE,
        )

        if fn_arg_match:
            magnitude = float(fn_arg_match.group(1))
        else:
            # Fallback for natural language action text
            numbers = re.findall(r'[-+]?\d*\.?\d+', action_text)
            if not numbers:
                if verbose:
                    print(f"   ⚠ Could not extract number from action: {action_text}")
                return None
            magnitude = float(numbers[0])

        if not isinstance(magnitude, float):
            if verbose:
                print(f"   ⚠ Could not extract number from action: {action_text}")
            return None
        
        # Forward/Backward movements (rotated by yaw)
        if "forward" in action_lower:
            waypoint["dx"] = magnitude * cos_y
            waypoint["dy"] = magnitude * sin_y
            if verbose:
                print(f"   📌 Forward {magnitude}m at yaw {yaw_degrees}°: dx={waypoint['dx']:.2f}, dy={waypoint['dy']:.2f}")
        
        elif "backward" in action_lower:
            waypoint["dx"] = -magnitude * cos_y
            waypoint["dy"] = -magnitude * sin_y
            if verbose:
                print(f"   📌 Backward {magnitude}m at yaw {yaw_degrees}°: dx={waypoint['dx']:.2f}, dy={waypoint['dy']:.2f}")
        
        # Lateral movements (perpendicular to yaw, with sign adjustment)
        elif "strafe right" in action_lower or ("right" in action_lower and "strafe" in action_lower):
            # Right strafe is perpendicular right to facing direction
            waypoint["dx"] = -magnitude * sin_y
            waypoint["dy"] = magnitude * cos_y
            if verbose:
                print(f"   📌 Strafe right {magnitude}m at yaw {yaw_degrees}°: dx={waypoint['dx']:.2f}, dy={waypoint['dy']:.2f}")
        
        elif "strafe left" in action_lower or ("left" in action_lower and "strafe" in action_lower):
            # Left strafe is perpendicular left to facing direction
            waypoint["dx"] = magnitude * sin_y
            waypoint["dy"] = -magnitude * cos_y
            if verbose:
                print(f"   📌 Strafe left {magnitude}m at yaw {yaw_degrees}°: dx={waypoint['dx']:.2f}, dy={waypoint['dy']:.2f}")
        
        # Vertical movements
        elif "ascend" in action_lower or "up" in action_lower:
            waypoint["dz"] = -magnitude  # NED: negative dz = up
            if verbose:
                print(f"   📌 Ascend {magnitude}m: dz={waypoint['dz']:.2f}")
        
        elif "descend" in action_lower or "down" in action_lower:
            waypoint["dz"] = magnitude  # NED: positive dz = down
            if verbose:
                print(f"   📌 Descend {magnitude}m: dz={waypoint['dz']:.2f}")
        
        # Rotations
        elif "turn left" in action_lower or ("left" in action_lower and ("turn" in action_lower or "rotate" in action_lower)):
            waypoint["dyaw"] = -magnitude  # Counter-clockwise
            if verbose:
                print(f"   📌 Turn left {magnitude}°: dyaw={waypoint['dyaw']:.2f}")
        
        elif "turn right" in action_lower or ("right" in action_lower and ("turn" in action_lower or "rotate" in action_lower)):
            waypoint["dyaw"] = magnitude  # Clockwise
            if verbose:
                print(f"   📌 Turn right {magnitude}°: dyaw={waypoint['dyaw']:.2f}")
        
        else:
            if verbose:
                print(f"   ⚠ Could not parse action type: {action_text}")
            return None
        
        return waypoint
        
    except Exception as e:
        if verbose:
            print(f"   ⚠ Error in parametric generation: {e}")
        return None


def node_generate_waypoints(
    state: StepByStepState,
    waypoint_generation_method: str = "parametric"
) -> Dict:
    """
    Node: Generate waypoints for current high-level action.

    Supports two waypoint generation methods:
    
    1. VLM-based (waypoint_generation_method="vlm"):
       - Initializes WaypointGenerator with specified model
       - Optionally formats spatial reasoning results as context
       - Calls VLM with image + drone_pose + current_action + spatial_context
       - Returns raw waypoint response from VLM
    
    2. Parametric/Mathematical (waypoint_generation_method="parametric"):
       - Parses high-level action using regex and natural language patterns
       - Applies rotation matrix transformation based on current yaw angle
       - Generates accurate (dx, dy, dz, dyaw) for ANY arbitrary yaw angle
       - Fast, deterministic, no VLM inference needed

    Args:
        state: Current two-stage workflow state containing:
            - image, drone_pose, current_action
            - low_level_model_name (or falls back to model_name)
            - spatial_results (optional, from pix2world)
        waypoint_generation_method: Either "vlm" (default) or "parametric"
            - "vlm": Use vision-language model for generation (slower, context-aware)
            - "parametric": Use mathematical transformation (fast, rotation-correct)

    Returns:
        Dict with 'raw_waypoint_response' key containing waypoint data
    """
    step_start = time.time()
    verbose = state.get("verbose", False)
    
    # Get current action(s) and yaw angle
    action_items = state["current_action"] if isinstance(state["current_action"], list) else [state["current_action"]]
    yaw = state.get("drone_pose", {}).get("yaw") or state.get("mission", {}).get("drone_start_pose", {}).get("yaw", 0.0)
    
    if waypoint_generation_method == "parametric":
        # ===== PARAMETRIC/MATHEMATICAL APPROACH =====
        if verbose:
            print(f"   🧮 Generating waypoints using parametric method")
            print(f"   📌 Action(s): {action_items}")
            print(f"   📌 Current yaw: {yaw}°")

        # Support multiple commands in one step (e.g., move_forward + move_down)
        combined_waypoint = {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0}
        parsed_any = False

        for action_text in action_items:
            waypoint = generate_waypoint_parametric(action_text, yaw, verbose=verbose)
            if waypoint is None:
                continue
            combined_waypoint["dx"] += waypoint["dx"]
            combined_waypoint["dy"] += waypoint["dy"]
            combined_waypoint["dz"] += waypoint["dz"]
            combined_waypoint["dyaw"] += waypoint["dyaw"]
            parsed_any = True

        if not parsed_any:
            if verbose:
                print(f"   ⚠ Parametric generation failed, falling back to VLM")
            waypoint_generation_method = "vlm"  # Fallback to VLM
            raw_response = f"Failed to parse actions: {action_items}. Fallback triggered."
        else:
            # Format waypoint as string response (matching VLM output format)
            raw_response = (
                f"dx={combined_waypoint['dx']:.2f} dy={combined_waypoint['dy']:.2f} "
                f"dz={combined_waypoint['dz']:.2f} dyaw={combined_waypoint['dyaw']:.2f}"
            )
            if verbose:
                print(f"   ✓ Parametric waypoint generated: {raw_response}")
    
    elif waypoint_generation_method == "vlm":
        # ===== VLM-BASED APPROACH =====
        model_name = state.get("low_level_model_name") or state["model_name"]
        
        if verbose:
            print(f"   🤖 Generating waypoints using VLM: {model_name}")

        # Initialize waypoint generator
        generator = WaypointGenerator(model_name=model_name, mission=state)

        # Generate waypoints
        raw_response = generator.generate_waypoints(
            mission=state,
            high_level_action=state["current_action"],
            prompt_type="basic_prompt_low_level",
            save_prompt_dir=state.get("scenario_results_dir"),
            step_index=state.get("current_action_index", 0),
        )

    else:
        raise ValueError(f"Unsupported waypoint_generation_method: {waypoint_generation_method}")
    
    all_raw_waypoint_response = state.get("all_raw_waypoint_response", [])
    all_raw_waypoint_response.append(raw_response)
    current_index = state["current_action_index"]
    step_elapsed = time.time() - step_start
    
    _append_step_payload(
        state.get("scenario_results_dir"),
        current_index,
        {
            "raw_waypoint_response": raw_response,
            "waypoint_generation_method": waypoint_generation_method,
            "time_wp_generation_node": round(step_elapsed, 2),
            
        },
    )
    return {"all_raw_waypoint_response": all_raw_waypoint_response}

def node_parse_waypoints(state: StepByStepState) -> Dict:
    """
    Node: Parse waypoint response into structured format.

    Extracts waypoints from text like:
        "dx=0 dy=-50 dz=0 dyaw=0"

    Into structured format:
        {"dx": 0.0, "dy": -50.0, "dz": 0.0, "dyaw": 0.0}

    Args:
        state: Current two-stage workflow state containing raw_waypoint_response

    Returns:
        Dict with 'waypoints' key containing list of waypoint dicts
    """
    step_start = time.time()
    if state.get("verbose"):
        print("   📋 Parsing waypoints...")

    raw_response = state["all_raw_waypoint_response"][-1]

    # Parse a single waypoint line containing dx, dy, dz, dyaw
    waypoint = {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0}
    try:
        import re

        for component in ["dx", "dy", "dz", "dyaw"]:
            match = re.search(rf"{component}=([\-\d\.]+)", raw_response)
            if match:
                waypoint[component] = float(match.group(1))
    except Exception as e:
        if state.get("verbose"):
            print(f"   ⚠ Failed to parse waypoint: {e}")
    
    # CRITICAL FIX: Clamp waypoint values to enforce constraints
    max_step_size = state.get("tolerances", {}).get("MAX_STEP_SIZE", 2.0)
    max_rotation = 30.0  # Max rotation per step (degrees)
    
    original_waypoint = waypoint.copy()
    
    waypoint["dx"] = max(-max_step_size, min(max_step_size, waypoint["dx"]))
    waypoint["dy"] = max(-max_step_size, min(max_step_size, waypoint["dy"]))
    waypoint["dz"] = max(-max_step_size, min(max_step_size, waypoint["dz"]))
    waypoint["dyaw"] = max(-max_rotation, min(max_rotation, waypoint["dyaw"]))
    
    if original_waypoint != waypoint:
        if state.get("verbose"):
            print(f"   ⚠ Clamped waypoint: {original_waypoint} → {waypoint}")
    
    if state.get("verbose"):
        print(f"   ✓ Parsed waypoint: {waypoint}")
    # Accumulate parsed waypoints across iterations
    all_parsed_waypoints = state.get("all_parsed_waypoints", [])
    all_parsed_waypoints.append(waypoint)
    current_index = state["current_action_index"]

        # Use absolute pose: start pose plus accumulated waypoint deltas
    start_pose = state.get("mission").get(
        "drone_start_pose"
    )
    all_parsed_waypoints = state.get("all_parsed_waypoints", [])

    current_x = start_pose.get("x", 0.0) + sum(wp.get("dx", 0.0) for wp in all_parsed_waypoints)
    current_y = start_pose.get("y", 0.0) + sum(wp.get("dy", 0.0) for wp in all_parsed_waypoints)
    current_z = start_pose.get("z", 0.0) + sum(wp.get("dz", 0.0) for wp in all_parsed_waypoints)
    current_yaw = start_pose.get("yaw", 0.0) + sum(wp.get("dyaw", 0.0) for wp in all_parsed_waypoints)
    drone_pose = {
        "x": current_x,
        "y": current_y,
        "z": current_z,
        "yaw": current_yaw,
    }

    # Track full history of absolute poses for evaluation metrics
    all_drone_poses = state.get("all_drone_poses", [])
    all_drone_poses.append(drone_pose)
    step_end = time.time()
    _append_step_payload(
        state.get("scenario_results_dir"),
        current_index,
        {
            "parsed_waypoint": waypoint,
            "drone_pose": drone_pose,
            "timing": {
                "time_wp_parse_node": round(step_end - step_start, 2),
            }
        },
    )
    # Also expose latest parsed waypoint for convenience
    return {
        "all_parsed_waypoints": all_parsed_waypoints,
        "all_drone_poses": all_drone_poses,
        "drone_pose": drone_pose,
    }

def should_continue_navigation_parser(mission_completed: str, state: StepByStepState, current_index: int) -> Dict:
    """
    Parse the VLM response for navigation continuation decision.
    
    Extracts reasoning and action from XML-formatted response and evaluates conditions
    for mission continuation vs completion.
    
    Args:
        mission_completed: Raw VLM response text
        state: Current workflow state containing tolerances
        current_index: Current action index for budget checking
        
    Returns:
        Dict containing:
        - reasoning_text: Extracted reasoning (if present)
        - action_text: Extracted action text (lowercase)
        - condition1: Boolean - whether to continue (not "navigation done")
        - condition2: Boolean - whether within action budget
        - stop_reason: Reason for stopping (if applicable)
    """
    # Parse XML-style Reasoning/Action if present
    reasoning_text = None
    task_gt_pred = None
    action_text = mission_completed
    try:
        import re
        reasoning_match = re.search(r"<Reasoning>(.*?)</Reasoning>", mission_completed, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r"<Action>(.*?)</Action>", mission_completed, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).strip()
            # Extract task_gt if present in reasoning (e.g., "task_gt: <value>")
            task_gt_match = re.search(r"task_gt\s*:\s*(.+)$", reasoning_text, re.IGNORECASE)
            if task_gt_match:
                task_gt_pred = task_gt_match.group(1).strip()
        if action_match:
            action_text = action_match.group(1).strip()
    except Exception as e:
        if state.get("verbose"):
            print(f"   ⚠ Failed to parse navigation XML: {e}")

    action_lower = action_text.lower()
    assert "continue" in action_lower or "navigation done" in action_lower, f"Unexpected response for mission completion: {mission_completed}"
    
    # Evaluate conditions
    condition1 = "navigation done" not in action_lower
    stop_reason = None
    if not condition1:
        stop_reason = "vlm_signaled_done"
    
    max_actions = state["tolerances"]["MAX_ACTIONS"]
    condition2 = current_index < max_actions
    if not condition2:
        stop_reason = "max_action_budget_reached"
    
    return {
        "reasoning_text": reasoning_text,
        "action_text": action_lower,
        "condition1": condition1,
        "condition2": condition2,
        "stop_reason": stop_reason,
        "max_actions": max_actions,
        "task_gt_pred": task_gt_pred,
    }

def should_continue_navigation(state: StepByStepState) -> str:
    """
    Conditional edge function: Determine if the navigation is completed or should continue.

    Args:
        state: Current two-stage workflow state

    Returns:
        Dict with 'nav_status' set to "continue" (loop) or "done" (finish)
    """
    step_start = time.time()
    current_index = state["current_action_index"]
    current_index += 1
    fetch_image_time_start = time.time()
    intermediate_capture = _capture_intermediate_step_frames(
        state,
        current_index=current_index,
        img_enhancement_type=state.get("img_enhancement_type"),
    )
    if intermediate_capture is not None:
        stepped_img_path, collision_info, intermediate_frame_count = intermediate_capture
    else:
        stepped_img_path, collision_info = fetch_image(state, img_enhancement_type=state["img_enhancement_type"])
        intermediate_frame_count = 0
    fetch_image_time_elapsed = time.time() - fetch_image_time_start
    collision_info = collision_info or []
    # for now lets not stop the mission if there is collision, lets get the collistion rate information and see how it correlates with the mission completion and success rate. 
    # if collision_info: # List of object names that the drone collided with
    #     if state.get("verbose"):
    #         print(f"   ⚠ Collision detected! Objects: {collision_info}")
    #     _append_step_payload(
    #         state.get("scenario_results_dir"),
    #         current_index - 1,
    #         {
    #             "collision_info": collision_info,
    #             "collided": True,
    #             "stop_reason": "collision_detected",
    #             "time_should_continue_node": round(time.time() - step_start, 2),
    #             "fetch_image_time": round(fetch_image_time_elapsed, 2),
    #         },
    #     )
    #     return {
    #         "nav_status": "done",
    #         "collision_info": collision_info,
    #         "collided": True,
    #         "stop_reason": "collision_detected",
    #         "time_should_continue_node": round(time.time() - step_start, 2),
    #         "fetch_image_time": round(fetch_image_time_elapsed, 2),
    #         "current_action_index": current_index,  # Return final index
    #         "image": stepped_img_path,
    #         "img_history": state["img_history"], 
    #     }
    # Copy/save the fetched image into scenario_results_dir/imgs with step suffix.
    # BB is drawn in-place on the exact image used by VLM in parse node,
    # so we do not draw prior-step BB onto this newly fetched frame.
    new_img = os.path.basename(stepped_img_path).replace(".png", f"_step_{current_index}.png")
    save_img_path = os.path.join(Path(state["scenario_results_dir"]/"imgs"), new_img)

    prompt_type = _resolve_high_level_prompt_type(state)
    shutil.copy(stepped_img_path, save_img_path)

    state["img_history"].append(save_img_path)
    # Single-VLM mode: completion decision already came from the action node.
    # Reuse this node only for stepping image/history and budget checks.
    if prompt_type in ("step_by_step_1vlm", "step_by_step_1vlm_with_bb"):
        max_actions = state["tolerances"]["MAX_ACTIONS"]
        condition2 = current_index < max_actions
        step_elapsed = time.time() - step_start

        _append_step_payload(
            state.get("scenario_results_dir"),
            current_index - 1,
            {
                "should_continue_response": "single_vlm_mode_no_extra_vlm_call",
                "collision_info": collision_info,
                "collided": False,
                "intermediate_frame_count": intermediate_frame_count,
                "parsed_action": "continue",
                "condition2_within_step_budget": f"{current_index} < {max_actions}" if condition2 else f"{current_index} >= {max_actions} therefore stop!",
                "time_should_continue_node": round(step_elapsed, 2),
                "fetch_image_time": round(fetch_image_time_elapsed, 2),
                "should_continue_vlm_inference_time": 0.0,
            },
        )

        if condition2:
            return {
                "nav_status": "continue",
                "current_action_index": current_index,
                "image": stepped_img_path,
                "img_history": state["img_history"],
                "collision_info": collision_info,
                "collided": False,
                "should_continue_response": "single_vlm_mode_no_extra_vlm_call",
                "task_gt_pred": state.get("task_gt_pred"),
            }

        return {
            "nav_status": "done",
            "stop_reason": "max_action_budget_reached",
            "current_action_index": current_index,
            "image": stepped_img_path,
            "img_history": state["img_history"],
            "collision_info": collision_info,
            "collided": False,
            "should_continue_response": "single_vlm_mode_no_extra_vlm_call",
            "task_gt_pred": state.get("task_gt_pred"),
        }

    # Two-VLM mode (legacy): keep existing mission-completion call.
    predictor = ActionPredictor(model_name=state["model_name"], mission=state)
    
    vlm_start = time.time()
    mission_completed = predictor.predict_actions(
        image=stepped_img_path,
        mission=state,
        prompt_type="should_continue_navigation_prompt",
        save_prompt_dir=state.get("scenario_results_dir"),
        step_index=current_index,
    )
    vlm_elapsed = time.time() - vlm_start
    state["should_continue_response"] = mission_completed
    
    # Parse the navigation continuation response
    parse_result = should_continue_navigation_parser(mission_completed, state, current_index)
    reasoning_text = parse_result["reasoning_text"]
    action_lower = parse_result["action_text"]
    condition1 = parse_result["condition1"]
    condition2 = parse_result["condition2"]
    stop_reason = parse_result["stop_reason"]
    max_actions = parse_result["max_actions"]
    task_gt_pred = parse_result.get("task_gt_pred")
    step_elapsed = time.time() - step_start
    # Save the navigation check response
    _append_step_payload(
        state.get("scenario_results_dir"),
        current_index - 1,
        {
            "should_continue_response": mission_completed,
            "collision_info": collision_info,
            "collided": bool(collision_info),
            "intermediate_frame_count": intermediate_frame_count,
            "parsed_reasoning_should_continue": reasoning_text,
            "parsed_action": action_lower,
            "task_gt_pred": task_gt_pred,
            "condition1_mission_completed": "navigation done" if "navigation done" in action_lower else "continue",
            "condition2_within_step_budget": f"{current_index} < {max_actions}" if condition2 else f"{current_index} >= {max_actions} therefore stop!",
            "time_should_continue_node": round(step_elapsed, 2),
            "fetch_image_time": round(fetch_image_time_elapsed, 2),
            "should_continue_vlm_inference_time": round(vlm_elapsed, 2),
        },
    )
    
    if condition1 and condition2:
        return {
            "nav_status": "continue",
            "current_action_index": current_index,  # Return incremented index
            "image": stepped_img_path,
            "img_history": state["img_history"], 
            "collision_info": collision_info,
            "collided": bool(collision_info),
            "should_continue_response": mission_completed,
            "task_gt_pred": task_gt_pred,
        }
    else:
        # Accumulate final waypoints before finishing
        if state.get("verbose"):
            print("\n✓ All actions processed, accumulating final waypoints...")
        return {
            "nav_status": "done",
            "stop_reason": stop_reason,
            "current_action_index": current_index,  # Return final index
            "image": stepped_img_path,
            "img_history": state["img_history"], 
            "collision_info": collision_info,
            "collided": bool(collision_info),
            "should_continue_response": mission_completed,
            "task_gt_pred": task_gt_pred,
        }

def node_evaluate_step_by_step(state: StepByStepState) -> Dict:
    """
    Node: Evaluate mission performance using step-by-step approach metrics.

    Calculates the following metrics:
    1. Oracle Success Rate (OSR): Whether agent gets within 5m of target at any point
    2. Success Rate (SR): Whether agent reaches final goal within tolerances
    3. Path Length Similarity: Placeholder for future implementation

    Args:
        state: Current workflow state containing drone_pose, position history, target info

    Returns:
        Dict with 'evaluation_result' containing metrics:
        - oracle_success: Boolean (1/0) if ever within 5m of target
        - oracle_success_step: Step at which oracle success was achieved (if any)
        - success_rate: Boolean (1/0) if final position within thresholds
        - x_diff, y_diff, z_diff, yaw_diff: Final deviations from target
        - min_distance_to_target: Minimum distance achieved during mission
    """
    
    print("\n📊 Evaluating mission performance...")
    step_start = time.time()
    # Extract mission goal/target information
    mission = state.get("mission", {})
    target_x = mission.get("ground_truth").get("waypoint_gt").get("x")
    target_y = mission.get("ground_truth").get("waypoint_gt").get("y")
    target_z = mission.get("ground_truth").get("waypoint_gt").get("z")
    target_yaw = mission.get("ground_truth").get("waypoint_gt").get("yaw")
    
    # Get final drone position
    final_pose = state.get("drone_pose", {})
    final_x = final_pose.get("x")
    final_y = final_pose.get("y")
    final_z = final_pose.get("z")
    final_yaw = final_pose.get("yaw")
    
    # ========== ORACLE SUCCESS RATE ==========
    # Use stored absolute drone poses at every step

    import math

    start_pose = mission.get("drone_start_pose", {})
    start_x = start_pose.get("x")
    start_y = start_pose.get("y")
    start_z = start_pose.get("z")

    trajectory_positions = [(start_x, start_y, start_z)]

    all_drone_poses = state.get("all_drone_poses", []) or []
    for pose in all_drone_poses:
        trajectory_positions.append((pose.get("x", 0.0), pose.get("y", 0.0), pose.get("z", 0.0)))

    # Calculate minimum distance to target across entire trajectory
    min_distance = float('inf')
    for pos_x, pos_y, pos_z in trajectory_positions:
        distance = math.sqrt(
            (pos_x - target_x)**2 + 
            (pos_y - target_y)**2 + 
            (pos_z - target_z)**2
        )
        min_distance = min(min_distance, distance)
    
    oracle_threshold_5 = 5.0  # meters
    oracle_success_5 = 1 if min_distance <= oracle_threshold_5 else 0
    oracle_success_step_5 = state.get("current_action_index") if oracle_success_5 else None
    oracle_success_10 = 1 if min_distance <= 10.0 else 0
    oracle_success_20 = 1 if min_distance <= 20.0 else 0
    
    # Also track final distance for reference
    final_distance = math.sqrt(
        (final_x - target_x)**2 + 
        (final_y - target_y)**2 + 
        (final_z - target_z)**2
    )
    
    print(f"   🎯 Oracle Success Rate (OSR):")
    print(f"      Trajectory length: {len(trajectory_positions)} positions")
    print(f"      Minimum distance to target: {min_distance:.2f}m")
    print(f"      Final distance to target: {final_distance:.2f}m")
    print(f"      Threshold: {oracle_threshold_5}m")
    print(f"      Status: {'✅ PASS' if oracle_success_5 else '❌ FAIL'}")
    
    # ========== SUCCESS RATE (SR5 and SR10) ==========
    # Calculate deviations from target
    x_diff = abs(final_x - target_x)
    y_diff = abs(final_y - target_y)
    z_diff = abs(final_z - target_z)
    yaw_diff = abs(normalize_angle_difference(final_yaw, target_yaw))
    
    # SR5: Success Rate with 5m position threshold and 15° yaw threshold
    x_threshold_sr5 = state.get("tolerances", {}).get("x", 5.0)
    y_threshold_sr5 = state.get("tolerances", {}).get("y", 5.0)
    z_threshold_sr5 = state.get("tolerances", {}).get("z", 5.0)
    yaw_threshold_sr5 = state.get("tolerances", {}).get("yaw", 15.0)
    
    sr5 = 1 if (
        x_diff <= x_threshold_sr5 and
        y_diff <= y_threshold_sr5 and
        z_diff <= z_threshold_sr5 and
        yaw_diff <= yaw_threshold_sr5
    ) else 0
    
    # SR10: Success Rate with 10m position threshold and 45° yaw threshold
    x_threshold_sr10 = 10.0
    y_threshold_sr10 = 10.0
    z_threshold_sr10 = 10.0
    yaw_threshold_sr10 = 45.0
    
    sr10 = 1 if (
        x_diff <= x_threshold_sr10 and
        y_diff <= y_threshold_sr10 and
        z_diff <= z_threshold_sr10 and
        yaw_diff <= yaw_threshold_sr10
    ) else 0

    # ========== TASK_GT MATCH (Optional, via small LLM) ==========
    task_gt_expected = mission.get("ground_truth", {}).get("task_gt").lower().strip() if mission.get("ground_truth", {}).get("task_gt") else None
    task_gt_pred = (
        state.get("task_gt_pred").lower().strip()
        if state.get("task_gt_pred") is not None
        else None
    )

    # Always initialize task_gt fields with sensible defaults
    task_gt_match = -1  # -1 = not evaluated, 0 = fail, 1 = success
    task_gt_model = None
    task_gt_response = "not_evaluated"
    print(f"\n   🎯 Evaluating Task GT Match:", task_gt_expected, task_gt_pred)
    eval_prompt = generate_prompt(
        prompt_type="task_gt_evaluation",
        task_gt_expected=task_gt_expected,
        task_gt_pred=task_gt_pred,
    )
    with open(Path(state["scenario_results_dir"])/f"task_gt_evaluation_prompt.txt", "w") as f:
                f.write(eval_prompt)
    if task_gt_expected and task_gt_pred:
        # Use a small model to evaluate match
        task_gt_model = state.get("mission", {}).get("models_related", {}).get("task_gt_eval_model", "gemini-3-flash-preview")
        task_gt_eval_timeout_sec = int(state.get("mission", {}).get("models_related", {}).get("task_gt_eval_timeout_sec", 30))
        try:
            # since we are not evaluating the ocr ability of the model, we can relax the exact match condition
            base_vision = VisionChain(mission=state)
            base_vision.set_model_name(task_gt_model)
            model = base_vision.create_model()
            content = [{"type": "text", "text": eval_prompt}]

            def _task_gt_timeout_handler(signum, frame):
                raise TimeoutError(f"task_gt evaluator timed out after {task_gt_eval_timeout_sec}s")

            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _task_gt_timeout_handler)
            signal.alarm(max(1, task_gt_eval_timeout_sec))
            try:
                response = model.invoke([HumanMessage(content=content)])
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)

            if isinstance(response.content, list):
                print("Warning: Received list response from model, taking first element")
                response = response.content[0]
            with open(Path(state["scenario_results_dir"])/f"task_gt_evaluation_response.txt", "w") as f:
                f.write(str(response))
            task_gt_response = (response['text'] or "").strip().lower()
            task_gt_match = 1 if task_gt_response.startswith("success") else 0
        except Exception as exc:
            print(f"   ⚠ task_gt evaluation failed: {exc}")
            task_gt_match = 0
            task_gt_response = f"error: {exc}"
            with open(Path(state["scenario_results_dir"])/f"task_gt_evaluation_response.txt", "w") as f:
                f.write(f"Error: {exc}")
            
        if state["mission"]["metadata"]["task_category"] in ["Visual_Inspection", "visual_inspection"]:
            # Incorporate task_gt result into both success rates
            sr5 = 1 if (sr5 == 1 or task_gt_match == 1) else 0
            sr10 = 1 if (sr10 == 1 or task_gt_match == 1) else 0
    
    
    print(f"\n   🎯 Success Rate SR5 (5m position, 15° yaw):")
    print(f"      X deviation: {x_diff:.2f}m (threshold: {x_threshold_sr5}m) {'✅' if x_diff <= x_threshold_sr5 else '❌'}")
    print(f"      Y deviation: {y_diff:.2f}m (threshold: {y_threshold_sr5}m) {'✅' if y_diff <= y_threshold_sr5 else '❌'}")
    print(f"      Z deviation: {z_diff:.2f}m (threshold: {z_threshold_sr5}m) {'✅' if z_diff <= z_threshold_sr5 else '❌'}")
    print(f"      Yaw deviation: {yaw_diff:.2f}° (threshold: {yaw_threshold_sr5}°) {'✅' if yaw_diff <= yaw_threshold_sr5 else '❌'}")
    if task_gt_expected and task_gt_pred:
        print(f"      Task GT Match: {'✅ PASS' if task_gt_match else '❌ FAIL'}")
    print(f"      SR5 Status: {'✅ PASS' if sr5 else '❌ FAIL'}")
    
    print(f"\n   🎯 Success Rate SR10 (10m position, 45° yaw):")
    print(f"      X deviation: {x_diff:.2f}m (threshold: {x_threshold_sr10}m) {'✅' if x_diff <= x_threshold_sr10 else '❌'}")
    print(f"      Y deviation: {y_diff:.2f}m (threshold: {y_threshold_sr10}m) {'✅' if y_diff <= y_threshold_sr10 else '❌'}")
    print(f"      Z deviation: {z_diff:.2f}m (threshold: {z_threshold_sr10}m) {'✅' if z_diff <= z_threshold_sr10 else '❌'}")
    print(f"      Yaw deviation: {yaw_diff:.2f}° (threshold: {yaw_threshold_sr10}°) {'✅' if yaw_diff <= yaw_threshold_sr10 else '❌'}")
    if task_gt_expected and task_gt_pred:
        print(f"      Task GT Match: {'✅ PASS' if task_gt_match else '❌ FAIL'}")
    print(f"      SR10 Status: {'✅ PASS' if sr10 else '❌ FAIL'}")
    
    """ 
    calculating the mission progress metric (mp) as the sum of total progress at each time step divided by the total number of steps, 
    where the progress at each time step is defined as the percentage decrease in distance to the target compared to the initial distance
    from the start position to the target. This metric provides insight into how effectively the agent is progressing towards the target
    throughout the mission, rather than just evaluating the final outcome
    """
    initial_distance = math.sqrt(
        (start_x - target_x)**2 + 
        (start_y - target_y)**2 + 
        (start_z - target_z)**2
    )
    mission_progress = -1.0  # Default to -1 if we cannot compute progress
    stepwise_binary_progress = ""
    try:
        actual_initial_distance = mission.get("ground_truth", {}).get("depth")
        progress_distances = []
        for pos_x, pos_y, pos_z in trajectory_positions[1:]:
            current_distance = math.sqrt(
                (pos_x - target_x)**2 + 
                (pos_y - target_y)**2 + 
                (pos_z - target_z)**2
            )
            progress = initial_distance - current_distance
            if progress > 0:
                stepwise_binary_progress += "1"
            else:
                stepwise_binary_progress += "0"
            progress_distances.append(progress)
            initial_distance = current_distance  # Update initial distance for next step
        
        mission_progress = sum(progress_distances) / actual_initial_distance
        # clip mission progress to be between 0 and 1
        mission_progress = max(0.0, min(1.0, mission_progress))
        print(f"\n   🎯 Clipped Mission Progress: {mission_progress:.3f} ")
    except Exception as exc:
        print(f"   ⚠ Failed to compute mission progress: {exc}")
        mission_progress = -1.0  # Indicate failure to compute

    # ========== COLLISION + STOP METRICS ==========
    steps_executed = state.get("current_action_index", 0)
    collision_count = 0
    collision_events = []
    stop_reason_final = state.get("stop_reason")
    try:
        inter_path = (
            Path(state.get("scenario_results_dir"))
            / "debug_outputs"
            / "step_by_step_intermediate_results.json"
        )
        if inter_path.exists():
            with open(inter_path, "r") as f:
                inter_data = json.load(f)
            # Backfill stop reason from latest step payload if state value is empty.
            if not stop_reason_final:
                for step_key in sorted(
                    inter_data.keys(), key=lambda k: int(k), reverse=True
                ):
                    step_payload = inter_data[step_key] or {}
                    candidate = step_payload.get("stop_reason")
                    if candidate:
                        stop_reason_final = candidate
                        break
                    # In single-VLM mode, termination can happen directly at
                    # parse-action node (without should_continue node), where
                    # nav_status_from_action is set to done.
                    nav_status = (
                        str(step_payload.get("nav_status_from_action", ""))
                        .strip()
                        .lower()
                    )
                    if nav_status == "done":
                        stop_reason_final = "vlm_signaled_done"
                        break
            for step_key, step_payload in inter_data.items():
                info = step_payload.get("collision_info", [])
                if info:
                    collision_count += 1
                    collision_events.append({
                        "step": int(step_key),
                        "objects": info,
                    })
    except Exception as exc:
        print(f"   ⚠ Failed to compute collision metrics: {exc}")

    collision_rate = (
        (collision_count / steps_executed)
        if steps_executed and steps_executed > 0
        else 0.0
    )
    # ========== BUILD EVALUATION RESULT ==========
    eval_result = {
        # Oracle Success Rate
        "oracle_success_5": oracle_success_5,
        "oracle_success_step_5": oracle_success_step_5,
        "oracle_success_10": oracle_success_10,
        "oracle_success_20": oracle_success_20,
        "min_distance_to_target": min_distance,
        "final_distance_to_target": final_distance,
        "oracle_threshold_5": oracle_threshold_5,
        "trajectory_length": len(trajectory_positions),
        "mission_progress": mission_progress,
        "stepwise_binary_progress": stepwise_binary_progress,
        # Success Rates (SR5 and SR10)
        "sr5": sr5,
        "sr10": sr10,
        "x_diff": x_diff,
        "y_diff": y_diff,
        "z_diff": z_diff,
        "yaw_diff": yaw_diff,
        "x_threshold_sr5": x_threshold_sr5,
        "y_threshold_sr5": y_threshold_sr5,
        "z_threshold_sr5": z_threshold_sr5,
        "yaw_threshold_sr5": yaw_threshold_sr5,
        "x_threshold_sr10": x_threshold_sr10,
        "y_threshold_sr10": y_threshold_sr10,
        "z_threshold_sr10": z_threshold_sr10,
        "yaw_threshold_sr10": yaw_threshold_sr10,

        # Task GT match
        "task_gt_expected": task_gt_expected,
        "task_gt_pred": task_gt_pred,
        "task_gt_match": task_gt_match,
        "task_gt_model": task_gt_model,
        "task_gt_response": task_gt_response,
        "collision_count": collision_count,
        "collision_rate": collision_rate,
        "collision_events": collision_events,
        
        # Final state
        "final_pose": {
            "x": final_x,
            "y": final_y,
            "z": final_z,
            "yaw": final_yaw,
        },
        "target_pose": {
            "x": target_x,
            "y": target_y,
            "z": target_z,
            "yaw": target_yaw,
        },
        "steps_executed": steps_executed,
        "stop_reason": stop_reason_final,
        "stop_condition": stop_reason_final,
    }
    
    # Log summary
    print(f"\n{'='*70}")
    print(f"📊 Mission Evaluation Summary")
    print(f"{'='*70}")
    print(f"Oracle Success Rate (OSR): {oracle_success_5} (min distance {min_distance:.2f}m ≤ {oracle_threshold_5}m)")
    print(f"Success Rate SR5: {sr5} (final position within 5m/15° thresholds)")
    print(f"Success Rate SR10: {sr10} (final position within 10m/45° thresholds)")
    print(f"Steps executed: {eval_result['steps_executed']}")
    print(f"Collisions: {collision_count} (rate={collision_rate:.3f})")
    print(f"Trajectory waypoints: {len(trajectory_positions)}")
    print(f"{'='*70}\n")
    current_index = state["current_action_index"]
    elapsed_time = time.time() - step_start
    # Save the navigation check response
    _append_step_payload(
        state.get("scenario_results_dir"),
        current_index - 1,
        {   "evaluation_result": eval_result,
            "time_evaluate_node": round(elapsed_time, 2),
        },
    )
    # if the mission type is patrol mission
    if state["mission"]["metadata"]["task_category"].lower() == "patrol":
        mission_name = state["mission"].get("name")
        env_name = state["mission"].get("metadata", {}).get("environment")
        dataset_root = Path(__file__).resolve().parents[1] / "dataset"

        gt_path = dataset_root / "Patrol" / str(env_name) / str(mission_name) / "airsim_rec.txt"
        if not gt_path.exists() and env_name and mission_name:
            matches = list(dataset_root.glob(f"**/{env_name}/{mission_name}/airsim_rec.txt"))
            if matches:
                gt_path = matches[0]

        if not gt_path.exists():
            raise FileNotFoundError(
                f"Could not resolve patrol GT file airsim_rec.txt for mission='{mission_name}', "
                f"environment='{env_name}'. Tried: {gt_path}"
            )

        print(f"   🧭 Computing patrol metrics using GT path: {gt_path}")
        patrol_metric = PatrolMetric(gt_path=gt_path,
                                     vlm_path=str(state.get("scenario_results_dir"))+"/debug_outputs/step_by_step_intermediate_results.json")
        buffer_radius = 20.0
        num_segments = 100
        iou = -1
        try:
            patrol_metric.plot_trajectories_with_buffers(buffer_radius=buffer_radius,
                                                         num_segments=num_segments,
                                                         save_path=state.get("scenario_results_dir")/f"patrol_buffer_visualization.png",
                                                         title="")
            iou = patrol_metric.compute_actual_iou(buffer_radius=buffer_radius, num_segments=num_segments)
            print(f"Actual area IoU: {iou:.4f}")
            eval_result["patrol_metrics"] = {"iou": iou}
            # Update SR5 and SR10 based on patrol IoU threshold
            patrol_success_5 = 1 if (iou is not None and iou >= 0.5) else 0
            patrol_success_10 = 1 if (iou is not None and iou >= 0.25) else 0
            eval_result["sr5"] = patrol_success_5
            eval_result["sr10"] = patrol_success_10
            eval_result["mission_progress"] = iou
            eval_result["stepwise_binary_progress"] = stepwise_binary_progress
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Actual area IoU: Error occurred: {e}")
    try:
        if state["mission"]["metadata"]["task_category"].lower() == "manipulation":
            # Update SR5 and SR10 based on manipulation instruction provided
            # the mission is successful only if the task_gt is successfully achieved, which is already incorporated into the task_gt_match metric, so we can use that to update the SR5 and SR10
            manip_success = 1 if task_gt_match == 1 else 0
            # easy metric and hard metric
            eval_result["sr5"] = manip_success and eval_result["sr5"]  # Keep original SR5 but also require task_gt match
            eval_result["sr10"] = manip_success and eval_result["sr10"]
    except Exception as e:
        print(f"   ⚠ Error in manipulation success evaluation: {e}")
        import traceback
        traceback.print_exc()
    if eval_result["sr5"] == 1:
        eval_result["mission_progress"] = 1
    return {"evaluation_result": eval_result}


if __name__ == "__main__":
    # For testing individual nodes
    # gpt4o-mini
    pickle_path = "/home/uam/taehyoung/suman/COLM/MissionBench/data/results/experiment_results_21-mar-20.48pm_/illegally_parked_car_mission30_gemini-3.1-pro-preview_exp_000012_gemini-3.1-pro-preview_rep3/scenario_results/final_state.pkl"
    final_state = pickle.load(open(pickle_path, "rb"))
    eval_result = node_evaluate_step_by_step(final_state)
    print(eval_result)