"""
Two-Stage Hierarchical Action Planner Demo

This script demonstrates the two-stage HLAP system:
    Stage 1: Image + Mission Instruction + Drone Pose → High-Level Actions
    Stage 2: Each High-Level Action → Low-Level Waypoints

The system uses two Vision-Language Models:
1. Stage 1 VLM: Analyzes image and generates strategic high-level actions
2. Stage 2 VLM: Converts each action to executable waypoints

Workflow:
- User provides: image, instruction, drone pose
- Stage 1 generates: ["turn right 30 degrees", "go straight 10 meters", ...]
- Stage 2 loops through each action and generates waypoints:
  - "turn right 30 degrees" → [{"dx": 0, "dy": 0, "dz": 0, "dyaw": 30}]
  - "go straight 10 meters" → [{"dx": 2, ...}, {"dx": 2, ...}, ...]

Author: Taehyoung Kim
Usage: python examples/two_stage_demo.py (from repository root)
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json
import time
# Add parent directory to path for imports
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
print(f"Added {parent_dir} to sys.path for module imports.")
# Import planner modules
from predictors.utils import setup_environment
from graphs import build_step_by_step_graph
from predictors.utils import (
    create_initial_step_by_step,
    display_results_two_stage,
    format_drone_pose,
)

# others
from utils.dataset_loader import MissionPlanningDataset
from predictors.utils import stitch_images_to_video

# =============================================================================
# PROGRESS & STREAMING UTILITIES
# =============================================================================

def stream_graph_with_progress(
    graph,
    initial_state: Dict,
    recursion_limit: int = 1000,
    show_node_details: bool = True,
    show_state_updates: bool = False
) -> Dict:
    """
    Execute graph with real-time progress streaming to terminal.
    
    Shows each node execution as it happens, making it easy to track progress
    during long-running model inferences.
    
    Args:
        graph: Compiled LangGraph workflow
        initial_state: Initial state dict
        recursion_limit: Max iterations
        show_node_details: Print detailed info for each node
        show_state_updates: Print state changes (verbose)
    
    Returns:
        Final state dict
    """
    node_counter = {}
    final_state = None
    
    try:
        # Use streaming to show progress
        for output in graph.stream(
            initial_state,
            config={"recursion_limit": recursion_limit}
        ):
            # output is a dict like: {"node_name": {"key": value}}
            for node_name, node_output in output.items():
                # Track node execution count
                node_counter[node_name] = node_counter.get(node_name, 0) + 1
                count = node_counter[node_name]
                
                # Friendly node name formatting
                display_name = node_name.replace("_", " ").title()
                
                if show_node_details:
                    print(f"\n   📍 [{count}] {display_name}")
                    
                    # Show key outputs from this node
                    if isinstance(node_output, dict):
                        for key in list(node_output.keys())[:3]:  # Show first 3 keys
                            value = node_output[key]
                            # Truncate long values
                            if isinstance(value, str) and len(value) > 100:
                                print(f"      • {key}: {value[:100]}...")
                            elif isinstance(value, list) and len(value) > 3:
                                print(f"      • {key}: {value[:2]} ... ({len(value)} items)")
                            else:
                                print(f"      • {key}: {value}")
                else:
                    print(f"   ✓ {display_name} (#{count})", end="", flush=True)
                    print("")  # Newline
                
                # Update final state after each node
                final_state = {**initial_state, **node_output}
                
                if show_state_updates and isinstance(node_output, dict):
                    print(f"      State updated with {len(node_output)} keys")
        
        print(f"\n   ✓ Workflow complete after {sum(node_counter.values())} node executions")
        print(f"      Nodes executed: {', '.join(f'{k} ({v}x)' for k, v in node_counter.items())}")
        
        return final_state or initial_state
        
    except Exception as e:
        print(f"\n   ❌ Error during graph execution: {e}")
        raise


def execute_graph_with_logging(
    graph,
    initial_state: Dict,
    recursion_limit: int = 1000,
    verbose: bool = True
) -> Dict:
    """
    Execute graph using invoke() with enhanced logging.
    
    This is the standard approach but with better console output.
    Faster than streaming but less real-time feedback.
    
    Args:
        graph: Compiled LangGraph workflow
        initial_state: Initial state dict
        recursion_limit: Max iterations
        verbose: Print execution details
    
    Returns:
        Final state dict
    """
    if verbose:
        print("🎬 Invoking graph (standard execution)...")
    
    start_time = time.time()
    final_state = graph.invoke(initial_state, {"recursion_limit": recursion_limit})
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"✓ Execution completed in {elapsed:.2f}s")
        print(f"  Total actions: {len(final_state.get('parsed_actions', []))}")
        print(f"  Total waypoints: {len(final_state.get('all_parsed_waypoints', []))}")
    
    return final_state


def execute_with_progress_spinner(
    graph,
    initial_state: Dict,
    recursion_limit: int = 1000
) -> Dict:
    """
    Execute graph with a simple spinner showing activity.
    
    Good for long operations where you just want to see something moving.
    
    Args:
        graph: Compiled LangGraph workflow
        initial_state: Initial state dict
        recursion_limit: Max iterations
    
    Returns:
        Final state dict
    """
    import sys
    import threading
    
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    stop_spinner = False
    
    def spin():
        i = 0
        while not stop_spinner:
            sys.stdout.write(f'\r   🤖 Processing {spinner_chars[i % len(spinner_chars)]}')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
    
    # Run spinner in background
    spinner_thread = threading.Thread(target=spin, daemon=True)
    spinner_thread.start()
    
    try:
        final_state = graph.invoke(initial_state, {"recursion_limit": recursion_limit})
    finally:
        stop_spinner = True
        spinner_thread.join(timeout=1)
        print(f"\r   ✓ Processing complete                    ")
    
    return final_state

# =============================================================================
# CORE FUNCTIONALITY
# =============================================================================


def save_step_by_step_results(
    final_state: Dict,
    scenario_name: str,
    scenario_results_dir: str
) -> str:
    """
    Save step-by-step execution results for debugging and analysis.
    
    Args:
        final_state: Final state dict from graph execution
        scenario_name: Name of the scenario for organization
        scenario_results_dir: Directory to save results
    Returns:
        Path to saved results directory
    """
    # Create results directory structure

    # Prepare data for serialization (remove non-serializable objects)
    serializable_state = {}
    non_serializable_keys = []

    for key, value in final_state.items():
        try:
            json.dumps(value, default=str)
            serializable_state[key] = value
        except (TypeError, ValueError):
            non_serializable_keys.append(key)
            # Try to convert to string representation
            serializable_state[f"{key}_str"] = str(value)

    # Save individual outputs for easier debugging
    debug_outputs = {
        "instruction": final_state.get("instruction"),
        "raw_response": final_state.get("raw_response"),
        "parsed_actions": final_state.get("parsed_actions"),
        "all_raw_waypoint_response": final_state.get("all_raw_waypoint_response", []),
        "model_name": final_state.get("model_name")
    }

    debug_file = scenario_results_dir / "debug_outputs.json"
    with open(debug_file, "w") as f:
        json.dump(debug_outputs, f, indent=2, default=str)
    print(f"   ✓ Saved debug outputs to: {debug_file}")

    # Save evaluation results if available
    if final_state.get("evaluation_result"):
        eval_file = scenario_results_dir / "evaluation_result.json"
        with open(eval_file, "w") as f:
            # Filter out non-serializable fields like PIL images
            eval_data = {k: v for k, v in final_state["evaluation_result"].items() 
                        if not isinstance(v, type(None)) and k != "final_step_result"}
            json.dump(eval_data, f, indent=2, default=str)
        print(f"   ✓ Saved evaluation results to: {eval_file}")

    # Save summary report
    summary = {
        "scenario_name": scenario_name,
        "instruction": final_state.get("instruction"),
        "model_name": final_state.get("model_name"),
        "low_level_model_name": final_state.get("low_level_model_name"),
        "num_actions": len(final_state.get("parsed_actions", [])),
        "actions": [a for a in final_state.get("parsed_actions", [])],
        "all_parsed_waypoints": final_state.get("all_parsed_waypoints", []),
        "evaluation": {
            "overall_similarity": final_state.get("evaluation_result", {}).get("overall_high_level_reasoning_similarity"),
            "exact_matches": final_state.get("evaluation_result", {}).get("exact_matches_high_level"),
            "mission_passed": final_state.get("evaluation_result", {}).get("mission_passed"),
            "waypoint_deviations": final_state.get("evaluation_result", {}).get("waypoint_deviation"),
        } if final_state.get("evaluation_result") else None,
    }

    summary_file = scenario_results_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   ✓ Saved summary to: {summary_file}")

    if non_serializable_keys:
        print(f"   ⚠ Note: {len(non_serializable_keys)} non-serializable fields excluded: {', '.join(non_serializable_keys)}")
    # save video of the step_by_step images
    base_dir = scenario_results_dir
    image_folder = os.path.join(base_dir, "imgs")
    video_path = os.path.join(image_folder, "mission_video.mp4")
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not os.path.exists(video_path):
        print(f"Stitching {len(image_files)} images to {video_path}...")
        ok = stitch_images_to_video(image_folder, image_files, output_path=video_path)
        if ok:
            print(f"Video saved to {video_path}")
        else:
            print("No images found or failed to create video.")
    else:
        print(f"Video already exists at {video_path}")
    return str(scenario_results_dir)


def run_step_by_step_algorithm(
    tolerances: dict,
    mission: dict,
    image: str,
    instruction: str,
    drone_pose: Dict[str, float],
    model_name: str = "gpt-4o-mini",
    low_level_model_name: Optional[str] = None,
    ground_truth: Optional[List[str]] = None,
    show_raw: bool = True,
    scenario_name: Optional[str] = None,
    scenario_metadata: Optional[Dict] = None,
    execution_mode: str = "spinner",
) -> Dict:
    """
    Execute two-stage mission planning using LangGraph workflow.

    Args:
        image: Path to drone camera image
        instruction: Mission objective
        drone_pose: Current drone position and orientation
        model_name: VLM model for high-level planning
        low_level_model_name: VLM model for waypoint generation (can use faster/cheaper model)
        ground_truth: Optional list of expected high-level actions for evaluation
        show_raw: Whether to display raw VLM responses
        scenario_name: Optional scenario name for LangSmith tracing
        scenario_metadata: Optional metadata dict for LangSmith organization
        execution_mode: How to display progress:
            - "stream": Real-time node-by-node progress (default, best for debugging)
            - "invoke": Standard execution with summary stats
            - "spinner": Simple spinning indicator
            - "silent": No progress output

    Returns:
        Final state dict with all results
    """
    ### USER CONFIGURATION ###
    
    # Configure LangSmith trace metadata for this scenario
    if scenario_name:
        os.environ["LANGCHAIN_RUN_NAME"] = scenario_name

        if scenario_metadata:
            tags = [
                scenario_name,
                scenario_metadata.get("task_category", "general"),
                scenario_metadata.get("environment", "unknown"),
            ]
            os.environ["LANGCHAIN_TAGS"] = ",".join(tags)

    # Display scenario info
    print(f"\n{'#' * 60}")
    print(f"# {scenario_name or 'Step by Step Mission Planning'}")
    print(f"{'#' * 60}")
    print(f"Image: {image}")
    print(f"Instruction: {instruction}")
    print(f"Stage 1 Model: {model_name}")
    print(f"Stage 2 Model: {low_level_model_name or model_name}")
    print(f"Drone Pose: {format_drone_pose(drone_pose)}")

    # Build graph
    print("\n🔧 Building step_by_step LangGraph workflow...")
    graph = build_step_by_step_graph(verbose=False)
    print("   ✓ Graph compiled")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_results_dir=Path(f"data/results/step_by_step/{scenario_name or 'unnamed_scenario'}/{model_name}_{low_level_model_name}_{timestamp}")
    scenario_results_dir.mkdir(parents=True, exist_ok=True)
    (scenario_results_dir/"imgs").mkdir(parents=True, exist_ok=True)
    # copy the start image to the scenario_results_dir as step 0
    shutil.copy(f"{image}", os.path.join(Path(scenario_results_dir/"imgs"), f"airsim_latest_0_step_0.png"))
    # save the mission yaml file to the scenario_results_dir
    mission_file = scenario_results_dir / "mission_config.yaml"
    with open(mission_file, "w") as f:
        yaml.dump(mission, f)
    print(f"   ✓ Saved mission config to: {mission_file}")
    # Create initial state
    initial_state = create_initial_step_by_step(
        tolerances=tolerances,
        mission=mission,
        image=image,
        instruction=instruction,
        drone_pose=drone_pose,
        camera_orientation={"roll": mission.get("camera", {}).get("roll", 0), "pitch": mission.get("camera", {}).get("pitch", 0), "yaw": mission.get("camera", {}).get("yaw", 0)},
        model_name=model_name,
        low_level_model_name=low_level_model_name,
        ground_truth=ground_truth,
        metadata=scenario_metadata,
        scenario_results_dir=scenario_results_dir,
    )

    # Execute graph
    print("\n🚀 Executing two-stage workflow...")
    
    if execution_mode == "stream":
        # Show real-time node-by-node progress (RECOMMENDED FOR DEBUGGING)
        print("   Mode: Real-time streaming (shows each node execution)\n")
        final_state = stream_graph_with_progress(
            graph, 
            initial_state, 
            recursion_limit=1000,
            show_node_details=True
        )
    elif execution_mode == "invoke":
        # Standard execution with summary
        print("   Mode: Standard execution with summary\n")
        final_state = execute_graph_with_logging(graph, initial_state, verbose=True)
    elif execution_mode == "spinner":
        # Simple spinner for long operations
        print("   Mode: Simple progress spinner\n")
        final_state = execute_with_progress_spinner(graph, initial_state)
    elif execution_mode == "silent":
        # No progress output
        final_state = graph.invoke(initial_state, {"recursion_limit": 1000})
    else:
        raise ValueError(f"Unknown execution_mode: {execution_mode}. Use 'stream', 'invoke', 'spinner', or 'silent'")
    
    print("   ✓ Workflow complete")
    # print(f"{final_state = }")
    # Save Results
    save_step_by_step_results(final_state, scenario_name or "unnamed_scenario", scenario_results_dir)
    
    return final_state


# =============================================================================
# EXAMPLE SCENARIOS
# =============================================================================


def main(benchmark_dir = "/media/uam/25560a30-d30f-4830-bcc4-31ffd6c66632/home/uam/nava/benchmark", mission_type="Visual_Inspection_high_altitude"):
    """
    Place the benchmark directory in the same level as the langgraph-ivi repository.
    Main function demonstrating two-stage planning.

    This script shows the complete hierarchical planning process:
    - Stage 1: Generate strategic high-level actions
    - Stage 2: Convert each action to executable waypoints
    """
    print("\n" + "=" * 60)
    print("TWO-STAGE HIERARCHICAL ACTION PLANNER DEMO")
    print("=" * 60)
    # CONFIGS
    
    dataset = MissionPlanningDataset(benchmark_dir)
    validation_missions = dataset.validation_dataset(path="validation_vi_missions.yaml").values()
    print(f"Validation missions ({len(validation_missions)}):")
    assert mission_type in ["Visual_Inspection_high_altitude", "Manipulation_Missions", "Patrol_Missions"], "Invalid mission type"
    evaluation_file = f"{benchmark_dir}/evaluation.yaml"
    with open(evaluation_file, 'r') as file:
            tolerances = yaml.safe_load(file)
    
    with open("configs/models.yaml", 'r') as file:
        models = yaml.safe_load(file)

    # Define test scenarios
    scenario_paths = validation_missions
    print(f"Found {len(scenario_paths)} scenarios in {benchmark_dir}")
    # Setup environment and optionally enable LangSmith tracing
    setup_done = False
    # Run each scenario
    for i, scenario_path in enumerate(scenario_paths, 1):
        with open(scenario_path, 'r') as file:
            scenario = yaml.safe_load(file)
        # if the tolerance values are in the mission_config_file, then we use them instead
        if "tolerances" in scenario:
            tolerances = scenario["tolerances"]

        model_short_name = scenario.get("models_related", {}).get("high_level_model", "gpt-4o-mini")  # Stage 1: High-level planning
        low_level_model_short_name = scenario.get("models_related", {}).get("low_level_model", "gpt-4o-mini")  # Stage 2: Waypoint generation
        low_level_model_family = low_level_model_short_name.split("_")[0]
        model_family = model_short_name.split("_")[0]
        assert model_family in models, f"Unsupported model family: {model_family}"
        assert low_level_model_family in models, f"Unsupported model family: {low_level_model_family}"
        model_name = models[model_family].get(model_short_name)["model_name"]
        low_level_model_name = models[low_level_model_family].get(low_level_model_short_name)["model_name"]

        if not setup_done:

            use_langsmith = scenario.get("langsmith_logging", False)
            setup_environment(use_langsmith=use_langsmith, langsmith_project=f"{scenario_path}_Missions_{mission_type}_cam_pitch=45", env_file_name=".env")
            setup_done = True
        mission_path = os.path.dirname(scenario_path)
        final_state = run_step_by_step_algorithm(
            tolerances=tolerances,
            mission = scenario,
            image=f"{mission_path}/imgs/start_fp_image.png",
            instruction=scenario["instruction"],
            drone_pose=scenario["drone_start_pose"],
            model_name=model_name,
            low_level_model_name=low_level_model_name,
            ground_truth=scenario.get("ground_truth"),
            show_raw=True,  # Set to True to see full VLM responses
            scenario_name=scenario["name"],
            scenario_metadata=scenario.get("metadata", {}),
            execution_mode="spinner",  # Options: "stream" (default), "invoke", "spinner", "silent"
        )
        break
    print("\n\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main(benchmark_dir="./dataset")
