import json
import os
import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphs.step_by_step_nodes import node_evaluate_step_by_step
from graphs.state import StepByStepState
import pickle

def create_results_json(exp_path: str):
    """Load step_by_step_intermediate_results.json and create a results.json file 
    in the experiment_results directory using the evaluation node.

    Args:
        exp_path (str): Path to the experiment directory (e.g., data/results/step_by_step/mission_name/timestamp)
    Returns:
        Path to the created results.json file
    """
    exp_path = Path(exp_path)
    
    # Load the intermediate results
    intermediate_file = exp_path / "debug_outputs" / "step_by_step_intermediate_results.json"
    if not intermediate_file.exists():
        raise FileNotFoundError(f"Intermediate results not found: {intermediate_file}")
    
    with open(intermediate_file, 'r') as f:
        intermediate_data = json.load(f)
    
    # Load mission config
    mission_config_file = exp_path / "mission_config.yaml"
    if not mission_config_file.exists():
        raise FileNotFoundError(f"Mission config not found: {mission_config_file}")
    
    
    with open(mission_config_file, 'r') as f:
        mission = yaml.safe_load(f)
    
    # Reconstruct state from intermediate results
    # Get the last step to extract final state
    step_keys = sorted([int(k) for k in intermediate_data.keys()])
    last_step_key = str(step_keys[-1])
    last_step = intermediate_data[last_step_key]
    
    # Build all_drone_poses from all steps
    state = pickle.load(open(exp_path / "final_state.pkl", "rb"))
    all_drone_poses = state["all_drone_poses"]
    
    mission = state["mission"]
    intermediate_data = state.get("intermediate_data", {})
    # Run evaluation
    eval_result_dict = node_evaluate_step_by_step(state)
    eval_result = eval_result_dict["evaluation_result"]
    
    # Get experiment info from path
    # Path structure: data/results/step_by_step/mission_name/model_timestamp/
    mission_name = exp_path.parent.name
    folder_name = exp_path.name
    # read total time from the summary.json
    summary_file = exp_path / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    total_time = summary_data.get("total_execution_time")
    assert total_time is not None, "total_execution_time not found in summary.json"
        
    # Create results structure matching the example
    results = {
        "success": eval_result["sr5"],
        "experiment_id": folder_name,
        "config": {
            "scenario_name": mission_name,
            "model_name": mission.get("models_related", {}).get("high_level_model", "unknown"),
            "experiment_id": folder_name,
            "max_step_size": state["tolerances"]["MAX_STEP_SIZE"],
            "max_actions": state["tolerances"]["MAX_ACTIONS"],
            "temperature": mission.get("temperature", 1.0),
            "top_p": mission.get("top_p", 0.95),
            "action_waypoint_validation_enabled": True,
            "spatial_reasoning_enabled": False,
            "repeat_number": 1,
            "random_seed": 42,
            "tags": [mission_name, mission.get("models_related", {}).get("high_level_model", "unknown")]
        },
        "total_time": total_time,
        "mission_result": {
            "mission_status": "completed",
            "scenario_name": mission_name,
            "model_name": mission.get("model_name", "unknown"),
            "steps_executed": eval_result["steps_executed"],
            "steps_budget": state["tolerances"]["MAX_ACTIONS"],
            "total_execution_time": sum(
                step_data.get("time_total_hlap_node", 0) + 
                step_data.get("time_wp_generation_node", 0) + 
                step_data.get("time_should_continue_node", 0)
                for step_data in intermediate_data.values()
            ),
            "drone_final_pose": eval_result["final_pose"],
            "actions_executed": [
                step_data.get("current_action", [""])[0] 
                for step_data in sorted(intermediate_data.values(), key=lambda x: int(x.get("step", 0)))
                if "current_action" in step_data
            ],
            "waypoints_generated": len(all_drone_poses),
            "scenario_results_dir": str(exp_path),
            "OSR_5": eval_result["oracle_success_5"],
            "OSR_10": eval_result["oracle_success_10"],
            "OSR_20": eval_result["oracle_success_20"],
            "sr5": eval_result["sr5"],
            "sr10": eval_result["sr10"],
            "min_distance_to_target": eval_result["min_distance_to_target"],
            "final_distance_to_target": eval_result["final_distance_to_target"],
            "oracle_threshold": eval_result["oracle_threshold_5"],
            "trajectory_length": eval_result["trajectory_length"],
            "x_diff": eval_result["x_diff"],
            "y_diff": eval_result["y_diff"],
            "z_diff": eval_result["z_diff"],
            "yaw_diff": eval_result["yaw_diff"],
            "x_threshold_sr5": eval_result["x_threshold_sr5"],
            "y_threshold_sr5": eval_result["y_threshold_sr5"],
            "z_threshold_sr5": eval_result["z_threshold_sr5"],
            "yaw_threshold_sr5": eval_result["yaw_threshold_sr5"],
            "x_threshold_sr10": eval_result["x_threshold_sr10"],
            "y_threshold_sr10": eval_result["y_threshold_sr10"],
            "z_threshold_sr10": eval_result["z_threshold_sr10"],
            "yaw_threshold_sr10": eval_result["yaw_threshold_sr10"],
            "steps_executed_evaluation": eval_result["steps_executed"],
            "stop_reason": eval_result["stop_reason"],
            "collision_count": eval_result.get("collision_count", 0),
            "collision_rate": eval_result.get("collision_rate", 0.0),
            "mission_progress": eval_result.get("mission_progress"),
            "stepwise_binary_progress": eval_result.get("stepwise_binary_progress"),
            "task_gt_expected": eval_result.get("task_gt_expected"),
            "task_gt_pred": eval_result.get("task_gt_pred"),
            "task_gt_match": eval_result.get("task_gt_match"),
            "task_gt_model": eval_result.get("task_gt_model"),
            "task_gt_response": eval_result.get("task_gt_response"),
            "stop_condition": eval_result.get("stop_reason"),
            "patrol_results": eval_result.get("patrol_metrics", {})
        },
        "error_message": None,
        "disable_from_dashboard": 0
    }
    
    # Save directly in the scenario result directory to avoid duplicating
    # artifacts between data/results/* and experiment_results/*.
    results_file = exp_path / "result.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Created results file: {results_file}")
    return str(results_file)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        exp_path = sys.argv[1]
    else:
        # Default example path
        raise ValueError("Please provide the experiment path as an argument.")
    
    print(f"Processing experiment: {exp_path}")
    result_file = create_results_json(exp_path)
    print(f"Done! Result saved to: {result_file}")
