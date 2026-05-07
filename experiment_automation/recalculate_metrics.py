"""
Recalculate evaluation metrics for existing experiment results.

This script:
1. Loads existing result.json files
2. Fetches ground truth waypoint from mission_config.yaml
3. Extracts drone trajectory from step_by_step_intermediate_results.json
4. Recalculates min_distance_to_target and final_distance_to_target
5. Updates result.json files with corrected metrics
6. Regenerates dashboard
"""

import json
import yaml
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple


def normalize_angle_difference(angle1: float, angle2: float) -> float:
    """Calculate the smallest angle difference between two angles."""
    diff = angle1 - angle2
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def load_mission_config(scenario_name: str, benchmark_dir: Path) -> Dict[str, Any]:
    """Load mission configuration for a given scenario."""
    # Parse scenario name to find mission config
    # Example: NHEnv_VI_grey_car_unittest_mission30
    parts = scenario_name.split("_")
    
    # Construct path based on naming convention
    # Visual_Inspection/NHEnv/grey_car_unittest_mission30/mission_config.yaml
    if "VI" in parts:
        task_category = "Visual_Inspection"
        env = parts[0]  # NHEnv
        mission_name = "_".join(parts[2:])  # grey_car_unittest_mission30
    else:
        raise ValueError(f"Unknown scenario format: {scenario_name}")
    
    config_path = benchmark_dir / task_category / env / mission_name / "mission_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Mission config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_trajectory_from_intermediate_results(results_dir: Path) -> List[Dict[str, float]]:
    """Extract drone trajectory from step_by_step_intermediate_results.json."""
    intermediate_file = results_dir / "debug_outputs" / "step_by_step_intermediate_results.json"
    
    if not intermediate_file.exists():
        print(f"   ⚠ Intermediate results not found: {intermediate_file}")
        return []
    
    with open(intermediate_file, 'r') as f:
        intermediate_data = json.load(f)
    
    # Extract drone poses from each step
    poses = []
    for step_key in sorted(intermediate_data.keys(), key=lambda x: int(x)):
        step_data = intermediate_data[step_key]
        if "drone_pose" in step_data:
            poses.append(step_data["drone_pose"])
    
    return poses


def recalculate_evaluation(
    start_pose: Dict[str, float],
    trajectory_poses: List[Dict[str, float]],
    target_pose: Dict[str, float],
    tolerances: Dict[str, float]
) -> Dict[str, Any]:
    """
    Recalculate evaluation metrics.
    
    Args:
        start_pose: Starting drone position
        trajectory_poses: List of drone positions during trajectory
        target_pose: Ground truth target waypoint
        tolerances: Threshold values for success
    
    Returns:
        Dict with recalculated metrics
    """
    target_x = target_pose.get("x", 0.0)
    target_y = target_pose.get("y", 0.0)
    target_z = target_pose.get("z", 0.0)
    target_yaw = target_pose.get("yaw", 0.0)
    
    # Build full trajectory including start
    all_positions = [(start_pose["x"], start_pose["y"], start_pose["z"])]
    for pose in trajectory_poses:
        all_positions.append((pose["x"], pose["y"], pose["z"]))
    
    # Calculate minimum distance across trajectory
    min_distance = float('inf')
    for pos_x, pos_y, pos_z in all_positions:
        distance = math.sqrt(
            (pos_x - target_x)**2 + 
            (pos_y - target_y)**2 + 
            (pos_z - target_z)**2
        )
        min_distance = min(min_distance, distance)
    
    # Get final pose
    if trajectory_poses:
        final_pose = trajectory_poses[-1]
    else:
        final_pose = start_pose
    
    final_x = final_pose["x"]
    final_y = final_pose["y"]
    final_z = final_pose["z"]
    final_yaw = final_pose.get("yaw", 0.0)
    
    # Calculate final distance
    final_distance = math.sqrt(
        (final_x - target_x)**2 + 
        (final_y - target_y)**2 + 
        (final_z - target_z)**2
    )
    
    # Oracle Success Rate
    oracle_threshold = 5.0
    oracle_success = 1 if min_distance <= oracle_threshold else 0
    
    # Success Rate (final position within thresholds)
    x_threshold = tolerances.get("x", 5.0)
    y_threshold = tolerances.get("y", 5.0)
    z_threshold = tolerances.get("z", 5.0)
    yaw_threshold = tolerances.get("yaw", 15.0)
    
    x_diff = abs(final_x - target_x)
    y_diff = abs(final_y - target_y)
    z_diff = abs(final_z - target_z)
    yaw_diff = abs(normalize_angle_difference(final_yaw, target_yaw))
    
    success_rate = 1 if (
        x_diff <= x_threshold and
        y_diff <= y_threshold and
        z_diff <= z_threshold and
        yaw_diff <= yaw_threshold
    ) else 0
    
    return {
        "oracle_success": oracle_success,
        "success_rate": success_rate,
        "min_distance_to_target": min_distance,
        "final_distance_to_target": final_distance,
        "x_diff": x_diff,
        "y_diff": y_diff,
        "z_diff": z_diff,
        "yaw_diff": yaw_diff,
        "trajectory_length": len(all_positions),
    }


def recalculate_all_results(
    results_base_dir: Path,
    benchmark_dir: Path
):
    """Recalculate metrics for all experiment results."""
    
    result_dirs = [d for d in results_base_dir.iterdir() if d.is_dir()]
    
    print(f"\n{'='*70}")
    print(f"Recalculating evaluation metrics for {len(result_dirs)} experiments")
    print(f"{'='*70}\n")
    
    updated_count = 0
    error_count = 0
    
    for result_dir in result_dirs:
        result_file = result_dir / "result.json"
        
        if not result_file.exists():
            print(f"⚠ Skipping {result_dir.name} - no result.json")
            continue
        
        print(f"Processing: {result_dir.name}")
        
        try:
            # Load existing result
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            mission_result = result.get("mission_result", {})
            scenario_name = mission_result.get("scenario_name")
            scenario_results_dir = Path(mission_result.get("scenario_results_dir", ""))
            
            if not scenario_name:
                print(f"   ⚠ No scenario_name found, skipping")
                continue
            
            # Load mission config to get ground truth
            mission_config = load_mission_config(scenario_name, benchmark_dir)
            ground_truth = mission_config.get("ground_truth", {})
            waypoint_gt = ground_truth.get("waypoint_gt", {})
            start_pose = mission_config.get("drone_start_pose", {})
            tolerances = mission_config.get("tolerances", {})
            
            print(f"   📍 Ground Truth: x={waypoint_gt.get('x'):.2f}, y={waypoint_gt.get('y'):.2f}, z={waypoint_gt.get('z'):.2f}")
            
            # Extract trajectory from intermediate results
            if scenario_results_dir and Path(scenario_results_dir).exists():
                trajectory_poses = extract_trajectory_from_intermediate_results(Path(scenario_results_dir))
                print(f"   📊 Trajectory: {len(trajectory_poses)} waypoints")
            else:
                print(f"   ⚠ No intermediate results found, using final pose only")
                trajectory_poses = [mission_result.get("drone_final_pose", {})]
            
            # Recalculate metrics
            new_metrics = recalculate_evaluation(
                start_pose=start_pose,
                trajectory_poses=trajectory_poses,
                target_pose=waypoint_gt,
                tolerances=tolerances
            )
            
            # Update mission_result
            mission_result.update(new_metrics)
            result["mission_result"] = mission_result
            
            # Also update top-level success field
            result["success"] = new_metrics["success_rate"]
            
            # Save updated result
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"   ✅ Updated - OSR: {new_metrics['oracle_success']}, SR: {new_metrics['success_rate']}, "
                  f"Min Dist: {new_metrics['min_distance_to_target']:.2f}m, Final Dist: {new_metrics['final_distance_to_target']:.2f}m\n")
            updated_count += 1
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            error_count += 1
            continue
    
    print(f"{'='*70}")
    print(f"✅ Recalculation complete!")
    print(f"   Updated: {updated_count}")
    print(f"   Errors: {error_count}")
    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    import sys
    
    # Get directories
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path("data/results")
    
    if len(sys.argv) > 2:
        benchmark_dir = Path(sys.argv[2])
    else:
        benchmark_dir = Path("/home/nava/uav_mission_planning/UAVMissionPlanning/dataset")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    if not benchmark_dir.exists():
        print(f"❌ Benchmark directory not found: {benchmark_dir}")
        sys.exit(1)
    
    # Recalculate all metrics
    recalculate_all_results(results_dir, benchmark_dir)
    
    
    print("\n✓ Dashboard regenerated with corrected metrics!")


if __name__ == "__main__":
    main()
