"""
ExperimentRunner: Executes a single experiment with a given configuration.
"""

import json
import time
import os
import sys
import shutil
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import yaml

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_automation.config import ExperimentConfig
from graphs import build_step_by_step_graph, build_step_by_step_1vlmgraph, build_step_by_step_1vlm_with_bb_graph
from predictors.utils import create_initial_step_by_step
from predictors.utils import setup_environment
from utils.image_utils import fetch_image
from simulator_interface.airsim_tools import reset_simulator
from utils.dataset_loader import MissionPlanningDataset
try:
    from utils.results_utils import save_step_by_step_results
except ImportError:
    print("Warning: utils module not found. Make sure to include it in the experiment_automation package.")
# def get_readable timestap()
def get_readable_timestamp():
    """Get a human-readable timestamp for directory naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class ExperimentRunner:
    """
    Executes a single experiment.
    
    Responsibilities:
    - Load experiment config
    - Run the mission with those parameters
    - Capture results (success, timing, debug info)
    - Save results to disk
    - Handle failures gracefully
    """
    
    def __init__(self, config: ExperimentConfig, results_dir: Path, benchmark_dir: Path = None):
        """
        Args:
            config: ExperimentConfig defining the experiment
            results_dir: Where to save results (e.g., /home/.../experiment_results/)
            benchmark_dir: Benchmark directory containing missions (e.g., UAVMissionPlanning/dataset)
        """
        self.config = config
        self.results_dir = Path(results_dir) / f"{self.config.scenario_name}_{self.config.model_name}_{self.config.experiment_id}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if benchmark_dir is None:
            raise ValueError("benchmark_dir must be provided by the caller")
        self.benchmark_dir = Path(benchmark_dir)
        
        self.start_time = None
        self.end_time = None
        self.result = None
    
    def _get_graph_builder(self):
        """
        Get the appropriate graph builder function based on strategy.
        
        Returns:
            Callable that builds and returns a LangGraph workflow
        """
        strategy = getattr(self.config, 'strategy', 'step_by_step').lower()
        
        graph_builders = {
            'vanilla_step_by_step': build_step_by_step_graph,
            'step_by_step_1vlm': build_step_by_step_1vlmgraph,
            'step_by_step_1vlm_with_bb': build_step_by_step_1vlm_with_bb_graph,
            'step_by_step_zeroshot': build_step_by_step_1vlmgraph,  # Use same graph but with zero-shot config
            # Add more strategies here as needed
        }
        
        if strategy not in graph_builders:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available strategies: {', '.join(graph_builders.keys())}"
            )
        
        return graph_builders[strategy]
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the experiment.
        
        Returns:
            Dict with keys:
            - success (bool)
            - experiment_id (str)
            - total_time (float)
            - error_message (str, if failed)
            - mission_result (dict, if successful)
        """
        try:
            self.start_time = time.time()
            with open("configs/models.yaml", 'r') as file:
                models = yaml.safe_load(file)
            print(f"\n{'='*60}")
            print(f"Running: {self.config}")
            print(f"Results dir: {self.results_dir}")
            print(f"{'='*60}")

            # This is where we'll inject the config into the existing demo code
            mission_result = self._run_mission()
            
            self.end_time = time.time()
            
            # Get config dict and add max_actions/max_step_size from mission
            config_dict = self.config.to_dict()
            if mission_result and "steps_budget" in mission_result:
                config_dict["max_actions"] = mission_result["steps_budget"]
                # Try to get max_step_size from somewhere (we'll add it to mission_result)
                config_dict["max_step_size"] = mission_result.get("max_step_size", None)
            
            self.result = {
                "success": mission_result.get("sr5", 0),
                "experiment_id": self.config.experiment_id,
                "config": config_dict,
                "total_time": self.end_time - self.start_time,
                "mission_result": mission_result,
                "error_message": None,
            }
            
            # Save result to disk
            self._save_result()
            
            return self.result
        except Exception as e:
            self.end_time = time.time()
            self.result = {
                "success": False,
                "experiment_id": self.config.experiment_id,
                "config": self.config.to_dict(),
                "total_time": self.end_time - (self.start_time or self.end_time),
                "mission_result": None,
                "error_message": str(e),
            }
            self._save_result()
            self._save_error(e)
            print(f"❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return self.result
    
    def _run_mission(self) -> Dict[str, Any]:
        """
        Execute the actual mission with current config using step_by_step algorithm.
        
        Loads mission config, builds LangGraph, and invokes the workflow.
        """
        # try:
        dataset = MissionPlanningDataset(self.benchmark_dir)
        validation_missions = dataset.validation_dataset(path=self.config.val_mission_file)
        # Load mission configuration
        mission_config_path = Path(validation_missions[self.config.scenario_name])
        if not mission_config_path:
            raise FileNotFoundError(f"Mission config not found: {mission_config_path}")
        
        with open(mission_config_path, 'r') as f:
            mission_config = yaml.safe_load(f)
        
        # Load tolerances
        tolerances = mission_config.get("tolerances", {})
        tolerances["MAX_STEP_SIZE"] = mission_config.get("tolerances", {}).get("MAX_STEP_SIZE")
        tolerances["MAX_ACTIONS"] = mission_config.get("tolerances", {}).get("MAX_ACTIONS")
        
        # Setup environment
        setup_environment(use_langsmith=False)
        
        # Override Google Drive results path if specified in config
        if self.config.model_results_drive_path:
            os.environ["GOOGLE_DRIVE_RESULTS_FOLDER_URL"] = self.config.model_results_drive_path
            print(f"   📂 Overriding Google Drive Folder URL: {self.config.model_results_drive_path}")

        # Build the graph based on strategy
        graph_builder = self._get_graph_builder()
        strategy = getattr(self.config, 'strategy', 'step_by_step')
        print(f"   🔧 Building {strategy} LangGraph workflow...")
        graph = graph_builder()
        print(f"   ✓ Graph compiled")
        
        # Keep all per-experiment artifacts co-located with result.json.
        scenario_results_dir = self.results_dir / "scenario_results"
        print(f"   📁 Creating results directory: {scenario_results_dir}")
        scenario_results_dir.mkdir(parents=True, exist_ok=True)
        (scenario_results_dir / "imgs").mkdir(parents=True, exist_ok=True)
        
        # Copy mission config to results directory
        mission_path = os.path.dirname(mission_config_path)
        shutil.copy(mission_config_path, scenario_results_dir / "mission_config.yaml")
        
        # Prepare initial state data
        instruction = mission_config.get("instruction", "")
        drone_pose = mission_config.get("drone_start_pose", {"x": 0, "y": 0, "z": 0, "yaw": 0})

        # Reset simulator for every mission and spawn at mission start pose.
        print(f"   🔄 Resetting simulator at start pose: {drone_pose}")
        reset_simulator(center_position=drone_pose)

        assert "models_related" in mission_config, "models_related section missing in mission config"
        mission_config["models_related"]["temperature"] = self.config.temperature
        mission_config["models_related"]["top_p"] = self.config.top_p
        mission_config["models_related"]["top_k"] = self.config.top_k
        mission_config["models_related"]["min_p"] = self.config.min_p
        mission_config["models_related"]["presence_penalty"] = self.config.presence_penalty
        mission_config["models_related"]["repetition_penalty"] = self.config.repetition_penalty
        mission_config["models_related"]["thinking_level"] = self.config.thinking_level
        mission_config["models_related"]["task_gt_eval_model"] = self.config.task_gt_eval_model
        
        print(f"   📋 Creating initial state...")
        img_enhancement_type = self.config.img_enhancement_type
        
        # Load static image as baseline for state initialization
        image_path = f"{mission_path}/imgs/start_fp_image.png"
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Create initial state with static image for proper initialization
        initial_state = create_initial_step_by_step(
            tolerances=tolerances,
            mission=mission_config,
            image=str(image_path),
            instruction=instruction,
            drone_pose=drone_pose,
            camera_orientation={
                "roll": mission_config.get("camera", {}).get("roll", 0),
                "pitch": mission_config.get("camera", {}).get("pitch", -45),
                "yaw": mission_config.get("camera", {}).get("yaw", 0),
            },
            model_name=self.config.model_name,
            low_level_model_name=self.config.model_name,  # Use same model for both stages
            ground_truth=None,
            metadata={
                "experiment_id": self.config.experiment_id,
                "repeat": self.config.repeat_number,
            },
            scenario_results_dir=scenario_results_dir,
            img_enhancement_type=img_enhancement_type,
            high_level_action_prompt_type=self.config.high_level_action_prompt_type,
            strategy=self.config.strategy,
            num_history_images=self.config.num_history_images,
            capture_interval=self.config.capture_interval,
        )
        
        # Fetch live image from simulator using actual drone start pose
        print(f"   📸 Fetching image from simulator at drone pose {drone_pose}...")
        first_image_dest = scenario_results_dir / "imgs" / "airsim_latest_0_step_0.png"
        shutil.copy(initial_state["image"], first_image_dest)
        initial_state["image"] = Path(first_image_dest)
        image_path_enhanced, _ = fetch_image(initial_state, img_enhancement_type=img_enhancement_type)
        first_image_dest.unlink(missing_ok=True)  # Remove initial static image file after enhancement step
        # shutil.copy(image_path_enhanced, first_image_dest)
        if image_path_enhanced:
            enhanced_image_name = Path(image_path_enhanced).name[:-4] + "_step_0.png"
            final_name = scenario_results_dir / "imgs" / enhanced_image_name
            # Copy fetched image to results
            shutil.copy(image_path_enhanced, final_name)
            initial_state["image"] = final_name
            initial_state["img_history"] = [final_name]
        else:
            # Fallback to static image if fetch fails
            print(f"   ⚠ Simulator fetch failed, using static image fallback")
            shutil.copy(image_path, scenario_results_dir / "imgs" / "airsim_latest_0_step_0.png")
        # Execute graph
        print(f"   🚀 Executing workflow...")
        workflow_start = time.time()
        final_state = graph.invoke(initial_state, {"recursion_limit": 1000})
        workflow_time = time.time() - workflow_start
        
        print(f"   ✓ Workflow complete in {workflow_time:.2f}s")
        
        # Extract evaluation metrics (OSR, SR, etc.)
        evaluation = final_state.get("evaluation_result", {}) or {}

        # Save step-by-step artifacts (evaluation, summaries, etc.)
        # Import here to avoid circular imports
        from utils.results_utils import save_step_by_step_results
        save_step_by_step_results(
            final_state,
            self.config.scenario_name,
            scenario_results_dir,
            total_execution_time=workflow_time,
            output_media_format=getattr(self.config, "output_media_format", "webp"),
        )

        # Extract results
        mission_result = {
            "mission_status": "completed",
            "scenario_name": self.config.scenario_name,
            "model_name": self.config.model_name,
            "steps_executed": len(final_state.get("parsed_actions", [])),
            "steps_budget": tolerances.get("MAX_ACTIONS"),
            "max_step_size": tolerances.get("MAX_STEP_SIZE"),  # Add for config
            "total_execution_time": workflow_time,
            "drone_final_pose": final_state.get("drone_pose", {}),
            "collided": bool(final_state.get("collided", False)),
            "collision_info": final_state.get("collision_info", []),
            "collision_count": evaluation.get("collision_count", 0),
            "collision_rate": evaluation.get("collision_rate", 0.0),
            "mission_progress": evaluation.get("mission_progress"),
            "stepwise_binary_progress": evaluation.get("stepwise_binary_progress"),
            "task_gt_expected": evaluation.get("task_gt_expected"),
            "task_gt_pred": evaluation.get("task_gt_pred"),
            "task_gt_match": evaluation.get("task_gt_match"),
            "task_gt_model": evaluation.get("task_gt_model"),
            "task_gt_response": evaluation.get("task_gt_response"),
            "actions_executed": final_state.get("parsed_actions", []),
            "waypoints_generated": len(final_state.get("all_parsed_waypoints", [])),
            "scenario_results_dir": str(scenario_results_dir),
            # Evaluation metrics
            "OSR_5": evaluation.get("oracle_success_5"),
            "OSR_10": evaluation.get("oracle_success_10"),
            "OSR_20": evaluation.get("oracle_success_20"),
            "sr5": evaluation.get("sr5"),
            "sr10": evaluation.get("sr10"),
            "min_distance_to_target": evaluation.get("min_distance_to_target"),
            "final_distance_to_target": evaluation.get("final_distance_to_target"),
            "oracle_threshold": evaluation.get("oracle_threshold"),
            "trajectory_length": evaluation.get("trajectory_length"),
            "x_diff": evaluation.get("x_diff"),
            "y_diff": evaluation.get("y_diff"),
            "z_diff": evaluation.get("z_diff"),
            "yaw_diff": evaluation.get("yaw_diff"),
            "x_threshold_sr5": evaluation.get("x_threshold_sr5"),
            "y_threshold_sr5": evaluation.get("y_threshold_sr5"),
            "z_threshold_sr5": evaluation.get("z_threshold_sr5"),
            "yaw_threshold_sr5": evaluation.get("yaw_threshold_sr5"),
            "x_threshold_sr10": evaluation.get("x_threshold_sr10"),
            "y_threshold_sr10": evaluation.get("y_threshold_sr10"),
            "z_threshold_sr10": evaluation.get("z_threshold_sr10"),
            "yaw_threshold_sr10": evaluation.get("yaw_threshold_sr10"),
            "steps_executed_evaluation": evaluation.get("steps_executed"),
            "stop_reason": evaluation.get("stop_reason"),
            "stop_condition": evaluation.get("stop_reason"),
            "patrol_results": evaluation.get("patrol_metrics", {})
        }
        
        return mission_result
        
        # except Exception as e:
        #     print(f"   ❌ Mission execution failed: {e}")
        #     raise
    
    def _save_result(self):
        """Save result JSON and metadata to disk."""
        result_file = self.results_dir / "result.json"
        
        with open(result_file, "w") as f:
            json.dump(self.result, f, indent=2)
        
        print(f"✓ Result saved to {result_file}")

    def _save_error(self, exc: Exception) -> None:
        """Persist error details for this experiment."""
        error_file = self.results_dir / "error.txt"
        with open(error_file, "w") as f:
            f.write(f"Experiment: {self.config.experiment_id}\n")
            f.write(f"Model: {self.config.model_name}\n")
            f.write(f"Error: {exc}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
