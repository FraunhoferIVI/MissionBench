"""
ResultsAggregator: Analyzes results and generates HTML dashboard.
"""

import json
import csv
import yaml
import subprocess
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class ResultsAggregator:
    """
    Analyzes experiment results and generates reports.
   
    Responsibilities:
    - Load all result files from disk
    - Aggregate statistics
    - Generate CSV summary
    - Generate HTML dashboard
    """
   
    def __init__(self, results_base_dir: Path):
        """
        Args:
            results_base_dir: Base directory containing all experiment results
        """
        self.results_base_dir = Path(results_base_dir)
        self.results: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {}
        self.model_name_mapping = self._load_model_name_mapping()
        self.git_commits: Dict[str, Dict[str, str]] = {}

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _compute_step_usage_percent(
        steps_taken: Any,
        steps_budget: Any,
    ) -> Optional[float]:
        """Return step usage as percentage of budget, or None if unavailable."""
        try:
            taken = float(steps_taken)
            budget = float(steps_budget)
        except (TypeError, ValueError):
            return None

        if budget <= 0:
            return None

        return (taken / budget) * 100.0

    @staticmethod
    def _normalize_task_category(raw_category: str) -> str:
        if not raw_category:
            return "Unknown"
        category = str(raw_category).strip().lower()
        if category in {"patrol", "patrol_missions"}:
            return "Patrol"
        if category in {"manipulation", "manipulation_missions", "manipualtion"}:
            return "Manipulation"
        if category in {"visual_inspection", "visual inspection", "navigation", "vi", "visual_inspection_high_altitude"}:
            return "Visual Inspection"
        return raw_category

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Normalize model names by replacing underscores with hyphens."""
        if not model_name:
            return model_name
        return model_name.replace("_", "-")

    def _get_task_category(self, result: Dict[str, Any]) -> str:
        cached = result.get("_task_category")
        if cached:
            return cached

        mission = result.get("mission_result", {}) or {}
        raw_category = mission.get("task_category")

        if not raw_category:
            scenario_results_dir = mission.get("scenario_results_dir")
            if scenario_results_dir:
                mission_config_path = Path(scenario_results_dir) / "mission_config.yaml"
                if mission_config_path.exists():
                    try:
                        with open(mission_config_path, "r") as f:
                            mission_config = yaml.safe_load(f)
                        raw_category = (
                            (mission_config.get("metadata") or {}).get("task_category")
                            or mission_config.get("task_category")
                        )
                    except Exception:
                        raw_category = None

        if not raw_category:
            scenario_name = (mission.get("scenario_name") or result.get("config", {}).get("scenario_name") or "").lower()
            if "patrol" in scenario_name:
                raw_category = "patrol"
            elif "manipulation" in scenario_name:
                raw_category = "manipulation"
            elif "visual" in scenario_name or "vi" in scenario_name or "inspection" in scenario_name:
                raw_category = "visual_inspection"

        normalized = self._normalize_task_category(raw_category)
        result["_task_category"] = normalized
        return normalized

    @staticmethod
    def _get_result_scenario_name(result: Dict[str, Any]) -> str:
        """Return scenario name from mission_result, then config, else unknown."""
        mission = result.get("mission_result", {}) or {}
        scenario = mission.get("scenario_name")
        if not scenario:
            scenario = (result.get("config", {}) or {}).get("scenario_name")
        return scenario or "unknown"

    def _resolve_task_gt_fields(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve task_gt fields with fallback to saved evaluation artifacts."""
        cached = result.get("_resolved_task_gt_fields")
        if isinstance(cached, dict):
            return cached

        mission = result.get("mission_result", {}) or {}
        expected = mission.get("task_gt_expected")
        predicted = mission.get("task_gt_pred")
        match = mission.get("task_gt_match")
        response = mission.get("task_gt_response")

        scenario_results_dir = mission.get("scenario_results_dir")
        if scenario_results_dir:
            scenario_path = Path(scenario_results_dir)

            eval_path = scenario_path / "evaluation_result.json"
            if eval_path.exists():
                try:
                    with open(eval_path, "r") as f:
                        eval_data = json.load(f) or {}
                    if expected in (None, ""):
                        expected = eval_data.get("task_gt_expected")
                    if predicted in (None, ""):
                        predicted = eval_data.get("task_gt_pred")
                    if match is None:
                        match = eval_data.get("task_gt_match")
                    if response in (None, ""):
                        response = eval_data.get("task_gt_response")
                except Exception:
                    pass

            if expected in (None, ""):
                mission_config_path = scenario_path / "mission_config.yaml"
                if mission_config_path.exists():
                    try:
                        with open(mission_config_path, "r") as f:
                            mission_config = yaml.safe_load(f) or {}
                        expected = (mission_config.get("ground_truth") or {}).get("task_gt")
                    except Exception:
                        pass

        resolved = {
            "task_gt_expected": expected,
            "task_gt_pred": predicted,
            "task_gt_match": match,
            "task_gt_response": response,
        }
        result["_resolved_task_gt_fields"] = resolved
        return resolved

    def _load_model_name_mapping(self) -> Dict[str, str]:
        """Load model name mappings from models.yaml."""
        mapping = {}
        models_yaml = Path(__file__).parent.parent / "data" / "models.yaml"

        if not models_yaml.exists():
            return mapping

        try:
            with open(models_yaml, "r") as f:
                models_config = yaml.safe_load(f)

            # Build reverse mapping: full_name -> short_name
            for provider, models in models_config.items():
                for short_name, model_config in models.items():
                    full_name = model_config.get("model_name")
                    if full_name:
                        mapping[full_name] = short_name
        except Exception as e:
            print(f"⚠ Failed to load models.yaml: {e}")

        return mapping
   
    def _get_short_model_name(self, full_name: str) -> str:
        """Get short model name from full name, or return full name if not found."""
        return self.model_name_mapping.get(full_name, full_name)
   
    def _get_git_commit(self, repo_path: str) -> Optional[str]:
        """
        Get the current git commit hash from a repository.
       
        Args:
            repo_path: Path to git repository
           
        Returns:
            Commit hash (short) or None if not a git repo
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
   
    def _get_git_branch(self, repo_path: str) -> Optional[str]:
        """Get the current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
   
    def _get_git_commit_message(self, repo_path: str) -> Optional[str]:
        """Get the latest commit message (short)."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%s"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
   
    def extract_git_info(self) -> Dict[str, Dict[str, str]]:
        """
        Extract git commit information for the current MissionBench repository.
       
        Returns:
            Dict mapping repository name to commit metadata
        """
        print("\\nExtracting git commit information...")
        repo_name = "MissionBench"
        repo_path = Path(__file__).resolve().parent.parent
        git_info: Dict[str, Dict[str, str]] = {}

        if repo_path.exists():
            commit = self._get_git_commit(str(repo_path))
            branch = self._get_git_branch(str(repo_path))
            message = self._get_git_commit_message(str(repo_path))
            git_info[repo_name] = {
                "commit": commit or "unknown",
                "branch": branch or "unknown",
                "message": message or "N/A",
            }
            print(f"  {repo_name}: {commit or 'N/A'} ({branch or 'N/A'})")
        else:
            git_info[repo_name] = {
                "commit": "N/A",
                "branch": "N/A",
                "message": "N/A",
            }
            print(f"  {repo_name}: Path not found - {repo_path}")
       
        self.git_commits = git_info
        return git_info
   
    def load_results(self) -> None:
        """Load all result.json files from results directory."""
        print(f"\nLoading results from {self.results_base_dir}...")

        # Primary layout: <results_dir>/<experiment_name>/result.json
        result_files = list(self.results_base_dir.glob("*/result.json"))

        # Fallback layout: <results_dir>/<run_folder>/<experiment_name>/result.json
        # (common when results are archived/zipped with one extra top-level folder)
        if not result_files:
            nested_result_files = [
                p
                for p in self.results_base_dir.glob("*/*/result.json")
                if "scenario_results" not in p.parts
            ]
            if nested_result_files:
                result_files = nested_result_files

        print(f"Found {len(result_files)} result files")
       
        disabled_count = 0
        for result_file in result_files:
            try:
                with open(result_file, "r") as f:
                    result = json.load(f)
                    # Skip experiments with disable_from_dashboard=1
                    if result.get("disable_from_dashboard", 0) == 1:
                        disabled_count += 1
                        continue
                    self.results.append(result)
            except Exception as e:
                print(f"  ⚠ Failed to load {result_file}: {e}")
       
        print(f"Loaded {len(self.results)} results ({disabled_count} disabled, {len(result_files) - disabled_count - (len(result_files) - len(self.results) - disabled_count)} enabled)")
   
    def compute_statistics(self) -> None:
        """Compute aggregate statistics."""
        if not self.results:
            print("No results to analyze")
            return
       
        successful_results = [r for r in self.results if r["success"]]
       
        success_count = len(successful_results)
        total_count = len(self.results)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0

        # Oracle Success Rate (OSR) and Success Rate (SR5/SR10) derived from mission_result
        osr_5_values = []  # 1/0 or None
        osr_10_values = []  # 1/0 or None
        osr_20_values = []  # 1/0 or None
        sr5_values = []   # 1/0 or None
        sr10_values = []  # 1/0 or None
        collision_rate_values = []  # float in [0,1]
        mission_progress_values = []  # float (can be negative/positive)
        for r in self.results:
            mission = r.get("mission_result", {}) or {}
            osr_5 = mission.get("OSR_5")
            osr_10 = mission.get("OSR_10")
            osr_20 = mission.get("OSR_20")
            sr5 = mission.get("sr5")
            sr10 = mission.get("sr10")
            collision_rate = mission.get("collision_rate")
            mission_progress = mission.get("mission_progress")
            if osr_5 is not None:
                osr_5_values.append(osr_5)
            if osr_10 is not None:
                osr_10_values.append(osr_10)
            if osr_20 is not None:
                osr_20_values.append(osr_20)
            if sr5 is not None:
                sr5_values.append(sr5)
            if sr10 is not None:
                sr10_values.append(sr10)
            if collision_rate is not None:
                collision_rate_values.append(collision_rate)
            if mission_progress is not None:
                mission_progress_values.append(mission_progress)

        osr_5_rate = (sum(osr_5_values) / len(osr_5_values)) if osr_5_values else 0
        osr_10_rate = (sum(osr_10_values) / len(osr_10_values)) if osr_10_values else 0
        osr_20_rate = (sum(osr_20_values) / len(osr_20_values)) if osr_20_values else 0
        sr5_rate = (sum(sr5_values) / len(sr5_values)) if sr5_values else 0
        sr10_rate = (sum(sr10_values) / len(sr10_values)) if sr10_values else 0
        avg_collision_rate = (
            sum(collision_rate_values) / len(collision_rate_values)
            if collision_rate_values
            else 0
        )
        avg_mission_progress = (
            sum(mission_progress_values) / len(mission_progress_values)
            if mission_progress_values
            else 0
        )
       
        # Group by model (excluding Patrol missions for dashboard display)
        by_model = {}
        by_model_non_patrol = {}
        for result in self.results:
            model = self._normalize_model_name(result["config"]["model_name"])
            category = self._get_task_category(result)
           
            if model not in by_model:
                by_model[model] = {"success": 0, "total": 0, "osr_5_count": 0, "osr_10_count": 0, "osr_20_count": 0}
            by_model[model]["total"] += 1
            if result["success"]:
                by_model[model]["success"] += 1
            mission_for_model = result.get("mission_result", {}) or {}
            if mission_for_model.get("OSR_5") == 1:
                by_model[model]["osr_5_count"] += 1
            if mission_for_model.get("OSR_10") == 1:
                by_model[model]["osr_10_count"] += 1
            if mission_for_model.get("OSR_20") == 1:
                by_model[model]["osr_20_count"] += 1
           
            # Track non-Patrol missions separately for dashboard
            if category != "Patrol":
                if model not in by_model_non_patrol:
                    by_model_non_patrol[model] = {"success": 0, "total": 0, "osr_5_count": 0, "osr_10_count": 0, "osr_20_count": 0}
                by_model_non_patrol[model]["total"] += 1
                if result["success"]:
                    by_model_non_patrol[model]["success"] += 1
                if mission_for_model.get("OSR_5") == 1:
                    by_model_non_patrol[model]["osr_5_count"] += 1
                if mission_for_model.get("OSR_10") == 1:
                    by_model_non_patrol[model]["osr_10_count"] += 1
                if mission_for_model.get("OSR_20") == 1:
                    by_model_non_patrol[model]["osr_20_count"] += 1
       
        # Compute model success rates and OSR (non-Patrol only for display)
        model_success_rates = {}
        for model, counts in by_model_non_patrol.items():
            rate = (counts["success"] / counts["total"] * 100) if counts["total"] > 0 else 0
            osr_5_rate_model = (counts["osr_5_count"] / counts["total"]) if counts["total"] > 0 else 0
            osr_10_rate_model = (counts["osr_10_count"] / counts["total"]) if counts["total"] > 0 else 0
            osr_20_rate_model = (counts["osr_20_count"] / counts["total"]) if counts["total"] > 0 else 0
            model_success_rates[model] = {
                "success": counts["success"],
                "total": counts["total"],
                "rate": rate,
                "osr_5_count": counts["osr_5_count"],
                "osr_5_rate": osr_5_rate_model,
                "osr_10_count": counts["osr_10_count"],
                "osr_10_rate": osr_10_rate_model,
                "osr_20_count": counts["osr_20_count"],
                "osr_20_rate": osr_20_rate_model,
            }
       
        # Group by mission scenario
        by_mission = {}
        by_category = {}
        by_mission_by_category = {}
        patrol_metric_keys = [
            "iou",
        ]
        for result in self.results:
            mission = result.get("mission_result", {}) or {}
            scenario = self._get_result_scenario_name(result)
            category = self._get_task_category(result)
            if scenario not in by_mission:
                by_mission[scenario] = {
                    "total": 0,
                    "osr_5_count": 0,
                    "osr_10_count": 0,
                    "osr_20_count": 0,
                    "sr5_count": 0,
                    "sr10_count": 0,
                    "min_distances": [],
                    "final_distances": [],
                }
            if category not in by_category:
                by_category[category] = {
                    "total": 0,
                    "success": 0,
                    "osr_5_count": 0,
                    "osr_10_count": 0,
                    "osr_20_count": 0,
                    "sr5_count": 0,
                    "sr10_count": 0,
                    "min_distances": [],
                    "final_distances": [],
                    "patrol_metrics": {k: [] for k in patrol_metric_keys},
                }
            if category not in by_mission_by_category:
                by_mission_by_category[category] = {}
            if scenario not in by_mission_by_category[category]:
                by_mission_by_category[category][scenario] = {
                    "total": 0,
                    "osr_5_count": 0,
                    "osr_10_count": 0,
                    "osr_20_count": 0,
                    "sr5_count": 0,
                    "sr10_count": 0,
                    "min_distances": [],
                    "final_distances": [],
                }
            by_mission[scenario]["total"] += 1
            by_category[category]["total"] += 1
            if result.get("success"):
                by_category[category]["success"] += 1
            by_mission_by_category[category][scenario]["total"] += 1
            if mission.get("OSR_5") == 1:
                by_mission[scenario]["osr_5_count"] += 1
                by_category[category]["osr_5_count"] += 1
                by_mission_by_category[category][scenario]["osr_5_count"] += 1
            if mission.get("OSR_10") == 1:
                by_mission[scenario]["osr_10_count"] += 1
                by_category[category]["osr_10_count"] += 1
                by_mission_by_category[category][scenario]["osr_10_count"] += 1
            if mission.get("OSR_20") == 1:
                by_mission[scenario]["osr_20_count"] += 1
                by_category[category]["osr_20_count"] += 1
                by_mission_by_category[category][scenario]["osr_20_count"] += 1
            if mission.get("sr5") == 1:
                by_mission[scenario]["sr5_count"] += 1
                by_category[category]["sr5_count"] += 1
                by_mission_by_category[category][scenario]["sr5_count"] += 1
            if mission.get("sr10") == 1:
                if "sr10_count" not in by_mission[scenario]:
                    by_mission[scenario]["sr10_count"] = 0
                by_mission[scenario]["sr10_count"] += 1
                by_category[category]["sr10_count"] += 1
                by_mission_by_category[category][scenario]["sr10_count"] += 1
            if mission.get("min_distance_to_target") is not None:
                by_mission[scenario]["min_distances"].append(mission.get("min_distance_to_target"))
                by_category[category]["min_distances"].append(mission.get("min_distance_to_target"))
                by_mission_by_category[category][scenario]["min_distances"].append(mission.get("min_distance_to_target"))
            if mission.get("final_distance_to_target") is not None:
                by_mission[scenario]["final_distances"].append(mission.get("final_distance_to_target"))
                by_category[category]["final_distances"].append(mission.get("final_distance_to_target"))
                by_mission_by_category[category][scenario]["final_distances"].append(mission.get("final_distance_to_target"))
            if category == "Patrol":
                patrol_results = mission.get("patrol_results", {}) or {}
                for key in patrol_metric_keys:
                    value = patrol_results.get(key)
                    if value is not None:
                        by_category[category]["patrol_metrics"][key].append(value)

        mission_breakdown = {}
        for scenario, data in by_mission.items():
            osr_5_rate_mission = (data["osr_5_count"] / data["total"]) if data["total"] > 0 else 0
            osr_10_rate_mission = (data["osr_10_count"] / data["total"]) if data["total"] > 0 else 0
            osr_20_rate_mission = (data["osr_20_count"] / data["total"]) if data["total"] > 0 else 0
            sr5_rate_mission = (data["sr5_count"] / data["total"]) if data["total"] > 0 else 0
            sr10_rate_mission = (data["sr10_count"] / data["total"]) if data["total"] > 0 else 0
            avg_min_distance = sum(data["min_distances"]) / len(data["min_distances"]) if data["min_distances"] else None
            avg_final_distance = sum(data["final_distances"]) / len(data["final_distances"]) if data["final_distances"] else None
            mission_breakdown[scenario] = {
                "total": data["total"],
                "osr_5_count": data["osr_5_count"],
                "osr_10_count": data["osr_10_count"],
                "osr_20_count": data["osr_20_count"],
                "sr5_count": data["sr5_count"],
                "sr10_count": data["sr10_count"],
                "osr_5_rate": osr_5_rate_mission,
                "osr_10_rate": osr_10_rate_mission,
                "osr_20_rate": osr_20_rate_mission,
                "sr5_rate": sr5_rate_mission,
                "sr10_rate": sr10_rate_mission,
                "avg_min_distance": avg_min_distance,
                "avg_final_distance": avg_final_distance,
            }

        category_breakdown = {}
        for category, data in by_category.items():
            osr_5_rate_cat = (data["osr_5_count"] / data["total"]) if data["total"] > 0 else 0
            osr_10_rate_cat = (data["osr_10_count"] / data["total"]) if data["total"] > 0 else 0
            osr_20_rate_cat = (data["osr_20_count"] / data["total"]) if data["total"] > 0 else 0
            sr5_rate_cat = (data["sr5_count"] / data["total"]) if data["total"] > 0 else 0
            sr10_rate_cat = (data["sr10_count"] / data["total"]) if data["total"] > 0 else 0
            avg_min_distance = sum(data["min_distances"]) / len(data["min_distances"]) if data["min_distances"] else None
            avg_final_distance = sum(data["final_distances"]) / len(data["final_distances"]) if data["final_distances"] else None
            patrol_metric_avgs = {
                key: self._safe_mean(values)
                for key, values in data["patrol_metrics"].items()
            } if data.get("patrol_metrics") else {}
            category_breakdown[category] = {
                "total": data["total"],
                "success": data["success"],
                "osr_5_rate": osr_5_rate_cat,
                "osr_10_rate": osr_10_rate_cat,
                "osr_20_rate": osr_20_rate_cat,
                "sr5_rate": sr5_rate_cat,
                "sr10_rate": sr10_rate_cat,
                "avg_min_distance": avg_min_distance,
                "avg_final_distance": avg_final_distance,
                "patrol_metric_avgs": patrol_metric_avgs,
            }

        mission_breakdown_by_category = {}
        for category, scenarios in by_mission_by_category.items():
            mission_breakdown_by_category[category] = {}
            for scenario, data in scenarios.items():
                osr_5_rate_mission = (data["osr_5_count"] / data["total"]) if data["total"] > 0 else 0
                osr_10_rate_mission = (data["osr_10_count"] / data["total"]) if data["total"] > 0 else 0
                osr_20_rate_mission = (data["osr_20_count"] / data["total"]) if data["total"] > 0 else 0
                sr5_rate_mission = (data["sr5_count"] / data["total"]) if data["total"] > 0 else 0
                sr10_rate_mission = (data["sr10_count"] / data["total"]) if data["total"] > 0 else 0
                avg_min_distance = sum(data["min_distances"]) / len(data["min_distances"]) if data["min_distances"] else None
                avg_final_distance = sum(data["final_distances"]) / len(data["final_distances"]) if data["final_distances"] else None
                mission_breakdown_by_category[category][scenario] = {
                    "total": data["total"],
                    "osr_5_count": data["osr_5_count"],
                    "osr_10_count": data["osr_10_count"],
                    "osr_20_count": data["osr_20_count"],
                    "sr5_count": data["sr5_count"],
                    "sr10_count": data["sr10_count"],
                    "osr_5_rate": osr_5_rate_mission,
                    "osr_10_rate": osr_10_rate_mission,
                    "osr_20_rate": osr_20_rate_mission,
                    "sr5_rate": sr5_rate_mission,
                    "sr10_rate": sr10_rate_mission,
                    "avg_min_distance": avg_min_distance,
                    "avg_final_distance": avg_final_distance,
                }
       
        # Calculate standard deviations from mission-level means
        # Rebuild mission grouping specifically for std dev calculation
        mission_means = {}
        for result in self.results:
            mission = result.get("mission_result", {}) or {}
            scenario = self._get_result_scenario_name(result)
           
            if scenario not in mission_means:
                mission_means[scenario] = {
                    "success_values": [],
                    "sr5_values": [],
                    "sr10_values": [],
                    "osr_5_values": [],
                    "osr_10_values": [],
                    "osr_20_values": [],
                }
           
            mission_means[scenario]["success_values"].append(1 if result.get("success") else 0)
            if mission.get("sr5") is not None:
                mission_means[scenario]["sr5_values"].append(mission.get("sr5"))
            if mission.get("sr10") is not None:
                mission_means[scenario]["sr10_values"].append(mission.get("sr10"))
            if mission.get("OSR_5") is not None:
                mission_means[scenario]["osr_5_values"].append(mission.get("OSR_5"))
            if mission.get("OSR_10") is not None:
                mission_means[scenario]["osr_10_values"].append(mission.get("OSR_10"))
            if mission.get("OSR_20") is not None:
                mission_means[scenario]["osr_20_values"].append(mission.get("OSR_20"))
       
        # Compute mission-level means for each metric
        mission_success_rates = []
        mission_sr5_rates = []
        mission_sr10_rates = []
        mission_osr5_rates = []
        mission_osr10_rates = []
        mission_osr20_rates = []
       
        for scenario, data in mission_means.items():
            if data["success_values"]:
                mission_success_rates.append(statistics.mean(data["success_values"]) * 100)
            if data["sr5_values"]:
                mission_sr5_rates.append(statistics.mean(data["sr5_values"]) * 100)
            if data["sr10_values"]:
                mission_sr10_rates.append(statistics.mean(data["sr10_values"]) * 100)
            if data["osr_5_values"]:
                mission_osr5_rates.append(statistics.mean(data["osr_5_values"]) * 100)
            if data["osr_10_values"]:
                mission_osr10_rates.append(statistics.mean(data["osr_10_values"]) * 100)
            if data["osr_20_values"]:
                mission_osr20_rates.append(statistics.mean(data["osr_20_values"]) * 100)
       
        # Calculate std devs from mission means
        success_std = statistics.stdev(mission_success_rates) if len(mission_success_rates) > 1 else 0
        sr5_std = statistics.stdev(mission_sr5_rates) if len(mission_sr5_rates) > 1 else 0
        sr10_std = statistics.stdev(mission_sr10_rates) if len(mission_sr10_rates) > 1 else 0
        osr_5_std = statistics.stdev(mission_osr5_rates) if len(mission_osr5_rates) > 1 else 0
        osr_10_std = statistics.stdev(mission_osr10_rates) if len(mission_osr10_rates) > 1 else 0
        osr_20_std = statistics.stdev(mission_osr20_rates) if len(mission_osr20_rates) > 1 else 0
       
        # Calculate timing stats
        time_values = [r.get("total_time", 0) for r in self.results if r.get("total_time")]
        avg_time = statistics.mean(time_values) if time_values else 0
        std_time = statistics.stdev(time_values) if len(time_values) > 1 else 0
        min_time = min(time_values) if time_values else 0
        max_time = max(time_values) if time_values else 0
       
        self.stats = {
            "total_experiments": total_count,
            "total_successful": success_count,
            "total_failed": total_count - success_count,
            "success_rate": success_rate,
            "success_std": success_std,
            "success_rate_by_model": model_success_rates,
            "osr_5_rate": osr_5_rate * 100,
            "osr_5_std": osr_5_std,
            "osr_10_rate": osr_10_rate * 100,
            "osr_10_std": osr_10_std,
            "osr_20_rate": osr_20_rate * 100,
            "osr_20_std": osr_20_std,
            "sr5_rate": sr5_rate * 100,
            "sr5_std": sr5_std,
            "sr10_rate": sr10_rate * 100,
            "sr10_std": sr10_std,
            "collision_rate": avg_collision_rate,
            "mission_progress": avg_mission_progress,
            "avg_time": avg_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "mission_breakdown": mission_breakdown,
            "category_breakdown": category_breakdown,
            "mission_breakdown_by_category": mission_breakdown_by_category,
        }
   
    def generate_csv_summary(self, output_file: Path = None) -> Path:
        """
        Generate comprehensive CSV summary of all results with metadata and statistics.
       
        Args:
            output_file: Output CSV file path
       
        Returns:
            Path to generated CSV file
        """
        if output_file is None:
            output_file = self.results_base_dir / "results_summary.csv"
       
        output_file.parent.mkdir(parents=True, exist_ok=True)
       
        print(f"\nGenerating comprehensive CSV summary: {output_file}")
       
        # Extract git info if not already done
        if not self.git_commits:
            self.extract_git_info()
       
        with open(output_file, "w", newline="") as f:
            if not self.results:
                print("  No results to write")
                return output_file
           
            # Get all config keys from first result
            config_keys = list(self.results[0]["config"].keys())
           
            # Comprehensive fieldnames including metadata
            fieldnames = [
                # Metadata
                "experiment_id",
                "timestamp",
                "git_missionbench",
            ] + config_keys + [
                # Results
                "success",
                "total_time",
                "error_message",
                # Metrics
                "OSR_5",
                "OSR_10",
                "OSR_20",
                "sr5",
                "sr10",
                "min_distance_to_target",
                "final_distance_to_target",
                "x_diff",
                "y_diff",
                "z_diff",
                "yaw_diff",
                "collision_count",
                "collision_rate",
                "step_usage",
                "mission_progress",
                "stepwise_binary_progress",
                "task_gt",
                "task_gt_predicted",
                "task_success_model_eval",
                "stop_reason",
                "stop_condition",
                "patrol_iou",
            ]
           
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            sorted_results = sorted(
                self.results,
                key=lambda result: str(
                    (result.get("mission_result") or {}).get(
                        "scenario_name",
                        result.get("config", {}).get("scenario_name", "")
                    )
                ).lower(),
            )
           
            for idx, result in enumerate(sorted_results, 1):
                row = {}
               
                # Add metadata
                row["experiment_id"] = f"exp_{idx:06d}"
                row["timestamp"] = result.get("timestamp", "")
               
                # Handle new git info structure (dict with commit, branch, message)
                def format_git_info(git_data):
                    if isinstance(git_data, dict):
                        return f"{git_data.get('commit', 'N/A')} ({git_data.get('branch', 'N/A')})"
                    return str(git_data) if git_data else "N/A"
               
                row["git_missionbench"] = format_git_info(self.git_commits.get("MissionBench"))
               
                # Copy all config keys, handling missing values
                for key in config_keys:
                    row[key] = result["config"].get(key, "")
                   
                # Add results
                row["success"] = result["success"]
                row["total_time"] = result["total_time"] / 60 if result["total_time"] else None
                row["error_message"] = result["error_message"] or ""
               
                # Add mission metrics
                mission = result.get("mission_result", {}) or {}
                row["OSR_5"] = mission.get("OSR_5")
                row["OSR_10"] = mission.get("OSR_10")
                row["OSR_20"] = mission.get("OSR_20")
                row["sr5"] = mission.get("sr5")
                row["sr10"] = mission.get("sr10")
                row["min_distance_to_target"] = mission.get("min_distance_to_target")
                row["final_distance_to_target"] = mission.get("final_distance_to_target")
                row["x_diff"] = mission.get("x_diff")
                row["y_diff"] = mission.get("y_diff")
                row["z_diff"] = mission.get("z_diff")
                row["yaw_diff"] = mission.get("yaw_diff")
                row["collision_count"] = mission.get("collision_count")
                row["collision_rate"] = mission.get("collision_rate")
                row["step_usage"] = self._compute_step_usage_percent(
                    mission.get("steps_executed", mission.get("steps_taken")),
                    mission.get("steps_budget", result.get("config", {}).get("max_actions")),
                )
                row["mission_progress"] = mission.get("mission_progress")
                row["stepwise_binary_progress"] = mission.get("stepwise_binary_progress")
                task_gt_fields = self._resolve_task_gt_fields(result)
                row["task_gt"] = task_gt_fields.get("task_gt_expected")
                row["task_gt_predicted"] = task_gt_fields.get("task_gt_pred")

                task_eval_response = task_gt_fields.get("task_gt_response")
                task_eval_match = task_gt_fields.get("task_gt_match")
                if isinstance(task_eval_response, str) and task_eval_response.strip():
                    response_text = task_eval_response.strip()
                    if response_text.lower().startswith("error"):
                        row["task_success_model_eval"] = response_text
                    elif response_text.lower() == "not_evaluated":
                        row["task_success_model_eval"] = "not_evaluated"
                    else:
                        row["task_success_model_eval"] = "success" if int(task_eval_match or 0) == 1 else "fail"
                else:
                    if task_eval_match == 1:
                        row["task_success_model_eval"] = "success"
                    elif task_eval_match == 0:
                        row["task_success_model_eval"] = "fail"
                    else:
                        row["task_success_model_eval"] = "not_evaluated"

                row["stop_reason"] = mission.get("stop_reason", "")
                row["stop_condition"] = mission.get("stop_condition", mission.get("stop_reason", ""))
                patrol_results = mission.get("patrol_results", {})
                row["patrol_iou"] = patrol_results.get("iou") if isinstance(patrol_results, dict) else None
               
                writer.writerow(row)
       
        print(f"  ✓ Comprehensive CSV written: {output_file}")
        # Also generate a summary statistics CSV
        self._generate_summary_stats_csv(output_file.parent / "statistics_summary.csv")
       
        return output_file
   
    def _generate_summary_stats_csv(self, output_file: Path) -> Path:
        """
        Generate a separate CSV with summary statistics (overall and mission-wise).
       
        Args:
            output_file: Output CSV file path
           
        Returns:
            Path to generated CSV file
        """
        print(f"\nGenerating statistics summary CSV: {output_file}")
       
        # First, compute mission-wise statistics to get per-mission rates
        by_mission = {}
        for result in self.results:
            mission = result.get("mission_result", {}) or {}
            scenario = self._get_result_scenario_name(result)
           
            if scenario not in by_mission:
                by_mission[scenario] = {
                    "success_values": [],
                    "time_values": [],
                    "sr5_values": [],
                    "sr10_values": [],
                    "osr_5_values": [],
                    "osr_10_values": [],
                    "collision_rate_values": [],
                    "mission_progress_values": [],
                    "stepwise_binary_progress_values": [],
                    "step_usage_values": [],
                }
           
            by_mission[scenario]["success_values"].append(1 if result.get("success") else 0)
            if result.get("total_time"):
                by_mission[scenario]["time_values"].append(result["total_time"])
            if mission.get("sr5") is not None:
                by_mission[scenario]["sr5_values"].append(mission["sr5"])
            if mission.get("sr10") is not None:
                by_mission[scenario]["sr10_values"].append(mission["sr10"])
            if mission.get("OSR_5") is not None:
                by_mission[scenario]["osr_5_values"].append(mission["OSR_5"])
            if mission.get("OSR_10") is not None:
                by_mission[scenario]["osr_10_values"].append(mission["OSR_10"])
            if mission.get("collision_rate") is not None:
                by_mission[scenario]["collision_rate_values"].append(
                    mission["collision_rate"]
                )
            if mission.get("mission_progress") is not None:
                by_mission[scenario]["mission_progress_values"].append(
                    mission["mission_progress"]
                )
            step_usage = self._compute_step_usage_percent(
                mission.get("steps_executed", mission.get("steps_taken")),
                mission.get("steps_budget", result.get("config", {}).get("max_actions")),
            )
            if step_usage is not None:
                by_mission[scenario]["step_usage_values"].append(step_usage)
            if mission.get("stepwise_binary_progress") is not None:
                by_mission[scenario]["stepwise_binary_progress_values"].append(
                    mission["stepwise_binary_progress"]
                )
       
        # Calculate per-mission rates for overall std dev calculation
        mission_success_rates = []
        mission_sr5_rates = []
        mission_sr10_rates = []
        mission_osr5_rates = []
        mission_osr10_rates = []
        mission_collision_rates = []
        mission_progress_means = []
        mission_step_usage_means = []
       
        for scenario, data in by_mission.items():
            if data["success_values"]:
                mission_success_rates.append(statistics.mean(data["success_values"]) * 100)
            if data["sr5_values"]:
                mission_sr5_rates.append(statistics.mean(data["sr5_values"]) * 100)
            if data["sr10_values"]:
                mission_sr10_rates.append(statistics.mean(data["sr10_values"]) * 100)
            if data["osr_5_values"]:
                mission_osr5_rates.append(statistics.mean(data["osr_5_values"]) * 100)
            if data["osr_10_values"]:
                mission_osr10_rates.append(statistics.mean(data["osr_10_values"]) * 100)
            if data["collision_rate_values"]:
                mission_collision_rates.append(
                    statistics.mean(data["collision_rate_values"])
                )
            if data["mission_progress_values"]:
                mission_progress_means.append(
                    statistics.mean(data["mission_progress_values"])
                )
            if data["step_usage_values"]:
                mission_step_usage_means.append(
                    statistics.mean(data["step_usage_values"])
                )
       
        # Compute overall statistics (pooled for rates, mission-to-mission for std dev)
        success_values = [1 if r.get("success") else 0 for r in self.results]
        time_values = [r.get("total_time", 0) for r in self.results if r.get("total_time")]
        sr5_values = [
            (r.get("mission_result") or {}).get("sr5", 0)
            for r in self.results
            if (r.get("mission_result") or {}).get("sr5") is not None
        ]
        sr10_values = [
            (r.get("mission_result") or {}).get("sr10", 0)
            for r in self.results
            if (r.get("mission_result") or {}).get("sr10") is not None
        ]
        osr_5_values = [
            (r.get("mission_result") or {}).get("OSR_5", 0)
            for r in self.results
            if (r.get("mission_result") or {}).get("OSR_5") is not None
        ]
        osr_10_values = [
            (r.get("mission_result") or {}).get("OSR_10", 0)
            for r in self.results
            if (r.get("mission_result") or {}).get("OSR_10") is not None
        ]
        collision_rate_values = [
            (r.get("mission_result") or {}).get("collision_rate", 0.0)
            for r in self.results
            if (r.get("mission_result") or {}).get("collision_rate") is not None
        ]
        mission_progress_values = [
            (r.get("mission_result") or {}).get("mission_progress")
            for r in self.results
            if (r.get("mission_result") or {}).get("mission_progress") is not None
        ]
        step_usage_values = []
        for r in self.results:
            mission = r.get("mission_result") or {}
            usage = self._compute_step_usage_percent(
                mission.get("steps_executed", mission.get("steps_taken")),
                mission.get("steps_budget", r.get("config", {}).get("max_actions")),
            )
            if usage is not None:
                step_usage_values.append(usage)

        avg_step_usage = statistics.mean(step_usage_values) if step_usage_values else None
       
        overall_stats = {
            "metric_type": "overall",
            "mission_name": "ALL_MISSIONS",
            "total_trials": len(self.results),
            "successful_trials": sum(success_values),
            "success_rate": statistics.mean(success_values) * 100 if success_values else 0,
            "success_std": statistics.stdev(mission_success_rates) if len(mission_success_rates) > 1 else 0,
            "sr5_rate": statistics.mean(sr5_values) * 100 if sr5_values else None,
            "sr5_std": statistics.stdev(mission_sr5_rates) if len(mission_sr5_rates) > 1 else 0,
            "sr10_rate": statistics.mean(sr10_values) * 100 if sr10_values else None,
            "sr10_std": statistics.stdev(mission_sr10_rates) if len(mission_sr10_rates) > 1 else 0,
            "osr_5_rate": statistics.mean(osr_5_values) * 100 if osr_5_values else None,
            "osr_5_std": statistics.stdev(mission_osr5_rates) if len(mission_osr5_rates) > 1 else 0,
            "osr_10_rate": statistics.mean(osr_10_values) * 100 if osr_10_values else None,
            "osr_10_std": statistics.stdev(mission_osr10_rates) if len(mission_osr10_rates) > 1 else 0,
            "collision_rate": statistics.mean(collision_rate_values) if collision_rate_values else None,
            "collision_rate_std": statistics.stdev(mission_collision_rates) if len(mission_collision_rates) > 1 else 0,
            "mission_progress": (statistics.mean(mission_progress_values) * 100) if mission_progress_values else None,
            "mission_progress_std": (statistics.stdev(mission_progress_means) * 100) if len(mission_progress_means) > 1 else 0,
            "avg_step_usage": avg_step_usage,
            "step_efficiency": (100 - avg_step_usage) if avg_step_usage is not None else None,
            "avg_time": (statistics.mean(time_values) / 60) if time_values else None,
            "std_time": (statistics.stdev(time_values) / 60) if len(time_values) > 1 else 0,
        }
       
        # Write statistics CSV
        with open(output_file, "w", newline="") as f:
            fieldnames = [
                "metric_type", "mission_name", "total_trials", "successful_trials",
                "success_rate", "success_std",
                "sr5_rate", "sr5_std", "sr10_rate", "sr10_std",
                "osr_5_rate", "osr_5_std", "osr_10_rate", "osr_10_std",
                "collision_rate", "collision_rate_std",
                "mission_progress", "mission_progress_std",
                "avg_step_usage", "step_efficiency",
                "avg_time", "std_time"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
           
            # Write overall stats
            writer.writerow(overall_stats)
           
            # Write mission-wise stats
            for scenario, data in sorted(by_mission.items()):
                success_vals = data["success_values"]
                times = data["time_values"]
                sr5 = data["sr5_values"]
                sr10 = data["sr10_values"]
                osr_5 = data["osr_5_values"]
                osr_10 = data["osr_10_values"]
                collision_rate = data["collision_rate_values"]
                mission_progress = data["mission_progress_values"]
                step_usage = data["step_usage_values"]
                avg_step_usage = statistics.mean(step_usage) if step_usage else None
               
                mission_stats = {
                    "metric_type": "mission",
                    "mission_name": scenario,
                    "total_trials": len(success_vals),
                    "successful_trials": sum(success_vals),
                    "success_rate": statistics.mean(success_vals) * 100 if success_vals else 0,
                    "success_std": statistics.stdev(success_vals) * 100 if len(success_vals) > 1 else 0,
                    "sr5_rate": statistics.mean(sr5) * 100 if sr5 else None,
                    "sr5_std": statistics.stdev(sr5) * 100 if len(sr5) > 1 else 0,
                    "sr10_rate": statistics.mean(sr10) * 100 if sr10 else None,
                    "sr10_std": statistics.stdev(sr10) * 100 if len(sr10) > 1 else 0,
                    "osr_5_rate": statistics.mean(osr_5) * 100 if osr_5 else None,
                    "osr_5_std": statistics.stdev(osr_5) * 100 if len(osr_5) > 1 else 0,
                    "osr_10_rate": statistics.mean(osr_10) * 100 if osr_10 else None,
                    "osr_10_std": statistics.stdev(osr_10) * 100 if len(osr_10) > 1 else 0,
                    "collision_rate": statistics.mean(collision_rate) if collision_rate else None,
                    "collision_rate_std": statistics.stdev(collision_rate) if len(collision_rate) > 1 else 0,
                    "mission_progress": (statistics.mean(mission_progress) * 100) if mission_progress else None,
                    "mission_progress_std": (statistics.stdev(mission_progress) * 100) if len(mission_progress) > 1 else 0,
                    "avg_step_usage": avg_step_usage,
                    "step_efficiency": (100 - avg_step_usage) if avg_step_usage is not None else None,
                    "avg_time": (statistics.mean(times) / 60) if times else None,
                    "std_time": (statistics.stdev(times) / 60) if len(times) > 1 else 0,
                }
                writer.writerow(mission_stats)
       
        print(f"  ✓ Statistics summary written: {output_file}")
        return output_file
   
    def generate_html_dashboard(self, output_file: Path = None) -> Path:
        """
        Generate HTML dashboard with visualizations.
       
        Args:
            output_file: Output HTML file path
       
        Returns:
            Path to generated HTML file
        """
        if output_file is None:
            output_file = self.results_base_dir / "dashboard.html"
       
        output_file.parent.mkdir(parents=True, exist_ok=True)
       
        print(f"\nGenerating HTML dashboard: {output_file}")
       
        # Generate HTML
        html_content = self._generate_html_content()
       
        with open(output_file, "w") as f:
            f.write(html_content)
       
        print(f"  ✓ Dashboard written: {output_file}")
        return output_file
   
    def _generate_html_content(self) -> str:
        """Generate HTML dashboard content."""
        total = self.stats.get("total_experiments", 0)
        successful = self.stats.get("successful_experiments", 0)
        failed = self.stats.get("failed_experiments", 0)
        success_rate = self.stats.get("overall_success_rate", 0)
        osr_5_rate = self.stats.get("OSR_5", 0)
        osr_10_rate = self.stats.get("OSR_10", 0)
        osr_20_rate = self.stats.get("OSR_20", 0)
        sr5_rate = self.stats.get("sr5_rate", 0)
        sr10_rate = self.stats.get("sr10_rate", 0)
       
        # Build model success rate rows (with OSR)
        def format_rate(value: float) -> str:
            formatted = f"{value:.2f}"
            if abs(value - 1.0) < 0.001:  # Check if value is 1.0
                return f'<span class="perfect-score">{formatted}</span>'
            return formatted
       
        model_rows = ""
        for model, data in self.stats.get("success_rate_by_model", {}).items():
            short_name = self._get_short_model_name(model)
            model_rows += f"""
            <tr>
                <td>{short_name}</td>
                <td>{data['success']}/{data['total']}</td>
                <td>{data['rate']:.1f}%</td>
                <td>{format_rate(data['osr_5_rate'])}</td>
                <td>{format_rate(data['osr_10_rate'])}</td>
                <td>{format_rate(data['osr_20_rate'])}</td>
            </tr>
            """

        def build_mission_rows(mission_breakdown: Dict[str, Any]) -> str:
            def format_rate(value: float) -> str:
                formatted = f"{value:.2f}"
                if abs(value - 1.0) < 0.001:  # Check if value is 1.0
                    return f'<span class="perfect-score">{formatted}</span>'
                return formatted
           
            rows = ""
            for idx, (mission_name, data) in enumerate(mission_breakdown.items(), 1):
                avg_min = f"{data.get('avg_min_distance', 0):.2f}" if data.get('avg_min_distance') is not None else "N/A"
                avg_final = f"{data.get('avg_final_distance', 0):.2f}" if data.get('avg_final_distance') is not None else "N/A"
                rows += f"""
                <tr>
                    <td>{idx}</td>
                    <td><strong>{mission_name}</strong></td>
                    <td>{format_rate(data['sr5_rate'])}</td>
                    <td>{format_rate(data['sr10_rate'])}</td>
                    <td>{format_rate(data['osr_5_rate'])}</td>
                    <td>{format_rate(data['osr_10_rate'])}</td>
                    <td>{format_rate(data['osr_20_rate'])}</td>
                    <td>{avg_min}m</td>
                    <td>{avg_final}m</td>
                </tr>
                """
            return rows

        def build_patrol_rows(results: List[Dict[str, Any]]) -> str:
            def format_metric(value: Any) -> str:
                if value is None or value == "N/A":
                    return "N/A"
                if isinstance(value, (int, float)):
                    return f"{value:.2f}"
                return str(value)

            rows = ""
            for idx, result in enumerate(results, 1):
                config = result["config"]
                mission = result.get("mission_result", {}) or {}
                patrol_results = mission.get("patrol_results", {}) or {}
                scenario = self._get_result_scenario_name(result)
                model_full = config.get("model_name", "unknown")
                model = self._get_short_model_name(model_full)
                rows += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{model}</td>
                    <td>{scenario}</td>
                    <td>{format_metric(patrol_results.get('gt_trajectory_length'))}</td>
                    <td>{format_metric(patrol_results.get('vlm_trajectory_length'))}</td>
                    <td>{format_metric(patrol_results.get('nn_gt_to_vlm'))}</td>
                    <td>{format_metric(patrol_results.get('nn_vlm_to_gt'))}</td>
                    <td>{format_metric(patrol_results.get('mean_trajectory_distance'))}</td>
                    <td>{format_metric(patrol_results.get('gt_path_length'))}</td>
                    <td>{format_metric(patrol_results.get('vlm_path_length'))}</td>
                </tr>
                """
            return rows

        category_order = ["Visual Inspection", "Patrol", "Manipulation"]
        results_by_category: Dict[str, List[Dict[str, Any]]] = {category: [] for category in category_order}
        for result in self.results:
            category = self._get_task_category(result)
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(result)

        for category in results_by_category.keys():
            if category not in category_order:
                category_order.append(category)

        category_sections = ""
        mission_breakdown_by_category = self.stats.get("mission_breakdown_by_category", {})
        category_breakdown = self.stats.get("category_breakdown", {})

        for category in category_order:
            category_results = results_by_category.get(category, [])
            if not category_results:
                continue
            category_stats = category_breakdown.get(category, {})
            mission_rows = build_mission_rows(mission_breakdown_by_category.get(category, {}))

            avg_min = (
                f"{category_stats.get('avg_min_distance', 0):.2f}"
                if category_stats.get('avg_min_distance') is not None else "N/A"
            )
            avg_final = (
                f"{category_stats.get('avg_final_distance', 0):.2f}"
                if category_stats.get('avg_final_distance') is not None else "N/A"
            )

            patrol_metrics_section = ""
            if category == "Patrol":
                patrol_rows = build_patrol_rows(category_results)
                patrol_avgs = category_stats.get("patrol_metric_avgs", {})
                patrol_metrics_section = f"""
                <h3>🛰️ Patrol Metrics</h3>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Model</th>
                        <th>Mission</th>
                        <th>GT Traj Len</th>
                        <th>VLM Traj Len</th>
                        <th>NN GT → VLM</th>
                        <th>NN VLM → GT</th>
                        <th>Mean Traj Dist</th>
                        <th>GT Path Len</th>
                        <th>VLM Path Len</th>
                        <th>Path Len Similarity</th>
                    </tr>
                    {patrol_rows}
                    <tr>
                        <td colspan="3"><strong>Average</strong></td>
                        <td>{patrol_avgs.get('gt_trajectory_length', 0):.2f}</td>
                        <td>{patrol_avgs.get('vlm_trajectory_length', 0):.2f}</td>
                        <td>{patrol_avgs.get('nn_gt_to_vlm', 0):.2f}</td>
                        <td>{patrol_avgs.get('nn_vlm_to_gt', 0):.2f}</td>
                        <td>{patrol_avgs.get('mean_trajectory_distance', 0):.2f}</td>
                        <td>{patrol_avgs.get('gt_path_length', 0):.2f}</td>
                        <td>{patrol_avgs.get('vlm_path_length', 0):.2f}</td>
                    </tr>
                </table>
                """

            if category == "Patrol":
                category_sections += f"""
                <h2>📂 {category} Missions</h2>
                {patrol_metrics_section}
                """
            else:
                model_section = ""
                if category == "Visual Inspection":
                    model_section = f"""
                    <h3>📊 Performance by Model</h3>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Success Rate</th>
                            <th>Success / Total</th>
                            <th>OSR_5 Rate</th>
                            <th>OSR_10 Rate</th>
                            <th>OSR_20 Rate</th>
                        </tr>
                        {model_rows}
                    </table>
                    """
                category_sections += f"""
                <h2>📂 {category} Missions</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{category_stats.get('total', 0)}</div>
                        <div class="stat-label">Total Experiments</div>
                    </div>
                    <div class="stat-card success">
                        <div class="stat-number">{category_stats.get('success', 0)}/{category_stats.get('total', 0)}</div>
                        <div class="stat-label">Execution Success</div>
                    </div>
                    <div class="stat-card osr">
                        <div class="stat-number">{category_stats.get('osr_5_rate', 0):.2f}</div>
                        <div class="stat-label">OSR_5 (within 5m)</div>
                    </div>
                    <div class="stat-card osr">
                        <div class="stat-number">{category_stats.get('osr_10_rate', 0):.2f}</div>
                        <div class="stat-label">OSR_10 (within 10m)</div>
                    </div>
                    <div class="stat-card osr">
                        <div class="stat-number">{category_stats.get('osr_20_rate', 0):.2f}</div>
                        <div class="stat-label">OSR_20 (within 20m)</div>
                    </div>
                    <div class="stat-card sr">
                        <div class="stat-number">{category_stats.get('sr5_rate', 0):.2f}</div>
                        <div class="stat-label">SR5 (5m/15°)</div>
                    </div>
                    <div class="stat-card sr">
                        <div class="stat-number">{category_stats.get('sr10_rate', 0):.2f}</div>
                        <div class="stat-label">SR10 (10m/45°)</div>
                    </div>
                </div>

                <h3>📍 Performance by Mission ({category})</h3>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Mission Name</th>
                        <th>SR5 Rate</th>
                        <th>SR10 Rate</th>
                        <th>OSR_5 Rate</th>
                        <th>OSR_10 Rate</th>
                        <th>OSR_20 Rate</th>
                        <th>Avg Min Distance (m)</th>
                        <th>Avg Final Distance (m)</th>
                    </tr>
                    {mission_rows}
                </table>

                <p style="font-size: 12px; color: #666;">
                    <strong>Category Averages:</strong> Min Distance {avg_min}m | Final Distance {avg_final}m
                </p>

                {patrol_metrics_section}

                {model_section}

                """
       
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Experiment Results Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-left: 4px solid #007bff;
            padding-left: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .stat-card.failed {{
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        }}
        .stat-card.osr {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .stat-card.sr {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        .stat-number {{
            font-size: 32px;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 14px;
            opacity: 0.9;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        table th {{
            background-color: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
            font-size: 13px;
        }}
        table td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        table tr:hover {{
            background-color: #f8f9fa;
        }}
        .perfect-score {{
            color: #28a745;
            font-weight: bold;
        }}
        .generated {{
            color: #666;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚁 Experiment Results Dashboard</h1>
       

       
        {category_sections}
       
        <div class="generated">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
        """
       
        return html


    def generate_latex_report(self, output_file: Path = None, title: str = "Experiment Results") -> Path:
        """
        Generate LaTeX report with comprehensive statistics.
       
        Args:
            output_file: Output .tex file path
            title: Report title
           
        Returns:
            Path to generated .tex file
        """
        try:
            from .latex_reporter import LaTeXReporter
        except ImportError:
            from latex_reporter import LaTeXReporter
       
        reporter = LaTeXReporter(self.results, self.results_base_dir)
        reporter.git_commits = self.git_commits  # Reuse git info
        reporter.overall_stats = self.stats  # Pass computed stats to avoid recalculation
        return reporter.generate_latex(output_file, title)
   
    def compile_pdf_report(self, tex_file: Path) -> Optional[Path]:
        """
        Compile LaTeX file to PDF.
       
        Args:
            tex_file: Path to .tex file
           
        Returns:
            Path to generated PDF or None if compilation failed
        """
        try:
            from .latex_reporter import LaTeXReporter
        except ImportError:
            from latex_reporter import LaTeXReporter
       
        reporter = LaTeXReporter(self.results, self.results_base_dir)
        return reporter.compile_pdf(tex_file)
   
    def send_report_email(
        self,
        pdf_file: Path,
        recipient: str,
        subject: str = None,
        smtp_config: Dict[str, Any] = None
    ) -> bool:
        """
        Send PDF report via email.
       
        Args:
            pdf_file: Path to PDF file
            recipient: Recipient email address
            subject: Email subject
            smtp_config: SMTP configuration
           
        Returns:
            True if sent successfully
        """
        try:
            from .latex_reporter import LaTeXReporter
        except ImportError:
            from latex_reporter import LaTeXReporter
       
        reporter = LaTeXReporter(self.results, self.results_base_dir)
        reporter.overall_stats = self.stats  # Reuse computed stats
        return reporter.send_email(pdf_file, recipient, subject, smtp_config)


def main():
    """
    Standalone script to regenerate results dashboard, CSV, and LaTeX/PDF reports.
   
    Usage:
        python aggregator.py [results_directory] [--latex] [--pdf] [--email recipient@example.com]
   
    If no directory is provided, uses ./experiment_results
    """
    import sys
    import os
    import argparse
   
    # Add current directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
   
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate experiment reports')
    parser.add_argument('results_dir', nargs='?', default=None,
                       help='Results directory (default: experiment_results)')
    parser.add_argument('--results_dir', dest='results_dir_flag', type=str, default=None,
                       help='Results directory (overrides positional results_dir if provided)')
    parser.add_argument('--latex', action='store_true',
                       help='Generate LaTeX report')
    parser.add_argument('--pdf', action='store_true',
                       help='Generate PDF report (requires pdflatex)')
    parser.add_argument('--email', type=str, metavar='RECIPIENT',
                       help='Send PDF report to email address')
    parser.add_argument('--title', type=str, default='Experiment Results',
                       help='Report title (default: Experiment Results)')
   
    args = parser.parse_args()
    results_dir_value = args.results_dir_flag or args.results_dir or 'experiment_results'
    results_dir = Path(results_dir_value)
   
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
   
    print(f"📂 Loading results from: {results_dir}")
   
    # Initialize and run aggregator
    aggregator = ResultsAggregator(results_dir)
    aggregator.load_results()
   
    if not aggregator.results:
        print("❌ No results found to analyze")
        sys.exit(1)
   
    # Extract git information
    aggregator.extract_git_info()
   
    # Compute statistics
    aggregator.compute_statistics()
   
    # Generate CSV summary
    csv_file = aggregator.generate_csv_summary()
   
    # Generate HTML dashboard
    dashboard_file = aggregator.generate_html_dashboard()
   
    print("\n" + "="*70)
    print("✓ Base reports generated successfully!")
    print("="*70)
    print(f"  CSV (detailed): {csv_file}")
    print(f"  CSV (statistics): {csv_file.parent / 'statistics_summary.csv'}")
    print(f"  Dashboard: {dashboard_file}")
   
    # Generate LaTeX report if requested
    tex_file = None
    pdf_file = None
   
    if args.latex or args.pdf or args.email:
        print("\n" + "="*70)
        print("📝 Generating LaTeX report...")
        print("="*70)
        tex_file = aggregator.generate_latex_report(title=args.title)
        print(f"  LaTeX: {tex_file}")
       
        # Compile PDF if requested
        if args.pdf or args.email:
            print("\n" + "="*70)
            print("📄 Compiling PDF...")
            print("="*70)
            pdf_file = aggregator.compile_pdf_report(tex_file)
           
            if pdf_file:
                print(f"  PDF: {pdf_file}")
            else:
                print("  ⚠ PDF compilation failed (LaTeX file is still available)")
                if args.email:
                    print("  ⚠ Cannot send email without PDF")
                    args.email = None
       
        # Send email if requested and PDF was generated
        if args.email and pdf_file:
            print("\n" + "="*70)
            print("📧 Sending email...")
            print("="*70)
           
            # Import email functionality from utils
            try:
                import sys
                import os
                # Direct import of email_utils to avoid importing whole utils package
                utils_path = Path(__file__).parent.parent / "utils"
                sys.path.insert(0, str(utils_path))
               
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                from email.mime.base import MIMEBase
                from email import encoders
               
                # Load email config from environment
                gmail_user = os.getenv('GMAIL_USER')
                gmail_password = os.getenv('GMAIL_PASSWORD')
               
                if not gmail_user or not gmail_password:
                    print("  ⚠ Email credentials not configured")
                    print("  Set GMAIL_USER and GMAIL_PASSWORD environment variables")
                    print("  For Gmail: Use an App Password (https://myaccount.google.com/apppasswords)")
                else:
                    # Create email body
                    html_body = f"""
                    <html>
                    <head>
                        <style>
                            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                            h1 {{ color: #2c3e50; }}
                            .summary {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                            .metric {{ margin: 10px 0; }}
                            .label {{ font-weight: bold; color: #34495e; }}
                            .value {{ color: #16a085; }}
                        </style>
                    </head>
                    <body>
                        <h1>🚁 {args.title}</h1>
                        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                       
                        <div class="summary">
                            <h2>📊 Summary</h2>
                            <div class="metric">
                                <span class="label">Total Experiments:</span>
                                <span class="value">{aggregator.stats.get('total_experiments', 0)}</span>
                            </div>
                            <div class="metric">
                                <span class="label">Success Rate:</span>
                                <span class="value">{aggregator.stats.get('overall_success_rate', 0):.2f}%</span>
                            </div>
                            <div class="metric">
                                <span class="label">OSR@10m:</span>
                                <span class="value">{aggregator.stats.get('OSR_10', 0):.2f}%</span>
                            </div>
                        </div>
                       
                        <h2>📎 Attachments</h2>
                        <ul>
                            <li><strong>PDF Report:</strong> Comprehensive experiment report with statistics and visualizations</li>
                            <li><strong>Detailed CSV:</strong> All experiment results with individual trial data</li>
                            <li><strong>Statistics CSV:</strong> Summary statistics for overall and mission-wise performance</li>
                            <li><strong>HTML Dashboard:</strong> Interactive web dashboard for results exploration</li>
                        </ul>
                       
                        <p style="color: #7f8c8d; margin-top: 30px; font-size: 0.9em;">
                            This is an automated report from the MissionBench experiment automation system.
                        </p>
                    </body>
                    </html>
                    """
                   
                    # Prepare attachments
                    attachments = [pdf_file, csv_file, csv_file.parent / 'statistics_summary.csv', dashboard_file]
                   
                    subject = f"{args.title} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                   
                    # Send email using direct SMTP implementation
                    try:
                        # Create message
                        msg = MIMEMultipart("alternative")
                        msg["Subject"] = subject
                        msg["From"] = gmail_user
                        msg["To"] = args.email
                       
                        # Attach HTML body
                        msg.attach(MIMEText(html_body, "html"))
                       
                        # Attach files
                        for file_path in attachments:
                            if file_path.exists():
                                with open(file_path, "rb") as attachment:
                                    part = MIMEBase("application", "octet-stream")
                                    part.set_payload(attachment.read())
                                encoders.encode_base64(part)
                                part.add_header(
                                    "Content-Disposition",
                                    f"attachment; filename= {file_path.name}",
                                )
                                msg.attach(part)
                                print(f"  ✓ Attached: {file_path.name}")
                       
                        # Send email via Gmail SMTP
                        print(f"  📧 Sending email to {args.email}...")
                        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                            server.login(gmail_user, gmail_password.replace(" ", ""))
                            server.sendmail(gmail_user, args.email, msg.as_string())
                       
                        print(f"  ✓ Email sent successfully to {args.email}")
                        success = True
                    except smtplib.SMTPAuthenticationError as e:
                        print(f"  ❌ Gmail authentication failed: {e}")
                        print(f"     Check that GMAIL_USER and GMAIL_PASSWORD are correct")
                        success = False
                    except Exception as e:
                        print(f"  ❌ Failed to send email: {e}")
                        success = False
                   
                    if not success:
                        print(f"  ⚠ Failed to send email to {args.email}")
            except ImportError as e:
                print(f"  ⚠ Could not import email libraries: {e}")
            except Exception as e:
                print(f"  ⚠ Error sending email: {e}")
   
    print("\n" + "="*70)
    print("✅ All requested reports generated!")
    print("="*70)
   
    # Print summary of what was generated
    print("\nGenerated files:")
    print(f"  - {csv_file}")
    print(f"  - {csv_file.parent / 'statistics_summary.csv'}")
    print(f"  - {dashboard_file}")
    if tex_file:
        print(f"  - {tex_file}")
    if pdf_file:
        print(f"  - {pdf_file}")
    print("="*70)


if __name__ == "__main__":
    main()