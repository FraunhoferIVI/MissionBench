"""
BatchExecutor: Runs multiple experiments in parallel with rate limiting.
"""

import time
import json
import yaml
import csv
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from experiment_automation.config import ExperimentConfig
from experiment_automation.runner import ExperimentRunner
from utils.environment_manager import get_environment_manager
from utils.environment_manager import SimulatorServerStartupError
from utils import load_mission_index, model_uses_gemini, is_gemini_quota_error, GeminiApiKeyRotator
from utils.data_collection_utils import upload_or_update_file_in_drive, _extract_google_drive_folder_id


class GeminiKeysExhaustedError(RuntimeError):
    """Raised when all configured Gemini API keys are exhausted by quota limits."""


class BatchExecutor:
    """
    Executes multiple experiments in parallel.
    
    Responsibilities:
    - Queue experiments
    - Execute with thread pool
    - Rate limit API calls
    - Track progress
    - Collect results
    - Update pending experiments file as experiments complete
    """
    
    def __init__(self, 
                 results_base_dir: Path = Path("data/results"),
                 rate_limit_delay: float = 0.5,
                 benchmark_dir: Path = None,
                 pending_file: Path = None,
                 project_root: Path | str = None,
                 gemini_key_rotator: Optional[GeminiApiKeyRotator] = None):
        """
        Args:
            results_base_dir: Base directory for all results
            rate_limit_delay: Seconds between experiment starts (to avoid overloading APIs)
            pending_file: Path to pending_experiments.json for incremental updates
            project_root: Path to MissionBench project root (for environment manager)
        """
        self.results_base_dir = Path(results_base_dir)
        self.results_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.rate_limit_delay = rate_limit_delay
        self.benchmark_dir = benchmark_dir
        self.pending_file = Path(pending_file) if pending_file else self.results_base_dir / "pending_experiments.json"
        
        self.results: List[Dict[str, Any]] = []
        self.failed_experiments: List[ExperimentConfig] = []
        self._mission_index_cache: Dict[str, Dict[str, str]] = {}
        self.gemini_key_rotator = gemini_key_rotator
        
        # Initialize environment manager for dynamic environment switching
        if project_root is None:
            project_root = Path.cwd()
        self.env_manager = get_environment_manager(project_root)
    
    def execute(self, configs: List[ExperimentConfig]) -> Dict[str, Any]:
        """
        Execute a batch of experiment configs sequentially.
        Removes completed experiments from pending_experiments.json incrementally.
        
        Args:
            configs: List of ExperimentConfig objects
        
        Returns:
            Summary dict with success count, timing, etc.
        """
        print(f"\n{'='*70}")
        print(f"BatchExecutor: Starting {len(configs)} experiments (SEQUENTIAL)")
        print(f"Rate limit: {self.rate_limit_delay}s between starts")
        print(f"Pending file: {self.pending_file}")
        print(f"{'='*70}\n")
        
        batch_start = time.time()
        self.results = []
        self.failed_experiments = []
        
        pending_configs: List[ExperimentConfig] = []

        current_index = 0
        try:
            for i, config in enumerate(configs, 1):
                current_index = i
                if i > 1:
                    time.sleep(self.rate_limit_delay)

                attempts = 0
                max_attempts = max(
                    1,
                    (self.gemini_key_rotator.total_keys if self.gemini_key_rotator else 1),
                )

                while True:
                    attempts += 1
                    try:
                        # Determine mission environment from scenario + val_mission_file.
                        target_env = self._resolve_environment_for_config(config)
                        print(
                            "   Target environment: "
                            f"scenario={config.scenario_name}, resolved={target_env}, "
                            f"current={self.env_manager.current_environment}"
                        )
                        self.env_manager.ensure_environment(target_env)

                        result = self._run_single_experiment(config)
                        if not result.get("success"):
                            err_msg = str(result.get("error_message", "")).lower()
                            if "/reset" in err_msg and (
                                "500" in err_msg
                                or "internal server error" in err_msg
                                or "unreachable" in err_msg
                            ):
                                raise SimulatorServerStartupError(
                                    "Fatal simulator reset/backend failure detected: "
                                    f"{result.get('error_message')}"
                                )

                            if (
                                attempts < max_attempts
                                and self._try_rotate_gemini_key(config, result.get("error_message"))
                            ):
                                print(
                                    f"[{i}/{len(configs)}] ↻ Retrying {config.experiment_id} "
                                    f"after Gemini key rotation ({attempts}/{max_attempts})"
                                )
                                continue

                            if self._gemini_keys_exhausted_for_error(
                                config,
                                result.get("error_message"),
                            ):
                                raise GeminiKeysExhaustedError(
                                    "All configured Gemini API keys were exhausted or invalid "
                                    "(quota/auth failures). "
                                    f"Stopping now and keeping remaining experiments in pending file: {self.pending_file}"
                                )

                        self.results.append(result)
                        self._update_running_averages_csv(result)

                        status = "✓ PASS" if result["success"] else "✗ FAIL"
                        print(f"[{i}/{len(configs)}] {status} {config.experiment_id}")

                        if not result["success"]:
                            self.failed_experiments.append(config)

                        # Remove completed experiment from pending file
                        self._remove_from_pending(config)
                        break

                    except Exception as e:
                        if isinstance(e, SimulatorServerStartupError):
                            print(f"[{i}/{len(configs)}] ✗ FATAL SERVER STARTUP ERROR: {e}")
                            raise

                        if (
                            attempts < max_attempts
                            and self._try_rotate_gemini_key(config, str(e))
                        ):
                            print(
                                f"[{i}/{len(configs)}] ↻ Retrying {config.experiment_id} "
                                f"after Gemini key rotation ({attempts}/{max_attempts})"
                            )
                            continue

                        if self._gemini_keys_exhausted_for_error(config, str(e)):
                            raise GeminiKeysExhaustedError(
                                "All configured Gemini API keys were exhausted or invalid "
                                "(quota/auth failures). "
                                f"Stopping now and keeping remaining experiments in pending file: {self.pending_file}"
                            )

                        print(f"[{i}/{len(configs)}] ✗ EXCEPTION {config.experiment_id}: {e}")
                        self.failed_experiments.append(config)
                        
                        error_result = {
                            "success": False,
                            "experiment_id": config.experiment_id,
                            "config": config.to_dict(),
                            "total_time": 0.0,
                            "mission_result": None,
                            "error_message": str(e),
                        }
                        self.results.append(error_result)
                        self._update_running_averages_csv(error_result)
                        
                        # Still remove from pending even if failed
                        self._remove_from_pending(config)
                        break
        except KeyboardInterrupt:
            pending_start = max(current_index - 1, 0)
            pending_configs = configs[pending_start:]
            print("\n⚠ Interrupted by user (Ctrl+C). Remaining experiments left in pending file.")
        
        batch_end = time.time()
        
        # Clean up pending file if all experiments are done
        if not pending_configs and self.pending_file.exists():
            remaining = self._read_pending_configs()
            if not remaining:
                print(f"✓ All experiments completed. Removing pending file: {self.pending_file}")
                self.pending_file.unlink()

        summary = {
            "total_experiments": len(configs),
            "successful": sum(1 for r in self.results if r["success"]),
            "failed": len(self.failed_experiments),
            "total_time": batch_end - batch_start,
            "results": self.results,
            "pending": len(pending_configs),
            "pending_file": str(self.pending_file) if pending_configs else None,
        }
        
        self._print_summary(summary)
        
        return summary

    @staticmethod
    def save_pending_configs(configs: List[ExperimentConfig], pending_path: Path) -> Path:
        """Save all configs to pending_experiments.json file."""
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "count": len(configs),
            "configs": [c.to_dict() for c in configs],
        }
        with open(pending_path, "w") as f:
            json.dump(payload, f, indent=2)
        return pending_path

    def _remove_from_pending(self, completed_config: ExperimentConfig):
        """Remove a completed experiment from the pending_experiments.json file."""
        if not self.pending_file.exists():
            return
        
        try:
            remaining = self._read_pending_configs()
            
            # Filter out the completed experiment
            updated_configs = [
                c for c in remaining 
                if ExperimentConfig.from_dict(c).experiment_id != completed_config.experiment_id
            ]
            
            # Update file
            if updated_configs:
                payload = {
                    "count": len(updated_configs),
                    "configs": updated_configs,
                }
                with open(self.pending_file, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"   ⟲ Updated pending file: {len(updated_configs)} remaining")
            else:
                # All done
                if self.pending_file.exists():
                    self.pending_file.unlink()
                print(f"   ⟲ Pending file cleared (all experiments done)")
        except Exception as e:
            print(f"   ⚠ Warning: Could not update pending file: {e}")

    def _read_pending_configs(self) -> List[Dict[str, Any]]:
        """Read pending configs from file without converting them."""
        if not self.pending_file.exists():
            return []
        with open(self.pending_file, "r") as f:
            payload = json.load(f)
        return payload.get("configs", [])

    @staticmethod
    def load_pending_configs(pending_file: Path) -> List[ExperimentConfig]:
        """Load pending configs saved by save_pending_configs."""
        with open(pending_file, "r") as f:
            payload = json.load(f)
        return [ExperimentConfig.from_dict(c) for c in payload.get("configs", [])]
    
    def _run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Execute a single experiment."""
        runner = ExperimentRunner(config, self.results_base_dir, benchmark_dir=self.benchmark_dir)
        return runner.run()

    def _try_rotate_gemini_key(self, config: ExperimentConfig, error_message: Optional[str]) -> bool:
        """Rotate Gemini key when a quota/limit error is detected and more keys exist."""
        if not self.gemini_key_rotator:
            return False

        uses_gemini = model_uses_gemini(getattr(config, "model_name", None)) or model_uses_gemini(
            getattr(config, "task_gt_eval_model", None)
        )
        if not uses_gemini:
            return False

        if not is_gemini_quota_error(error_message):
            return False

        rotated = self.gemini_key_rotator.rotate_to_next_key()
        if rotated:
            print(
                "   🔁 Gemini quota limit detected. Switched to key "
                f"{self.gemini_key_rotator.active_index + 1}/{self.gemini_key_rotator.total_keys} "
                f"({self.gemini_key_rotator.active_key_masked})"
            )
            return True

        print("   ⚠ Gemini quota limit detected but no additional Gemini keys are available.")
        return False

    def _gemini_keys_exhausted_for_error(self, config: ExperimentConfig, error_message: Optional[str]) -> bool:
        """Return True when quota error happened on Gemini and there is no next key."""
        if not self.gemini_key_rotator:
            return False

        uses_gemini = model_uses_gemini(getattr(config, "model_name", None)) or model_uses_gemini(
            getattr(config, "task_gt_eval_model", None)
        )
        if not uses_gemini:
            return False

        if not is_gemini_quota_error(error_message):
            return False

        return self.gemini_key_rotator.active_index >= (self.gemini_key_rotator.total_keys - 1)

    def _resolve_environment_for_config(self, config: ExperimentConfig) -> Any:
        """Resolve mission environment from mission index + mission config metadata."""
        val_file = str(config.val_mission_file)
        if val_file not in self._mission_index_cache:
            self._mission_index_cache[val_file] = load_mission_index(
                val_file, project_root=self.env_manager.project_root
            )

        mission_map = self._mission_index_cache[val_file]
        mission_path = mission_map.get(config.scenario_name)
        if not mission_path:
            raise SimulatorServerStartupError(
                "Could not resolve mission path for scenario "
                f"'{config.scenario_name}' from {config.val_mission_file}"
            )

        resolved_mission_path = self._resolve_mission_config_path(
            mission_path=str(mission_path),
            val_mission_file=val_file,
        )

        try:
            with open(resolved_mission_path, "r", encoding="utf-8") as f:
                mission_cfg = yaml.safe_load(f) or {}
            return self.env_manager.get_mission_environment(mission_cfg)
        except ValueError as e:
            raise SimulatorServerStartupError(
                "Invalid mission environment metadata for scenario "
                f"'{config.scenario_name}' in {resolved_mission_path}: {e}"
            ) from e
        except Exception as e:
            raise SimulatorServerStartupError(
                "Failed to load mission config for scenario "
                f"'{config.scenario_name}' from {resolved_mission_path}: {e}"
            ) from e

    def _resolve_mission_config_path(self, mission_path: str, val_mission_file: str) -> Path:
        """Resolve mission config path using index file location and project conventions."""
        raw = Path(mission_path)
        if raw.is_absolute():
            if raw.exists():
                return raw
            raise SimulatorServerStartupError(f"Mission config path does not exist: {raw}")

        project_root = Path(self.env_manager.project_root)
        val_file_path = Path(val_mission_file)
        if not val_file_path.is_absolute():
            val_file_path = project_root / val_file_path

        candidates = [
            val_file_path.parent / raw,
            project_root / raw,
            project_root / "dataset" / raw,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        raise SimulatorServerStartupError(
            "Mission config path not found. "
            f"raw={mission_path}, checked={[str(c) for c in candidates]}"
        )
    
    def _update_running_averages_csv(self, current_result: Dict[str, Any]):
        """Update the running averages CSV with new results and upload to Drive."""
        csv_path = self.results_base_dir / "model_running_averages.csv"
        
        # Load existing rows, excluding the AVERAGES row if present
        existing_rows = []
        if csv_path.exists():
            try:
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("Experiment_ID") == "AVERAGES":
                            continue
                        existing_rows.append(row)
            except Exception:
                pass # If file is corrupt or empty, start fresh
        
        # Format current result
        mission_res = current_result.get("mission_result") or {}
        config = current_result.get("config", {})
        
        error_msg = current_result.get("error_message") or ""
        # Clean error message for CSV
        error_msg = str(error_msg).replace("\n", " ").replace("\r", "")[:100]

        new_row = {
            "Experiment_ID": current_result.get("experiment_id"),
            "Scenario": config.get("scenario_name", "unknown"),
            "Model": config.get("model_name", "unknown"),
            "Status": "PASS" if current_result.get("success") else "FAIL",
            "SR5": f"{float(mission_res.get('sr5', 0)):.2f}",
            "SR10": f"{float(mission_res.get('sr10', 0)):.2f}",
            "OSR5": f"{float(mission_res.get('osr5', 0)):.2f}",
            "Progress": f"{float(mission_res.get('mission_progress', 0)):.2f}",
            "Collision_Rate": f"{float(mission_res.get('collision_rate', 0)):.2f}",
            "Steps_Taken": str(mission_res.get("steps_taken", 0)),
            "Steps_Budget": str(mission_res.get("steps_budget", 0)),
            "Time": f"{float(current_result.get('total_time', 0)):.2f}",
            "Error": error_msg
        }
        
        # Check for duplicates (based on ID) and append
        existing_ids = {r["Experiment_ID"] for r in existing_rows}
        if new_row["Experiment_ID"] not in existing_ids:
            existing_rows.append(new_row)
        
        # Compute averages
        if not existing_rows:
            return

        def safe_mean(key):
            values = []
            for r in existing_rows:
                try:
                    values.append(float(r.get(key, 0)))
                except ValueError:
                    pass
            return statistics.mean(values) if values else 0.0

        avg_sr5 = safe_mean("SR5")
        avg_sr10 = safe_mean("SR10")
        avg_osr5 = safe_mean("OSR5")
        avg_progress = safe_mean("Progress")
        avg_collision = safe_mean("Collision_Rate")
        
        avg_row = {
            "Experiment_ID": "AVERAGES",
            "Scenario": "-",
            "Model": "-",
            "Status": f"Total: {len(existing_rows)}",
            "SR5": f"{avg_sr5:.2f}",
            "SR10": f"{avg_sr10:.2f}",
            "OSR5": f"{avg_osr5:.2f}",
            "Progress": f"{avg_progress:.2f}",
            "Collision_Rate": f"{avg_collision:.2f}",
            "Steps_Taken": "-",
            "Steps_Budget": "-",
            "Time": "-",
            "Error": "-"
        }
        
        fieldnames = ["Experiment_ID", "Scenario", "Model", "Status", "SR5", "SR10", "OSR5", "Progress", "Collision_Rate", "Steps_Taken", "Steps_Budget", "Time", "Error"]
        
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(avg_row)
                writer.writerows(existing_rows)
            print(f"   📊 Updated results CSV: {csv_path}")
        except Exception as e:
            print(f"   ⚠ Failed to update CSV: {e}")

        # Upload to Drive if configured
        drive_path = config.get("model_results_drive_path")
        if drive_path:
            folder_id = _extract_google_drive_folder_id(drive_path)
            if folder_id:
                print(f"   ☁ Uploading CSV to Drive folder: {folder_id}...")
                upload_or_update_file_in_drive(str(csv_path), folder_id)
            else:
                print(f"   ⚠ Invalid Drive path ID in config: {drive_path}")

    def _print_summary(self, summary: Dict[str, Any]):
        """Print execution summary."""
        success_rate = (summary["successful"] / summary["total_experiments"] * 100) if summary["total_experiments"] > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"Batch Execution Summary")
        print(f"{'='*70}")
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total time: {summary['total_time']:.2f}s")
        if summary['pending'] > 0:
            print(f"Pending: {summary['pending']}")
        print(f"{'='*70}\n")

