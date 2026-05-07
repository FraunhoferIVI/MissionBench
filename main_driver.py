#!/usr/bin/env python3
"""
Main driver for experiment automation.
Orchestrates experiment definition, execution, and analysis.
"""
import os
import yaml
import time
import argparse
import re
from pathlib import Path
from experiment_automation import (
    GridSearchDefinition,
    BatchExecutor,
    ResultsAggregator,
)
from experiment_automation.executor import GeminiKeysExhaustedError
from utils.environment_manager import SimulatorServerStartupError
from utils import (
    validate_main_driver_config,
    get_high_level_prompt_type_for_strategy,
    send_experiment_results_email,
    get_readable_timestamp,
    ensure_required_envs_present,
    ensure_required_dataset_present,
    ensure_model_api_keys_present,
    sync_runtime_sim_config,
    validate_google_drive_destination_for_sync,
    sync_mission_folder_to_google_drive,
    load_mission_index,
    model_uses_gemini,
    GeminiApiKeyRotator,
)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, load manually
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def main(config_path=None):
    """Main entry point for experiment automation.
    
    Args:
        config_path: Optional path to experiment_configs.yaml. If not provided,
                    uses default configs/experiment_configs.yaml
    """
    if config_path is None:
        project_root = Path(__file__).resolve().parent
        config_path = project_root / "configs" / "experiment_configs.yaml"
    else:
        config_path = Path(config_path)
        project_root = config_path.resolve().parent.parent
    
    print("\n🚁 UAV Mission Planning - Experiment Automation")
    print("=" * 70)

    project_root = Path(__file__).resolve().parent

    # Load main driver config
    if not config_path.exists():
        raise FileNotFoundError(f"experiment_configs.yaml not found at {config_path}")
    with open(config_path, "r") as f:
        data_configs = yaml.safe_load(f) or {}
    validate_main_driver_config(data_configs)
    main_cfg = data_configs.get("main_driver", {})

    model_related_cfg = (
        main_cfg.get("model_related")
        if isinstance(main_cfg.get("model_related"), dict)
        else main_cfg.get("models_related")
        if isinstance(main_cfg.get("models_related"), dict)
        else {}
    )
    image_related_cfg = (
        main_cfg.get("image_related")
        if isinstance(main_cfg.get("image_related"), dict)
        else {}
    )

    def _normalize_thinking_level(value):
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "none", "null"}:
                return None
            return normalized
        return value

    resolved_temperature = model_related_cfg.get("temperature", main_cfg.get("temperature", 0.7))
    resolved_top_p = model_related_cfg.get("top_p", main_cfg.get("top_p", 0.95))
    resolved_top_k = model_related_cfg.get("top_k", main_cfg.get("top_k", 20))
    resolved_min_p = model_related_cfg.get("min_p", main_cfg.get("min_p", 0.0))
    resolved_presence_penalty = model_related_cfg.get(
        "presence_penalty", main_cfg.get("presence_penalty", 1.5)
    )
    resolved_repetition_penalty = model_related_cfg.get(
        "repetition_penalty", main_cfg.get("repetition_penalty", 1.0)
    )
    resolved_model_names = list(
        model_related_cfg.get("model_names", main_cfg.get("model_names", ["gemini-2.5-flash-lite"])) or []
    )
    resolved_task_gt_eval_model = model_related_cfg.get(
        "task_gt_eval_model", main_cfg.get("task_gt_eval_model", "gemini-3-flash-preview")
    )
    resolved_thinking_level = _normalize_thinking_level(
        model_related_cfg.get("thinking_level", main_cfg.get("thinking_level"))
    )
    resolved_num_history_images = image_related_cfg.get(
        "num_history_images", main_cfg.get("num_history_images", 1)
    )
    resolved_img_enhancement_type = image_related_cfg.get(
        "img_enhancement_type", main_cfg.get("img_enhancement_type")
    )
    resolved_capture_interval = image_related_cfg.get(
        "capture_interval", main_cfg.get("capture_interval", 0)
    )
    resolved_output_media_format = image_related_cfg.get(
        "output_media_format", main_cfg.get("output_media_format", "webp")
    )

    configured_models = list(resolved_model_names)
    task_gt_eval_model = resolved_task_gt_eval_model
    if isinstance(task_gt_eval_model, str) and task_gt_eval_model.strip():
        configured_models.append(task_gt_eval_model.strip())

    # For Gemini runs, ignore inherited shell Gemini vars and rebuild from .env for this process.
    if any(model_uses_gemini(name) for name in configured_models):
        key_pattern = re.compile(r"^(GOOGLE|GEMINI)_API_KEY(?:_?\w+)?$")
        removed = [k for k in list(os.environ.keys()) if key_pattern.fullmatch(k)]
        for k in removed:
            os.environ.pop(k, None)

        loaded = []
        env_file = project_root / ".env"
        if env_file.exists():
            with open(env_file, "r") as file:
                for raw_line in file:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if not key_pattern.fullmatch(key):
                        continue
                    value = re.sub(r"#.*$", "", value).strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                        value = value[1:-1].strip()
                    if value:
                        os.environ[key] = value
                        loaded.append(key)

        if removed:
            print(f"✓ Cleared inherited Gemini env var(s) for this run: {len(removed)}")
        if loaded:
            print(f"✓ Loaded Gemini env var(s) from .env for this run: {', '.join(loaded)}")

    # Keep backward compatibility: sync simulator runtime config used by API server.
    sync_runtime_sim_config(project_root, main_cfg)

    ensure_required_envs_present(project_root)
    ensure_required_dataset_present(project_root)
    ensure_model_api_keys_present(config_path)

    gemini_key_rotator = None
    if any(model_uses_gemini(name) for name in configured_models):
        gemini_key_rotator = GeminiApiKeyRotator.from_environment()
        if gemini_key_rotator.activate_first_key():
            print(
                "✓ Gemini API key rotation enabled "
                f"({gemini_key_rotator.total_keys} key(s); active={gemini_key_rotator.active_key_masked})"
            )

    strategy = main_cfg.get("strategy", "step_by_step")
    if isinstance(strategy, list):
        high_level_action_prompt_type = [
            get_high_level_prompt_type_for_strategy(item) for item in strategy
        ]
    else:
        high_level_action_prompt_type = get_high_level_prompt_type_for_strategy(strategy)
    
    # Step 1: Define experiment grid or resume pending
    print("\n[Step 1] Defining experiment grid...")
    resume_exp = bool(main_cfg.get("resume_exp", False))
    timestamp = get_readable_timestamp()
    
    # Handle resume: use existing results_dir instead of creating new one
    if resume_exp:
        results_dir_to_resume = main_cfg.get("results_dir_to_resume")
        if not results_dir_to_resume:
            print("✗ resume_exp=true but results_dir_to_resume not specified in config")
            return
        results_dir = Path(results_dir_to_resume)
        pending_file = results_dir / "pending_experiments.json"
        if not pending_file.exists():
            print(f"✗ Pending file not found at {pending_file}")
            return
        # Resuming from a previous run
        print(f"⚠ Pending experiments detected. Resuming from {pending_file}")
        configs = BatchExecutor.load_pending_configs(pending_file)
        print(f"✓ Loaded {len(configs)} pending experiment configurations")
        # copy the config file to the results directory for record-keeping
        pending_file_backup = results_dir / f"pending_experiments_backup_{timestamp}.json"
        pending_file_backup.write_text(pending_file.read_text())
        print(f"✓ Backed up pending experiments to {pending_file_backup}")
        # copy the configs/experiment_configs.yaml to results directory for record-keeping
        configs_yaml_backup = results_dir / f"configs_backup_{timestamp}.yaml"
        configs_yaml_backup.write_text(config_path.read_text())
        print(f"✓ Backed up configs.yaml to {configs_yaml_backup}")
    else:
        # Fresh run - create new results directory with timestamp
        results_dir = Path(f"data/results/experiment_results_{timestamp}_")
        pending_file = results_dir / "pending_experiments.json"
        results_dir.mkdir(parents=True, exist_ok=True)
        val_mission_file_name = main_cfg.get("val_mission_file")
        try:
            val_data = load_mission_index(val_mission_file_name, project_root=project_root)
            print(f"✓ Mission loader mode: nested-index ({val_mission_file_name})")
        except Exception:
            # Backward compatibility: legacy mission-map loading path.
            with open(val_mission_file_name, "r") as file:
                val_data = yaml.safe_load(file) or {}
            print(f"✓ Mission loader mode: legacy-flat-map ({val_mission_file_name})")
        val_names = val_data.keys()
        if main_cfg.get("full_validation_set"):
            main_cfg["scenario_names"] = list(val_names)
            print(f"✓ Loaded all {len(val_names)} validation scenarios from dataset.")
        else:
            print(f"✓ Using specified validation scenarios: {main_cfg.get('scenario_names', [])}")
        grid = GridSearchDefinition(
            name="test_grid_v1",
            scenario_names=main_cfg.get("scenario_names"),
            model_names=resolved_model_names,
            repeat=int(main_cfg.get("repeat", 1)),  # 1 repeat for testing
            temperature=resolved_temperature,
            top_p=resolved_top_p,
            top_k=resolved_top_k,
            min_p=resolved_min_p,
            presence_penalty=resolved_presence_penalty,
            repetition_penalty=resolved_repetition_penalty,
            thinking_level=resolved_thinking_level,
            task_gt_eval_model=resolved_task_gt_eval_model,
            val_mission_file=val_mission_file_name,
            strategy=strategy,
            num_history_images=resolved_num_history_images,  # Number of images for ablation
            img_enhancement_type=resolved_img_enhancement_type,
            high_level_action_prompt_type=high_level_action_prompt_type,
            capture_interval=resolved_capture_interval,
            collision_check_mode=main_cfg.get("simulator", {}).get("CollisionCheckMode", False),
            output_media_format=str(resolved_output_media_format),
            model_results_drive_path=main_cfg.get("model_results_drive_path"),
        )
        # copy the configs/experiment_configs.yaml to results directory for record-keeping
        configs_yaml_backup = results_dir / f"configs_backup_{timestamp}.yaml"
        configs_yaml_backup.write_text(config_path.read_text())
        print(f"✓ Backed up configs.yaml to {configs_yaml_backup}")
        # Validate grid inputs and explain zero-config causes
        zero_causes = []
        if not grid.scenario_names:
            zero_causes.append("scenario_names is empty")
        if not grid.model_names:
            zero_causes.append("model_names is empty")
        if grid.repeat <= 0:
            zero_causes.append(f"repeat must be > 0 (got {grid.repeat})")

        if zero_causes:
            print("\n⚠ Grid definition will generate 0 configurations:")
            for cause in zero_causes:
                print(f"  - {cause}")
            print("  Fix: provide at least one value for each list and repeat > 0.")
        
        # Generate all experiment configs
        configs = grid.generate_configs()
        print(f"✓ Generated {len(configs)} experiment configurations")
        print(f"  Expected total: {grid.count_experiments()}")

        if len(configs) == 0:
            print("\n✗ Step 1 failed: No experiment configurations generated. Stopping.")
            return
        
        # Print first few configs for review
        print("\nFirst 3 configurations:")
        for config in configs[:3]:
            print(f"  - {config}")

    run_logs_dir = results_dir / "logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    print("\n[Initialization] Configuring simulator log directory...")
    print(f"  Logs directory: {run_logs_dir}")
    # Share log directory with AirSim API server and Unreal process logs.
    # Server lifecycle is managed by EnvironmentManager during Step 2.
    os.environ["MISSIONBENCH_LOG_DIR"] = str(run_logs_dir)
    
    # Always create/update pending_experiments.json with all configs to run
    print(f"\n[Step 1] Creating pending experiments file: {pending_file}")
    BatchExecutor.save_pending_configs(configs, pending_file)
    print(f"✓ Pending experiments file saved with {len(configs)} configurations")

    # Optional results sync to Google Drive/local path.
    drive_sync_enabled = bool(main_cfg.get("upload_results_to_google_drive", False))
    drive_destination = (
        str(main_cfg.get("google_drive_destination") or os.getenv("GOOGLE_DRIVE_RESULTS_FOLDER_URL", ""))
    ).strip()
    if drive_sync_enabled:
        if not drive_destination:
            raise ValueError(
                "upload_results_to_google_drive=true but no destination configured. "
                "Set main_driver.google_drive_destination or GOOGLE_DRIVE_RESULTS_FOLDER_URL."
            )
        validate_google_drive_destination_for_sync(drive_destination)
        print(f"✓ Google Drive sync enabled: {drive_destination}")
    
    # Step 2: Execute experiments sequentially
    print("\n[Step 2] Executing experiments...")
    benchmark_dir = (project_root / "dataset")
    executor = BatchExecutor(
        results_base_dir=results_dir,
        rate_limit_delay=float(main_cfg.get("rate_limit_delay", 0.5)),
        benchmark_dir=benchmark_dir,
        pending_file=pending_file,
        project_root=project_root,
        gemini_key_rotator=gemini_key_rotator,
    )

    # Run experiments sequentially and sync each mission folder as soon as it completes.
    batch_start = time.time()
    results = []
    failed_configs = []
    pending_configs = []
    completed_count = 0
    current_index = 0
    try:
        for i, config in enumerate(configs, 1):
            current_index = i
            if i > 1:
                time.sleep(executor.rate_limit_delay)

            try:
                mission_summary = executor.execute([config])
                result = (mission_summary.get("results") or [{}])[0]
                results.append(result)
                status = "✓ PASS" if result.get("success") else "✗ FAIL"
                print(f"[{i}/{len(configs)}] {status} {config.experiment_id}")
                if not result.get("success"):
                    failed_configs.append(config)
            except GeminiKeysExhaustedError as e:
                print(f"\n✗ Execution stopped: {e}")
                print(
                    "⚠ All Gemini API keys appear exhausted for today. "
                    f"Please run again tomorrow with this pending file: {pending_file}"
                )
                return
            except Exception as e:
                if isinstance(e, SimulatorServerStartupError):
                    raise
                print(f"[{i}/{len(configs)}] ✗ EXCEPTION {config.experiment_id}: {e}")
                failed_configs.append(config)
                result = {
                    "success": False,
                    "experiment_id": config.experiment_id,
                    "config": config.to_dict(),
                    "total_time": 0.0,
                    "mission_result": None,
                    "error_message": str(e),
                }
                results.append(result)

            completed_count += 1

            # Per-mission Google Drive sync: upload each experiment folder right away.
            if drive_sync_enabled:
                mission_folder = results_dir / f"{config.scenario_name}_{config.model_name}_{config.experiment_id}"
                if mission_folder.exists():
                    try:
                        synced_path = sync_mission_folder_to_google_drive(mission_folder, drive_destination)
                        print(f"   ☁ Synced mission to Google Drive: {synced_path}")
                    except Exception as e:
                        print(f"   ⚠ Failed to sync mission to Google Drive ({mission_folder.name}): {e}")
                else:
                    print(f"   ⚠ Mission folder not found for sync: {mission_folder}")
    except KeyboardInterrupt:
        pending_start = max(current_index - 1, 0)
        pending_configs = configs[pending_start:]
        print("\n⚠ Interrupted by user (Ctrl+C). Remaining experiments left in pending file.")

    batch_end = time.time()
    batch_summary = {
        "total_experiments": len(configs),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": len(failed_configs),
        "completed": completed_count,
        "total_time": batch_end - batch_start,
        "results": results,
        "pending": len(pending_configs),
        "pending_file": str(pending_file) if pending_configs else None,
    }

    # Continue to analysis even if some experiments failed (we still have data)
    # Only stop if ALL experiments are pending or none executed at all
    if batch_summary.get("pending", 0) > 0 and batch_summary.get("completed", 0) == 0:
        print("\n⚠ Step 2 incomplete: All experiments are pending. Saved for resume.")
        return
    if batch_summary.get("total_experiments", 0) == 0:
        print("\n✗ Step 2 failed: No experiments executed. Stopping.")
        return
    
    # Warn about failures but continue
    if batch_summary.get("failed", 0) > 0:
        print(f"\n⚠ Step 2 incomplete: {batch_summary.get('failed', 0)} experiments failed.")
        print(f"   Continuing to analysis with {batch_summary.get('completed', 0)} successful experiments.")
    else:
        print(f"\n✓ Step 2 complete: All {batch_summary.get('completed', 0)} experiments succeeded!")
    
    # Step 3: Analyze and visualize results
    print("\n[Step 3] Analyzing results...")
    aggregator = ResultsAggregator(results_dir)
    aggregator.load_results()
    aggregator.extract_git_info()
    aggregator.compute_statistics()
    
    csv_file = aggregator.generate_csv_summary()
    html_file = aggregator.generate_html_dashboard()
    
    # Generate PDF report
    print("\n[Step 3.1] Generating PDF report...")
    pdf_file = None
    tex_file = None
    try:
        report_title = main_cfg.get("report_title", "UAV Mission Planning - Experiment Results")
        tex_file = aggregator.generate_latex_report(title=report_title)
        print(f"  ✓ LaTeX generated: {tex_file}")
        
        pdf_file = aggregator.compile_pdf_report(tex_file)
        if pdf_file:
            print(f"  ✓ PDF generated: {pdf_file}")
        else:
            print(f"  ⚠ PDF compilation failed (LaTeX file available at {tex_file})")
            compile_log = tex_file.with_suffix(".compile.log") if tex_file else None
            if compile_log and compile_log.exists():
                print(f"  ⚠ Compile log: {compile_log}")
    except Exception as e:
        print(f"  ⚠ Failed to generate PDF report: {e}")

    # Build summary for email from aggregated results (covers resumed + previous runs)
    total_results = len(aggregator.results)
    successful_results = sum(1 for r in aggregator.results if r.get("success"))
    failed_results = total_results - successful_results
    email_summary = {
        "total_experiments": total_results,
        "successful": successful_results,
        "failed": failed_results,
        "pending": 0,
        "results": aggregator.results,
        "total_time": batch_summary.get("total_time"),
    }
    
    print(f"\n✓ Results generated:")
    print(f"  - CSV (detailed): {csv_file}")
    print(f"  - CSV (statistics): {csv_file.parent / 'statistics_summary.csv'}")
    print(f"  - Dashboard: {html_file}")
    if pdf_file:
        print(f"  - PDF Report: {pdf_file}")
    
    print("\n" + "=" * 70)
    print("✓ Experiment run complete!")
    print("=" * 70 + "\n")
    
    # Step 4: Send results via email
    print("[Step 4] Sending results email...")
    recipient_email = os.getenv("RESULTS_EMAIL") or main_cfg.get("results_email", "sumanrbt1997@gmail.com")
    gmail_user = os.getenv("GMAIL_USER") or main_cfg.get("gmail_user")
    gmail_password = os.getenv("GMAIL_PASSWORD") or main_cfg.get("gmail_password")
    
    if send_experiment_results_email(
        to_email=recipient_email,
        csv_file=Path(csv_file),
        html_file=Path(html_file),
        batch_summary=email_summary,
        gmail_user=gmail_user,
        gmail_password=gmail_password,
        pdf_file=pdf_file,
        tex_file=tex_file,
    ):
        print(f"✓ Results email sent to {recipient_email}")
        if pdf_file:
            print(f"  Including PDF report: {pdf_file.name}")
        elif tex_file:
            print(f"  PDF unavailable; included LaTeX source: {Path(tex_file).name}")
    else:
        print(f"⚠ Failed to send results email. Check GMAIL_USER and GMAIL_PASSWORD in .env file or environment variables.")
        print(f"  See .env_sample for configuration template.")
        print(f"  For Gmail App Password: https://myaccount.google.com/apppasswords")
    
    print("\n" + "=" * 70)
    print("✓ All done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV Mission Planning - Experiment Automation")
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to experiment_configs.yaml (default: configs/experiment_configs.yaml)"
    )
    args = parser.parse_args()
    main(config_path=args.config_file)