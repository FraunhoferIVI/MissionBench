import json
import os
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    from utils.image_utils import stitch_images_to_video
except ModuleNotFoundError:
    # Support direct script execution: `python utils/results_utils.py ...`
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Import sibling module directly to avoid shadowing `utils` as a package name.
    from image_utils import stitch_images_to_video


def _normalize_name_token(value: str) -> str:
    token = (value or "unknown").strip().lower()
    token = token.replace("-", "_").replace(" ", "_")
    token = re.sub(r"[^a-z0-9._]+", "", token)
    token = token.replace("_preview", "")
    if token.startswith("gemini_"):
        token = "gemini" + token[len("gemini_"):]
    return token or "unknown"


def _metric_token(value, decimals: int = 2) -> str:
    if value is None:
        return "na"
    try:
        if isinstance(value, str) and value.strip() == "":
            return "na"
        num = float(value)
        if num.is_integer():
            return str(int(num))
        return f"{num:.{decimals}f}"
    except Exception:
        return "na"


def _build_media_filename(scenario_results_dir: Path, media_format: str) -> str:
    mission_name = "unknown_mission"
    model_name = "unknown_model"
    try:
        mission_name = _normalize_name_token(scenario_results_dir.parents[2].name)
        model_name = _normalize_name_token(scenario_results_dir.parents[1].name)
    except Exception:
        pass

    mp = "na"
    sr5 = "na"
    sr10 = "na"
    eval_path = scenario_results_dir / "evaluation_result.json"
    if eval_path.exists():
        try:
            with open(eval_path, "r") as f:
                eval_data = json.load(f) or {}
            mp = _metric_token(eval_data.get("mission_progress"), decimals=2)
            sr5 = _metric_token(eval_data.get("sr5"), decimals=0)
            sr10 = _metric_token(eval_data.get("sr10"), decimals=0)
        except Exception:
            pass

    return (
        f"{mission_name}_{model_name}_"
        f"mp_{mp}_sr5_{sr5}_sr10_{sr10}.{media_format}"
    )


def generate_mission_media(
    scenario_results_dir: str | Path,
    output_media_format: str = "webp",
    fps: int = 1,
    overwrite: bool = False,
) -> Optional[str]:
    """Generate stitched mission media from imgs/ for an existing run.

    Returns media path on success, else None.
    """
    scenario_results_dir = Path(scenario_results_dir)
    image_folder = scenario_results_dir / "imgs"
    if not image_folder.exists():
        print(f"Image folder not found: {image_folder}")
        return None
    media_format = (output_media_format or "webp").strip().lower()
    if media_format not in {"webp", "mp4"}:
        media_format = "webp"

    media_path = image_folder / _build_media_filename(
        scenario_results_dir,
        media_format,
    )
    image_files = sorted(
        [
            f
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    if not image_files:
        print(f"No images found in {image_folder}")
        return None

    if media_path.exists() and not overwrite:
        print(f"Media already exists at {media_path} (use --overwrite to rebuild)")
        return str(media_path)

    print(f"Stitching {len(image_files)} images to {media_path}...")
    ok = stitch_images_to_video(
        str(image_folder),
        image_files,
        output_path=str(media_path),
        fps=fps,
        output_format=media_format,
    )
    if ok:
        print(f"Media saved to {media_path}")
        return str(media_path)

    print("Failed to create mission media.")
    return None


def save_step_by_step_results(
    final_state: Dict,
    scenario_name: str,
    scenario_results_dir: str,
    total_execution_time: Optional[float] = None,
    output_media_format: str = "webp",
) -> str:
    """Save step-by-step execution results for debugging and analysis."""
    import pickle
    from experiment_automation.create_results import create_results_json

    scenario_results_dir = Path(scenario_results_dir)
    scenario_results_dir.mkdir(parents=True, exist_ok=True)
    pickle_file = scenario_results_dir / "final_state.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(final_state, f)
    print(f"\n💾 Saved final state to: {pickle_file}")

    # Prepare data for serialization (remove non-serializable objects)
    serializable_state = {}
    non_serializable_keys = []

    for key, value in final_state.items():
        try:
            json.dumps(value, default=str)
            serializable_state[key] = value
        except (TypeError, ValueError):
            non_serializable_keys.append(key)
            serializable_state[f"{key}_str"] = str(value)

    # Save individual outputs for easier debugging
    debug_outputs = {
        "instruction": final_state.get("instruction"),
        "raw_response": final_state.get("raw_response"),
        "parsed_actions": final_state.get("parsed_actions"),
        "all_raw_waypoint_response": final_state.get("all_raw_waypoint_response", []),
        "model_name": final_state.get("model_name"),
    }

    debug_file = scenario_results_dir / "debug_outputs.json"
    with open(debug_file, "w") as f:
        json.dump(debug_outputs, f, indent=2, default=str)
    print(f"   ✓ Saved debug outputs to: {debug_file}")

    # Save evaluation results if available
    if final_state.get("evaluation_result"):
        eval_file = scenario_results_dir / "evaluation_result.json"
        with open(eval_file, "w") as f:
            eval_data = {
                k: v
                for k, v in final_state["evaluation_result"].items()
                if not isinstance(v, type(None)) and k != "final_step_result"
            }
            json.dump(eval_data, f, indent=2, default=str)
        print(f"   ✓ Saved evaluation results to: {eval_file}")

    # Save summary report
    summary = {
        "total_execution_time": total_execution_time,
        "scenario_name": scenario_name,
        "instruction": final_state.get("instruction"),
        "model_name": final_state.get("model_name"),
        "low_level_model_name": final_state.get("low_level_model_name"),
        "num_actions": len(final_state.get("parsed_actions", [])),
        "actions": [a for a in final_state.get("parsed_actions", [])],
        "all_parsed_waypoints": final_state.get("all_parsed_waypoints", []),
        "evaluation": {
            "mission_passed": final_state.get("evaluation_result", {}).get("mission_passed"),
            "waypoint_deviations": final_state.get("evaluation_result", {}).get("waypoint_deviation"),
        }
        if final_state.get("evaluation_result")
        else None,
    }

    summary_file = scenario_results_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   ✓ Saved summary to: {summary_file}")

    if non_serializable_keys:
        print(
            f"   ⚠ Note: {len(non_serializable_keys)} non-serializable fields excluded: {', '.join(non_serializable_keys)}"
        )

    # Save stitched mission media (WebP by default)
    generate_mission_media(
        scenario_results_dir=scenario_results_dir,
        output_media_format=output_media_format,
        fps=1,
        overwrite=False,
    )
    create_results_json(scenario_results_dir)
    return str(scenario_results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate stitched mission media for an existing scenario results directory."
    )
    parser.add_argument(
        "scenario_results_dir",
        type=str,
        nargs="?",
        default=None,
        help="Path to scenario results directory (must contain imgs/).",
    )
    parser.add_argument(
        "--scenario_results_dir",
        dest="scenario_results_dir_flag",
        type=str,
        default=None,
        help="Path to scenario results directory (must contain imgs/).",
    )
    parser.add_argument(
        "--format",
        dest="output_media_format",
        type=str,
        default="webp",
        choices=["webp", "mp4"],
        help="Output media format (default: webp).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Playback fps (default: 1  for normal playback).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mission_video.<format> if present.",
    )
    parser.add_argument(
        "--create-results-json",
        action="store_true",
        help="Also regenerate result.json from existing debug outputs.",
    )

    args = parser.parse_args()

    target_scenario_results_dir = args.scenario_results_dir_flag or args.scenario_results_dir
    if not target_scenario_results_dir:
        parser.error("Provide scenario results path as positional argument or --scenario_results_dir")

    output = generate_mission_media(
        scenario_results_dir=target_scenario_results_dir,
        output_media_format=args.output_media_format,
        fps=args.fps,
        overwrite=args.overwrite,
    )
    if output and args.create_results_json:
        from experiment_automation.create_results import create_results_json
        create_results_json(target_scenario_results_dir)
