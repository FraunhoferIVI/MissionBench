"""Run LLM-as-judge over a CSV of ground-truth / predictions.

Reads CSV rows with columns `task_gt` and `task_gt_pred`, generates an
evaluation prompt for each pair, invokes a chat model, saves the model's
text response in a new column `model_response`, and writes the CSV back.

Usage:
  python -m predictors.llm_as_judge_runner /path/to/file.csv --model gemini-3-flash-preview
"""
from pathlib import Path
import csv
import argparse
import os
import time
import sys
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Add parent directory to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from base import VisionChain
from prompter.prompt_generator import generate_prompt


def process_csv(csv_path: Path, model_name: str, write_back: bool = True, dry_run: bool = False):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for r in reader:
            rows.append(r)

    out_col = "model_response"
    if out_col not in fieldnames:
        fieldnames.append(out_col)

    base_vision = VisionChain(mission={"mission": {"models_related": {"temperature": 0.0}}})
    base_vision.set_model_name(model_name)
    try:
        model = base_vision.create_model()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize model '{model_name}': {exc}")

    for idx, row in enumerate(rows):
        task_gt = (row.get("task_gt") or "").strip()
        task_gt_pred = (row.get("task_gt_pred") or "").strip()

        if not task_gt:
            rows[idx][out_col] = ""
            continue

        prompt = generate_prompt(
            prompt_type="task_gt_evaluation",
            task_gt_expected=task_gt,
            task_gt_pred=task_gt_pred,
        )

        if dry_run:
            rows[idx][out_col] = "[dry-run]"
            print(f"{idx+1}/{len(rows)} dry-run: {task_gt} -> {task_gt_pred}")
            continue

        try:
            messages = [HumanMessage(content=[{"type": "text", "text": prompt}])]
            response = model.invoke(messages)
            response_text = base_vision.validate_response(response)
            rows[idx][out_col] = response_text
            print(f"{idx+1}/{len(rows)} OK")
            time.sleep(0.1)
        except Exception as exc:
            rows[idx][out_col] = f"[error] {exc}"
            print(f"{idx+1}/{len(rows)} ERROR: {exc}")

    if write_back:
        tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        tmp_path.replace(csv_path)
        print(f"Wrote updated CSV to {csv_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Path to CSV file")
    p.add_argument("--model", default=os.getenv("JUDGE_MODEL", "gemini-3-flash-preview"), help="Model name to use")
    p.add_argument("--dry-run", action="store_true", help="Do not call model, only simulate")
    args = p.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")
    process_csv(csv_path, model_name=args.model, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
