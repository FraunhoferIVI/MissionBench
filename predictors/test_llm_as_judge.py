"""this is for testing the different precdictors separately from hlap/vision.py for easier debugging"""
from datetime import datetime
from pathlib import Path
import sys
from PIL import Image
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
# Add parent directory to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))
from vision import PixelPredictor
# Load environment variables
load_dotenv()
import yaml
from base import VisionChain
from prompter.prompt_generator import generate_prompt

if __name__ == "__main__":
    task_gt_model = "gemini-3-flash-preview"
    task_gt_expected_list = ["cell towers or antennas", "drop", "3 elephants", "camp fire by people", "1 missing person", "1 missing person", "2 rhinos"]
    task_gt_pred_list = ["two red and white lattice towers", "drop", "3 elephants", "campfire", "1", "1", ""]
    for task_gt_expected, task_gt_pred in zip(task_gt_expected_list, task_gt_pred_list):

        eval_prompt = generate_prompt(
                prompt_type="task_gt_evaluation",
                task_gt_expected=task_gt_expected,
                task_gt_pred=task_gt_pred,
            )

        # USER INPUTS #
        mission_yaml = "dataset/Visual_Inspection/NHEnv/grey_car_unittest_mission30/mission_config.yaml"
        with open(mission_yaml, "r") as f:
            test_mission = yaml.safe_load(f)
        model_name = "gemini-3-pro-preview"
        pp = PixelPredictor(model_name=model_name, mission={"mission": test_mission})
        if task_gt_expected:
            # Use a small model to evaluate match
            try:
                # since we are not evaluating the ocr ability of the model, we can relax the exact match condition
                base_vision = VisionChain(mission={"mission": test_mission})
                base_vision.set_model_name(task_gt_model)
                model = base_vision.create_model()
                content = [{"type": "text", "text": eval_prompt}]
                response = model.invoke([HumanMessage(content=content)])
                if isinstance(response.content, list):
                    print("Warning: Received list response from model, taking first element")
                    response = response.content[0]
                task_gt_response = (response['text'] or "").strip().lower()
                task_gt_match = 1 if task_gt_response.startswith("success") else 0

                print(f"Task GT Evaluation:\nExpected: {task_gt_expected}\nPredicted: {task_gt_pred}\nModel Response: {task_gt_response}\nMatch: {task_gt_match}")
            except Exception as exc:
                print(f"   ⚠ task_gt evaluation failed: {exc}")