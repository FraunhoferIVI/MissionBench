"""this is for testing the different precdictors separately from hlap/vision.py for easier debugging"""
from datetime import datetime
from pathlib import Path
import sys
from PIL import Image
import yaml
import cv2
import numpy as np
import cv2

from dotenv import load_dotenv
# Add parent directory to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))
from vision import PixelPredictor
# Load environment variables
load_dotenv()

from predictors.base import VisionChain
from prompt_generator import generate_prompt

if __name__ == "__main__":
    # USER INPUTS #
    mission_yaml = "/home/nava/uav_mission_planning/UAVMissionPlanning/dataset/Visual_Inspection/NHEnv/grey_car_mission29/mission_config.yaml"
    input_image = "/home/nava/uav_mission_planning/UAVMissionPlanning/dataset/Visual_Inspection/NHEnv/grey_car_mission29/imgs/start_fp_image.png"
    model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    # amz nova premier
    # model_name = "us.amazon.nova-premier-v1:0"
    # gpt 5
    # model_name = "gpt-5-2025-08-07"
    model_name = "gemini-3-pro-preview"
    # model_name = "gemini-3-flash-preview"
    # model_name = "us.amazon.nova-pro-v1:0"
    predictor = "bb_and_depth_prediction"
    # USER INPUTS END #
    # mk results dir
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"data/results/test_predictors/{predictor}/{model_name}/{time_stamp}"
    
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    mission_name = Path(mission_yaml).parent.name
    
    with open(mission_yaml, "r") as f:
        test_mission = yaml.safe_load(f)
    pp = PixelPredictor(model_name=model_name, mission={"mission": test_mission})
    parsed_response_dict = pp.predict_bb_and_depth(image=input_image, results_dir=results_dir)

    output_img = pp.draw_bounding_boxes(Image.open(input_image), (parsed_response_dict['bb'], parsed_response_dict['depth'], parsed_response_dict['label']))
    # save image
    img_file_name = f"{results_dir}/{mission_name}_{model_name}_annotated_image.png"
    output_img.save(img_file_name)

    print(f"Annotated image generated and saved to {img_file_name}")
    
    
    