from pathlib import Path
import re
import ast
from typing import Union
from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from predictors.base import VisionChain
from predictors.utils import format_drone_pose, add_module_path
import numpy as np
import cv2

from prompter.prompt_generator import generate_prompt

class PixelPredictor(VisionChain):
    def __init__(
        self,
        model_name="gpt-5o-nano",
        mission: dict = None,
    ):
        super().__init__(mission=mission)
        self.set_model_name(model_name)

    def analyze_image(self, image: Union[str, Image.Image], prompt: str) -> str:
        """
        Analyze a single image with the given prompt.

        Args:
            image: PIL Image, file path, or URL
            prompt: Text prompt for analysis

        Returns:
            Analysis result as string
        """
        model = self.create_model()

        # Create message content
        content = [{"type": "text", "text": prompt}, self._image_to_content(image)]

        # Create message and invoke model
        message = HumanMessage(content=content)
        response = model.invoke([message])

        return response.content

    def analyze_image_full(self, image: Union[str, Image.Image], prompt: str) -> str:
        """
        Analyze a single image with the given prompt returning full response object.

        Args:
            image: PIL Image, file path, or URL
            prompt: Text prompt for analysis

        Returns:
            Analysis result as string
        """
        model = self.create_model()

        # Create message content
        content = [{"type": "text", "text": prompt}, self._image_to_content(image)]

        # Create message and invoke model
        message = HumanMessage(content=content)
        return model.invoke([message])

    def predict_bb_and_depth(self, image: Union[str, Image.Image], results_dir=None) -> str:
        """
        Predict high-level actions based on image and mission.

        Args:
            image: Current camera frame (PIL Image, file path, or URL)
            mission: the mission yaml file)

        Returns:
            List of high-level action commands
        """
        prompt = generate_prompt(prompt_type="bb_and_depth_prediction",
                                 mission=self.mission["mission"])

        response = self.analyze_image(image, prompt.strip())

        # Validate and clean response
        response = self.validate_response(response)

        if results_dir is not None:
            self.save_prompt_to_disk(file_name=Path(results_dir)/"bb_raw_response.txt", prompt_text=response)
        bb, depth, label = self.parse_bb_depth_info(response)   
        assert len(bb) == 4, "Bounding box should have 4 coordinates"
        print(f"Predicted Bounding Box: {bb}, Depth: {depth}, Label: {label}")
        return {"bb": bb, "depth": depth, "label": label}
    
    def parse_bb_depth_info(self, response: str) -> tuple[list[int], float, str]:
        """Parse single bounding box and depth from model output.

        Expected format:
        <bounding_boxes_and_depth>
        [400, 450, 600, 550, 10.0, 'car']
        </bounding_boxes_and_depth>
        
        Returns:
            Tuple of (bbox_coords, depth, label)
        """
 
        # Validate and clean response
        response = self.validate_response(response)

        # Extract inner content if wrapped in tags
        xml_match = re.search(r'<bounding_boxes_and_depth>\s*(.*?)\s*</bounding_boxes_and_depth>', response, re.DOTALL)
        content = xml_match.group(1) if xml_match else response

        # Try structured parsing via literal_eval
        try:
            data = ast.literal_eval(content.strip())
            if isinstance(data, (list, tuple)) and len(data) >= 6:
                bbox_coords = list(map(int, data[:4]))
                depth = float(data[4])
                label = str(data[5]).strip().strip('\'"')
                return (bbox_coords, depth, label)
        except Exception:
            pass
        
        # Fallback to regex parsing
        match = re.search(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\d.]+,\s*[\'\"]?[\w\s]+[\'\"]?)\]', content)
        if match:
            parts = match.group(1).split(',')
            bbox_coords = list(map(int, parts[:4]))
            depth = float(parts[4].strip())
            label = parts[5].strip().strip('\'"')
            return (bbox_coords, depth, label)
        
        return ([], 0.0, "")
    
    
    def draw_bounding_boxes(self, image, bounding_box_data, save=False):
        """Draw a single bounding box with depth label on image.
        
        Args:
            image: PIL Image
            bounding_box_data: Tuple of (bbox_coords, depth, label)
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = np.array(image)
        bounding_box, depth, label = bounding_box_data

        # Normalize the bounding box coordinates.
        width, height = image.shape[1], image.shape[0]
        xmin, ymin, xmax, ymax = bounding_box
        x1 = int(xmin / 1000 * width)
        y1 = int(ymin / 1000 * height)
        x2 = int(xmax / 1000 * width)
        y2 = int(ymax / 1000 * height)

        # Generate color for label
        color = np.random.randint(0, 256, (3,)).tolist()

        # Create label with depth information
        label_text = f"{label} ({depth}m)"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        box_thickness = 2
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]

        text_bg_x1 = x1
        text_bg_y1 = y1 - text_size[1] - 5
        text_bg_x2 = x1 + text_size[0] + 8
        text_bg_y2 = y1

        cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        cv2.putText(image, label_text, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

        image = Image.fromarray(image)
        return image
    
class ActionPredictor(VisionChain):
    def __init__(
        self,
        model_name="gpt-5o-nano",
        mission: dict = None,
    ):
        super().__init__(mission=mission)
        self.set_model_name(model_name)

    def analyze_image(
        self,
        image: Union[str, Image.Image, list],
        prompt: str,
        system_prompt_type: str = None,
    ) -> str:
        """
        Analyze one or more images with the given prompt.

        Args:
            image: Single image (PIL Image, file path, or URL) or list of images
            prompt: Text prompt for analysis

        Returns:
            Analysis result as string
        """
        model = self.create_model()

        # Create message content starting with prompt
        content = [{"type": "text", "text": prompt}]
        
        # Add image(s) to content
        if isinstance(image, list):
            for img in image:
                content.append(self._image_to_content(img))
        else:
            content.append(self._image_to_content(image))


        # Create message list and invoke model
        messages = [HumanMessage(content=content)]
        if isinstance(system_prompt_type, str):
            system_prompt = generate_prompt(prompt_type=system_prompt_type, mission=self.mission["mission"])
            messages.insert(0, SystemMessage(content=system_prompt.strip()))

        response = model.invoke(messages)

        # Fallback: if content is empty but reasoning is present, use reasoning
        if hasattr(response, "content") and not response.content and hasattr(response, "additional_kwargs"):
             reasoning = response.additional_kwargs.get("reasoning_content", "")
             if reasoning:
                 print(f"[Info] analyze_image: Content empty, using reasoning_content ({len(reasoning)} chars)")
                 return reasoning

        return response.content


    def predict_actions(self, image: Union[str, Image.Image, list], mission: dict, prompt_type="basic", save_prompt_dir: str = None, step_index: int = 0, system_prompt_type=None) -> str:
        """
        Predict high-level actions based on image(s) and mission.

        Args:
            image: Current camera frame (PIL Image, file path, or URL) or list of images
            mission: the mission yaml file
            prompt_type: Type of prompt to generate
            save_prompt_dir: Optional directory to save prompt text file
            step_index: Step index for filename when saving prompt

        Returns:
            List of high-level action commands
        """
        # Ensure model_name is available for prompt generation logic
        if hasattr(self, "model_name"):
            mission["model_name"] = self.model_name

        if prompt_type in ["basic"]:
            prompt = generate_prompt(prompt_type=prompt_type,
                                    mission=mission, high_level=True)
        else:
            prompt = generate_prompt(prompt_type=prompt_type,
                                     mission=mission)

        # Save prompt to disk if directory is provided
        if save_prompt_dir:
            self._save_prompt_to_disk(save_prompt_dir, prompt_type, step_index, prompt)
        response_model = self.analyze_image_full(image, prompt.strip(), system_prompt_type=system_prompt_type)
        response = response_model.content
        
        # Extract Dictionary/Reasoning if available
        reasoning_content = response_model.additional_kwargs.get("reasoning_content", "")
        
        # If content is empty but reasoning_content is present, use reasoning as content
        # This handles cases where llama-server puts everything in reasoning_content
        if not response and reasoning_content:
            response = reasoning_content
            print(f"[Info] Content empty, using reasoning_content ({len(response)} chars) instead.")

        # --- DEBUG LOGGING ---
        if save_prompt_dir:
            try:
                response_log_path = Path(save_prompt_dir) / "debug_outputs" / f"step_{step_index}_{prompt_type}_raw_response.txt"
                response_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(response_log_path, "w", encoding="utf-8") as f:
                    f.write(f"--- Full Model Response ({len(response)} chars) ---\n")
                    f.write(f"Metadata:\n{response_model.response_metadata}\n")
                    f.write(f"Full Response Object:\n{response_model}\n")
                    if reasoning_content:
                         f.write(f"\n--- Reasoning Process ---\n{reasoning_content}\n")
                    f.write(f"\n--- Response Content ---\n{response}\n")
            except Exception as e:
                print(f"[Warning] Could not save raw response to file: {e}")
        # ---------------------

        num_tries = 3

        # Validate and clean response
        response = self.validate_response(response)

        if response is None or response.strip() == "":
            # retry a few timesstep_by_step_system_prompt
            for _ in range(num_tries):
                response = self.analyze_image(image, prompt.strip(), system_prompt_type=system_prompt_type)
                
                # --- DEBUG LOGGING (RETRY) ---
                if save_prompt_dir:
                    try:
                        retry_log_path = Path(save_prompt_dir) / "debug_outputs" / f"step_{step_index}_{prompt_type}_raw_response_retry.txt"
                        retry_log_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(retry_log_path, "a", encoding="utf-8") as f:
                            f.write(f"\n--- RETRY RESPONSE ({len(response) if response else 0} chars) ---\n")
                            f.write(f"Metadata:\n{response_model.response_metadata}\n")
                            f.write(f"Full Response Object:\n{response_model}\n")
                            f.write(f"Content:\n{response}\n")
                    except Exception as e:
                        print(f"[Warning] Could not save retry response to file: {e}")
                # -----------------------------

                if response and response.strip() != "":
                    break
        return response


    def analyze_scene(self, image: Union[str, Image.Image, list]) -> str:
        """
        Analyze the scene to understand objects and terrain.

        Args:
            image: Current camera frame (single image or list of images)

        Returns:
            Scene analysis description
        """
        prompt = """
        Analyze this drone camera view and describe:
        1. What objects/landmarks do you see?
        2. What is the terrain like?
        3. Are there any obstacles?
        4. What's the overall environment (urban, nature, indoor, etc.)?

        Be concise and focus on information useful for drone navigation.
        """

        return self.analyze_image(image, prompt.strip())


class WaypointGenerator(VisionChain):
    """
    Low-Level Waypoint Generator for Two-Stage Planning.

    This class converts high-level actions into executable waypoints.
    Each waypoint is a relative movement command: {dx, dy, dz, dyaw}

    Example:
        High-level: "turn right 30 degrees"
        Waypoints: [{"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 30.0}]

        High-level: "go straight 10 meters"
        Waypoints: [
            {"dx": 2.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0},
            {"dx": 2.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0},
            ...  # 5 waypoints of 2m each
        ]
    """

    def __init__(self, model_name="gpt-4o-mini", mission: dict = None):
        super().__init__()
        self.set_model_name(model_name)
        self.mission = mission

    def analyze_image(self, image,  prompt: str, image2=None) -> str:
        """
        Analyze a single image with the given prompt.

        Args:
            image: PIL Image, file path, or URL
            prompt: Text prompt for analysis

        Returns:
            Analysis result as string
        """
        model = self.create_model()
        # Create message content
        content = [{"type": "text", "text": prompt}, self._image_to_content(image)]
        if image2:          
            content.append(self._image_to_content(image2))
        # Create message and invoke model
        message = HumanMessage(content=content)
        response = model.invoke([message])

        return response.content

    def predict_actions(self, image: Union[str, Image.Image], instruction: str) -> str:
        """
        Predict actions - for WaypointGenerator, this is not the primary method.
        Use generate_waypoints() instead.

        This method exists to satisfy the ABC interface.
        """
        # This is just a stub to satisfy abstract method requirement
        # The real functionality is in generate_waypoints()
        raise NotImplementedError("Use generate_waypoints() for waypoint generation")

    def generate_waypoints(self, mission: dict, high_level_action: str, save_prompt_dir: str = None, prompt_type: str = "basic", step_index: int = 0) -> str:
        """
        Generate low-level waypoints from a high-level action.

        Args:
            image: Current camera frame (PIL Image, file path, or URL)
            drone_pose: Current drone state
                Format: {"x": float, "y": float, "z": float, "roll": float, "pitch": float, "yaw": float}
            high_level_action: Single high-level action to execute (e.g., "turn right 30 degrees")
            spatial_context: Optional spatial reasoning results from pix2world (e.g., distance and world coordinates)

        Returns:
            Text response with waypoints in structured format
        """
        prompt = generate_prompt(prompt_type="basic",
                                 mission=mission, high_level=False, high_level_action=high_level_action)
        # Save prompt to disk if directory is provided
        if save_prompt_dir:
            self._save_prompt_to_disk(save_prompt_dir, prompt_type, step_index, prompt)
        return self.analyze_image(mission["image"], prompt.strip())

    def refine_waypoints(self, mission: dict, stepped_img_path: str) -> str:
        """
        Refine raw waypoint response to ensure structured format.

        Args:
            raw_response: Raw text response from the model

        Returns:
            Refined text response with waypoints in structured format
        """
        prompt = generate_prompt(prompt_type="refine",
                                 mission=mission)

        return self.analyze_image(mission["image"], prompt.strip(), stepped_img_path)