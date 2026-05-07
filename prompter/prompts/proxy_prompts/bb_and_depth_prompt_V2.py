import yaml
def bb_and_depth_prediction_prompt_V2(
    mission: dict
) -> str:
    """
    Generate a prompt for bounding box and depth prediction tasks.

    Args:
        mission (dict): Mission details including instruction and camera specs.

    Returns:
        str: Generated prompt text.
    """
    # Extract target object from mission if available
    target_object = mission.get("target_object", "target object")
    instruction = mission.get("instruction", "complete the mission")
    
    context = f"""You are in command of a UAV equipped with a camera of resolution {mission["camera"]["width"]}x{mission["camera"]["height"]} and a horizontal field of view of {mission["camera"]["fov"]} degrees.
    The camera has a pitch of {mission["camera"].get("pitch", -45)} degrees. So the images captured by the camera resemble a bird's eye view looking downward.
    The mission instruction is: {instruction}
    
    The PRIMARY TARGET OBJECT to locate is: {target_object}
    
    You need to identify and locate ONLY the target object in the image by predicting its bounding box and estimating its depth from the drone.
    """

    return f"""
    <Context>
    {context}
    </Context>

    <Objective>
    Your goal is to analyse the current image and find ONLY the TARGET OBJECT: {target_object}
    
    **CRITICAL - CORRECT TARGET IDENTIFICATION**:
    - You MUST find: {target_object}
    - Read the mission instruction carefully: "{instruction}"
    - Identify what "{target_object}" looks like and find it in the image
    - Do NOT confuse it with other objects in the scene (other buildings, water bodies, vehicles, etc.)
    
    Draw a bounding box ONLY around the target object: {target_object}
    </Objective>

    <Instructions>
    1. Read the mission instruction: "{instruction}"
    2. Understand what "{target_object}" is - what does it look like from a bird's eye view?
    3. Scan the image carefully to find the target object
    4. IMPORTANT: Do NOT annotate other objects that are NOT the target
       - If target is furniture, don't annotate pools/buildings
       - If target is a vehicle, don't annotate buildings/trees
       - If target is a landing pad, don't annotate roads/parking lots
    5. The camera is pitched down at {mission["camera"].get("pitch", -45)} degrees, so objects appear from above
    6. Estimate depth (distance from drone) considering:
       - Objects appearing smaller are farther away
       - Objects in the lower part of the image are generally closer
       - Use scene context to gauge scale
    7. Return bounding box coordinates as integers normalized to range (0-1000)
    8. Return depth in meters as a float rounded to two decimal places
    </Instructions>

    <Output Format>
    Only provide the output in the following format:
        <reasoning>
        I am looking for "{target_object}" based on the mission: "{instruction}"
        Describe what the target object looks like and where you found it in the image.
        Explain how you distinguished it from other objects in the scene.
        </reasoning>
        <bounding_boxes_and_depth>
        [xmin, ymin, xmax, ymax, depth, '{target_object}']
        </bounding_boxes_and_depth>
        
    Example:
        <reasoning>
        The mission is to reach the {target_object}. I identified it in the image as [describe appearance]. 
        It is located in the [position] portion of the image. I estimated the depth to be approximately X meters 
        based on its apparent size relative to other objects.
        </reasoning>
        
        <bounding_boxes_and_depth>
        [xmin, ymin, xmax, ymax, depth, '{target_object}']
        </bounding_boxes_and_depth>
        
    </Output Format>
    """
    
if __name__ == "__main__":
    # Example usage
    # load mission file from yaml file
    with open("/home/nava/uav_mission_planning/UAVMissionPlanning/dataset/Manipulation/CityEnv/drone_landing_pad_mission4/mission_config.yaml", "r") as file:
        mission_example = yaml.safe_load(file)
    prompt = bb_and_depth_prediction_prompt(mission_example)
    print(prompt)