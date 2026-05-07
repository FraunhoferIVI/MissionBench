import yaml
def bb_and_depth_prediction_prompt(
    mission: dict
) -> str:
    """
    Generate a prompt for bounding box and depth prediction tasks.

    Args:
        mission (dict): Mission details including instruction and camera specs.
        image_path (str): Path to the current image.
        stepped_img_path (str): Path to save debug image with annotations.

    Returns:
        str: Generated prompt text.
    """
    context = f"""You are in command of a UAV equipped with a camera of resolution {mission["camera"]["width"]}x{mission["camera"]["height"]} and a horizontal field of view of {mission["camera"]["fov"]} degrees.
    The camera has a pitch of {mission["camera"].get("pitch", -45)} degrees. So the images captured by the camera resemble a bird's eye view.

    The ultimate mission is: {mission["instruction"]}, for this purpose you need to identify and locate target objects in the image by predicting their bounding boxes and estimating their depth from the drone.
    """

    return f"""
    <Context>
    {context}
    </Context>

    <Objective>
    Your goal is to analyse the current image and determine the bounding boxes around the target objects as well as estimate their depth from the drone.
    Output only the bounding box coordinates in the format (x_min, y_min, x_max, y_max) along with the estimated depth in meters for each detected object.
    </Objective>

    <Instructions>
    Analyse the image and the target objects and the required mission. Orientation and positioning of objects should be inferred from the image.
    Return bounding boxes for the main object relevant to the mission.
    Provide the bounding box coordinate as integers normalised in the range (0-1000) and depth in meters as a float rounded to two decimal places.
    </Instructions>

    <Output Format>
    Only provid the output in the following format:
        <reasoning>
        Provide your reasoning here if necessary.
        </reasoning>
        <bounding_boxes_and_depth>
        " a list. \n [xmin, ymin, xmax, ymax, depth, object_name].
        </bounding_boxes_and_depth>
    e.g.,
        <reasoning>
        The car is located near the center of the image, slightly towards the bottom right. The tree is on the left side of the image.
        </reasoning>
        
        <bounding_boxes_and_depth>
        [400, 450, 600, 550, 10.0, 'car']
        </bounding_boxes_and_depth>
        
    </Output Format>
    """
    
if __name__ == "__main__":
    # Example usage
    # load mission file from yaml file
    with open("/home/nava/uav_mission_planning/UAVMissionPlanning/dataset/Visual_Inspection/AirSimNHEnv/red_car_mission1/mission_config.yaml", "r") as file:
        mission_example = yaml.safe_load(file)
    prompt = bb_and_depth_prediction_prompt(mission_example)
    print(prompt)