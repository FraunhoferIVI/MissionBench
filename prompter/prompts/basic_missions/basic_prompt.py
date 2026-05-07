from typing import Dict, List
from .prompt_utils import yaw_to_facing_direction

def basic_prompt(
    mission: dict,
    high_level: bool = True,
    high_level_action: str = None,
):      
    if high_level:
        return basic_prompt_high_level(mission)
    else:
        return basic_prompt_low_level(mission, high_level_action)
        
def basic_prompt_high_level(
    mission: dict
) -> str:
    mission_benchmark = mission["mission"]
    yaw = mission_benchmark.get("drone_start_pose").get("yaw")
    facing_direction = yaw_to_facing_direction.get(yaw, "")
    if facing_direction != "":
        facing_direction = f"You're are currently facing the {facing_direction}."
    # print(f"{facing_direction = }")
    if "spatial_reasoning" in mission:
        context = f"""You are in command of a UAV equipped with a camera of resolution {mission_benchmark["camera"]["width"]}x{mission_benchmark["camera"]["height"]} and a horizontal field of view of {mission_benchmark["camera"]["fov"]} degrees.
    The drone is currently at position (x={mission_benchmark["drone_start_pose"]["x"]:.1f}, y={mission_benchmark["drone_start_pose"]["y"]:.1f}, z={mission_benchmark["drone_start_pose"]["z"]:.1f}) with yaw={mission_benchmark["drone_start_pose"]["yaw"]:.1f}.
    {facing_direction}
    Your mission is: {mission_benchmark["instruction"]}
    You are also provided with the following information from a spatial reasoning tool that computes the real-world distance and angle to the target object from the drone's current position:
    {mission["spatial_results"]} \n {mission["spatial_reasoning"]}
    The distance provided is the euclidean distance in meters from the drone to the target object in 3D space.
    Use the coordinates of the target point and the drone current position to plan the mission.
    """
    
    else:
        context = f"""You are in command of a UAV equipped with a camera of resolution {mission_benchmark["camera"]["width"]}x{mission_benchmark["camera"]["height"]} and a horizontal field of view of {mission_benchmark["camera"]["fov"]} degrees.
        The camera has a pitch of {mission_benchmark["camera"].get("pitch", -45)} degrees. So the images captured by the camera resemble a bird's eye view.
    The drone is currently at position (x={mission_benchmark["drone_start_pose"]["x"]:.1f}, y={mission_benchmark["drone_start_pose"]["y"]:.1f}, z={mission_benchmark["drone_start_pose"]["z"]:.1f}) with yaw={mission_benchmark["drone_start_pose"]["yaw"]:.1f}.
    {facing_direction}. 
    Your mission is: {mission_benchmark["instruction"]}
"""

    return f"""
    <Context>
    {context}
    </Context>

    <Objective>
    Your goal is to analyse the image and produce a high level action plan to accomplish the mission given the drone's current position and orientation.
    The action plan should consist of a sequence of high-level reasoning steps that will lead to the successful completion of the mission.
    Use the spatial reasoning information provided to inform your action plan. Include distance information where relevant to make it easier for the
    waypoint generator to produce accurate waypoints.
    </Objective>

    <Instructions>
    Analyse the image and the target objects and the required mission. Orientation and positioning of objects should be inferred from the image.
    Every action plan MUST INCLUDE approximate DISTANCES and ANGLES to accomplish the mission based on visual reasonnig and provided information.
    The action plan should be detailed enough to guide a low-level controller to execute the mission.
    Look carefully at the facing directions of the objects of interest, as well as the drone's facing direction.
    In this mission, the agent needs to precisely view the target object from the specified viewpoint.
    Provide a one sentence high-level action plan without any additional explanations or commentary.

    Then list the high-level actions as a numbered list, with one action per line.
    Hovering, capturing images, landing and taking off DO NOT need to be included in the action plan.
    Just focus on the navigation  actions required to complete the mission.
    You need to output an action plan to execute the full mission.
    The high-level actions should be concise and actionable.Choose actions such as forward, backwards, turn left, ascend, descend, turn right, 
    Important: when specifying distances/angles, use approximate values based on the image analysis and spatial reasoning information provided.
    The distances should be in meters and angles in degrees.
    Example format:

    Action Plan:
    1. Fly forward <approximate distance> until the target object is in view
    2. Turn right <approximate angle> until the target object is visible
    3. Descend <approximate distance> enough to view the target object
            
    For example if the drone is facing the positive x-direction and there is a car facing directly towards the drone and the mission is to inspect the 
        rear license plate, then the drone needs to move past the car <x> meters and then turn 180 degrees to face the rear of the car.

    ...

    </Instructions>
    """

def basic_prompt_low_level(
    mission: dict,
    high_level_action: str
) -> str:
    mission_benchmark = mission["mission"]
    try:
        yaw = mission.get("drone_current_pose").get("yaw")
        x = mission.get("drone_current_pose").get("x")
        y = mission.get("drone_current_pose").get("y")
        z = mission.get("drone_current_pose").get("z")
    except Exception:
        yaw = mission_benchmark.get("drone_start_pose").get("yaw")
        x = mission_benchmark["drone_start_pose"]["x"]
        y = mission_benchmark["drone_start_pose"]["y"]
        z = mission_benchmark["drone_start_pose"]["z"]
    facing_direction = yaw_to_facing_direction.get(yaw, "")
    if facing_direction != "":
        facing_direction = f"You're are currently facing the {facing_direction}."
    else:
        facing_direction = f"deduce the facing direction using the following information that maps yaw angle to facing direction: {str(yaw_to_facing_direction)} choose the closest one based on the current yaw of {yaw} degrees."
    return f"""<Context>
    You are in command of a UAV equipped with a camera of resolution {mission_benchmark["camera"]["width"]}x{mission_benchmark["camera"]["height"]} and a horizontal field of view of {mission_benchmark["camera"]["fov"]} degrees.
    The drone is currently at position (x={x:.1f}, y={y:.1f}, z={z:.1f}) with yaw={yaw:.1f}.
    Your mission is: {mission_benchmark["instruction"]}

    {facing_direction}
    </Context>

    <Objective>
    In order to accomplish the mission, you have been given the following high-level action to execute:
    "{high_level_action}"
    Your task is to convert this high-level action into precise low-level waypoint commands that the drone can follow.

    </Objective>

    <Instructions>

    Based on the image and current drone state, generate waypoints as RELATIVE movements.

    ...
    Guidelines:
    1. For each high-level action, generate ONLY ONE waypoint that accomplishes that action.
    2. Make use of the distance information provided in the high-level action to inform your waypoint generation.
    3. For turns: ONLY dyaw should be non-zero and a right turn is a clockwise rotation (postive dyaw) and left turn is an anticlockwise rotation (negative dyaw)
    4. For forward movements: ONLY dx should be non-zero
    5. Provide dz in NED coordinates (positive dz = move down/Descend) - so to move up/ascend, dz SHOULD BE NEGATIVE
    6. For lateral movements: ONLY dy should be non-zero, right side movement is positive dy and left side movement is negative dy
    7. Consider safety: avoid obstacles visible in the image
    8. Keep movements smooth and achievable
    9. THERE SHOULD BE ONE WAYPOINT PER HIGH-LEVEL ACTION.
    
    10. HEADING DIRECTION IS CRITICAL - All waypoints are RELATIVE to the drone's current facing direction:
        
        UNDERSTANDING RELATIVE MOVEMENT:
        - "Forward" = direction the drone is currently facing
        - "Backward" = opposite to facing direction
        - "Right" = perpendicular right from facing direction
        - "Left" = perpendicular left from facing direction
        
        The coordinate system (dx, dy) rotates with the drone's heading:
        - When facing POSITIVE X-direction (yaw=0°):
          * Forward → dx=positive, dy=0
          * Backward → dx=negative, dy=0
          * Right → dx=0, dy=positive
          * Left → dx=0, dy=negative
        
        - When facing POSITIVE Y-direction (yaw=90°):
          * Forward → dx=0, dy=positive
          * Backward → dx=0, dy=negative
          * Right → dx=negative, dy=0
          * Left → dx=positive, dy=0
        
        - When facing NEGATIVE X-direction (yaw=180°):
          * Forward → dx=negative, dy=0
          * Backward → dx=positive, dy=0
          * Right → dx=0, dy=negative
          * Left → dx=0, dy=positive
        
        - When facing NEGATIVE Y-direction (yaw=-90°):
          * Forward → dx=0, dy=negative
          * Backward → dx=0, dy=positive
          * Right → dx=positive, dy=0
          * Left → dx=negative, dy=0
        
        DETAILED EXAMPLES BY HEADING DIRECTION:
        
        Example Set A - Facing Positive X (yaw=0°):
        a1) Action: "move forward 10m" → dx=10, dy=0, dz=0, dyaw=0
        a2) Action: "move backward 5m" → dx=-5, dy=0, dz=0, dyaw=0
        a3) Action: "strafe right 3m" → dx=0, dy=3, dz=0, dyaw=0
        a4) Action: "strafe left 4m" → dx=0, dy=-4, dz=0, dyaw=0
        a5) Action: "turn right 90 deg" → dx=0, dy=0, dz=0, dyaw=90
        
        Example Set B - Facing Positive Y (yaw=90°):
        b1) Action: "move forward 10m" → dx=0, dy=10, dz=0, dyaw=0
        b2) Action: "move backward 5m" → dx=0, dy=-5, dz=0, dyaw=0
        b3) Action: "strafe right 3m" → dx=-3, dy=0, dz=0, dyaw=0
        b4) Action: "strafe left 4m" → dx=4, dy=0, dz=0, dyaw=0
        b5) Action: "turn left 45 deg" → dx=0, dy=0, dz=0, dyaw=-45
        
        Example Set C - Facing Negative X (yaw=180°):
        c1) Action: "move forward 10m" → dx=-10, dy=0, dz=0, dyaw=0
        c2) Action: "move backward 5m" → dx=5, dy=0, dz=0, dyaw=0
        c3) Action: "strafe right 3m" → dx=0, dy=-3, dz=0, dyaw=0
        c4) Action: "strafe left 4m" → dx=0, dy=4, dz=0, dyaw=0
        c5) Action: "turn right 90 deg" → dx=0, dy=0, dz=0, dyaw=90
        
        Example Set D - Facing Negative Y (yaw=-90°):
        d1) Action: "move forward 10m" → dx=0, dy=-10, dz=0, dyaw=0
        d2) Action: "move backward 5m" → dx=0, dy=5, dz=0, dyaw=0
        d3) Action: "strafe right 3m" → dx=3, dy=0, dz=0, dyaw=0
        d4) Action: "strafe left 4m" → dx=-4, dy=0, dz=0, dyaw=0
        d5) Action: "ascend 5m" → dx=0, dy=0, dz=-5, dyaw=0 (NED: negative dz = up)
        
        CRITICAL REMINDERS:
        ✓ ALWAYS check the current facing direction first
        ✓ "Forward" is ALWAYS in the direction the drone is facing, NOT global positive-x
        ✓ Right/Left are PERPENDICULAR to the facing direction
        ✓ After a turn, the facing direction changes for subsequent movements
        ✓ Vertical movements (dz) are independent of heading direction (NED coordinates)

    11. After every waypoint generation sequentially, update the heading direction and then think the next high level action with respect to the heading direction.
    Examples
    Output format (one waypoint per high-level action):
    1. dx=<forward_meters> dy=<right_meters> dz=<up_meters> dyaw=<rotation_degrees>
    2. dx=<forward_meters> dy=<right_meters> dz=<up_meters> dyaw=<rotation_degrees>
    ...

    Output ONLY the waypoints in the format shown above. No explanations.
    </Instructions>
    """


def refine_prompt_low_level(
    mission: dict
) -> str:
    mission_benchmark = mission["mission"]
    return f"""<Context>
    You are in command of a UAV equipped with a camera of resolution {mission_benchmark["camera"]["width"]}x{mission_benchmark["camera"]["height"]} and a horizontal field of view of {mission_benchmark["camera"]["fov"]} degrees.
    The drone is currently at position (x={mission_benchmark["drone_start_pose"]["x"]:.1f}, y={mission_benchmark["drone_start_pose"]["y"]:.1f}, z={mission_benchmark["drone_start_pose"]["z"]:.1f}) with yaw={mission_benchmark["drone_start_pose"]["yaw"]:.1f}.
    Your mission is: {mission_benchmark["instruction"]}
    </Context>

    <Objective>
    In order to accomplish the mission, you have already generated the following final waypoint: {mission["predicted_waypoints"]}.
    The first image shows the starting position and the second image shows the final position after executing the predicted waypoints.
    Your task is to refine this waypoint to ensure it is accurate and achievable for the drone to follow.
    </Objective>

    <Instructions>

    Just look at the previously generated waypoints and update it so that the mission is achievable, if the mission was \
    already achieved using the previous waypoints, just return the same waypoints.
    Analyse the before and after images to understand the drone's movement and positioning.
    If the target object is not properly aligned in the after image, adjust the waypoints accordingly to improve alignment.
    If the target object is not at all visible in the after image, adjust the waypoints significantly to bring the target object into view.
    If the target object is partially visible but not well-centered, make minor adjustments to center it better in the frame.
    If the target object is well-centered and clearly visible, return the same waypoints.
    ...
    Guidelines:
    1. For turns: ONLY dyaw should be non-zero
    2. For forward movements: ONLY dx should be non-zero
    3. Provide dz in NED coordinates (positive dz = move down) - so to move up, dz SHOULD BE NEGATIVE
       for example if the drone is at z = -5 and you want to move it a bit up for better alignment, the refined waypoint should have dz = -7 (to move up 2 meter to dz = -2)
    4. if the image is distorted or upside down, it is probably becuase dz is positive and it crashed the drone. Always keep the dz negative to be above ground.
    5. For lateral movements: ONLY dy should be non-zero
    6. Keep movements smooth and achievable
    Examples:

    Output format: {{'dx': <float>, 'dy': <float>, 'dz': <float>, 'dyaw': <float>}}
    ...

    Output the final refined waypoint in the format shown above.
    So modify the original waypoint for better alignment and return the updated waypoint, NOT RELATIVE WAYPOINT
    Provide a short summary of changes made to the waypoint and the corresponding reasons.

    Example ouput:
    "{{'dx': <>, 'dy': <>, 'dz': <>, 'dyaw': 0.0}}\n\nSummary of changes:\n- Switched to a pure lateral move left (dy = <>) to clear the utility pole that ..."
    </Instructions>
    """

def pixel_detection_prompt(
    mission: dict
) -> str:
    mission_benchmark = mission["mission"]
    return f"""<Context>
    You are in command of a UAV equipped with a camera of resolution {mission_benchmark["camera"]["width"]}x{mission_benchmark["camera"]["height"]} and a horizontal field of view of {mission_benchmark["camera"]["fov"]} degrees.
    The drone is currently at position (x={mission_benchmark["drone_start_pose"]["x"]:.1f}, y={mission_benchmark["drone_start_pose"]["y"]:.1f}, z={mission_benchmark["drone_start_pose"]["z"]:.1f}) with yaw={mission_benchmark["drone_start_pose"]["yaw"]:.1f}.
    Your mission is: {mission_benchmark["instruction"]}
    </Context>

    <Objective>
    Your goal is to analyse the image and produce the estimated pixel coordinates (x, y) of the target object required to accomplish the mission.
    Use the image and mission description to identify the target object and provide its pixel coordinates in the image.
    </Objective>

    <Instructions>
    Analyse the image and the target query, identify the target object and its estimated pixel position.
    Required Output format:
    (x_pixel, y_pixel)
    Provide ONLY the output in the format shown above. No explanations.
    </Instructions>
    """

def select_next_action(
    mission: dict,
    high_level_action: str,
    action_history: List[Dict[str, str]]
) -> str:
    mission_benchmark = mission["mission"]
    mission_benchmark = mission = mission["mission"]
    yaw = mission_benchmark.get("drone_start_pose").get("yaw")
    facing_direction = yaw_to_facing_direction.get(yaw, "")
    if facing_direction != "":
        facing_direction = f"You're are currently facing the {facing_direction}."
    # print(f"{facing_direction = }")
    return f"""<Context>
    You are in command of a UAV equipped with a camera of resolution {mission["camera"]["width"]}x{mission["camera"]["height"]} and a horizontal field of view of {mission["camera"]["fov"]} degrees.
    The drone is currently at position (x={mission["drone_start_pose"]["x"]:.1f}, y={mission["drone_start_pose"]["y"]:.1f}, z={mission["drone_start_pose"]["z"]:.1f}) with yaw={mission["drone_start_pose"]["yaw"]:.1f}.
    Your mission is: {mission["instruction"]}

    {facing_direction}
    </Context>

    <Objective>
    In order to accomplish the mission, you have been given the following high-level action to execute:
    "{high_level_action}"
    Your task is to select the next high level action to execute based on the history of executions {action_history}.
    The provided high-level plan is just a rough plan and we need to execute step by step and refine our high-level plan.
    Select the next action - based on the action history and provide a description of the next action to try, it can be a subwaypoint of a high level action.


    </Objective>

    <Instructions>

    Based on the image and current drone state, generate waypoints as RELATIVE movements.

    ...
    Guidelines:
    1. For the selected next action, generate a subwaypoint to try out.
    Examples
    Output format (one waypoint per high-level action):
    Move forward <> meters
    ...

    Output ONLY the action in the format shown above. No explanations.
    </Instructions>
    """

def select_next_action_prompt(
    mission: dict,
    high_level_action: str,
    action_history: List[Dict[str, str]],
    explanation_history: List[str]
) -> str:
    mission_benchmark = mission["mission"]
    yaw = mission.get("drone_current_pose").get("yaw")
    facing_direction = yaw_to_facing_direction.get(yaw, "")
    if facing_direction != "":
        facing_direction = f"You're are currently facing the {facing_direction}."
    # print(f"{facing_direction = }")
    x= mission["drone_current_pose"]["x"]
    y=mission["drone_current_pose"]["y"]
    z=mission["drone_current_pose"]["z"]
    yaw=mission["drone_current_pose"]["yaw"]
    drone_current_position = f"x = {x}, y={y}, z={z}) with yaw={yaw}"
    img_widht = mission_benchmark["camera"]["width"]
    img_height = mission_benchmark["camera"]["height"]
    fov = mission_benchmark["camera"]["fov"]
    return f"""<Context>
    You are in command of a UAV equipped with a camera of resolution {img_widht}x{img_height} and a horizontal field of view of {fov} degrees.
    The drone is currently at position {drone_current_position}.
    Your mission is: {mission["instruction"]}

    {facing_direction}
    </Context>

    <Objective>
    In order to accomplish the mission, you have been given the following high-level action to execute:
    "{high_level_action}"
    Your task is to select the next high level action to execute based on the history of executions {action_history} and the history of explanations {explanation_history}.
    The provided high-level plan is just a rough plan and we need to execute step by step and refine our high-level plan.
    Select the next action - based on the action history and provide a description of the next action to try, it can be a subwaypoint of a high level action.


    </Objective>

    <Instructions>

    Based on the image and current drone state, generate waypoints as RELATIVE movements.

    ...
    Guidelines:
    1. For the selected next action, generate a subwaypoint to try out.
    Examples
    Output format (one waypoint per high-level action):
    Move forward <> meters
    ...

    Output ONLY the action in the format shown above. No explanations.
    </Instructions>
    """

def should_continue_waypoint_generation_prompt(
    mission: dict,
    high_level_action: str,
    action_history: str,
    explanation_history: str
) -> str:
    mission_benchmark = mission["mission"]
    yaw = mission.get("drone_current_pose").get("yaw")
    facing_direction = yaw_to_facing_direction.get(yaw, "")
    print(f"{facing_direction = }")
    x=mission["drone_current_pose"]["x"]
    y=mission["drone_current_pose"]["y"]
    z=mission["drone_current_pose"]["z"]
    yaw=mission["drone_current_pose"]["yaw"]
    drone_current_position = f"x = {x}, y={y}, z={z}) with yaw={yaw}"
    img_widht = mission_benchmark["camera"]["width"]
    img_height = mission_benchmark["camera"]["height"]
    fov = mission_benchmark["camera"]["fov"]
    MAX_STEP_SIZE = mission["MAX_STEP_SIZE"]
    return f"""
    <Context>
    You are in command of a UAV equipped with a camera of resolution {img_widht}x{img_height} and a horizontal field of view of {fov} degrees.
    The drone is started at position {drone_current_position} and facing {facing_direction}
    The original mission was: {mission_benchmark["instruction"]}
    The following is the history of actions executed so far and the observations and explanations after each stepping.
    {action_history = }
    {explanation_history = }
    </Context>

    <Objective>
    In order to accomplish the mission, you have already executed the following actions {mission.get("executed_actions", [])}.
    The first image shows the starting position and the second image shows the final position after executing previous actions.
    Your task is to determine whether the mission has been accomplished or if there is some adjustment needed in the original high-level action plan {high_level_action}
    to execute the mission.
    If there needs to be further adjustments, provide the next action to execute to continue the mission.
    </Objective>

    <Instructions>
    1.  Always think directions with respect the current facing direction of the drone.
         BE CAREFUL ABOUT THE HEADING DIRECTION, consider the facing direction of the drone when generating the waypoints.
         for example, if the drone is facing the positive x-direction and the action is "fly forward 10 meters", then dx=10, dy=0, dz=0, dyaw=0
         but if the drone is facing the positive y-direction and the action is "fly forward 10 meters", then dx=0, dy=10, dz=0, dyaw=0

    2. EXACTLY follow the output format requested! The output must be on separate lines, numbered 1, 2, 3.
    3. The NEW ACTION MUST be a simple sentence like "move forward 35m", "turn 90 deg", "strafe right 3m", etc., WITHOUT DETAILED EXPLANATION
    
    4. ADAPTIVE STEP SIZE STRATEGY (CRITICAL):

        This section applied only during the finetuning stage, in the begning only the high level actions should be executed.
        When fine-tuning the drone's position to keep the target object in view and properly aligned, it is ESSENTIAL to adjust the step sizes based on the target's visibility and positioning:

             - Maximum movement in any direction: {MAX_STEP_SIZE}m (turns have no restrictions)
             - INITIAL SEARCH PHASE (target not visible): Take LARGER steps (0.75-1.0 × max = {MAX_STEP_SIZE * 0.75:.1f}m to {MAX_STEP_SIZE}m)
                 * Use longer movements to efficiently search for the target
                 * Example: "move forward {int(MAX_STEP_SIZE * 0.8)}m", "strafe right {int(MAX_STEP_SIZE * 0.9)}m", "ascend {int(MAX_STEP_SIZE)}m"
       
             - TARGET ACQUISITION PHASE (target becomes visible): Reduce to MEDIUM steps (0.4-0.6 × max = {MAX_STEP_SIZE * 0.4:.1f}m to {MAX_STEP_SIZE * 0.6:.1f}m)
                 * Target is visible but not well-positioned
                 * Example: "move forward {int(MAX_STEP_SIZE * 0.5)}m", "strafe left {int(MAX_STEP_SIZE * 0.6)}m", "descend {int(MAX_STEP_SIZE * 0.4)}m"
       
             - FINE-TUNING PHASE (target visible and close): Take VERY SMALL steps (0.15-0.25 × max = {MAX_STEP_SIZE * 0.15:.1f}m to {MAX_STEP_SIZE * 0.25:.1f}m)
                 * CRITICAL: Once target is clearly visible, use small movements to avoid overshooting
                 * Small steps prevent losing track of the target object
                 * Example: "move forward {int(MAX_STEP_SIZE * 0.2)}m", "strafe right {int(MAX_STEP_SIZE * 0.25)}m", "ascend {int(MAX_STEP_SIZE * 0.15)}m"
       
             STEP SIZE EXAMPLES WITH MAX={MAX_STEP_SIZE}m:
             ✓ Target not in view yet → "move forward {int(MAX_STEP_SIZE * 0.8)}m" (large step, 80% of max)
             ✓ Target just appeared in view → "move forward {int(MAX_STEP_SIZE * 0.5)}m" (medium step, 50% of max)
             ✓ Target visible, needs positioning → "strafe right {int(MAX_STEP_SIZE * 0.25)}m" (small step, 25% of max)
             ✓ Target well-centered, final touch → "move forward {int(MAX_STEP_SIZE * 0.2)}m" (very small step, 20% of max)
             ✗ WRONG: Target visible but "move forward {MAX_STEP_SIZE}m" (too large, will overshoot and lose track)
    
    5. Explanations should be kept short and precise (1-2 sentences max)
    6. <new_action> must be MAXIMUM 5 words, keep it short with no explanations
    7. Only ONE SINGLE HIGH LEVEL ACTION at a time in either direction or a turn
    
    8. VISUAL TRACKING IS CRITICAL - NEVER LOSE SIGHT OF THE TARGET:
       ⚠️ MOST IMPORTANT RULE: Once the target object becomes visible, you MUST keep it within the camera frame at all times
       
       - If target disappears after an action: IMMEDIATELY review the action history to find the last step where it was visible
       - Recovery strategy: Return to the previous known good position by reversing or adjusting the last action
       - Prevention: When target is visible, use VERY SMALL steps (refer to FINE-TUNING PHASE) to avoid overshooting
       
       EXAMPLES OF TARGET LOSS AND RECOVERY:
       
       Scenario 1 - Overshoot:
       * History: "move forward 8m" (target visible) → "move forward 8m" (target lost, moved too far past it)
       * Recovery: "move backward 5m" to re-acquire target
       * Explanation: "Overshot target, reversing to reacquire"
       
       Scenario 2 - Wrong direction:
       * History: "turn right 45 deg" (target visible) → "move forward 10m" (target lost, wrong angle)
       * Recovery: "turn left 45 deg" to undo the turn, then reassess
       * Explanation: "Lost target after turn, reversing rotation"
       
       Scenario 3 - Too aggressive lateral movement:
       * History: "strafe right 3m" (target centered) → "strafe right 6m" (target lost, moved too far)
       * Recovery: "strafe left 4m" to return near last good position
       * Explanation: "Target left frame, correcting lateral position"
       
       CRITICAL GUIDELINES FOR MAINTAINING VISUAL CONTACT:
       ✓ Once target appears, ALL subsequent actions must be conservative (small steps)
       ✓ Before each action, predict: "Will this keep the target in frame?"
       ✓ If unsure, take a smaller step first (e.g., 2m instead of 5m)
       ✓ After ANY action, check: "Is target still visible?" If not, immediately recover
    ✓ NEVER take large steps (>50% of MAX_STEP_SIZE) when target is already visible
       ✓ Always reference the action history to understand what caused target loss
       
       ✗ WRONG: Target visible → "move forward 10m" → Target lost → "move forward 10m again" (keeps getting lost!)
       ✓ CORRECT: Target visible → "move forward 2m" → Target still visible → "move forward 2m" → Perfect positioning
    
    9. LATERAL MOVEMENT FOR CENTERING - UNDERSTAND LEFT/RIGHT CORRECTLY:
       ⚠️ CRITICAL FAILURE MODE: Many agents confuse which direction to strafe when centering objects
       
       CORRECT LOGIC FOR CENTERING:
       - If target is on the LEFT side of frame → Strafe LEFT to center it (move drone toward the target)
       - If target is on the RIGHT side of frame → Strafe RIGHT to center it (move drone toward the target)
       - Think of it as: "Move the drone in the direction where the target appears"
       
       DETAILED EXAMPLES:
       
       ✗ WRONG - Common mistake:
       * Observation: "Car is on the left side of the frame"
       * Wrong action: "strafe right 2m" ← This moves drone AWAY from the car, making it more off-center!
       * Why wrong: Drone moves right, making the car appear even more to the left
       
       ✓ CORRECT:
       * Observation: "Car is on the left side of the frame"
       * Correct action: "strafe left 3m" ← This moves drone TOWARD the car, centering it
       * Why correct: Drone moves left, bringing the car toward center of frame
       
       ✗ WRONG - Another common mistake:
       * Observation: "License plate is on the right edge of the image"
       * Wrong action: "strafe left 2m" ← This moves drone away from target
       
       ✓ CORRECT:
       * Observation: "License plate is on the right edge of the image"
       * Correct action: "strafe right 2m" ← This moves drone toward the target
       
       MENTAL MODEL - CAMERA FRAME CENTERING:
       - Camera shows what's in front of the drone
       - If object appears LEFT in image → Object is physically to the LEFT of drone → Strafe LEFT to align
       - If object appears RIGHT in image → Object is physically to the RIGHT of drone → Strafe RIGHT to align
       - You're not pushing the object, you're moving the DRONE to change the viewing angle
       
       VERIFICATION QUESTION:
       Before strafing, ask yourself: "Am I moving the DRONE toward where the target appears in the frame?"
       If answer is NO, you're moving in the WRONG direction!
    
    10. if you are lost at any point, you can always check the history of actions taken and go back in history and start with different approaches.
    
    10. AVOID REDUNDANT MOVEMENTS - ANALYZE ACTION HISTORY CAREFULLY:
       ⚠️ CRITICAL: Before suggesting any new action, review the action_history to prevent wasteful back-and-forth movements
       
       - If the last action was "strafe right 3m", DO NOT immediately suggest "strafe left 3m" (this cancels out the movement)
       - If you moved forward, then backward, you've wasted actions and made no progress
       - Look for PATTERNS in the history: repeated oscillations indicate poor planning
       
       EXAMPLES OF REDUNDANT MOVES TO AVOID:
       
       ✗ WRONG - Redundant oscillation:
       * Action 1: "strafe right 5m"
       * Action 2: "strafe left 4m" ← This nearly cancels action 1!
       * Better: Commit to one direction or take smaller initial steps
       
       ✗ WRONG - Forward/backward cycling:
       * Action 1: "move forward 8m"
       * Action 2: "move backward 6m" ← Wasted 14m of movement for 2m net progress!
       * Better: Take smaller forward steps (e.g., 3m) when uncertain
       
       ✗ WRONG - Aimless wandering:
       * Action 1: "turn right 45 deg"
       * Action 2: "turn left 90 deg"
       * Action 3: "turn right 45 deg" ← Back to starting orientation!
       * Better: Plan rotation carefully based on target location
       
       GUIDELINES TO PREVENT REDUNDANCY:
       ✓ Before each action, check if it contradicts recent moves (last 2-3 actions)
       ✓ If you need to reverse direction, question WHY the previous action was wrong
       ✓ Commit to a strategy for 2-3 actions before changing approach drastically
       ✓ Small corrective moves are OK (e.g., "strafe right 1m" after "strafe right 5m")
       ✓ Large opposing moves indicate poor initial judgment - learn from this
       
       ✓ CORRECT - Progressive refinement:
       * Action 1: "move forward 8m" (target becomes visible)
       * Action 2: "move forward 2m" (getting closer, same direction)
       * Action 3: "strafe right 1m" (minor lateral adjustment)
       ← All actions build on each other, no wasted movement
    
    11. DO NOT OVER-OPTIMIZE - ACCEPT "GOOD ENOUGH" MISSION COMPLETION:
       ⚠️ IMPORTANT: Once the mission is ROUGHLY achieved, declare it DONE - do not endlessly fine-tune for perfect alignment
       
       - If the target object is clearly visible and identifiable in the frame, that's usually sufficient
       - If the inspection target (e.g., license plate, window, door) is legible and well-framed, the mission is complete
       - Spending 5+ actions on minor positioning adjustments wastes resources
       - Perfect center alignment is NOT required unless explicitly stated in the mission
       
       WHEN TO DECLARE MISSION COMPLETE:
       
       ✓ License plate inspection: Plate is visible, text is readable (even if slightly off-center) → DONE
       ✓ Window inspection: Window is in frame, details are clear → DONE
       ✓ Object identification: Object is recognizable and properly visible → DONE
       ✓ Viewpoint verification: Viewing from correct side/angle, object clearly visible → DONE
       
       ✗ DO NOT keep iterating for:
       - Pixel-perfect centering (unless mission explicitly requires "centered view")
       - Minor angle adjustments that don't improve visibility
       - Small distance tweaks when object is already clear
       - Aesthetic framing improvements
       
       EXAMPLES:
       
       Scenario 1 - License plate at 60% of frame width, clearly readable:
       ✓ CORRECT: "1. <DONE>\n2. License plate is clearly visible and readable\n3. None"
       ✗ WRONG: "1. <NOT DONE>\n2. Plate could be more centered\n3. strafe left 1m"
       
       Scenario 2 - Building entrance visible but slightly to the right:
       ✓ CORRECT: "1. <DONE>\n2. Entrance is clearly visible and identifiable\n3. None"
       ✗ WRONG: "1. <NOT DONE>\n2. Can center it better\n3. strafe left 2m"
       
       Scenario 3 - Car from correct side, 70% in frame:
       ✓ CORRECT: "1. <DONE>\n2. Vehicle inspection angle achieved, details clear\n3. None"
       ✗ WRONG: "1. <NOT DONE>\n2. Could fit entire car in frame\n3. move backward 3m"
       
       RULE OF THUMB:
       - If you can complete the mission objective (read, identify, inspect) with current view → Declare DONE
       - If you've made 3+ small adjustments and view keeps improving but slowly → Declare DONE (good enough)
       - Save MAX_ACTIONS budget for missions that truly need repositioning
    
    OUTPUT FORMAT (CRITICAL - Follow this exact structure):
    
    If the mission needs more actions to complete:
    1. <NOT DONE>
    2. <short explanation of what still needs to be done>
    3. <simple next action command>

    If the mission has been accomplished:
    1. <DONE>
    2. <short explanation of mission completion>
    3. None

    EXAMPLES:

    Example 1 - Mission not complete, need to continue forward:
    1. <NOT DONE>
    2. Target is visible but still too far away
    3. move forward 20m

    Example 2 - Mission not complete, need to adjust position:
    1. <NOT DONE>
    2. Target partially visible, need better angle
    3. turn right 45 deg

    Example 3 - Mission not complete, need altitude adjustment:
    1. <NOT DONE>
    2. Target obscured, need higher viewpoint
    3. ascend 10m

    Example 4 - Mission not complete, lateral movement needed:
    1. <NOT DONE>
    2. Object visible but off-center to left
    3. strafe right 5m

    Example 5 - Mission complete:
    1. <DONE>
    2. Target clearly visible at optimal viewing angle
    3. None

    Example 6 - Mission complete:
    1. <DONE>
    2. License plate inspection completed successfully
    3. None

    Example 7 - Mission not complete, combination needed:
    1. <NOT DONE>
    2. Need to circle around obstacle
    3. turn left 90 deg

    Example 8 - Mission not complete, approach target:
    1. <NOT DONE>
    2. Target identified, moving closer for inspection
    3. move forward 15m

    CRITICAL REMINDERS:
    - Output MUST be exactly 3 numbered lines
    - Line 1: ONLY "<DONE>" or "<NOT DONE>" with angle brackets
    - Line 2: Brief explanation (1-2 sentences)
    - Line 3: Simple action (max 5 words) or "None" if done
    - NO additional text, commentary, or formatting
    </Instructions>
    """


# if __name__ == "__main__":
    # basic_prompt = basic_prompt(
    #     mission="Inspect the front headlight of the red car.",
    #     target_object="front headlight of the red car",
    #     target_location="on the road near the swimming pool",
    #     camera_width=1920,
    #     camera_height=1080,
    #     fov_deg=90,
    #     drone_x=-0.0241438,
    #     drone_y=0.0450883,
    #     drone_z=4.31142,
    #     drone_yaw=0.004,
    #     facing_direction="positive x-direction"
    # )
    # print(basic_prompt)