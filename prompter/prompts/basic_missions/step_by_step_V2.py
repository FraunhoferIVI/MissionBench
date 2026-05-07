from typing import Dict, List
from .prompt_utils import yaw_to_facing_direction

def step_by_step_prompt_V2(
    mission: dict
) -> str:
    mission_benchmark = mission["mission"]
    tolerances = mission.get("tolerances", {})
    max_step_size = tolerances.get("MAX_STEP_SIZE", mission_benchmark.get("MAX_STEP_SIZE", 5))
    max_actions = tolerances.get("MAX_ACTIONS", "N/A")
    current_action_index = mission.get("current_action_index", 0)
    action_history = mission.get("parsed_actions")
    # Calculate current drone position from start + accumulated waypoints
    drone_pose = mission.get("drone_pose", {})
    current_x = drone_pose.get("x", 0.0)
    current_y = drone_pose.get("y", 0.0)
    current_z = drone_pose.get("z", 0.0)
    current_yaw = drone_pose.get("yaw", 0.0)
    facing_direction = yaw_to_facing_direction.get(int(current_yaw), "")
    should_continue_response = mission.get("should_continue_response")
    extra_continue_info = ""
    if should_continue_response is not None:
        extra_continue_info = f"\n    Previous navigation assessment: {should_continue_response}\n    "
    if facing_direction != "":
        facing_direction = f"You're are currently facing the {facing_direction}."
    
    context = f"""You are in command of a UAV equipped with a camera of resolution {mission_benchmark["camera"]["width"]}x{mission_benchmark["camera"]["height"]} and a horizontal field of view of {mission_benchmark["camera"]["fov"]} degrees.
    The camera has a pitch of {mission_benchmark["camera"].get("pitch", -45)} degrees. So the images captured by the camera resemble a bird's eye view.
    The drone is currently at position (x={current_x:.1f}, y={current_y:.1f}, z={current_z:.1f}) with yaw={current_yaw:.1f}.
    {facing_direction}. 
    Your mission is: {mission_benchmark["instruction"]}
    Mission constraints: maximum single-step movement {max_step_size} meters; action budget {max_actions} (current count {current_action_index}).
    """
    if "spatial_results" in mission:
        context = f"""You are in command of a UAV equipped with a camera of resolution {mission_benchmark["camera"]["width"]}x{mission_benchmark["camera"]["height"]} and a horizontal field of view of {mission_benchmark["camera"]["fov"]} degrees.
    You are provided with the following information from a spatial reasoning tool that computes the real-world distance information to the target object from the drone's current position:
    {mission["spatial_results"]}"""
    
    if extra_continue_info != "":
        context += extra_continue_info

    context += f"""
    The distance provided is the euclidean distance in meters from the drone to the target object in 3D space.
    Use the coordinates of the target point and the drone current position to plan the mission.
    
    CRITICAL - Action History and Visual Continuity:
    You are provided with the history of actions taken so far: {action_history}
    You are also provided with the last images captured from the simulator corresponding to the last actions taken. Each action was applied to the simulator, 
    the drone moved to the new position, and a new observation is fetched.
    
    ALWAYS review the action history and the latest image to generate your next action:
    1. Understand how you arrived at the current state
    2. Identify if the target has drifted out of view or if you've lost track of it
    3. Make corrections based on past movements (e.g., if you turned left and lost the target, turn right to recover it)
    4. Avoid repeating unsuccessful movement patterns
    5. Learn from previous actions to make better-informed next steps
    6. Given the spatial results information, use it to inform your next action precisely.
    7. The spatial results provide critical information about the distance to the target object, use this to plan your movements accurately.
    8. It is not necessary to always use the max_step_size, use smaller steps when precision is needed.
    
    If the target is no longer visible in the current image but was visible in previous images:
    - Review the action history to determine which movement caused you to lose sight
    - Plan corrective actions to return to a position where the target was last clearly visible
    - Use reverse movements intelligently (e.g., if last 3 actions were "forward", consider "backwards" to recover view)
    - If the drone heading is not aligned with the target, consider slight turning and then strafing to directly look at the target.
    NEVER continue moving forward blindly if the target is lost - always prioritize regaining visual on the target before proceeding.
    """
    # Get target object name
    target_object = mission_benchmark.get("target_object", "target")
    
    return f"""
    <Context>
    {context}
    </Context>

    <Objective>
    Navigate the drone to reach the target: **{target_object}**
    
    Use the spatial_results distance information to decide your action.
    </Objective>

    <Instructions>
    Check the spatial_results for distance to target.
    
    **APPROACH STRATEGY - Mix forward and descend:**
    
    1. **FAR from target** (horizontal > 10m):
       - Mostly FLY FORWARD to close distance
       - Occasional small descend (every 2-3 forwards)
    
    2. **MEDIUM distance** (horizontal 5-10m):
       - Mix FORWARD and DESCEND actions
       - Alternate: forward, forward, descend, forward, descend...
    
    3. **CLOSE to target** (horizontal < 5m):
       - Mostly DESCEND to reach the target
       - Small forward adjustments if needed
    
    **KEY POINTS**:
    - The target ({target_object}) is on the GROUND
    - You need BOTH horizontal movement AND descent
    - Don't just fly forward forever - mix in descends!
    - Don't just descend without getting close - fly forward too!
    
    Action history: {action_history}
    
    Look at your action history - if you've done 2-3 forwards, consider a descend next.
    If you've done 2-3 descends, consider forward next (unless very close).
    
    **Output Format**:
    <Reasoning>Horizontal distance: Xm. Recent actions: [list]. Based on distance and history, next action should be [forward/descend].</Reasoning>
    <Action>[action] [distance]</Action>
    
    Actions: forward, backwards, descend, ascend, strafe left, strafe right, turn left, turn right
    Max step: {max_step_size}m. Budget: {max_actions} actions (current: {current_action_index}).
    
    Examples:
    <Reasoning>Horizontal 15m, far away. Last actions: forward, forward. Continuing forward to close distance.</Reasoning>
    <Action>forward {max_step_size} meters</Action>
    
    <Reasoning>Horizontal 8m, medium distance. Last actions: forward, forward, forward. Time to descend while approaching.</Reasoning>
    <Action>descend {max_step_size} meters</Action>
    
    <Reasoning>Horizontal 3m, close to target. Need to descend to reach it.</Reasoning>
    <Action>descend {max_step_size} meters</Action>
    </Instructions>
    """


def should_continue_navigation_prompt(
    mission: dict
) -> str:
    """Prompt to determine if the mission is complete based on current action and image analysis."""
    mission_benchmark = mission["mission"]
    action_history = mission.get("parsed_actions", [])
    tolerances = mission.get("tolerances", {})
    max_actions = tolerances.get("MAX_ACTIONS", "N/A")
    
    current_action_index = mission.get("current_action_index", 0)
    
    # Calculate current drone position from start + accumulated waypoints
    start_pose = mission_benchmark.get("drone_start_pose")
    all_waypoints = mission.get("all_parsed_waypoints", [])
    
    current_x = start_pose["x"] + sum(wp.get("dx", 0) for wp in all_waypoints)
    current_y = start_pose["y"] + sum(wp.get("dy", 0) for wp in all_waypoints)
    current_z = start_pose["z"] + sum(wp.get("dz", 0) for wp in all_waypoints)
    
    # Get target object name
    target_object = mission_benchmark.get("target_object", "target")
    
    # Calculate remaining actions
    remaining_actions = max_actions - current_action_index if isinstance(max_actions, int) else "N/A"
    
    context = f"""You are analyzing whether a UAV mission has been completed.
    The drone started at position (x={start_pose["x"]:.1f}, y={start_pose["y"]:.1f}, z={start_pose["z"]:.1f}).
    The drone is currently at position (x={current_x:.1f}, y={current_y:.1f}, z={current_z:.1f}).
    The target object is: {target_object}
    The mission instruction was: {mission_benchmark["instruction"]}
    Actions executed so far: {action_history}
    
    Mission Constraints:
    - Maximum allowed actions: {max_actions}
    - Current action count: {current_action_index}
    - Remaining actions: {remaining_actions}
    """
    
    return f"""
    <Context>
    {context}
    </Context>

    <Objective>
    Look at the current image and determine: Have you REACHED the target ({target_object})?
    </Objective>

    <Instructions>
    **LOOK AT THE IMAGE CAREFULLY!**
    
    Based on the current image and the mission objective, assess whether:
    1. The mission objective has been achieved (target inspected, location reached, etc.)
    2. The drone is in the correct position to fulfill the mission requirement
    3. No further navigation is needed
    4. Check if action budget ({max_actions} actions) has been exceeded
    
    Ask yourself these questions:
    1. Can I see the {target_object} in the image?
    2. How LARGE does the {target_object} appear? (Small = far, Large = close)
    3. Does the {target_object} fill a significant portion of the image?
    4. Am I close enough to complete the mission (delivery/inspection)?
    
    **For visual inspection missions (reading license plates, identifying text, etc.):**
    - Examine the target object closely. Can you clearly see and read the required details?
    - The text/details MUST be clearly readable - not blurry or too small
    - If YES and readable: Provide the information (e.g., "license plate: ABC123") in your reasoning
    - If NO or unclear: Respond with "continue"
    
    **OUTPUT "navigation done" if:**
    - The {target_object} appears LARGE in the image (fills >25% of frame)
    - You are clearly close to the {target_object}
    - You can see the {target_object} in detail
    - The {target_object} is directly below or in front of you at close range
    - For inspection missions: you can clearly read/see the required details
    - Action budget exceeded ({current_action_index} >= {max_actions})
    
    **OUTPUT "continue" if:**
    - The {target_object} appears SMALL in the image (far away)
    - You need to get closer to the {target_object}
    - The {target_object} is not yet in close range
    - For inspection missions: text/details are not readable yet
    
    **IMPORTANT**: 
    - Base your decision on the IMAGE, not just coordinates
    - If the target looks close/large in the image = STOP
    - If the target looks small/distant in the image = CONTINUE
    
    Output Format (STRICT):
    <Reasoning>I can see the {target_object}. It appears [large/small] in the image, filling approximately [X]% of the frame. [For inspection: I can/cannot read the details.] [I am close enough / I need to get closer].</Reasoning>
    <Action>navigation done</Action>
    OR
    <Reasoning>Explain why more navigation is needed: target appears too small/distant, text not readable, etc.</Reasoning>
    <Action>continue</Action>
    
    Be strict: only output "navigation done" when visual evidence clearly satisfies the mission objective or the action budget is exhausted.
    </Instructions>
    """
