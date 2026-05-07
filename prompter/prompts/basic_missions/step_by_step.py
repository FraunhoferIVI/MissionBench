from typing import Dict, List
from .prompt_utils import yaw_to_facing_direction

def step_by_step_prompt(
    mission: dict,
    next_conversation_request: bool = False
) -> str:
    mission_benchmark = mission["mission"]
    if next_conversation_request:
        prompt = """The suggested action was applied to a simulator and the resulting stepped image is here. Based on the current state and the previous conversation determine the next immediate action to take towards accomplishing the original mission given the drone's current position and orientation. Output only the next single action that should be executed now based on the current state and the previous conversation history."""
        return prompt
    tolerances = mission.get("tolerances", mission_benchmark.get("tolerances", {}))
    max_step_size = tolerances.get("MAX_STEP_SIZE")
    max_actions = tolerances.get("MAX_ACTIONS")
    assert isinstance(max_actions, (int, float)), "MAX_STEP_SIZE should be a number"
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
    target_object = mission_benchmark.get("metadata").get("target_object")
    context = f"""You are in command of a UAV equipped with a camera of resolution {mission_benchmark["camera"]["width"]}x{mission_benchmark["camera"]["height"]} and a horizontal field of view of {mission_benchmark["camera"]["fov"]} degrees.
    The camera has a pitch of {mission_benchmark["camera"].get("pitch", -45)} degrees. So the images captured by the camera resemble a bird's eye view, it is an egocentric view and the top of the image as "forward" for the drone.

    The drone is currently at position (x={current_x:.1f}, y={current_y:.1f}, z={current_z:.1f}) with yaw={current_yaw:.1f}.
    {facing_direction}. 
    Your mission is: {mission_benchmark["instruction"]}
    Mission constraints: maximum single-step movement {max_step_size} meters max_angle_step = 3 x {max_step_size}; action budget {max_actions} (current count {current_action_index}).
    <image1> shows the initial image at the start of the mission and <image2> shows the current image from the drone's camera after executing the following actions: {action_history}
    """
    if "spatial_results" in mission:
        context += f"""
    You are provided with the following information for <image1> from a spatial reasoning tool that computes the real-world depth information to the target object from the drone's current position:
    {mission["spatial_results"]}
    The provided depth is just a reference for the initial estimation of the euclidean distance to some random point of the target object, 
    this information does not get updated as the drone progresses, so just use it a reference and there could be deviations from the
    actual distance to the target object."""
    
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
    The PRIMARY TARGET OBJECT to locate is: {target_object}
    1. Understand how you arrived at the current state
    2. Identify if the target has drifted out of view or if you've lost track of it
    3. Make corrections based on past movements (e.g., if you turned left and lost the target, turn right to recover it)
    4. Avoid repeating unsuccessful movement patterns
    5. Learn from previous actions to make better-informed next steps
    6. Given the spatial results information, use it to inform your next action precisely.
    7. The spatial results provide critical information about the distance to the target object, use this to plan your movements accurately.
    8. It is not necessary to always use the max_step_size max_angle_step = 3 x {max_step_size}, use smaller steps when precision is needed.
    
    If the target is no longer visible in the current image but was visible in previous images:
    - Review the action history to determine which movement caused you to lose sight
    - Plan corrective actions to return to a position where the target was last clearly visible
    - Use reverse movements intelligently (e.g., if last 3 actions were "forward", consider "backwards" to recover view)
    - If the drone heading is not aligned with the target, consider slight turning and then strafing to directly look at the target.
    NEVER continue moving forward blindly if the target is lost - always prioritize regaining visual on the target before proceeding.
    """
    prompt = f"""
    <Context>
    {context}
    </Context>

    <Objective>
    Your goal is to analyse the current image <image2> and determine the next immediate action to take towards accomplishing the mission given the drone's current position and orientation.
    Output only the next single action that should be executed now based on the current state.
    Use the spatial reasoning information provided to inform your decision if it is provided. Include distance information where relevant to make it easier for the waypoint generator to produce accurate waypoints.
    To accomplish the mission successfully, get as close as possible to the target object without collision and ensure the target object 
    covers a significant portion of the <image2> frame for detailed inspection.
    CRITICAL PRIORITIES (IN ORDER):
    1. Alwway keep the target object in view, there could be multiple other objects in the scene, maybe be similar objects with different
         colors, shapes, sizes - make sure you always keep the correct target object in view.
    2. NEVER lose sight of the target object - keeping it in the camera view is MORE important than making progress towards it
    3. If the target has left the frame, IMMEDIATELY stop forward movement and execute corrective actions to re-acquire the target
    4. Review action history to understand what caused the target to be lost
    5. Only after ensuring the target is visible and centered, proceed with navigation towards the mission goal
    6. Use past mistakes to inform better decisions - if a movement pattern failed before, try a different approach
    7. Note that the spatial reasoning information is provided only for the start of the mission, as the drone steps forward,
       the image gets updated but the spatial reasoning info does not - use it only as a reference for the initial positioning and to reason
       across the history of actions taken.
    8. Always produce actions with respect to the current heading direction of the drone.
    9. Always keep the object of interest in the center of the frame when possible by strafing left and right.
    10. Adjust the heading direction by turning left and right as needed to keep the target in perpendicular view.
    </Objective>

    <Approach Strategy - Distance-Based Navigation>
    Check the spatial_results for distance to target and use this strategy:
    
    1. **FAR from target** (horizontal distance > 10m):
       - Mostly FLY FORWARD to close horizontal distance
       - Occasional small DESCEND action (every 2-3 forwards)
    
    2. **MEDIUM distance** (horizontal distance 5-10m):
       - Mix FORWARD and DESCEND actions equally
       - Pattern: forward, forward, descend, forward, descend...
    
    3. **CLOSE to target** (horizontal distance < 5m):
       - Mostly DESCEND to reach ground level
       - Small forward adjustments if needed
    
    **KEY POINTS**:
    - The target object is typically on or near the GROUND
    - You need BOTH horizontal movement AND vertical descent to reach it
    - Don't just fly forward forever - mix in descends to reach ground level!
    - Don't just descend without getting close horizontally - fly forward too!
    - Review your action history: if you've done 2-3 forwards, consider a descend next
    - If you've done 2-3 descends, consider forward next (unless already very close to target)
    </Approach Strategy>

    <Instructions>
    Analyse the image <image2> and the target objects and the required mission. Orientation and positioning of objects should be inferred from the <image2>.
    Based on the current state, determine what the NEXT IMMEDIATE action should be.
    The action MUST INCLUDE approximate DISTANCES and ANGLES based on visual reasoning and provided information.
    Look carefully at the facing directions of the objects of interest, as well as the drone's facing direction.
    In this mission, the agent needs to precisely view the target object from the specified viewpoint.
    
    <Output Format>
    Your response MUST contain two XML tags:
    1. <Reasoning> tag: Explain your analysis of the current state and why you chose this action
    2. <Action> tag: The specific action to execute
    
    Hovering, capturing images, landing and taking off DO NOT need to be included.
    Just focus on the navigation action required to progress towards the mission goal.
    Respect constraints: no single move longer than {max_step_size} meters and max_angle_step = 3 x {max_step_size}; stay within the action budget of {max_actions} (current count {current_action_index}).
    The action should be concise and actionable. Choose actions such as: forward, backwards, turn left, ascend, descend, turn right, strafe right, strafe left.
    CRITICAL ACTION FORMAT RULE:
    - Every navigation action MUST include a numeric magnitude + unit.
    - Distance actions MUST include meters (e.g., "Fly forward 2 meters", "Strafe right 1.5 meters", "Descend 1 meter").
    - Rotation actions MUST include degrees (e.g., "Turn left 15 degrees").
    - NEVER output bare action tokens without magnitude (invalid examples: "fly_forward", "forward", "turn_left", "descend").
    Important: when specifying distances/angles, use approximate values based on the image analysis and spatial reasoning information provided.
    The distances should be in meters and angles in degrees.
    
    Example format:
    ONLY PROVIDE STRING RESPONSES, DO NOT OUTPUT ANY JSON OR OTHER FORMATS, JUST STRINGS IN THE SPECIFIED XML FORMAT:
    <Reasoning>The grey car is visible ahead at approximately 12 meters distance. The license plate is not yet readable from this altitude and distance. I need to fly forward to close the gap while maintaining the car in my field of view.</Reasoning>
    <Action>Fly forward 2 meters</Action>
    
    or
    
    <Reasoning>The target car has drifted to the left edge of my view. To keep it centered and avoid losing sight of it, I need to adjust my heading by turning left approximately 15 degrees.</Reasoning>
    <Action>Turn left 15 degrees</Action>
            
    For example if the drone is facing the positive x-direction and there is a car facing directly towards the drone and the mission is to inspect the 
        rear license plate, and the car is 10 meters away:
    
    <Reasoning>I am facing the front of the car at 10 meters distance. To inspect the rear license plate, I need to fly past the car (approximately 12 meters forward) to position myself behind it.</Reasoning>
    <Action>Fly forward 2 meters</Action>
    </Output Format>
    Guidelines:
    0. Think in termns of left and right, instead of compass directions (north, south, etc.) since the drone heading can change
    1. **TARGET VISIBILITY IS PARAMOUNT**: ALWAYS keep the object of interest in view - if it leaves frame, STOP and immediately correct direction
    2. **USE HISTORY INTELLIGENTLY**: Before each action, review the action history and <image1> sequence to understand:
       - What movements brought you to the current state
       - Whether previous actions successfully maintained target visibility
       - If you're repeating a pattern that previously lost the target
    3. **AVOID OSCILLATING MOVEMENTS**: Do NOT alternate between opposite actions (e.g., ascend then descend repeatedly):
       - Example of WRONG: "Ascend 8m" → "Descend 8m" → "Ascend 5m" → "Descend 5m" (cancels out, wastes actions)
       - Example of WRONG: "Forward 10m" → "Backward 10m" (no net progress, action budget wasted)
       - Example of WRONG: "Turn right 20°" → "Turn left 20°" (ends at same heading, wasted action)
       - If you need to adjust altitude, commit to it - use ONE descent or ONE ascent, not alternating
       - If you need forward progress, move forward decisively - don't second-guess with backward movements
       - otherwise you can decrease the step size to make finer adjustments
       
    4. **COMMIT TO DECISIONS**: Once you choose a direction (forward/backward/up/down), continue in that direction for 1-2 more steps before changing:
       - If target is visible ahead: move forward decisively, don't retreat
       - If target is above: ascend decisively, then assess before descending
       - If target is to the side: strafe in that direction, don't oscillate left-right
    5. **ACTION BUDGET IS LIMITED**: Every action counts toward your total ({max_actions} max):
       - Wasting 10 actions on oscillation leaves you only {max_actions - 10} for actual progress
       - CRITICAL: Cancel out any recent opposite movements (if last action was "ascend 5m" and you want to descend, just descend less)
    6. **RECOGNIZE STUCK PATTERNS**: If you see 3+ alternating opposite movements in history, STOP and reassess:
       - This means the current strategy is failing
       - Look at where the target actually is and take ONE clear corrective action
       - Don't keep wiggling in the same bad pattern
    7. **RECOVER FROM LOSSES**: If the object of interest is not visible in current <image1>:
       - Analyze which recent action(s) caused the loss of view
       - Execute reverse/corrective movements to return to last known good position
       - Do NOT continue blindly forward when target is not visible
    8. **LEARN FROM MISTAKES**: If an action pattern failed (e.g., multiple forward movements lost the target), try alternative approach:
       - Adjust heading before moving forward
       - Use slower, smaller movements
       - Strafe instead of moving forward/backward
    9. Use precise distance and angle estimates based on visual cues and spatial reasoning data
    10. Avoid unnecessary movements that do not contribute to mission progress
    11. Ensure each action is feasible given the drone's current position and orientation
    12. Do not exceed the maximum single-step movement of {max_step_size} meters but for angel max_angle_step = 3 x {max_step_size}
    13. Analyse the current drone position in the context and beware that the height MUST always be at least -1 above the ground level
        So when decending actions, consider the current drone pose and plan accordingly.
    13. AVOID COLLISIONS at all costs
    14. Avoid aggressive altitude changes - descend/ascend gradually (max {max_step_size} meters per step, max_angle_step = 3 x {max_step_size}) to maintain visibility of target
    15. Do not repeat the same action more than 2 consecutive times without reassessing - if stuck in a pattern, consult action history and try a different direction
    16. ONLY provide the answer in the specified XML format - no additional text outside the tags
    17. For patrol missions, ensure to cover the entire area methodically without missing sections or overlapping excessively and dont get distracted by non-target objects or intermediate routes.
    </Instructions>
    """
    model_name_local = mission.get("model_name", mission.get("mission", {}).get("model_name", ""))
    if "qwen" in model_name_local.lower() or "local" in model_name_local.lower():
         prompt += "\n18. End your response with the token END_OF_TURN\n"
         prompt += "19. CRITICAL: Keep your internal reasoning/thinking VERY concise (max 3-5 sentences). Do not generate long chain-of-thought analysis. Focus on the immediate next action.\n"

    return prompt


def should_continue_navigation_prompt(
    mission: dict,
    next_conversation_request: bool = False
) -> str:
    """Prompt to determine if the mission is complete based on current action and image analysis."""
    if next_conversation_request:
        return "Based on the current state and the previous conversation, determine if the mission has been successfully completed or if more navigation actions are needed. Output 'navigation done' if the mission is complete or 'continue' if more navigation is required."
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
    context = f"""You are analyzing whether a UAV mission has been completed.
    The drone started at position (x={start_pose["x"]:.1f}, y={start_pose["y"]:.1f}, z={start_pose["z"]:.1f}).
    The drone is currently at position (x={current_x:.1f}, y={current_y:.1f}, z={current_z:.1f}).
    The original mission was: {mission_benchmark["instruction"]}
    The executed actions are : {action_history}
    
    Mission Constraints:
    - Maximum allowed actions: {max_actions}
    - Current action count: {current_action_index}
    """
    
    return f"""
    <Context>
    {context}
    </Context>

    <Objective>
    Determine if the mission has been successfully completed or if more actions are needed.
    </Objective>

    <Instructions>
    Based on the current image and the mission objective, assess whether:
    1. The mission objective has been achieved (target inspected, location reached, etc.)
    2. The drone is in the correct position to fulfill the mission requirement
    3. No further navigation is needed
    4. Check if action budget ({max_actions} actions) has been exceeded
    
    IMPORTANT - For visual inspection missions (reading licenses plates, identifying text, etc.):
    - Examine the target object closely. Can you clearly see and read the required details?
    - If YES and the mission asks for specific information: Provide the information (e.g., "license plate: ABC123")
    - If NO: Respond with "continue"
    
    Output Format (STRICT):
    <Reasoning>Explain your visual check and budget check. Wether this pose is sufficient for solcing the mission (e.g., license plate), include it here. End the reasoning with a new line formatted exactly as: task_gt: <value></Reasoning>
    <Action>navigation done</Action>
    OR
    <Reasoning>Explain why more navigation is needed and what is missing from view. End the reasoning with a new line formatted exactly as: task_gt: <value></Reasoning>
    <Action>continue</Action>
    
    For manipulation missions like package delivery, sample collection, etc.:
    - Assess whether the drone is in the correct position to complete the task (e.g., above the drop zone, close enough to the target, etc.)
    - If YES: output with the following for e.g a package delivery mission:
    <Reasoning> explain how the drone is in the correct position and the task is complete: task_gt: drop </Reasoning>
    <Action>navigation done</Action>

    For patrol/surveillance missions:
    - Assess whether the drone has covered the required area and observed the necessary viewpoints
    - If YES: output with the following:
    <Reasoning> explain how the drone has successfully patrolled the area and observed the required viewpoints: task_gt: patrol complete </Reasoning>
    <Action>navigation done</Action>

    Respond with:
    - If the navigation is completed and the viewpoint is good enough to solve the mission (and text is readable when required), output <Action>navigation done</Action>
    - If navigation is not complete and constraints allow continuing, output <Action>continue</Action>
    - If action budget exceeded, output <Action>navigation done</Action>

    Be strict: only output "navigation done" when visual evidence clearly satisfies the mission objective or the action budget is exhausted.
    </Instructions>
    """


def step_by_step_1vlm_prompt(
    mission: dict,
    num_images: int = None,
) -> str:
    """
    Simplified prompt for single-VLM that predicts next action and checks completion in one call.
    Assumes system prompt contains all formatting and rule instructions.
    
    Args:
        mission: Mission dictionary with all state information
        num_images: Number of most recent images to reference (sliding history window)
                   If None, uses value from mission state (default 1)
                   - At timestep 1: [image1]
                   - At timestep 2 and num_images=2: [image2, image1] (newest first)
                   - At timestep t and num_images=N: latest N images, newest first
    """
    mission_benchmark = mission["mission"]
    tolerances = mission.get("tolerances", mission_benchmark.get("tolerances", {}))
    max_step_size = tolerances.get("MAX_STEP_SIZE", 10)
    max_actions = tolerances.get("MAX_ACTIONS", 20)
    
    current_action_index = mission.get("current_action_index", 0)
    action_history = mission.get("parsed_actions", [])
    
    # Get drone pose
    drone_pose = mission.get("drone_pose", {})
    current_x = drone_pose.get("x", 0.0)
    current_y = drone_pose.get("y", 0.0)
    current_z = drone_pose.get("z", 0.0)
    current_yaw = drone_pose.get("yaw", 0.0)
    
    facing_direction = yaw_to_facing_direction.get(int(current_yaw), "unknown direction")
    target_object = mission_benchmark.get("metadata", {}).get("target_object", "target")
    
    # Get num_images from mission state if not provided
    if num_images is None:
        num_images = mission.get("num_history_images", 1)
    num_images = max(1, int(num_images))

    # Align prompt image references with actually available history (if present)
    img_history = mission.get("img_history", [])
    if isinstance(img_history, list) and len(img_history) > 0:
        num_images = min(num_images, len(img_history))
    
    # Build context
    prompt = f"""Mission: {mission_benchmark["instruction"]}

Camera: {mission_benchmark["camera"]["width"]}x{mission_benchmark["camera"]["height"]}, FOV: {mission_benchmark["camera"]["fov"]}°, Pitch: {mission_benchmark["camera"].get("pitch", -45)}°

Current State:
- Position: (x={current_x:.1f}, y={current_y:.1f}, z(-
Altitude)={current_z:.1f})
- Note the AirSim uses a NED coordinate system where z is negative when above ground level, so altitude is -z.
- If the z is positive, then it means it crashed into the ground, so be careful with descending actions and always consider the current altitude when planning descends to avoid crashing.
- Yaw: {current_yaw:.1f}° (facing {facing_direction})
- Action count: {current_action_index}/{max_actions}

Constraints:
- Max step size: {max_step_size}m (angles: {3 * max_step_size}°)
- Action budget: {max_actions} actions

Action History: {action_history if action_history else "No actions taken yet"}
"""

    if "spatial_results" in mission and mission.get("spatial_results"):
        prompt += f"""

Spatial reference (from initial stage):
- {mission.get("spatial_results")}
- This is a coarse initial reference and may be stale after movement; use it as guidance, not absolute truth.
"""
    
    # Image references based on sliding last-N history window (most-recent-first)
    prompt += "\nImages:\n"
    
    if num_images == 1:
        # Current image only
        prompt += "- <image1>: Current view from drone camera\n\nAnalyze the current image and determine the next action OR declare mission complete.\n"
    
    else:
        prompt += "- <image1>: Most recent frame (latest/current view)\n"
        for i in range(2, num_images):
            prompt += f"- <image{i}>: Intermediate recent frame\n"
        prompt += f"- <image{num_images}>: Oldest frame in this recent history window\n\n"
        prompt += f"Use <image1> as the primary decision frame and use <image2> ... <image{num_images}> as short-term history context. Determine the next action OR declare mission complete.\n"

    prompt += """
    - the most important thing is to remember the target object, it is the same throughout the mission and it is 
    important to alway keep it in view and not get distracted by other objects in the scene.
    - CRITICAL: the action MUST include a numeric magnitude and unit:
    - Distance moves use meters ("Fly forward 2 meters", "Strafe left 1 meter", "Ascend 0.5 meters").
    - Turns use degrees ("Turn right 20 degrees").
    - INVALID outputs: "fly_forward", "forward", "turn_left", "descend" (missing magnitude/unit).
    <Instructions>
    Analyse the image <image2> and the target objects and the required mission. Orientation and positioning of objects should be inferred from the <image2>.
    Based on the current state, determine what the NEXT IMMEDIATE action should be.
    The action MUST INCLUDE approximate DISTANCES and ANGLES based on visual reasoning and provided information.
    Look carefully at the facing directions of the objects of interest, as well as the drone's facing direction.
    In this mission, the agent needs to precisely view the target object from the specified viewpoint.
    
    <Output Format>
    Your response MUST contain two XML tags:
    1. <Reasoning> tag: Explain your analysis of the current state and why you chose this action
    2. <Action> tag: The specific action to execute
    
    Hovering, capturing images, landing and taking off DO NOT need to be included.
    Just focus on the navigation action required to progress towards the mission goal.
    Respect constraints: no single move longer than {max_step_size} meters and max_angle_step = 3 x {max_step_size}; stay within the action budget of {max_actions} (current count {current_action_index}).
    The action should be concise and actionable. Choose actions such as: forward, backwards, turn left, ascend, descend, turn right, strafe right, strafe left.
    CRITICAL ACTION FORMAT RULE:
    - Every navigation action MUST include a numeric magnitude + unit.
    - Distance actions MUST include meters (e.g., "Fly forward 2 meters", "Strafe right 1.5 meters", "Descend 1 meter").
    - Rotation actions MUST include degrees (e.g., "Turn left 15 degrees").
    - NEVER output bare action tokens without magnitude (invalid examples: "fly_forward", "forward", "turn_left", "descend").
    Important: when specifying distances/angles, use approximate values based on the image analysis and spatial reasoning information provided.
    The distances should be in meters and angles in degrees.
    
    Guidelines:
    0. Think in termns of left and right, instead of compass directions (north, south, etc.) since the drone heading can change
    1. **TARGET VISIBILITY IS PARAMOUNT**: ALWAYS keep the object of interest in view - if it leaves frame, STOP and immediately correct direction
    2. **USE HISTORY INTELLIGENTLY**: Before each action, review the action history and <image1> sequence to understand:
       - What movements brought you to the current state
       - Whether previous actions successfully maintained target visibility
       - If you're repeating a pattern that previously lost the target
    3. **AVOID OSCILLATING MOVEMENTS**: Do NOT alternate between opposite actions (e.g., ascend then descend repeatedly):
       - Example of WRONG: "Ascend 8m" → "Descend 8m" → "Ascend 5m" → "Descend 5m" (cancels out, wastes actions)
       - Example of WRONG: "Forward 10m" → "Backward 10m" (no net progress, action budget wasted)
       - Example of WRONG: "Turn right 20°" → "Turn left 20°" (ends at same heading, wasted action)
       - If you need to adjust altitude, commit to it - use ONE descent or ONE ascent, not alternating
       - If you need forward progress, move forward decisively - don't second-guess with backward movements
       - otherwise you can decrease the step size to make finer adjustments
       
    4. **COMMIT TO DECISIONS**: Once you choose a direction (forward/backward/up/down), continue in that direction for 1-2 more steps before changing:
       - If target is visible ahead: move forward decisively, don't retreat
       - If target is above: ascend decisively, then assess before descending
       - If target is to the side: strafe in that direction, don't oscillate left-right
    5. **ACTION BUDGET IS LIMITED**: Every action counts toward your total ({max_actions} max):
       - Wasting 10 actions on oscillation leaves you only {max_actions - 10} for actual progress
       - CRITICAL: Cancel out any recent opposite movements (if last action was "ascend 5m" and you want to descend, just descend less)
    6. **RECOGNIZE STUCK PATTERNS**: If you see 3+ alternating opposite movements in history, STOP and reassess:
       - This means the current strategy is failing
       - Look at where the target actually is and take ONE clear corrective action
       - Don't keep wiggling in the same bad pattern
    7. **RECOVER FROM LOSSES**: If the object of interest is not visible in current <image1>:
       - Analyze which recent action(s) caused the loss of view
       - Execute reverse/corrective movements to return to last known good position
       - Do NOT continue blindly forward when target is not visible
    8. **LEARN FROM MISTAKES**: If an action pattern failed (e.g., multiple forward movements lost the target), try alternative approach:
       - Adjust heading before moving forward
       - Use slower, smaller movements
       - Strafe instead of moving forward/backward
    9. Use precise distance and angle estimates based on visual cues and spatial reasoning data
    10. Avoid unnecessary movements that do not contribute to mission progress
    11. Ensure each action is feasible given the drone's current position and orientation
    12. Do not exceed the maximum single-step movement of {max_step_size} meters but for angel max_angle_step = 3 x {max_step_size}
    13. Analyse the current drone position in the context and beware that the height MUST always be at least -1 above the ground level
        So when decending actions, consider the current drone pose and plan accordingly.
    13. AVOID COLLISIONS at all costs
    14. Avoid aggressive altitude changes - descend/ascend gradually (max {max_step_size} meters per step, max_angle_step = 3 x {max_step_size}) to maintain visibility of target
    15. Do not repeat the same action more than 2 consecutive times without reassessing - if stuck in a pattern, consult action history and try a different direction
    16. ONLY provide the answer in the specified XML format - no additional text outside the tags
    17. For patrol missions, ensure to cover the entire area methodically without missing sections or overlapping excessively and dont get distracted by non-target objects or intermediate routes.
    </Instructions>

    
Format reminder:
- Return only <Reasoning> and <Action> tags.
- End <Reasoning> with: task_gt: <value> (e.g., license plate text if readable, otherwise N/A).
- If task_gt has a specific value (not N/A), you MUST output <Action>navigation done</Action> - do NOT continue.
- If the task can be solved, you can direclty ouput the task_dt value and end the mission with <Action>navigation done</Action>.
  You're not required to keep achieving the most perfect viewpoint, as long as the task can be solved with the current view, you can end the mission.
- If not done, output exactly ONE action command in <Action>.
- If mission is complete (target details readable or budget exhausted), output <Action>navigation done</Action>.

For manipulation missions like package delivery, sample collection, etc.:
- Assess whether the drone is in the correct position to complete the task (e.g., above the drop zone, close enough to the target, etc.)
- If YES: output with the following for e.g a package delivery mission:
<Reasoning> explain how the drone is in the correct position and the task is complete: task_gt: drop </Reasoning>
<Action>navigation done</Action>

For patrol/surveillance missions:
- Assess whether the drone has covered the required area and observed the necessary viewpoints
- If YES: output with the following:
<Reasoning> explain how the drone has successfully patrolled the area and observed the required viewpoints: task_gt: patrol complete </Reasoning>
<Action>navigation done</Action>
"""
    
    model_name_local = mission.get("model_name", mission.get("mission", {}).get("model_name", ""))
    if "qwen" in model_name_local.lower() or "local" in model_name_local.lower():
         prompt += "\n18. End your response with the token END_OF_TURN\n"

    return prompt


def step_by_step_1vlm_with_bb_prompt(
    mission: dict,
    num_images: int = None,
) -> str:
    """Prompt variant that additionally asks the VLM to predict a bounding box.

    Builds on top of ``step_by_step_1vlm_prompt`` and appends an extra
    requirement: before the ``<Reasoning>`` tag the model must output a
    ``<BoundingBox>`` tag containing the 0-1000 normalised coordinates of the
    target object visible in the current image.

    Output format expected from the model::

        <BoundingBox>x1,y1,x2,y2</BoundingBox>
        <Reasoning>... task_gt: value</Reasoning>
        <Action>move forward 2 meters</Action>

    This predicted bounding box is parsed by the downstream
    ``node_parse_step_by_step_action_with_bb`` node and is later drawn onto
    the saved step image via ``node_draw_predicted_bounding_box``.
    """
    mission_benchmark = mission["mission"]

    # Re-use the base 1-VLM prompt and append the BB instructions
    base_prompt = step_by_step_1vlm_prompt(mission=mission, num_images=num_images)

    bb_instructions = """

---
BOUNDING BOX PREDICTION REQUIREMENT
You MUST predict the location of the target object, its depth in m and the target_object_name
 in the CURRENT image using a bounding box.
Use coordinates normalised to 0-1000 where:
  - (0, 0)    → top-left corner of the image
  - (1000, 1000) → bottom-right corner of the image
  - Format: [x_min, y_min, x_max, y_max, depth, target_object_name]

Include this prediction as the FIRST XML tag in your response:
    <BoundingBox>[400, 450, 600, 550, 10.0, 'car']</BoundingBox>

If the target is NOT visible in the current image use:

    <BoundingBox>-1,-1,-1,-1, -1, 'unknown'</BoundingBox>

FULL output format (in this exact order):

    <BoundingBox>[x_min, y_min, x_max, y_max, depth, target_object_name]</BoundingBox>
    <Reasoning>Your analysis here. End with: task_gt: <value></Reasoning>
    <Action>Your single navigation action OR "navigation done"</Action>
"""

    # Ensure stop token instruction is at the very end
    stop_token_msg = "\n18. End your response with the token END_OF_TURN\n"
    if stop_token_msg in base_prompt:
        base_prompt = base_prompt.replace(stop_token_msg, "")
        bb_instructions += stop_token_msg

    return base_prompt + bb_instructions