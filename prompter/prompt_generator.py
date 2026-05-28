from .prompts.basic_missions.basic_prompt import basic_prompt, refine_prompt_low_level, pixel_detection_prompt, select_next_action_prompt, should_continue_waypoint_generation_prompt
from .prompts.basic_missions.step_by_step import step_by_step_prompt, should_continue_navigation_prompt, step_by_step_1vlm_prompt, step_by_step_1vlm_with_bb_prompt
from .prompts.basic_missions.evaluation_prompt import task_gt_evaluation_prompt
from .prompts.proxy_prompts.bb_and_depth_prompt import bb_and_depth_prediction_prompt
from .prompts.system_prompt import step_by_step_1vlm_system_prompt, step_by_step_1vlm_system_zeroshot
from .prompts.zeroshot_prompt import step_by_step_zeroshot
import yaml

def generate_prompt(
    prompt_type: str,
    **kwargs
) -> str:
    """
    Super function to generate prompts.
    Args:
        prompt_type (str): "basic" or "step_by_step" or "step_by_step_1vlm_system" or "step_by_step_1vlm" or "step_by_step_1vlm_with_bb" or "should_continue_navigation_prompt" or "bb_and_depth_prediction" or "refine" or "select_next_action" or "should_continue_waypoint_generation"
        **kwargs: Arguments for the specific prompt function
    Returns:
        str: Generated prompt
    """
    if prompt_type == "basic":
        return basic_prompt(**kwargs)
    elif prompt_type == "step_by_step":
        return step_by_step_prompt(**kwargs)
    elif prompt_type == "step_by_step_zeroshot":
        return step_by_step_zeroshot(**kwargs)
    elif prompt_type == "step_by_step_1vlm_system":
        return step_by_step_1vlm_system_prompt(**kwargs)
    elif prompt_type == "step_by_step_1vlm_system_zeroshot":
        return step_by_step_1vlm_system_zeroshot(**kwargs)
    elif prompt_type == "step_by_step_1vlm":
        return step_by_step_1vlm_prompt(**kwargs)
    elif prompt_type == "step_by_step_1vlm_with_bb":
        return step_by_step_1vlm_with_bb_prompt(**kwargs)
    elif prompt_type == "step_by_step_next_conversation_request":
        return step_by_step_prompt(**kwargs, next_conversation_request=True)
    elif prompt_type == "should_continue_navigation_next_conversation_request_prompt":
        return should_continue_navigation_prompt(**kwargs, next_conversation_request=True)
    elif prompt_type == "should_continue_navigation_prompt":
        return should_continue_navigation_prompt(**kwargs)
    elif prompt_type == "bb_and_depth_prediction":
        return bb_and_depth_prediction_prompt(**kwargs)
    elif prompt_type == "refine":
        return refine_prompt_low_level(**kwargs)
    elif prompt_type == "select_next_action":
        return select_next_action_prompt(**kwargs)
    elif prompt_type == "should_continue_waypoint_generation":
        return should_continue_waypoint_generation_prompt(**kwargs)
    elif prompt_type == "task_gt_evaluation":
        return task_gt_evaluation_prompt(**kwargs)
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")
    

if __name__ == "__main__":
    # read a sample mission dict from yaml file
    mission_yaml = "/home/nava/uav_mission_planning/UAVMissionPlanning/dataset/Visual_Inspection/NHEnv/grey_car_unittest_mission30/mission_config.yaml"
    with open(mission_yaml, "r") as f:
        mission = yaml.safe_load(f)
    prompt = generate_prompt(prompt_type="bb_and_depth_prediction", mission=mission)
    print(prompt)