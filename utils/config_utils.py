from typing import Dict, Any


STRATEGY_TO_HIGH_LEVEL_PROMPT_TYPE = {
    "spf_step_by_step": "step_by_step",
    "vanilla_step_by_step": "step_by_step",
    "step_by_step_1vlm": "step_by_step_1vlm",
    "step_by_step_1vlm_with_bb": "step_by_step_1vlm_with_bb",
}


def get_high_level_prompt_type_for_strategy(strategy: str) -> str:
    """Return the predefined high-level prompt type for a given strategy."""
    if strategy not in STRATEGY_TO_HIGH_LEVEL_PROMPT_TYPE:
        raise ValueError(
            "main_driver.strategy must be one of: "
            + ", ".join(STRATEGY_TO_HIGH_LEVEL_PROMPT_TYPE.keys())
        )
    return STRATEGY_TO_HIGH_LEVEL_PROMPT_TYPE[strategy]


def validate_configs_schema(data: Dict[str, Any]) -> None:
    """Validate configs.yaml base schema."""
    if not isinstance(data, dict):
        raise ValueError("configs.yaml must be a mapping")


def validate_main_driver_config(data: Dict[str, Any]) -> None:
    """Validate configs.yaml schema for main_driver."""
    validate_configs_schema(data)
    if "main_driver" not in data or not isinstance(data["main_driver"], dict):
        raise ValueError("configs.yaml missing required key: main_driver (mapping)")

    main_cfg = data["main_driver"]
    for key in ["resume_exp", "scenario_names", "repeat", "rate_limit_delay"]:
        if key not in main_cfg:
            raise ValueError(f"configs.yaml missing main_driver.{key}")
    strategy = main_cfg.get("strategy")
    if isinstance(strategy, list):
        if not strategy:
            raise ValueError("main_driver.strategy list must not be empty")
        if not all(isinstance(item, str) for item in strategy):
            raise ValueError("main_driver.strategy list must contain only strings")
        for item in strategy:
            get_high_level_prompt_type_for_strategy(item)
    elif isinstance(strategy, str):
        get_high_level_prompt_type_for_strategy(strategy)
    else:
        raise ValueError("main_driver.strategy must be string or list of strings")
    if not isinstance(main_cfg["resume_exp"], bool):
        raise ValueError("main_driver.resume_exp must be boolean")
    if not isinstance(main_cfg["results_dir_to_resume"], str):
        raise ValueError("main_driver.results_dir_to_resume must be string")
    if not isinstance(main_cfg["scenario_names"], list):
        raise ValueError("main_driver.scenario_names must be list")
    if not isinstance(main_cfg["repeat"], int):
        raise ValueError("main_driver.repeat must be int")
    if not isinstance(main_cfg["val_mission_file"], str):
        raise ValueError("main_driver.val_mission_file must be string")
    if not isinstance(main_cfg["rate_limit_delay"], (int, float)):
        raise ValueError("main_driver.rate_limit_delay must be number")
    image_related_cfg = main_cfg.get("image_related")
    if image_related_cfg is not None and not isinstance(image_related_cfg, dict):
        raise ValueError("main_driver.image_related must be mapping")
    image_cfg = image_related_cfg if isinstance(image_related_cfg, dict) else main_cfg

    if "capture_interval" in image_cfg and not isinstance(image_cfg["capture_interval"], (int, float)):
        raise ValueError("main_driver.image_related.capture_interval must be number")
    if "num_history_images" in image_cfg:
        num_history_images = image_cfg["num_history_images"]
        if isinstance(num_history_images, list):
            if not num_history_images:
                raise ValueError("main_driver.image_related.num_history_images list must not be empty")
            if not all(isinstance(num, int) for num in num_history_images):
                raise ValueError("main_driver.image_related.num_history_images list must contain only ints")
        elif not isinstance(num_history_images, int):
            raise ValueError("main_driver.image_related.num_history_images must be int or list of ints")
    if "temperature" in main_cfg:
        temperature = main_cfg["temperature"]
        if isinstance(temperature, list):
            if not temperature:
                raise ValueError("main_driver.temperature list must not be empty")
            if not all(isinstance(temp, (int, float)) for temp in temperature):
                raise ValueError("main_driver.temperature list must contain only numbers")
        elif not isinstance(temperature, (int, float)):
            raise ValueError("main_driver.temperature must be number or list of numbers")
    model_related_cfg = main_cfg.get("model_related")
    models_related_cfg = main_cfg.get("models_related")
    nested_model_cfg = model_related_cfg if model_related_cfg is not None else models_related_cfg
    if nested_model_cfg is None:
        nested_model_cfg = {}

    if not isinstance(nested_model_cfg, dict):
        raise ValueError("main_driver.model_related/models_related must be mapping")

    resolved_model_names = nested_model_cfg.get("model_names", main_cfg.get("model_names"))
    if not isinstance(resolved_model_names, list):
        raise ValueError("main_driver.model_related.model_names must be list")

    resolved_task_gt_eval_model = nested_model_cfg.get("task_gt_eval_model", main_cfg.get("task_gt_eval_model"))
    if not isinstance(resolved_task_gt_eval_model, str):
        raise ValueError("main_driver.model_related.task_gt_eval_model must be string")

    if "temperature" in nested_model_cfg:
        nested_temperature = nested_model_cfg["temperature"]
        if isinstance(nested_temperature, list):
            if not nested_temperature:
                raise ValueError("main_driver.model_related.temperature list must not be empty")
            if not all(isinstance(temp, (int, float)) for temp in nested_temperature):
                raise ValueError("main_driver.model_related.temperature list must contain only numbers")
        elif not isinstance(nested_temperature, (int, float)):
            raise ValueError("main_driver.model_related.temperature must be number or list of numbers")
    if "thinking_level" in nested_model_cfg:
        thinking_level = nested_model_cfg["thinking_level"]
        if thinking_level is not None and not isinstance(thinking_level, str):
            raise ValueError("main_driver.model_related.thinking_level must be string or null")
        if isinstance(thinking_level, str) and thinking_level.strip().lower() not in {"", "none", "null", "low", "medium", "high"}:
            raise ValueError("main_driver.model_related.thinking_level must be one of: low, medium, high, null")

    if "top_p" in nested_model_cfg and not isinstance(nested_model_cfg["top_p"], (int, float)):
        raise ValueError("main_driver.model_related.top_p must be number")
    if "top_k" in nested_model_cfg and not isinstance(nested_model_cfg["top_k"], int):
        raise ValueError("main_driver.model_related.top_k must be int")
    if "min_p" in nested_model_cfg and not isinstance(nested_model_cfg["min_p"], (int, float)):
        raise ValueError("main_driver.model_related.min_p must be number")
    if "presence_penalty" in nested_model_cfg and not isinstance(nested_model_cfg["presence_penalty"], (int, float)):
        raise ValueError("main_driver.model_related.presence_penalty must be number")
    if "repetition_penalty" in nested_model_cfg and not isinstance(nested_model_cfg["repetition_penalty"], (int, float)):
        raise ValueError("main_driver.model_related.repetition_penalty must be number")

    # Backward-compatible top-level checks (deprecated)
    if "top_p" in main_cfg and not isinstance(main_cfg["top_p"], (int, float)):
        raise ValueError("main_driver.top_p must be number")
    if "top_k" in main_cfg and not isinstance(main_cfg["top_k"], int):
        raise ValueError("main_driver.top_k must be int")
    if "min_p" in main_cfg and not isinstance(main_cfg["min_p"], (int, float)):
        raise ValueError("main_driver.min_p must be number")
    if "presence_penalty" in main_cfg and not isinstance(main_cfg["presence_penalty"], (int, float)):
        raise ValueError("main_driver.presence_penalty must be number")
    if "repetition_penalty" in main_cfg and not isinstance(main_cfg["repetition_penalty"], (int, float)):
        raise ValueError("main_driver.repetition_penalty must be number")
    if "thinking_level" in main_cfg:
        thinking_level = main_cfg["thinking_level"]
        if thinking_level is not None and not isinstance(thinking_level, str):
            raise ValueError("main_driver.thinking_level must be string or null")
        if isinstance(thinking_level, str) and thinking_level.strip().lower() not in {"", "none", "null", "low", "medium", "high"}:
            raise ValueError("main_driver.thinking_level must be one of: low, medium, high, null")
    if "img_enhancement_type" in image_cfg:
        img_enhancement_type = image_cfg["img_enhancement_type"]
        if isinstance(img_enhancement_type, list):
            if not img_enhancement_type:
                raise ValueError("main_driver.image_related.img_enhancement_type list must not be empty")
            if not all(isinstance(item, dict) for item in img_enhancement_type):
                raise ValueError("main_driver.image_related.img_enhancement_type list must contain only mappings")
        elif not isinstance(img_enhancement_type, dict):
            raise ValueError("main_driver.image_related.img_enhancement_type must be mapping or list of mappings")
    if "upload_results_to_google_drive" in main_cfg and not isinstance(main_cfg["upload_results_to_google_drive"], bool):
        raise ValueError("main_driver.upload_results_to_google_drive must be boolean")
    if "output_media_format" in main_cfg:
        media_format = main_cfg["output_media_format"]
        if not isinstance(media_format, str):
            raise ValueError("main_driver.output_media_format must be string")
        if media_format.lower() not in {"webp", "mp4"}:
            raise ValueError(
                "main_driver.output_media_format must be one of: webp, mp4"
            )
    if "simulator" in main_cfg:
        simulator_cfg = main_cfg["simulator"]
        if not isinstance(simulator_cfg, dict):
            raise ValueError("main_driver.simulator must be mapping")
        if "environment" in simulator_cfg and not isinstance(simulator_cfg["environment"], str):
            raise ValueError("main_driver.simulator.environment must be string")
        if "rendering" in simulator_cfg and not isinstance(simulator_cfg["rendering"], bool):
            raise ValueError("main_driver.simulator.rendering must be boolean")
        if "ActiveSimMode" in simulator_cfg and not isinstance(simulator_cfg["ActiveSimMode"], str):
            raise ValueError("main_driver.simulator.ActiveSimMode must be string")
        if "active_sim_mode" in simulator_cfg and not isinstance(simulator_cfg["active_sim_mode"], str):
            raise ValueError("main_driver.simulator.active_sim_mode must be string")
        if "CollisionCheckMode" in simulator_cfg and not isinstance(simulator_cfg["CollisionCheckMode"], bool):
            raise ValueError("main_driver.simulator.CollisionCheckMode must be boolean")


def validate_example_demo_config(data: Dict[str, Any]) -> None:
    """Validate configs.yaml schema for example_demo."""
    validate_configs_schema(data)
    example = data.get("example_demo")
    if example is None or not isinstance(example, dict):
        raise ValueError("configs.yaml missing required key: example_demo (mapping)")
    if "resume_path" not in example:
        raise ValueError("configs.yaml missing example_demo.resume_path")
    if not isinstance(example["resume_path"], str):
        raise ValueError("example_demo.resume_path must be string")
