"""
Configuration schema for experiments.
Defines what parameters can be configured and their default values.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union


@dataclass
class ExperimentConfig:
    """
    Single experiment configuration.
    Represents one run with specific parameters.
    """
    # Identifiers
    scenario_name: str  # e.g., "grey_car_mission30"
    model_name: str  # e.g., "gpt-4o-mini"
    experiment_id: str  # e.g., "exp_001_gpt4o_step2.0_actions50"
    val_mission_file: str  # e.g., "validation_vi_missions.json"
    
    # Strategy/Algorithm selection
    strategy: str = "step_by_step"  # "step_by_step" or "spf" or other graph names
    
    # Model parameters
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    thinking_level: Optional[str] = None
    task_gt_eval_model: str = "gemini-3-flash-preview"
    # top_p: float = 0.95
    img_enhancement_type: Optional[Dict[str, Any]] = None
    high_level_action_prompt_type: str = "step_by_step"  # e.g., "step_by_step_1vlm" or "step_by_step"
    num_history_images: int = 1  # Number of images to pass (1=current, 2=initial+current, 3+)
    capture_interval: float = 0.0  # If >0, capture intermediate frames every N meters and N*10 degrees
    
    
    # Tracking
    repeat_number: int = 1  # which repeat (1, 2, 3 if repeat=3)
    random_seed: int = 42
    # tags: List[str] = field(default_factory=list)
    
    collision_check_mode: bool = False  # Whether to perform collision checking in the simulator
    output_media_format: str = "webp"  # "webp" (default) or "mp4"
    model_results_drive_path: Optional[str] = None  # Google Drive folder URL for results upload

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create ExperimentConfig from a dictionary."""
        return cls(**data)
    
    def __str__(self) -> str:
        return f"Exp({self.model_name}, repeat={self.repeat_number},\
              val_mission={self.val_mission_file})"



@dataclass
class GridSearchDefinition:
    """
    Defines a parameter sweep grid.
    Cartesian product of all parameters = N experiments.
    """
    name: str  # e.g., "grid_search_v1"
    scenario_names: List[str]
    model_names: List[str]
    val_mission_file: str
    repeat: int = 1  # How many times to repeat each combination
    temperature: Union[float, List[float]] = 0.7
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    thinking_level: Optional[str] = None
    task_gt_eval_model: str = "gemini-3-flash-preview"
    strategy: Union[str, List[str]] = "step_by_step"  # Navigation strategy: "step_by_step" or "spf"
    img_enhancement_type: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    high_level_action_prompt_type: Union[str, List[str]] = "step_by_step"  # e.g., "step_by_step_1vlm" or "step_by_step"
    num_history_images: Union[int, List[int]] = 1  # Number of images to pass to VLM for ablation studies
    capture_interval: float = 0.0  # If >0, capture intermediate frames every N meters and N*10 degrees
    collision_check_mode: bool = False  # Whether to perform collision checking in the simulator
    output_media_format: str = "webp"  # "webp" (default) or "mp4"
    model_results_drive_path: Optional[str] = None  # Google Drive folder URL for results upload

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        """Normalize scalar/list inputs for Cartesian product generation."""
        return value if isinstance(value, list) else [value]

    def count_experiments(self) -> int:
        """Calculate total number of experiments in this grid."""
        temperatures = self._as_list(self.temperature)
        strategies = self._as_list(self.strategy)
        history_image_counts = self._as_list(self.num_history_images)
        image_enhancements = self._as_list(self.img_enhancement_type) if self.img_enhancement_type is not None else [None]
        return (
            len(self.scenario_names) * 
            len(self.model_names) * 
            self.repeat *
            len(temperatures) *
            len(strategies) *
            len(history_image_counts) *
            len(image_enhancements)
        )
    
    def generate_configs(self) -> List[ExperimentConfig]:
        """Generate all experiment configs from the grid."""
        configs = []
        exp_id = 0
        temperatures = self._as_list(self.temperature)
        strategies = self._as_list(self.strategy)
        prompt_types = self._as_list(self.high_level_action_prompt_type)
        history_image_counts = self._as_list(self.num_history_images)
        image_enhancements = self._as_list(self.img_enhancement_type) if self.img_enhancement_type is not None else [None]

        if len(prompt_types) not in {1, len(strategies)}:
            raise ValueError(
                "high_level_action_prompt_type must be a single value or a list with the same length as strategy"
            )

        strategy_prompt_pairs = []
        for idx, strategy in enumerate(strategies):
            prompt_type = prompt_types[idx] if len(prompt_types) > 1 else prompt_types[0]
            strategy_prompt_pairs.append((strategy, prompt_type))
        
        for scenario in self.scenario_names:
            for model in self.model_names:
                for repeat_num in range(1, self.repeat + 1):
                    for temperature in temperatures:
                        for strategy, prompt_type in strategy_prompt_pairs:
                            for num_history_images in history_image_counts:
                                for img_enhancement in image_enhancements:
                                    exp_id += 1
                                    config = ExperimentConfig(

                                        scenario_name=scenario,
                                        model_name=model,
                                        experiment_id=f"exp_{exp_id:06d}_{model}_rep{repeat_num}",
                                        val_mission_file=self.val_mission_file,
                                        repeat_number=repeat_num,
                                        temperature=float(temperature),
                                        top_p=float(self.top_p),
                                        top_k=int(self.top_k),
                                        min_p=float(self.min_p),
                                        presence_penalty=float(self.presence_penalty),
                                        repetition_penalty=float(self.repetition_penalty),
                                        thinking_level=self.thinking_level,
                                        task_gt_eval_model=self.task_gt_eval_model,
                                        strategy=str(strategy),
                                        img_enhancement_type=img_enhancement,
                                        high_level_action_prompt_type=str(prompt_type),
                                        num_history_images=int(num_history_images),
                                        capture_interval=self.capture_interval,
                                        collision_check_mode=self.collision_check_mode,
                                        output_media_format=self.output_media_format,
                                        model_results_drive_path=self.model_results_drive_path,
                                    )
                                    configs.append(config)
        
        return configs
