"""
Simple Dataset and DataLoader for UAV Mission Planning
"""
import math
from random import sample
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import quaternion_to_eulerian_angle, load_mission_index
from utils.convert_pose import parse_airsim_file_and_get_movements
from prompter.prompt_generator import generate_prompt
from PIL import Image

class MissionPlanningDataset(Dataset):
    """
    Simple dataset for loading UAV mission planning configuration files.
    """  
    def __init__(
        self,
        root_dir: str,
        task_categories: Optional[List[str]] = None,
        environments: Optional[List[str]] = None,
        transform=None
    ):
        """
        Args:
            root_dir: Root directory of the dataset
              (e.g., UAVMissionPlanning/dataset)
            task_categories: List of task categories to include 
            (e.g., ['Visual_Inspection', 'Manipulation'. 'Patrol'])
            environments: List of environments to include 
            (e.g., ['AirSimNHEnv', 'City'])
            transform: Optional transform to be applied on a sample
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.data_samples = []    
        self.finetune_samples = []
        # Default to all task categories if not specified
        if task_categories is None:
            task_categories = ['Visual_Inspection', 'Patrol', 'Manipulation']       
        # Load all mission files
        self._load_missions(task_categories, environments)   
        self.check_benchmark_integrity()
        
    def _load_missions(self, task_categories: List[str],
                       environments: Optional[List[str]]):
        """Load all mission YAML files from specified categories and
          environments."""
        for category in task_categories:
            category_path = self.root_dir / category
            if not category_path.exists():
                continue
            # Walk through the category directory
            for yaml_file in category_path.rglob('*.yaml'):
                # Skip template files
                if 'template' in yaml_file.name.lower():
                    continue               
                # Check environment filter if specified
                if environments is not None:
                    env_match = any(env in str(yaml_file) 
                                    for env in environments)
                    if not env_match:
                        continue              
                self.data_samples.append({
                    'file_path': str(yaml_file),
                    'category': category,
                    'environment': self._extract_environment_from_yaml(yaml_file),
                    'relative_path': str(yaml_file.relative_to(self.root_dir))
                })

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data_samples)   
    def extract_all_instructions_into_csv(self, output_csv: str):
        """Extract all instructions and metadata into a CSV file for analysis."""
        import csv
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['mission_name', 'category', 'environment', 'instruction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for sample in self.data_samples:
                with open(sample['file_path'], 'r') as f:
                    mission_config = yaml.safe_load(f)
                instruction = mission_config.get('instruction', '')
                writer.writerow({
                    'mission_name': Path(sample['file_path']).parent.name,
                    'category': sample['category'],
                    'environment': sample['environment'],
                    'instruction': instruction
                })
        print(f"Instructions extracted to CSV: {output_csv}")
    def _extract_environment_from_yaml(self, yaml_file: Path) -> str:
        """Extract environment name from the YAML file."""
        with open(yaml_file, 'r') as f:
            mission_config = yaml.safe_load(f)
        env_name = mission_config.get('metadata', {}).get('environment', 'unknown')
        return env_name
    
    def check_tolrances_key(self):
        """Check if all mission yaml files have the 'tolerances' key."""
        for sample in self.data_samples:
            with open(sample['file_path'], 'r') as f:
                mission_config = yaml.safe_load(f)
            assert 'tolerances' in mission_config, f"⚠ 'tolerances' key missing in file: {sample['file_path']}"
            
    def check_benchmark_integrity(self) -> bool:
        self._check_root_dir_exists()
        self._check_yaml_file_integrity()
        self._check_task_category_names()
        self._check_naming_conventions()
        self._check_env_names()
        self.check_tolrances_key()
        self.check_pose_consistency()
        self._check_target_object_naming_issue()
        # self._check_topdown_map_presence()
        return True
    
    def _check_topdown_map_presence(self) -> bool:
        """Check if all mission yaml files have the 'camera' key with 'topdown_map'."""
        for sample in self.data_samples:
            with open(sample['file_path'], 'r') as f:
                imgs_folder = Path(sample['file_path']).parent / "imgs"
                if not imgs_folder.exists():
                    assert False, f"⚠ 'imgs' folder missing in mission folder: {imgs_folder}"
                imgs_available = list(imgs_folder.glob("*.png"))
                if len(imgs_available) == 0:
                    assert False, f"⚠ No PNG images found in 'imgs' folder: {imgs_folder}"
                assert "topdown_map_start.png" in [img.name for img in imgs_available], f"⚠ 'topdown_map_start.png' missing in 'imgs' folder: {imgs_folder}"
                assert "topdown_map_end.png" in [img.name for img in imgs_available], f"⚠ 'topdown_map_end.png' missing in 'imgs' folder: {imgs_folder}"
        return True
    
    def check_pose_consistency(self) -> bool:
        """Check if all mission yaml files have consistent pose formats."""
        for sample in self.data_samples:
            with open(sample['file_path'], 'r') as f:
                mission_config = yaml.safe_load(f)
            start_pose = mission_config.get('drone_start_pose', {})
            gt_pose = mission_config.get('ground_truth', {}).get('waypoint_gt', {})
            # Check if both poses have x, y, z keys
            for key in ['x', 'y', 'z']:
                assert key in start_pose, f"⚠ Missing '{key}' in drone_start_pose of file: {sample['file_path']}"
                assert key in gt_pose, f"⚠ Missing '{key}' in ground_truth waypoint_gt of file: {sample['file_path']}"
                assert isinstance(start_pose[key], (int, float)), f"⚠ '{key}' in drone_start_pose of file {sample['file_path']} is not a number."
                assert isinstance(gt_pose[key], (int, float)), f"⚠ '{key}' in ground_truth waypoint_gt of file {sample['file_path']} is not a number."
                assert start_pose['z'] < 0, f"⚠ 'z' in drone_start_pose of file {sample['file_path']} should be negative (above ground level)."

        return True
    
    def _check_root_dir_exists(self) -> bool:
        """Check if the root directory exists."""
        assert self.root_dir.exists(), f"⚠ Root directory does not exist: {self.root_dir}"
        return True
    
    def _check_naming_conventions(self) -> bool:
        """Ensure all mission names start with lowercase letters and contain no spaces.
        Just assert when False"""
        for sample in self.data_samples:
            with open(sample['file_path'], 'r') as f:
                mission_config = yaml.safe_load(f)
            # only check the starting letters and spaces of the name of the folder only
            folder_name = Path(sample['file_path']).parent.name
            assert folder_name[0] == folder_name[0].lower(), f"⚠ Mission name '{folder_name}' in file {sample['file_path']} does not follow naming conventions."
            assert ' ' not in folder_name, f"⚠ Mission name '{folder_name}' in file {sample['file_path']} contains spaces."

    def _check_target_object_naming_issue(self) -> bool:
        """Check if target object names in the mission configurations follow expected naming conventions."""
        for sample in self.data_samples:
            with open(sample['file_path'], 'r') as f:
                mission_config = yaml.safe_load(f)
            metadata = mission_config.get('metadata', {})
            if 'target_object' in metadata:
                target_object = metadata['target_object']
                if "red_car" in target_object.lower() and "red" not in mission_config["instruction"]:
                    print(sample['file_path'])
                    assert False
        return True

    def _check_yaml_file_integrity(self) -> bool:
        """
        Load all the mission yaml files and make sure they are named as mission_config.yaml
        """
        all_good = True
        for sample in self.data_samples:
            file_path = sample['file_path']
            assert Path(file_path).name == "mission_config.yaml", f"⚠ YAML file not named mission_config.yaml: {file_path}"

    def _check_env_names(self) -> bool:
        """
        Load all the mission yaml files and make sure they are named as mission_config.yaml
        """
        all_good = True
        for sample in self.data_samples:
            with open(sample['file_path'], 'r') as f:
                mission_config = yaml.safe_load(f)
            env_name = mission_config.get('metadata', {}).get('environment', 'unknown')
            valid_envs = ['AirSimNHEnv', 'CityEnv', 'NHEnv', 'ForestEnv', 'Africa_SavannahEnv']
            assert env_name in valid_envs, f"⚠ Environment name '{env_name}' in file {sample['file_path']} is not valid."
        return all_good
    
    def _check_task_category_names(self) -> bool:
        """
        Load all the mission yaml files and make sure they are named as mission_config.yaml
        """
        all_good = True
        valid_categories = ['Visual_Inspection', 'manipulation', 'patrol']
        for sample in self.data_samples:
            with open(sample['file_path'], 'r') as f:
                mission_config = yaml.safe_load(f)
            category = mission_config.get('metadata', {}).get('task_category', 'unknown')
            # assert category in valid_categories, f"⚠ Task category '{category}' in file {sample['file_path']} is not valid. Valid categories are: {valid_categories}"
            # applying fix
            if not category in valid_categories:
                if input(f"Task category '{category}' in file {sample['file_path']} is not valid. Valid categories are: {valid_categories}. Do you want to fix it? (y/n)") == 'y':
                    if "v" in category.lower()[0]:
                        corrected_category = valid_categories[0] 
                    if "m" in category.lower()[0]:
                        corrected_category = valid_categories[1]
                    if "p" in category.lower()[0]:
                        corrected_category = valid_categories[2]

                    assert corrected_category in valid_categories, f"⚠ Corrected task category '{corrected_category}' is not valid. Valid categories are: {valid_categories}"
                    mission_config['metadata']['task_category'] = corrected_category
                    with open(sample['file_path'], 'w') as f:
                        yaml.safe_dump(mission_config, f)
                    print(f"✓ Task category in file {sample['file_path']} has been updated to '{corrected_category}'.")
        return all_good
    
    def category_counts(self) -> Dict[str, int]:
        """Return a summary count of missions per category."""
        counts: Dict[str, int] = {}
        VI = []
        M = []
        P = []
        for sample in self.data_samples:
            env = sample.get('environment', 'unknown')
            if env != 'NHEnv':
                continue
            category = sample.get('category', 'unknown')
            counts[category] = counts.get(category, 0) + 1
            print(sample['file_path'])
            with open(sample['file_path'], 'r') as f:
                mission_config = yaml.safe_load(f)     
            if category == 'Visual_Inspection' and mission_config.get('metadata').get('environment') == 'NHEnv':
                scenario_name = sample['relative_path']
                # print(f"{sample['relative_path'].split('/')[2]}: {sample['relative_path']}")
                VI.append(scenario_name)
            elif category == 'Manipulation' and mission_config.get('metadata').get('environment') == 'NHEnv':
                scenario_name = sample['relative_path']
                M.append(scenario_name)
            elif category == 'Patrol' and mission_config.get('metadata').get('environment') == 'NHEnv':
                scenario_name = sample['relative_path']
                P.append(scenario_name)
        print(f"Visual Inspection Missions: {len(VI)}")
        print(f"Manipulation Missions: {len(M)}")
        print(f"Patrol Missions: {len(P)}")

    
    def validation_dataset(self, path: str = None) -> Dict[str, str]:
        # read validation yaml file and return dataset
        if path is None:       
            raise ValueError("Must specify the path to the validation mission file!")

        project_root = self.root_dir.parent
        try:
            val_missions = load_mission_index(path, project_root=project_root)
            print(f"✓ Validation mission loader mode: nested-index ({path})")
        except Exception:
            # Backward compatibility: allow legacy flat mission maps.
            with open(path, 'r') as f:
                val_missions = yaml.safe_load(f) or {}
            print(f"✓ Validation mission loader mode: legacy-flat-map ({path})")
        # prepend the root dir to every mission path
        for key in val_missions.keys():
            mission_path = Path(val_missions[key])
            if mission_path.is_absolute():
                val_missions[key] = str(mission_path)
            elif str(mission_path).startswith("dataset/"):
                val_missions[key] = str(project_root / mission_path)
            else:
                val_missions[key] = str(self.root_dir / mission_path)
        return val_missions 
    
    def curated_dataset(self, path: str = None) -> Dict[str, str]:
        # read curated yaml file and return dataset
        if path is None:
            raise ValueError("Must specify the path to the curated mission file!")
        
        with open(path, 'r') as f:
            curated_missions = yaml.safe_load(f)
        # prepend the root dir to every mission path
        for key in curated_missions.keys():
            curated_missions[key] = f"{self.root_dir}/{curated_missions[key]}"
        return curated_missions
    
    def finetuning_dataset(self, path: str = None) -> Dict[str, str]:
        """
        Read finetuning yaml file and return dataset.
        
        Args:
            path: Path to the finetuning_mission.yaml file
            
        Returns:
            Dictionary mapping mission names to their full paths
        """
        if path is None:
            raise ValueError("Must specify the path to the finetuning mission file!")
        
        with open(path, 'r') as f:
            finetuning_missions = yaml.safe_load(f)
        
        # Prepend the root dir to every mission path
        for key in finetuning_missions.keys():
            finetuning_missions[key] = f"{self.root_dir}/{finetuning_missions[key]}"
        
        return finetuning_missions
    
    def get_finetuning_samples(self, finetuning_missions_path: str = None, resize = None) -> List[Dict[str, Any]]:
        """
        Generate finetuning samples from missions specified in finetuning_mission.yaml.
        
        Args:
            finetuning_missions_path: Path to finetuning_mission.yaml file
            
        Returns:
            List of finetuning samples with prompts, images, and ground truth
        """
        finetuning_missions = self.finetuning_dataset(path=finetuning_missions_path)
        
        for mission_name, mission_path in finetuning_missions.items():
            # Find the corresponding sample index
            sample_index = None
            for idx, sample in enumerate(self.data_samples):
                if sample['file_path'] == mission_path:
                    sample_index = idx
                    break
            
            if sample_index is not None:
                ft_samples = self.get_ft_sample(sample_index, resize=resize)
        return self.finetune_samples
    
    def test_dataset(self, path: str = None) -> List[str]:
        # return everything else not in the validation dataset
        val_missions = self.validation_dataset(path=path)
        test_samples = []
        val_file_paths = [mission for mission in val_missions]
        for sample in self.data_samples:
            if sample['file_path'] not in val_file_paths:
                test_samples.append(sample['file_path'])
        return test_samples
    
    def get_mission_names(self) -> List[str]:
        """Return list of all mission folder names."""
        mission_names = []
        for sample in self.data_samples:
            folder_name = Path(sample['file_path']).parent.name
            mission_names.append(folder_name)
        return sorted(mission_names)
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive distribution statistics of the dataset.
        
        Returns:
            Dictionary containing mission counts, task distribution, 
            environment distribution, and object type counts
        """
        mission_meta = []
        for sample in self.data_samples:
            with open(sample['file_path'], 'r') as f:
                mission_config = yaml.safe_load(f)
            task = sample['category']
            env = mission_config.get('metadata', {}).get('environment', 'unknown')
            env = env.replace('Env', '')  # Normalize environment names
            mission_name = Path(sample['file_path']).parent.name
            mission_meta.append((task, env, mission_name))
        
        # Task and environment distribution
        task_counts = Counter([t for t, _, _ in mission_meta])
        env_counts = Counter([e for _, e, _ in mission_meta])
        task_env_counts = Counter([(t, e) for t, e, _ in mission_meta])
        
        # Object type extraction from mission names
        obj_counts = {}
        for _, _, mission in mission_meta:
            # Try to extract base object name
            m = re.match(r"(.*)_mission\d+$", mission)
            if m:
                base = m.group(1)
            else:
                m2 = re.match(r"(.*)_\d+$", mission)
                base = m2.group(1) if m2 else mission
            groups = ['car', 'van', 'ship', 'turbine', 'fire', 'board', 'basket', 'bull', 'statue', 'house', 'roof', 'sign']
            for item in groups:
                if item in base.lower():
                    base = item
                    break
            obj_counts[base] = obj_counts.get(base, 0) + 1
        
        # Sort results
        task_counts_sorted = sorted(task_counts.items(), key=lambda x: (-x[1], x[0]))
        env_counts_sorted = sorted(env_counts.items(), key=lambda x: (-x[1], x[0]))
        task_env_counts_sorted = sorted(task_env_counts.items(), key=lambda x: (-x[1], x[0]))
        obj_counts_sorted = sorted(obj_counts.items(), key=lambda x: (-x[1], x[0]))
        
        return {
            "mission_count": len(mission_meta),
            "mission_names": self.get_mission_names(),
            "task_counts": dict(task_counts_sorted),
            "env_counts": dict(env_counts_sorted),
            "task_env_counts": {f"{t}+{e}": count for (t, e), count in task_env_counts_sorted},
            "object_type_counts": dict(obj_counts_sorted)
        }
    
    def generate_distribution_plots(self, output_dir: Optional[str] = None):
        """
        Generate distribution plots for the dataset.
        
        Args:
            output_dir: Directory to save plots. If None, saves to root_dir/report_plots
        """
        if output_dir is None:
            output_dir = self.root_dir / "report_plots"
        else:
            output_dir = Path(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get statistics
        stats = self.get_distribution_stats()
        
        # Plot 1: Task distribution
        plt.figure(figsize=(6, 4))
        tasks, counts = zip(*sorted(stats['task_counts'].items(), 
                                    key=lambda x: (-x[1], x[0])))
        plt.bar(tasks, counts, color="#6E6E6E")
        plt.ylabel("# Missions")
        plt.title("Mission distribution by task")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_dir / "missions_by_task.png", dpi=200)
        plt.close()
        
        # Plot 2: Environment distribution
        plt.figure(figsize=(6, 4))
        envs, counts = zip(*sorted(stats['env_counts'].items(), 
                                   key=lambda x: (-x[1], x[0])))
        plt.bar(envs, counts, color="#6E6E6E")
        plt.ylabel("# Missions")
        plt.title("Mission distribution by environment")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_dir / "missions_by_environment.png", dpi=200)
        plt.close()
        
        # Plot 3: Top 20 object types
        obj_items = sorted(stats['object_type_counts'].items(), 
                          key=lambda x: (-x[1], x[0]))[:20]
        if len(obj_items) > 0:
            plt.figure(figsize=(8, 6))
            labels, values = zip(*obj_items)
            plt.barh(labels[::-1], values[::-1], color="#6E6E6E")
            plt.xlabel("# Missions")
            plt.title("Top 20 object types by mission count")
            plt.tight_layout()
            plt.savefig(output_dir / "object_types_top20.png", dpi=200)
            plt.close()
        
        # Plot 4: All object types
        obj_items_all = sorted(stats['object_type_counts'].items(), 
                              key=lambda x: (-x[1], x[0]))
        if len(obj_items_all) > 0:
            plt.figure(figsize=(10, max(6, 0.25 * len(obj_items_all))))
            labels, values = zip(*obj_items_all)
            plt.barh(labels[::-1], values[::-1], color="#B279A2")
            plt.xlabel("# Missions")
            plt.title("Object types by mission count")
            plt.tight_layout()
            plt.savefig(output_dir / "object_types_all.png", dpi=200)
            plt.close()
        
        print(f"Plots saved to: {output_dir}")
        return output_dir
    
    def print_statistics(self):
        """Print comprehensive dataset statistics."""
        stats = self.get_distribution_stats()
        
        print("=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"\nTotal missions: {stats['mission_count']}")
        
        print("\n" + "-" * 60)
        print("TASK DISTRIBUTION")
        print("-" * 60)
        for task, count in stats['task_counts'].items():
            percentage = (count / stats['mission_count']) * 100
            print(f"  {task:<25} {count:>3} ({percentage:>5.1f}%)")
        
        print("\n" + "-" * 60)
        print("ENVIRONMENT DISTRIBUTION")
        print("-" * 60)
        for env, count in stats['env_counts'].items():
            percentage = (count / stats['mission_count']) * 100
            print(f"  {env:<25} {count:>3} ({percentage:>5.1f}%)")
        
        print("\n" + "-" * 60)
        print("TOP 10 OBJECT TYPES")
        print("-" * 60)
        for i, (obj, count) in enumerate(list(stats['object_type_counts'].items())[:10], 1):
            print(f"  {i:>2}. {obj:<25} {count:>3}")
        
        print("\n" + "=" * 60)
    
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        Args:
            idx: Index of the sample          
        Returns:
            Dictionary containing mission configuration and metadata
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()       
        sample_info = self.data_samples[idx]       
        # Load YAML file
        with open(sample_info['file_path'], 'r') as f:
            mission_config = yaml.safe_load(f)       
        # Create sample dictionary
        sample = {
            'file_path': sample_info['file_path'],
            'category': sample_info['category'],
            'name': mission_config.get('name', 'unknown'),
            'instruction': mission_config.get('instruction', ''),
            'drone_start_pose': mission_config.get('drone_start_pose', {}),
            'ground_truth': mission_config.get('ground_truth', {}),
            'metadata': mission_config.get('metadata', {}),
            'environment': mission_config.get('metadata', {}).get('environment', 'unknown'),
            'camera': mission_config.get('camera', {}),
            'models_related': mission_config.get('models_related', {}),
        }       
        # Apply transform if specified
        if self.transform:
            sample = self.transform(sample)       
        return sample

    def get_ft_sample(self, index: int, resize: tuple = None) -> Dict[str, float]:
        
        finetune_sample = self.data_samples[index]
        mission_name = Path(finetune_sample['file_path']).parent.name
        airsim_rec_path = Path(finetune_sample['file_path']).parent / mission_name / "airsim_rec.txt"
        img_paths, movements = parse_airsim_file_and_get_movements(airsim_rec_path, img_root=Path(finetune_sample['file_path']).parent / mission_name / "images")
        with open(finetune_sample['file_path'], 'r') as f:
            mission_config = yaml.safe_load(f)
        mission = {"mission": mission_config}
        # create pairs of img paths and movements
        
        prompt = generate_prompt(prompt_type="step_by_step", mission=mission)
        for img_path, movement in zip(img_paths[:-1], movements):
            if "hover" in movement.lower():
                continue
            single_sample = {
                "prompt": prompt,
                "img_path": img_path,
                "GT": f"<Action>{movement}</Action>"
            }
            self.finetune_samples.append(single_sample)
        return self.finetune_samples
    
    def name_to_index(self):
        """Helper function to create a mapping from mission names to indices."""
        name_index_map = {}
        for idx, sample in enumerate(self.data_samples):
            mission_name = Path(sample['file_path']).parent.name
            name_index_map[mission_name] = idx
        return name_index_map
    
    @staticmethod
    def convert_to_conversation(ft_sample):
        conversation = [
            { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : ft_sample["prompt"]},
                {"type" : "image", "image" : ft_sample["img_path"]} ]
            },
            { "role" : "assistant",
            "content" : [
                {"type" : "text",  "text"  : ft_sample["GT"]} ]
            },
        ]
        return { "messages" : conversation }
    
# Example usage
if __name__ == "__main__":
    # Example 1: Load all Visual Inspection missions
    dataset_root = "./dataset"  
    dataset = MissionPlanningDataset(
        root_dir=dataset_root,
        task_categories=None
    )   
    dataset.print_statistics()
    # ft_data_samples = dataset.get_finetuning_samples("dataset/finetuning_missions.yaml")
    # print(f"Total finetuning samples: {len(ft_data_samples)}")
    # print(f"Sample finetuning data: {ft_data_samples[0]}")
    print(dataset[0])
    # dataset.extract_all_instructions_into_csv("mission_instructions.csv")
