import re
import json
import os
import csv
import pandas as pd
import yaml

def parse_mission_name(scenario_name):
    """
    Extracts 'mission_name_repX' from strings like:
    'toppled_car_mission2_gemini-3.1-pro-preview_exp_000084_gemini-3.1-pro-preview_rep3'
    """
    base_match = re.match(r"^(.*?)_gemini", scenario_name)
    rep_match = re.search(r"_(rep\d+)$", scenario_name)

    if base_match and rep_match:
        return f"{base_match.group(1)}_{rep_match.group(1)}"
    return scenario_name


def extract_raw_mission_name(name: str) -> str:
    """Return the mission base ending with _mission<number> when available."""
    match = re.search(r"(.*?_mission\d+)", str(name))
    return match.group(1) if match else str(name)

def classify_mission(metrics):
    """
    Returns a single main mission outcome/failure reason.
    No priority ranking or multi-label aggregation is used.
    """
    print(f"  [Debug] Metrics for classification: {metrics}")
    # 1. Success Check (Mutually Exclusive to Failures)
    if metrics.get('success') == 1 or metrics.get('sr5') == 1:
        return "Success", f"Task completed successfully. Final distance: {metrics.get('final_dist', 999.0):.1f}m."

    # Variables
    collision_count = metrics.get('collision_count', 0)
    stop_reason = str(metrics.get('stop_reason', ''))
    final_dist = float(metrics.get('final_dist', 999.0))
    z_diff = float(metrics.get('z_diff', 0.0))
    yaw_diff = float(metrics.get('yaw_diff', 0.0))
    y_diff = float(metrics.get('y_diff', 0.0))
    x_diff = float(metrics.get('x_diff', 0.0))
    progress = float(metrics.get('progress', 0.0))
    steps = int(metrics.get('steps', 0))
    budget = int(metrics.get('budget', 50))
    task_gt_match = int(metrics.get('task_gt_match', -1))
    task_type = str(metrics.get('task_type').lower())
    # Single-label failure classification.
    if collision_count >= 3:
        return "Continuous Collision", f"{collision_count} collisions recorded."

    if stop_reason == "max_action_budget_reached" or (steps >= budget and budget > 0):
        if progress < 0.4:
            return "Step Budget Not Enough", f"Ran out of steps {final_dist:.1f}m away."
        return "Drift and oscillation", "Ran out of steps near target (likely spinning)."

    if stop_reason == "vlm_signaled_done":
        if task_gt_match == 0 and task_type in ['Visual_Inspection', 'visual_inspection']:
            return "Last-mile perception", "Misread text or hallucinated object."
        if task_gt_match == -1 and task_type in ['manipulation', 'Manipulation']:
            return "Last-mile perception", "Failed to execute final manipulation command."
        if task_type in ["visual_inspection", "Visual_Inspection", "manipulation", "Manipulation"]:
            if z_diff > 5.0 or yaw_diff > 15.0 or y_diff > 5.0 or x_diff > 5.0:
                return "Altitude / Align Failure", f"Z-diff: {z_diff:.1f}m, Yaw-diff: {yaw_diff:.1f}°."
        if task_type in ["Patrol", "patrol"]:
            if progress < 0.3:
                return "Premature termination", f"Stopped before patrol completion"
        if final_dist > 15.0:
            return "Premature termination", f"Stopped {final_dist:.1f}m away."

    return "Unclassified / Manual Review Required", f"Stop: '{stop_reason}', Prog: {progress:.2f}."


def extract_metrics_from_scenario(scenario_path):
    """
    Safely loads and merges metrics from the specific JSONs inside scenario_results.
    """
    metrics = {}

    files_to_check = ['evaluation_result.json']
    mission_config_path = os.path.join(scenario_path, 'mission_config.yaml')
    for filename in files_to_check:
        file_path = os.path.join(scenario_path, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)

                    # Sometimes metrics are nested under 'mission_result'
                    m_data = data.get('mission_result', data)

                    # Fetch important data values
                    if 'success' in data: metrics['success'] = data['success']
                    if 'sr5' in m_data: metrics['sr5'] = m_data['sr5']
                    if 'collision_count' in m_data: metrics['collision_count'] = m_data['collision_count']
                    if 'stop_reason' in m_data: metrics['stop_reason'] = m_data['stop_reason']
                    if 'final_distance_to_target' in m_data: metrics['final_dist'] = m_data['final_distance_to_target']
                    if 'z_diff' in m_data: metrics['z_diff'] = m_data['z_diff']
                    if 'yaw_diff' in m_data: metrics['yaw_diff'] = m_data['yaw_diff']
                    if 'x_diff' in m_data: metrics['x_diff'] = m_data['x_diff']
                    if 'y_diff' in m_data: metrics['y_diff'] = m_data['y_diff']
                    if 'mission_progress' in m_data: metrics['progress'] = m_data['mission_progress']
                    if 'steps_executed' in m_data: metrics['steps'] = m_data['steps_executed']
                    if 'steps_budget' in m_data: metrics['budget'] = m_data['steps_budget']
                    if 'task_gt_match' in data: metrics['task_gt_match'] = data['task_gt_match']
                
            except Exception as e:
                print(f"  [Warning] Could not parse {filename}: {e}")
    with open(mission_config_path, 'r', encoding='utf-8') as f:
        task_type = yaml.safe_load(f).get('metadata').get('task_category')
    metrics['task_type'] = task_type
    return metrics


def analyze_experiment_folder(root_folder, output_csv):
    """
    Walks through the mission directories. Checks for evaluation_result.json FIRST.
    If missing, falls back to error.txt.
    """
    headers = ["Mission Name", "Reason for failure", "Comments"]
    print(f"Scanning directory: {root_folder}...")

    rows = []

    # Iterate over immediate subdirectories (mission folders)
    for mission_dir_name in os.listdir(root_folder):
        if mission_dir_name.lower() == 'logs':
            continue
        
        mission_path = os.path.join(root_folder, mission_dir_name)

        if not os.path.isdir(mission_path):
            continue  # Skip stray files in the root dir

        formatted_name = parse_mission_name(mission_dir_name)
        scenario_path = os.path.join(mission_path, 'scenario_results')
        eval_file_path = os.path.join(scenario_path, 'evaluation_result.json')
        print(f"Processing mission: {formatted_name}...")

        # --- Primary Check: Did the mission actually finish and evaluate? ---
        if os.path.exists(eval_file_path):
            metrics = extract_metrics_from_scenario(scenario_path)
            print(mission_dir_name)
            # if mission_dir_name == "window_mission9_human_brain_exp_000007_human_brain_rep1":
            #     breakpoint()
            reason, comment = classify_mission(metrics)
            rows.append([formatted_name, reason, comment])

        # --- Fallback Check: Mission crashed or aborted before evaluation ---
        else:
            error_file = os.path.join(mission_path, 'error.txt')
            if os.path.exists(error_file):
                try:
                    with open(error_file, 'r', encoding='utf-8') as ef:
                        error_text = ef.read().strip().replace('\n', ' ')
                    rows.append([
                        formatted_name,
                        "Server/Client Error",
                        f"Failed before evaluation. Error: {error_text[:150]}..."
                    ])
                except Exception as e:
                    print(f"Error reading {error_file}: {e}")
            else:
                # Catch edge cases where it didn't evaluate and there is no error text
                rows.append([formatted_name, "Unclassified / Manual Review Required", "Missing result files."])
    files_processed = len(rows)
    success_count = sum(1 for _, reason, _ in rows if reason == "Success")
    failure_count = files_processed - success_count
    success_pct = (success_count / files_processed * 100.0) if files_processed else 0.0
    failure_pct = (failure_count / files_processed * 100.0) if files_processed else 0.0

    # Always rewrite so summary remains near the top while preserving a valid CSV header.
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(["SUMMARY", "Total Missions", files_processed])
        writer.writerow(["SUMMARY", "Total Success", f"{success_count} ({success_pct:.2f}%)"])
        writer.writerow(["SUMMARY", "Total Failure", f"{failure_count} ({failure_pct:.2f}%)"])
        writer.writerows(rows)

    print(f"Processing complete! {files_processed} missions analyzed and saved to {output_csv}.")

if __name__ == "__main__":
    root_folder = "/home/uam/taehyoung/suman/COLM/MissionBench/data/results/final_expts/sonnet_4.6"
    model_names_input = "gemini3.1_pro, sonnet_4.6, gemini_robotics_rer, human_evaluation_p/experiment_results_28-mar-10.15am_praveen, human_evaluation_p/experiment_results_29-mar-07.24am_sulojan, gpt5.4_mini, gpt5.4, anthropic.claude-opus-4-6-v1, Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-BF16, gemini_2.5_flash, nova pro, nova_premier, gemini_3.1_flash_lite-preview, gemini3_flash"
    model_name = "human_brain"
    os.makedirs(root_folder, exist_ok=True)
    output_csv = f"{root_folder}/failure_case_analysis.csv"
    analyze_experiment_folder(root_folder, output_csv)
    
    output_csv = f'{root_folder}/failure_case_analysis.csv'
    df_failures = pd.read_csv(output_csv)
    df_missions = df_failures[df_failures['Mission Name'] != 'SUMMARY'].copy()

    # Group by 'Reason for failure' and count occurrences
    failure_counts = df_missions['Reason for failure'].value_counts().reset_index()
    failure_counts.columns = ['Reason for failure', 'Count']

    mission_names_by_reason = (
        df_missions
        .groupby('Reason for failure')['Mission Name']
        .apply(
            lambda s: ', '.join(
                sorted(set(s.astype(str).tolist()))
            )
        )
        .reset_index(name='Mission Names')
    )
    failure_counts = failure_counts.merge(
        mission_names_by_reason,
        on='Reason for failure',
        how='left',
    )

    # Put Success first, then the rest by count descending.
    success_rows = failure_counts[failure_counts['Reason for failure'] == 'Success']
    non_success_rows = failure_counts[failure_counts['Reason for failure'] != 'Success']
    if not non_success_rows.empty:
        non_success_rows = non_success_rows.sort_values(by='Count', ascending=False)
    failure_counts = pd.concat([success_rows, non_success_rows], ignore_index=True)

    total_missions = len(df_missions)
    failure_counts['Percentage'] = (
        (failure_counts['Count'] / total_missions * 100.0) if total_missions else 0.0
    )
    failure_counts['Percentage'] = failure_counts['Percentage'].round(2)

    total_row = pd.DataFrame([
        {
            'Reason for failure': 'Total',
            'Count': total_missions,
            'Percentage': 100.00 if total_missions else 0.00,
            'Mission Names': '',
        }
    ])
    failure_counts = pd.concat([failure_counts, total_row], ignore_index=True)
    failure_counts = failure_counts[
        ['Reason for failure', 'Count', 'Percentage', 'Mission Names']
    ]

    print(f"Total missions analyzed: {total_missions}")
    print(failure_counts.round(2))
    failure_counts.to_csv(f"{root_folder}/failure_summary.csv", index=False)
