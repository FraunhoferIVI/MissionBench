def update_osr(experiment_results_dir_path: str):
    """_summary_

    Args:
        experiment_results_dir_path (str): path to experiment results directory
        
    Returns:
        updates the result.json file in place
    """
    # go through all the result.json files and add the OSR_10 and OSR_20 fields
    import os
    import json
    for root, dirs, files in os.walk(experiment_results_dir_path):
        for file in files:
            if file == "result.json":
                results_json_path = os.path.join(root, file)
                with open(results_json_path, "r") as f:
                    results = json.load(f)
                
                # Check if OSR_10 and OSR_20 are already present
                # get the min_distance_to_target and compute OSR_10 and OSR_20 
                # if the key "disable_from_dashboard" is available, set the default value to 0
                # if "disable_from_dashboard" not in results:
                #     results["disable_from_dashboard"] = 0
                results["disable_from_dashboard"] = 1
                # if "OSR_10" not in results.get("mission_result", {}) and "OSR_20" not in results.get("mission_result", {}):
                #     min_dist = results.get("mission_result").get("min_distance_to_target")
                #     assert min_dist is not None, f"min_distance_to_target not found in {results_json_path}"
                #     osr_10 = 1 if min_dist is not None else 0
                #     osr_20 = 1 if min_dist is not None else 0
                #     results["mission_result"]["OSR_10"] = osr_10
                #     results["mission_result"]["OSR_20"] = osr_20
                # # replace oracle_success_rate with OSR_5
                # if "oracle_success" in results.get("mission_result"):
                #     oracle_success_rate = results["mission_result"].pop("oracle_success")
                #     results["mission_result"]["OSR_5"] = oracle_success_rate
                with open(results_json_path, "w") as f:
                    json.dump(results, f, indent=4)
                print(f"Updated {results_json_path} with OSR_10 and OSR_20")
                    
if __name__ == "__main__":
    experiment_results_dir_path = "/home/nava/uav_mission_planning/langgraph-ivi/experiment_results"  # replace with actual path
    update_osr(experiment_results_dir_path)