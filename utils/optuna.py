import shutil
import optuna
import os

def delete_pruned_trial_dirs(root_path, study):
    pruned_numbers = {
        trial.number 
        for trial in study.trials 
        if trial.state == optuna.trial.TrialState.PRUNED
    }

    removed = []

    if not os.path.isdir(root_path):
        raise ValueError(f"{root_path} is not a valid directory.")

    for entry in os.listdir(root_path):
        full_path = os.path.join(root_path, entry)

        if entry.startswith("."):
            continue  # skip hidden folders
            
        if entry.count("_OPTUNA_TRIAL#") == 1:
            try:
                trial_num = int(entry.split("_OPTUNA_TRIAL#")[1])
            except ValueError:
                continue

            if trial_num in pruned_numbers:
                shutil.rmtree(full_path)
                removed.append(entry)

    return removed