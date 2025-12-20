import shutil
import optuna
import os
import optuna
import pandas as pd
from optuna.trial import create_trial
import numpy as np

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

def import_csv(file_name, target_column="value", direction="minimize"):
    """
    Imports a CSV into an Optuna study for plotting.
    Assumes hyperparameter columns start with 'params_' (standard Optuna export format).
    """
    # 1. Load the data
    df = pd.read_csv(file_name)
    
    # 2. Create a new study in-memory
    study = optuna.create_study(direction=direction)
    
    # 3. Identify parameter columns
    # This filters for columns that start with 'params_'
    param_cols = [c for c in df.columns if c.startswith('params_')]
    
    for _, row in df.iterrows():
        # Clean the column names (removing 'params_') for the study
        params = {c.replace('params_', ''): row[c] for c in param_cols}
        
        distributions = {}
        for name, val in params.items():
            if isinstance(val, float):
                distributions[name] = optuna.distributions.FloatDistribution(df[f'params_{name}'].min(), df[f'params_{name}'].max())
            elif isinstance(val, (int, np.integer)):
                distributions[name] = optuna.distributions.IntDistribution(int(df[f'params_{name}'].min()), int(df[f'params_{name}'].max()))
            else:
                distributions[name] = optuna.distributions.CategoricalDistribution(df[f'params_{name}'].unique())

        # 4. Create and add the trial
        trial = create_trial(
            params=params,
            distributions=distributions,
            value=row[target_column]
        )
        study.add_trial(trial)
        
    return study