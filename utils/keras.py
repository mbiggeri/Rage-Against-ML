import keras, keras_tuner
import json
import os
import datetime
from keras.callbacks import EarlyStopping
from losses import MeanEuclidianError
import numpy as np
import matplotlib.pyplot as plt

def load_saved_model(model_path: str):
    """
    Loads a full Keras SavedModel (.keras file or directory).
    """
    model = keras.models.load_model(model_path)
    return model

def save_hyperparameters(hp: keras_tuner.HyperParameters, name="best_hp", dir="keras/hyperparameters"):
    os.makedirs(dir, exist_ok=True)
    hp_values = hp.values
    with open(os.path.join(dir, f"{name}.json"), "w") as f:
        json.dump(hp_values, f, indent=2)
    with open(os.path.join(dir, f"{name}_config.json"), "w") as f:
        json.dump(hp.get_config(), f, indent=2)

def load_hyperparameters(basepath: str, hp_filename="best_hp_config.json") -> keras_tuner.HyperParameters:
    basepath += f"/{hp_filename}"
    with open(basepath, "r") as f:
        hp_config = json.load(f)
    print("hp", hp_config)
    return keras_tuner.HyperParameters.from_config(hp_config)

def dict_to_filename(hyperparams: dict, prefix: str = "sgd_fold") -> str:
    KEY_ABBREVIATIONS = {
        'model__units_1': 'uu1',
        'model__units_2': 'uu2',
        'model__lambda_1': 'll1',
        'model__lambda_2': 'll2',
        'model__learning_rate': 'lr',
    }
    result_parts = [prefix]
    sorted_items = sorted(hyperparams.items())
    for key, value in sorted_items:
        abbr_key = KEY_ABBREVIATIONS.get(key, key)
        str_value = str(value)
        result_parts.append(f"{abbr_key}_{str_value}")
    return "_".join(result_parts)

def log_dir(name, kind="fit"):
    return f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}"

def build_results_json(
        mee_single, 
        mee_ensemble, 
        mee_baseline, 
        mse_single, 
        mse_ensemble, 
        mse_baseline, 
        prefix="tr", 
        print_baseline=False):
    def unwrap(v):
        return v.item() if hasattr(v, "item") else float(v)

    result = {
        prefix: {
                "mee_single": unwrap(mee_single),
                "mee_ensemble": unwrap(mee_ensemble),
                "mse_single": unwrap(mse_single),
                "mse_ensemble": unwrap(mse_ensemble)
            },
    }

    if print_baseline:
        result["mee_baseline"] = unwrap(mee_baseline)
        result["mse_baseline"] = unwrap(mse_baseline)

    return result

def make_early_stopping(baseline, monitor="val_loss"):
    return EarlyStopping(
        monitor=monitor,
        patience=15,
        mode="min",
        restore_best_weights=True,
        baseline=baseline,
        verbose=1
    )

def assessment(model, X_tr, y_tr, X_ts, y_ts, mee):
    y_pred_tr = model.predict(X_tr)
    y_pred_ts = model.predict(X_ts)

    mee_single = mee.call(y_ts, y_pred_ts)
    mse_single = np.square(y_ts - y_pred_ts).mean()

    tr_mee_single = mee.call(y_tr, y_pred_tr)
    tr_mse_single = np.square(y_tr - y_pred_tr).mean()

    return tr_mee_single, mee_single, tr_mse_single, mse_single

def save_history(history, base_path):
    if type(history) != dict:
        hist_dict = history.history
    else:
        hist_dict = history

    hist_dict_clean = {k: [float(x) for x in v] for k, v in hist_dict.items()}

    with open(base_path + "/history.json", "w") as f:
        json.dump(hist_dict_clean, f, indent=2)

def save_history_from_dict(history_dict, base_path):
    """Save a history dict (like KerasRegressor.history_) to JSON."""
    os.makedirs(base_path, exist_ok=True)
    # ensure plain Python floats
    hist_dict_clean = {k: [float(x) for x in v] for k, v in history_dict.items()}
    with open(os.path.join(base_path, "history.json"), "w") as f:
        json.dump(hist_dict_clean, f, indent=2)

def load_history(base_path):
    with open(base_path + "/history.json", "r") as f:
        hist_dict = json.load(f)

    hist_clean = {k: [float(x) for x in v] for k, v in hist_dict.items()}

    return hist_clean