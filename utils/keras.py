import keras, keras_tuner
import json
import os
import datetime

def load_saved_model(model_path: str):
    """
    Loads a full Keras SavedModel (.keras file or directory).
    """
    model = keras.models.load_model(model_path)
    return model

def save_hyperparameters(tuner: keras_tuner.Tuner, name="best_hp", dir="keras/hyperparameters"):
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    os.makedirs(dir, exist_ok=True)
    hp_values = best_hp.values
    with open(os.path.join(dir, f"{name}.json"), "w") as f:
        json.dump(hp_values, f, indent=2)
    with open(os.path.join(dir, f"{name}_config.json"), "w") as f:
        json.dump(best_hp.get_config(), f, indent=2)

def load_hyperparameters(basepath: str, hp_filename="best_hp_config.json") -> keras_tuner.HyperParameters:
    basepath += f"/{hp_filename}"
    with open(basepath, "r") as f:
        hp_config = json.load(f)
    print("hp", hp_config)
    return keras_tuner.HyperParameters.from_config(hp_config)

def dict_to_filename(hyperparams: dict, prefix: str = "sgd_fold") -> str:
    KEY_ABBREVIATIONS = {
        'lambda_layer1': 'll1',
        'lambda_layer2': 'll2',
        'learning_rate': 'lr',
    }
    result_parts = [prefix]
    sorted_items = sorted(hyperparams.values.items())
    for key, value in sorted_items:
        abbr_key = KEY_ABBREVIATIONS.get(key, key)
        str_value = str(value)
        result_parts.append(f"{abbr_key}_{str_value}")
    return "_".join(result_parts)

def log_dir(name, kind="fit"):
    return f"logs/fit/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{name}"

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