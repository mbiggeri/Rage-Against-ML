import keras, keras_tuner
import json
import os
import datetime
from keras.callbacks import EarlyStopping
from losses import MeanEuclidianError
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay

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

def plot_prediction_error(y, y_pred):
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0],
        scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
        line_kwargs={"color": "tab:red"},
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1],
        scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
        line_kwargs={"color": "tab:red"},
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle("Plotting predictions")
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_cv_bar_per_fold(fold_mees, fold_mses, model_name="BO Best Model"):
    fold_mees = np.array(fold_mees, dtype=float)
    fold_mses = np.array(fold_mses, dtype=float)

    k = len(fold_mees)
    folds = np.arange(1, k + 1)

    mean_mee = fold_mees.mean()
    mean_mse = fold_mses.mean()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # --- MEE per fold ---
    ax[0].bar(folds, fold_mees)
    ax[0].axhline(mean_mee, color="red", linestyle="--", label=f"mean = {mean_mee:.2f}")
    ax[0].set_title(f"{model_name} — MEE per fold")
    ax[0].set_xlabel("Fold")
    ax[0].set_ylabel("MEE")
    ax[0].set_xticks(folds)
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    # --- MSE per fold ---
    ax[1].bar(folds, fold_mses, color="orange")
    ax[1].axhline(mean_mse, color="red", linestyle="--", label=f"mean = {mean_mse:.2f}")
    ax[1].set_title(f"{model_name} — MSE per fold")
    ax[1].set_xlabel("Fold")
    ax[1].set_ylabel("MSE")
    ax[1].set_xticks(folds)
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def plot_cv_line(fold_mees, fold_mses, model_name="BO Model"):
    k = len(fold_mees)
    folds = np.arange(1, k+1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(folds, fold_mees, marker="o")
    ax[0].set_title(f"{model_name} — MEE per fold")
    ax[0].set_xlabel("Fold")
    ax[0].set_ylabel("MEE")
    ax[0].grid(alpha=0.3)

    ax[1].plot(folds, fold_mses, marker="o", color="orange")
    ax[1].set_title(f"{model_name} — MSE per fold")
    ax[1].set_xlabel("Fold")
    ax[1].set_ylabel("MSE")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_loss_curve(history, title="Loss over epochs", log_y=False):
    if type(history) != dict:
        history = history.history

    loss = history.get("loss")
    val_loss = history.get("val_loss")

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Train Loss")
    if val_loss is not None:
        plt.plot(val_loss, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    if log_y:
        plt.yscale("log")

    plt.tight_layout()
    plt.show()

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