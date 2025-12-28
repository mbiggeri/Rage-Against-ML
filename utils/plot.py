import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


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

def plot_optuna_vs_random(optuna_csv_path, rs_csv_path, baseline=None):
       # --- 1. CARICAMENTO E PULIZIA OPTUNA ---
    df_optuna = pd.read_csv(optuna_csv_path)
    
    # Ordiniamo cronologicamente per numero di trial
    df_optuna = df_optuna.sort_values('number')
    
    # Prendiamo i valori (che contengono NaN per i trial falliti)
    values_optuna = df_optuna['value'].values
    
    # TRUCCO: Sostituiamo i NaN con infinito
    # Così il "minimo" non cambierà e la linea resterà piatta nei buchi
    values_optuna_no_nan = np.where(np.isnan(values_optuna), np.inf, values_optuna)
    
    # Calcolo del minimo cumulativo (Best so far)
    conv_optuna = np.minimum.accumulate(values_optuna_no_nan)
    
    first_valid = \
        df_optuna['value'].dropna().iloc[0] if not df_optuna['value'].dropna().empty else 300
    conv_optuna[conv_optuna == np.inf] = first_valid

    # --- 2. CARICAMENTO E PULIZIA RANDOM SEARCH ---
    df_rs = pd.read_csv(rs_csv_path)
    
    # Sklearn di solito non ha NaN, ma applichiamo la stessa logica per sicurezza
    scores_rs = df_rs['mean_test_score'].values
    if np.all(scores_rs < 0): scores_rs = -scores_rs # Invertiamo se neg_mse
    
    scores_rs_no_nan = np.where(np.isnan(scores_rs), np.inf, scores_rs)
    conv_rs = np.minimum.accumulate(scores_rs_no_nan)

    # --- 3. PLOT ---
    plt.figure(figsize=(11, 6))

    if baseline:
        plt.axhline(y=baseline, color='gray', linestyle=':', label='Baseline (Mean Prediction)')
    
    # Plot Optuna
    plt.plot(range(len(conv_optuna)), conv_optuna, label='Optuna Optimization (TPE)', 
             color='#1f77b4', linewidth=2.5, zorder=3)
    
    # Plot Randomized Search
    plt.plot(range(len(conv_rs)), conv_rs, label='Standard Randomized Search', 
             color='#d62728', linestyle='--', linewidth=2, zorder=2)
    
    # Estetica
    plt.title('Convergence Analysis: Best Score Found Over Iterations', fontsize=14)
    plt.xlabel('Number of Iterations', fontsize=12)
    plt.ylabel('Best MSE found so far', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.show()