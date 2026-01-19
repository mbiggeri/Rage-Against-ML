import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def update_json(task, model_name, metrics_dict, filename='all_results.json'):
    """
    Reads all_results.json, updates the entry for the given task and model, and saves it.
    
    Args:
        task (str): The task name (e.g., "Monk-1", "CUP").
        model_name (str): The model name (e.g., "PyTorch", "Keras", "SVM").
        metrics_dict (dict): Dictionary containing metrics (e.g., {"test_accuracy": 0.95, "MEE": 1.2}).
        filename (str): Path to the JSON file.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    if task not in data:
        data[task] = {}
    
    # Update or create the model entry
    if model_name not in data[task]:
        data[task][model_name] = {}
        
    data[task][model_name].update(metrics_dict)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Updated {filename} for Task: {task}, Model: {model_name}")

def standard_plot(history_dict, title, filename):
    """
    Generates a 2-subplot graph (Loss | Metric).
    Checks if val_loss exists (skip plotting validation if not).
    
    Args:
        history_dict (dict): Dictionary containing history keys (loss, val_loss, accuracy, val_accuracy, etc.).
                             Expected keys: 'loss', 'val_loss' (optional), and one metric key pair.
        title (str): Title for the plot.
        filename (str): Filename to save the plot.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Train Loss')
    if 'val_loss' in history_dict:
        plt.plot(history_dict['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Determine Metric to plot (anything that is not loss/val_loss)
    metric_keys = [k for k in history_dict.keys() if 'loss' not in k and not k.startswith('val_')]
    
    if metric_keys:
        metric = metric_keys[0]
        val_metric = f"val_{metric}"
        
        plt.subplot(1, 2, 2)
        plt.plot(history_dict[metric], label=f'Train {metric}')
        if val_metric in history_dict:
            plt.plot(history_dict[val_metric], label=f'Val {metric}')
        elif 'test_' + metric in history_dict: # Handle test_metric naming if applicable
             plt.plot(history_dict['test_' + metric], label=f'Test {metric}')

        plt.title(f'{title} - {metric}')
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """
    Computes and plots the confusion matrix.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels (will be thresholded at 0.5).
        title (str): Title for the plot.
        filename (str): Filename to save the plot.
    """
    # Ensure binary predictions
    y_pred_bin = (np.array(y_pred) > 0.5).astype(int)
    y_true_bin = (np.array(y_true) > 0.5).astype(int)
    
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title(f"{title}\nConfusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix to {filename}")
