import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# --- Import Models ---
from models import IterativeRefinementNet, StandardFeedForwardNet

# --- Import Data Loaders ---
from data_loader import get_monk1_data, get_ml_cup_data

def get_data(dataset_name, batch_size):
    """
    Selects and loads the correct dataset based on the name.
    """
    data_root = './data'
    
    # 1. Custom Datasets (MONK & ML-CUP)
    if dataset_name == 'monk1':
        # Returns: train_loader, test_loader, input_size, output_size
        return get_monk1_data(batch_size, data_root)
    
    elif dataset_name == 'mlc25':
        return get_ml_cup_data(batch_size, data_root)
    
    # 2. TorchVision Datasets (Images)
    elif dataset_name in ['mnist', 'fmnist', 'kmnist']:
        # Define specific statistics for normalization
        if dataset_name == 'mnist':
            dataset_class = torchvision.datasets.MNIST
            # Mean and Std for MNIST
            stats = ((0.1307,), (0.3081,))
        elif dataset_name == 'fmnist':
            dataset_class = torchvision.datasets.FashionMNIST
            # Mean and Std for FashionMNIST
            stats = ((0.2860,), (0.3530,))
        elif dataset_name == 'kmnist':
            dataset_class = torchvision.datasets.KMNIST
            # Mean and Std for KMNIST
            stats = ((0.1918,), (0.3483,))
        
        # Standard transform: Convert to Tensor -> Normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        
        # Download and load
        train_d = dataset_class(root=data_root, train=True, transform=transform, download=True)
        test_d = dataset_class(root=data_root, train=False, transform=transform, download=True)
        
        # Image datasets inputs are 28x28 = 784, output is 10 classes
        return (DataLoader(train_d, batch_size=batch_size, shuffle=True),
                DataLoader(test_d, batch_size=batch_size, shuffle=False),
                784, 10)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def objective(trial):
    """
    The main function Optuna optimizes.
    1. Optuna 'suggests' hyperparameters.
    2. We build and train a model with those parameters.
    3. We return a score (accuracy or MSE) to Optuna.
    """
    
    # --- 1. Define the Search Space ---
    # You can edit these ranges to control what Optuna explores.
    
    # Model Architecture
    model_type = trial.suggest_categorical("model", ["standard", "step_out"])
    
    # Activation functions
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid", "gelu"])
    
    # Learning Rate: log=True means it explores scales (0.001, 0.01, 0.1) evenly
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    
    # Batch Size
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Dynamic Layers:
    # "n_layers" decides how deep the network is (1 to 3 hidden layers)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_sizes = []
    for i in range(n_layers):
        # For each layer, suggest a width (neurons)
        hidden_sizes.append(trial.suggest_int(f"n_units_l{i}", 16, 256))

    # --- 2. Setup Data ---
    # We get data based on the suggested batch_size
    train_loader, val_loader, input_size, output_size = get_data(args.dataset, batch_size)
    
    # Detect Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # --- 3. Build Model ---
    if model_type == 'standard':
        model = StandardFeedForwardNet(input_size, hidden_sizes, output_size, activation).to(device)
    else:
        # Step_out needs num_iterations. We set it equal to the number of hidden layers.
        model = IterativeRefinementNet(input_size, hidden_sizes, len(hidden_sizes), output_size, activation).to(device)

    # --- 4. Loss & Optimizer ---
    is_regression = (args.dataset == 'mlc25')
    
    # MSE for regression, CrossEntropy for classification
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- 5. Training Loop (with Pruning) ---
    # We train for a limited number of epochs to speed up search.
    epochs = args.epochs 

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # --- Validation Step ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if is_regression:
                    val_loss += criterion(output, target).item()
                else:
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)

        # Calculate the metric to report
        if is_regression:
            metric = val_loss / len(val_loader) # Mean Squared Error
        else:
            metric = correct / total # Accuracy (0.0 to 1.0)

        # --- OPTUNA PRUNING ---
        # Report the current epoch's score to Optuna
        trial.report(metric, epoch)

        # If this trial is performing very poorly compared to others, stop it early.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the final metric of this trial
    return metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Search')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['monk1', 'mlc25', 'mnist', 'fmnist', 'kmnist'],
                        help='Dataset to use.')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials (experiments) to run')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per trial')
    args = parser.parse_args()

    # Define optimization direction
    # Regression (ML-CUP): Lower MSE is better -> minimize
    # Classification: Higher Accuracy is better -> maximize
    direction = "minimize" if args.dataset == 'mlc25' else "maximize"
    
    print(f"Starting optimization for {args.dataset.upper()}...")
    print(f"Goal: {direction} metric.")
    
    # Create study with a "Median Pruner" (stops trials worse than the median of previous ones)
    study = optuna.create_study(direction=direction, pruner=optuna.pruners.MedianPruner())
    
    try:
        study.optimize(objective, n_trials=args.trials)
    except KeyboardInterrupt:
        print("\nSearch stopped by user.")

    # --- Summary ---
    print("\n" + "="*40)
    print(f"Optimization Finished. Trials completed: {len(study.trials)}")
    
    if len(study.trials) > 0:
        best_trial = study.best_trial
        print(f"Best Value: {best_trial.value:.4f}")
        print("Best Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

        # --- Save Results ---
        os.makedirs("results", exist_ok=True)
        
        filename = f"results/best_params_{args.dataset}.json"
        with open(filename, "w") as f:
            json.dump(best_trial.params, f, indent=4)
        print(f"Results saved to {filename}")
        print("="*40)