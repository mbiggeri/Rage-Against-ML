import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
import os
import requests
import sys
import numpy as np

# --- 1. Import Models ---
from models import IterativeRefinementNet, StandardFeedForwardNet

# Import SVM and metrics from scikit-learn
try:
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
except ImportError:
    print("scikit-learn not found. SVM model will not be available.")
    print("Please install it with: pip install scikit-learn")
    SVC = None
    accuracy_score = None


# --- 2. Data Loading ---

def get_monk1_data(batch_size, data_root='./data'):
    monk_dir = os.path.join(data_root, 'monk')
    os.makedirs(monk_dir, exist_ok=True)
    
    train_file = os.path.join(monk_dir, 'monks-1.train')
    test_file = os.path.join(monk_dir, 'monks-1.test')
    
    # Download if files don't exist
    if not os.path.exists(train_file):
        print("Downloading MONK-1 train data...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train"
        r = requests.get(url)
        with open(train_file, 'w') as f:
            f.write(r.text)
            
    if not os.path.exists(test_file):
        print("Downloading MONK-1 test data...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test"
        r = requests.get(url)
        with open(test_file, 'w') as f:
            f.write(r.text)

    # One-hot encoding definitions for 6 attributes (Total: 17 features)
    attr_dims = [3, 3, 2, 3, 4, 2]
    
    def parse_monk_file(file_path):
        features = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                label = int(parts[0])
                attrs = [int(a) for a in parts[1:-1]] # last part is ID
                
                # One-hot encode features
                one_hot_features = []
                for i, attr_val in enumerate(attrs):
                    one_hot = torch.zeros(attr_dims[i])
                    one_hot[attr_val - 1] = 1.0 # Values are 1-based
                    one_hot_features.append(one_hot)
                
                features.append(torch.cat(one_hot_features))
                labels.append(label)
                
        return torch.stack(features), torch.tensor(labels, dtype=torch.long)

    print("Parsing MONK-1 data...")
    train_x, train_y = parse_monk_file(train_file)
    test_x, test_y = parse_monk_file(test_file)
    
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = 17
    output_size = 2
    
    return train_loader, test_loader, input_size, output_size

def get_ml_cup_data(batch_size, data_root='./data'):
    # data will be in ./MLC25/
    ml_cup_dir = os.path.join(data_root, 'MLC25')
    os.makedirs(ml_cup_dir, exist_ok=True)
    
    train_file = os.path.join(ml_cup_dir, 'ML-CUP25-TR.csv')
    test_file = os.path.join(ml_cup_dir, 'ML-CUP25-TS.csv')

    if not os.path.exists(train_file):
        print("Downloading ML-CUP25-TR train data...")
        url = "https://gist.githubusercontent.com/FlavRomano/a19771d5c67f71dad557e5fa384db38b/raw/7290bff843b8a5c3a650457281c93c1d54e55f51/ML-CUP25-TR.csv"
        r = requests.get(url)
        with open(train_file, 'w') as f:
            f.write(r.text)
            
    if not os.path.exists(test_file):
        print("Downloading ML-CUP25-TS test data...")
        url = "https://gist.githubusercontent.com/FlavRomano/453dc2affc584028cb122d6b52cec295/raw/1cb1e84b26f8efd2ac081701d610c94498f988e1/ML-CUP25-TS.csv"
        r = requests.get(url)
        with open(test_file, 'w') as f:
            f.write(r.text)
    
    # --- Check if files exist ---
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("---" * 20)
        print(f"ERROR: ML-CUP dataset files not found.")
        print(f"This script cannot download the ML-CUP dataset automatically.")
        print(f"Please manually place your dataset files at these locations:")
        print(f"Training data: {os.path.abspath(train_file)}")
        print(f"Test data:     {os.path.abspath(test_file)}")
        print("---" * 20)
        sys.exit(1) # Stop the script

    # --- Parser for ML-CUP data ---
    # This parser assumes the standard ML-CUP format:
    # - Lines starting with '#' are comments
    # - Data is comma-separated
    # - Column 0: ID (ignored)
    # - Columns 1-10: 10 input features
    # - Columns 11-12: 2 output targets (regression)
    def parse_ml_cup_file(file_path):
        features = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                if len(parts) < 13:
                    print(f"Warning: Skipping malformed line: {line}")
                    continue
                
                try:
                    # Features are columns 1 through 10 (10 features)
                    feature_values = [float(p) for p in parts[1:11]]
                    # Labels are columns 11 and 12 (2 targets)
                    label_values = [float(p) for p in parts[11:13]]
                    
                    features.append(torch.tensor(feature_values, dtype=torch.float32))
                    labels.append(torch.tensor(label_values, dtype=torch.float32))
                except ValueError as e:
                    print(f"Warning: Skipping line due to parsing error ({e}): {line}")

        return torch.stack(features), torch.stack(labels)

    print("Parsing ML-CUP data...")
    train_x, train_y = parse_ml_cup_file(train_file)
    test_x, test_y = parse_ml_cup_file(test_file)
    
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # Based on the parser above
    input_size = 10
    output_size = 2
    
    print("---" * 10)
    print("WARNING: The 'step_out' model uses a final-state activation (e.g., ReLU).")
    print("This is likely unsuitable for the ML-CUP regression task (which can have")
    print("negative targets). The 'standard' model is recommended.")
    print("---" * 10)
    
    return train_loader, test_loader, input_size, output_size


# --- 3. Main Training Script ---
if __name__ == "__main__":

    # --- 3.1 Setup Command-Line Arguments ---
    parser = argparse.ArgumentParser(description='Train a Neural Network or SVM.')
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['step_out', 'standard'],
                        help='Type of model to train (iterative, standard)')
    
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fmnist', 'kmnist', 'monk1', 'mlc25'],
                        help='Dataset to use (mnist, f_mnist, kmnist, monk1 or mlc25).')
    
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001).')
    
    parser.add_argument('--hidden_sizes', type=int, nargs='+', required=True,
                        help='A list of hidden layer sizes (e.g. --hidden_sizes 256 256 256).')
    
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh', 'sigmoid'],
                        help='Activation function for hidden layers (default: relu).')


    args = parser.parse_args()
    
    # --- 3.2 Set Hyperparameters and Load Data ---
    NUM_ITERATIONS = len(args.hidden_sizes) 
    
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    ACTIVATION = args.activation

    train_loader = None
    test_loader = None
    INPUT_SIZE = None
    OUTPUT_SIZE = None
    data_root = './data'
    
    print(f"Loading dataset: {args.dataset.upper()}")
    if args.dataset == 'monk1':
        train_loader, test_loader, INPUT_SIZE, OUTPUT_SIZE = get_monk1_data(BATCH_SIZE, data_root)
        
    elif args.dataset == 'mlc25':
        train_loader, test_loader, INPUT_SIZE, OUTPUT_SIZE = get_ml_cup_data(BATCH_SIZE)
        is_regression_task = True
        metric_name = "Test MSE" # Update metric name for regression (not classification accuracy)
        
    else:
        if args.dataset == 'mnist':
            normalize_mean, normalize_std = (0.1307,), (0.3081,)
            dataset_class = torchvision.datasets.MNIST
        elif args.dataset == 'fmnist':
            normalize_mean, normalize_std = (0.2860,), (0.3530,)
            dataset_class = torchvision.datasets.FashionMNIST
        elif args.dataset == 'kmnist':
            normalize_mean, normalize_std = (0.1918,), (0.3483,)
            dataset_class = torchvision.datasets.KMNIST

        INPUT_SIZE = 784
        OUTPUT_SIZE = 10

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])

        train_dataset = dataset_class(root=data_root, train=True, transform=transform, download=True)
        test_dataset = dataset_class(root=data_root, train=False, transform=transform, download=True)

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    
    # --- 3.3 SVM MODEL PATH ---
    # TODO: Implement SVM training and evaluation
        

    # --- 3.4 PYTORCH MODEL PATH ---
    # This code only runs if args.model is 'step_out' or 'standard'

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: Apple MPS (M1 GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using device: NVIDIA CUDA")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
        
    print(f"Detected Input Size: {INPUT_SIZE}, Output Size: {OUTPUT_SIZE}")
    print(f"Using Model: {args.model.upper()}")
    print(f"Using Activation: {ACTIVATION.upper()}")

    model = None
    if args.model == 'step_out':
        model = IterativeRefinementNet(
            input_size=INPUT_SIZE,
            hidden_sizes=args.hidden_sizes,
            num_iterations=NUM_ITERATIONS,
            output_size=OUTPUT_SIZE,
            activation_str=ACTIVATION
        ).to(device)
    
    elif args.model == 'standard':
        model = StandardFeedForwardNet(
            input_size=INPUT_SIZE,
            hidden_sizes=args.hidden_sizes,
            output_size=OUTPUT_SIZE,
            activation_str=ACTIVATION
        ).to(device)
        
    if model is None:
        raise ValueError(f"Unknown PyTorch model_type: {args.model}")

    # --- Select Loss Function based on task ---
    if is_regression_task:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Model architecture:")
    print(model)
    print(f"Training for {NUM_EPOCHS} epochs...")

    epoch_train_losses = []
    epoch_test_metrics = []

    # --- 3.5 Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train() 
        running_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            
            # For classification, data needs flattening (e.g., MNIST)
            # For regression, data is already flat (e.g., ML-CUP)
            # The models handle this flattening internally
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            log_frequency = max(1, len(train_loader) // 4)
            if (i+1) % log_frequency == 0 or (i+1) == len(train_loader):
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                
        avg_train_loss = running_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] complete. Average Training Loss: {avg_train_loss:.4f}')

        # --- 3.6 Evaluation Loop ---
        model.eval()
        running_metric = 0.0    # Could be accuracy or MSE
        correct = 0     # For classification accuracy
        total = 0   # For classification accuracy
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)
                
                outputs = model(data)
                
                if is_regression_task:
                    # Calculate loss (MSE) for regression
                    loss = criterion(outputs, labels)
                    running_metric += loss.item()
                else:
                    # Calculate accuracy for classification
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        if is_regression_task:
            metric = running_metric / len(test_loader) # Average Test MSE
            epoch_test_metrics.append(metric)
            print(f'Average Test MSE (lower is better) after epoch {epoch+1}: {metric:.4f}')
        else:
            metric = 100 * correct / total # Test Accuracy
            epoch_test_metrics.append(metric)
            print(f'Accuracy of the network after epoch {epoch+1}: {metric:.2f} %')


    print('Finished Training!')

    # --- 3.7 Plotting Results ---
    print("Generating plot...")
    epochs_range = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, epoch_train_losses, 'o-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, epoch_test_metrics, 'o-', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(f'Training Metrics ({args.dataset.upper()} - {args.model.title()})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    plt.show()