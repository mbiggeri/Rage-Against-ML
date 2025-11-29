import os
import torch
import requests
import sys
from torch.utils.data import DataLoader, TensorDataset
from dataloaders import MLCupDataLoader, MLCupDataset

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin

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

def get_ml_cup_data(batch_size, data_root='./data', validation: bool=True, test_ratio=.10, validation_ratio=.15, scaler: TransformerMixin=None) -> tuple[MLCupDataLoader, MLCupDataLoader, MLCupDataLoader, int, int]:
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

    if validation:
        val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=test_ratio/(test_ratio + validation_ratio)) 

    if scaler:
        print(f"applying scaling {scaler.__class__.__name__} on train_X and test_X")
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        if validation:
            val_x = scaler.fit_transform(val_x)
        test_x = scaler.transform(test_x)
    
    validation_loader = None
    if validation:
        val_dataset = MLCupDataset(val_x, val_y)
        validation_loader = MLCupDataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    train_dataset = MLCupDataset(train_x, train_y)
    test_dataset = MLCupDataset(test_x, test_y)
    
    train_loader = MLCupDataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = MLCupDataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Based on the parser above
    input_size = 10
    output_size = 2
    
    return train_loader, validation_loader, test_loader, input_size, output_size