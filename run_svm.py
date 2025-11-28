import argparse
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Import data loading functions
from data_loader import get_monk1_data, get_ml_cup_data

# Import SVM models and metrics
try:
    from models import SVCModel, SVRModel
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.multioutput import MultiOutputRegressor
except ImportError:
    print("FATAL ERROR: Scikit-learn or SVM Models not found.")
    print("Please ensure 'scikit-learn' is installed and 'svm_models.py' is in the 'models' directory.")
    sys.exit(1)


# --- Utility function to extract data from DataLoader to NumPy arrays ---
# SVM needs the full dataset loaded into memory as NumPy arrays (no batching)
def extract_data_to_numpy(data_loader):
    """
    Converts data from a PyTorch DataLoader into a flattened NumPy array pair (X, y).
    The targets (y) are returned in their original dimensionality (e.g., [N, M] for multi-output).
    """
    X_list = []
    y_list = []
    for X, y in data_loader:
        # Flatten the input (e.g., 28x28 image -> 784 features)
        X_list.append(X.view(X.size(0), -1).numpy()) 
        # Convert labels to NumPy
        y_list.append(y.numpy())
    
    X_data = np.concatenate(X_list)
    y_data = np.concatenate(y_list)
    
    # We return y_data as is (2D array). Dimensionality handling (e.g., .ravel()) 
    # for single-output tasks is done in the training block.
    return X_data, y_data

# --- Main Execution ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train and Evaluate SVM Models (SVC/SVR).')
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['svc', 'svr'],
                        help='Type of SVM model to train (svc for classification, svr for regression).')
    
    parser.add_argument('--dataset', type=str, default='monk1',
                        choices=['monk1', 'mlc25', 'mnist', 'fmnist', 'kmnist'],
                        help='Dataset to use (monk1, mlc25, mnist, fmnist, kmnist ).')
    
    parser.add_argument('--svm_kernel', type=str, default='rbf',
                        choices=['rbf', 'linear', 'poly', 'sigmoid'],
                        help='Kernel for SVM models (default: rbf).')
    
    # Other SVM hyperparameters (optional)
    parser.add_argument('--C', type=float, default=1.0,
                        help='Regularization parameter C (default: 1.0).')
    parser.add_argument('--gamma', type=str, default='scale',
                        help='Kernel coefficient gamma (default: scale).')

    args = parser.parse_args()
    
    # --- 1. Load Data ---
    BATCH_SIZE = 1024 # Use large batch size as we extract all data anyway
    data_root = './data'
    
    print(f"Loading dataset: {args.dataset.upper()}...")
    
    # Determine task type and load data (DataLoader objects are returned)
    if args.dataset == 'monk1':
        train_loader, test_loader, INPUT_SIZE, OUTPUT_SIZE = get_monk1_data(BATCH_SIZE, data_root)
        is_regression_task = False
        metric_name = "Test Accuracy (%)"
    elif args.dataset == 'mlc25':
        train_loader, test_loader, INPUT_SIZE, OUTPUT_SIZE = get_ml_cup_data(BATCH_SIZE, data_root)
        is_regression_task = True
        metric_name = "Test MSE" 
    # --- LOGIC FOR IMAGE DATASETS (MNIST, FMNIST, KMNIST) ---
    elif args.dataset in ['mnist', 'fmnist', 'kmnist']:
        
        # 1. Define specific parameters and dataset class
        if args.dataset == 'mnist':
            normalize_mean, normalize_std = (0.1307,), (0.3081,)
            dataset_class = torchvision.datasets.MNIST
        elif args.dataset == 'fmnist':
            normalize_mean, normalize_std = (0.2860,), (0.3530,)
            dataset_class = torchvision.datasets.FashionMNIST
        elif args.dataset == 'kmnist':
            normalize_mean, normalize_std = (0.1918,), (0.3483,)
            dataset_class = torchvision.datasets.KMNIST
            
        #INPUT_SIZE = 784 # 28 * 28 pixels
        #OUTPUT_SIZE = 10 # 10 classes
        is_regression_task = False
        metric_name = "Test Accuracy (%)"
        
        # 2. Transformations and DataLoaders
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])

        train_dataset = dataset_class(root=data_root, train=True, transform=transform, download=True)
        test_dataset = dataset_class(root=data_root, train=False, transform=transform, download=True)

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # --- END IMAGE DATASET LOGIC ---
    else:
        # This block handles the error if the dataset is outside the specified choices (monk1, mlc25).
        print("Unsupported dataset for SVM.")
        sys.exit(1)

    # --- 2. Data Preparation ---
    
    # Convert DataLoaders (PyTorch) to NumPy arrays (Scikit-learn)
    X_train, y_train = extract_data_to_numpy(train_loader)
    X_test, y_test = extract_data_to_numpy(test_loader)
    
    print(f"Data loaded: Training samples={X_train.shape[0]}, Test samples={X_test.shape[0]}")
    
    # --- 3. Model Initialization ---
    
    # Create the base model with hyperparameters
    base_svm_model = None
    if args.model == 'svc':
        if is_regression_task:
            print("ERROR: SVC (Classification) cannot be used for Regression (mlc25).")
            sys.exit(1)
        # Get the scikit-learn SVC object
        base_svm_model = SVCModel(kernel=args.svm_kernel, C=args.C, gamma=args.gamma).model
        
    elif args.model == 'svr':
        # Get the scikit-learn SVR object
        base_svm_model = SVRModel(kernel=args.svm_kernel, C=args.C, gamma=args.gamma).model
        
    # --- Assign the Final Model Container (Handle Multi-Output) ---
    if is_regression_task and y_train.shape[1] > 1:
        # If Regression AND Multi-Output (MLC25), use the MultiOutputRegressor wrapper
        model_container = MultiOutputRegressor(base_svm_model)
    else:
        # If Classification (MONK1) or Single-Output Regression, use the base model
        model_container = base_svm_model
    
    print(f"Using Model: {args.model.upper()} (Kernel: {args.svm_kernel}, Multi-Output: {is_regression_task and y_train.shape[1] > 1})")

    # --- 4. Training and Evaluation ---
    
    # 1. Prepare Target Data Format (y)
    
    if is_regression_task and y_train.shape[1] > 1:
        # Multi-Output Regression: target must be 2D [N, M]
        y_train_fit = y_train 
        y_test_eval = y_test
    else:
        # Classification (SVC) or Single-Output Regression: target must be 1D [N,]
        y_train_fit = y_train.ravel() 
        y_test_eval = y_test.ravel() 
    
    print("\n--- Starting SVM Training ---")
    
    # 2. Training (model_container is SVC, SVR, or MultiOutputRegressor)
    model_container.fit(X_train, y_train_fit) 
    
    print("--- Training Completed ---")

    # 3. Evaluation
    y_test_pred = model_container.predict(X_test)
    
    if is_regression_task:
        # Calculate MSE for Regression
        final_metric = mean_squared_error(y_test_eval, y_test_pred)
    else:
        # Calculate Accuracy for Classification
        final_metric = accuracy_score(y_test_eval, y_test_pred) * 100.0
    
    # --- 5. Print Results ---
    print("\n--- Final Results ---")
    print(f"Model: {args.model.upper()} (Kernel: {args.svm_kernel})")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"{metric_name}: {final_metric:.4f}")
    print("---------------------")