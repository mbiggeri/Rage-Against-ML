# Rage Against the Machine Learning - Project Overview

This repository contains the code for the **Machine Learning 2025 Project**. It includes scripts for data analysis, model training (SVM, Neural Networks, Ensembles), hyperparameter optimization, and final model evaluation.

## 📂 1. Notebooks (`.ipynb`)

These notebooks are used for exploration, analysis, and running experiments interactively.

* **`dataset-analysis.ipynb`**
* Performs exploratory data analysis (EDA) on the Monk and CUP datasets. It includes visualization of feature distributions, correlation matrices, and PCA (Principal Component Analysis) to understand the data structure.


* **`svm_analysis.ipynb`**
* Dedicated to Support Vector Machines (SVM). It handles data loading, preprocessing, hyperparameter search using Grid Search, Random Search or Optuna and performance evaluation for SVM models on both Monk and CUP tasks.


* **`keras_nn.ipynb`**
* The main notebook for Neural Network experiments using Keras/TensorFlow. It defines model architectures, runs training loops, and evaluates results. It likely integrates with the optimization results found by Optuna.


* **`ensemble_mlcup.ipynb`**
* Focuses on building and evaluating a **Heterogeneous Ensemble** model for the ML-CUP task. It loads the best trained models (PyTorch, Keras, SVM), combines their predictions (e.g., via averaging), and generating the final blind test submission file.



## 📜 2. Main Scripts (`.py`)

Executable scripts for running long-duration tasks like hyperparameter search.

* **`utils/optuna_search_mlcup_nn.py`**
* A script dedicated to automating hyperparameter optimization for Neural Networks using **Optuna**. It defines the search space (layers, units, learning rates) and runs many trials to find the best configuration for the ML-CUP dataset.


* **`utils/training_utils.py`**
* Contains helper functions used across notebooks and scripts for training loops, including functions to split data, handle callbacks, or manage experiment logging.



## 🛠️ 3. Modules & Libraries

Reusable code organized by functionality.

### `models/`

* **`standard.py`**: Defines standard/baseline model classes or factories.
* **`svm_models.py`**: Wrappers and utility classes specifically for creating and managing SVM models (SVR/SVC).
* **`ensemble.py`**: Implements the `Ensemble` class logic, allowing you to combine predictions from multiple base estimators.

### `utils/`

* **`data_loader.py`**: Functions to load CSV files (`ML-CUP25-TR.csv`, `monks-*.csv`), handle parsing, and perform initial preprocessing.
* **`plot.py`**: Visualization utilities to generate consistent plots for learning curves (Loss/MEE vs. Epochs) and model comparisons.
* **`optuna_utils.py`**: Helper functions to streamline Optuna studies, such as saving/loading study databases or defining objective functions.
* **`keras.py`**: Utilities specific to Keras models, possibly custom callbacks or layer definitions.

### `losses/`

* **`MeanEuclidianError.py`**: Implementation of the **MEE (Mean Euclidean Error)** loss function, which is the specific metric required for the ML-CUP competition.

### `executors/`

* **`executors.py`**: Contains `Executor` classes (e.g. `OptunaRegressorExecutor`) that encapsulate the complex logic for running training loops, cross-validation, and hyperparameter searches. They help keep the notebooks clean by handling the execution flow (data splitting, model building, trial execution) behind the scenes.

---

### 🚀 Quick Start

1. **Explore Data:** Start with `dataset-analysis.ipynb` for EDA.
2. **Optimize Models:** Run `optuna_search_mlcup_nn.py` (or use the cells in `pytorch_nn.ipynb`) to find the best hyperparameters.
3. **Train & Evaluate:** 
   - Run `pytorch_nn.ipynb` and `keras_nn.ipynb` to train the best Neural Network models.
   - Run `svm_analysis.ipynb` to find and train the best SVM models.
   - All notebooks will save their best models to the `models/` directory and log results to `all_results.json`.
4. **Compare & Ensemble:**
   - Run `compare_models.ipynb` to visualize performance across frameworks.
   - Run `ensemble_mlcup.ipynb` to load the best saved models, evaluate the heterogeneous ensemble, and generate the final `submission_ensemble.csv` for the blind test set.
