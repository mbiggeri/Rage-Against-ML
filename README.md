# Rage Against the Machine Learning - Project Overview

This repository contains the code for the **Machine Learning 2025 Project**. It includes scripts for data analysis, model training (SVM, Neural Networks, Ensembles), hyperparameter optimization, and final model evaluation.

## 📂 1. Notebooks (`.ipynb`)

These notebooks are used for exploration, analysis, and running experiments interactively.

* **`dataset-analysis.ipynb`**
* Performs exploratory data analysis (EDA) on the Monk and CUP datasets. It includes visualization of feature distributions, correlation matrices, and PCA (Principal Component Analysis) to understand the data structure.


* **`svm_analysis.ipynb`**
* Dedicated to Support Vector Machines (SVM). It handles data loading, preprocessing, hyperparameter tuning (using Random Search), and performance evaluation for SVM models on both Monk and CUP tasks.


* **`keras_nn.ipynb`**
* The main notebook for Neural Network experiments using Keras/TensorFlow. It defines model architectures, runs training loops, and evaluates results. It likely integrates with the optimization results found by Optuna.


* **`ensemble_optimization.ipynb`**
* Focuses on building and optimizing an Ensemble model. It combines multiple trained models (e.g., best neural networks) to improve prediction stability and accuracy, using techniques like averaging or weighted voting.



## 📜 2. Main Scripts (`.py`)

Executable scripts for running long-duration tasks like hyperparameter search.

* **`optuna_search_mlcup_nn.py`**
* A script dedicated to automating hyperparameter optimization for Neural Networks using **Optuna**. It defines the search space (layers, units, learning rates) and runs many trials to find the best configuration for the ML-CUP dataset.


* **`training_utils.py`**
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
* **`optuna.py`**: Helper functions to streamline Optuna studies, such as saving/loading study databases or defining objective functions.
* **`keras.py`**: Utilities specific to Keras models, possibly custom callbacks or layer definitions.

### `losses/`

* **`MeanEuclidianError.py`**: Implementation of the **MEE (Mean Euclidean Error)** loss function, which is the specific metric required for the ML-CUP competition.

### `executors/`

* **`executors.py`**: Classes or functions that manage the execution flow of training and testing, helping to decouple the model logic from the running logic.

---

### 🚀 Quick Start

1. **Explore Data:** Start with `dataset-analysis.ipynb`.
2. **Optimize Models:** Run `optuna_search_mlcup_nn.py` to find the best hyperparameters.
3. **Train & Evaluate:** Use `keras_nn.ipynb` or `svm_analysis.ipynb` to train your final models using the best parameters found.
4. **Final Submission:** (To be implemented) Use the best models to generate predictions on the blind test set.