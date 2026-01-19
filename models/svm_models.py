import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.model_selection import learning_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, RocCurveDisplay

class BaseSVMWrapper:
    """
    Base class for SVM model wrappers.
    """
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.params = kwargs

        if model_type == 'svc':
            self.model = SVC(**kwargs)
        elif model_type == 'svr':
            self.model = SVR(**kwargs)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)  

    def decision_function(self, X): 
        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        else:
            raise NotImplementedError("Decision function not available for this model type.")
    
    def save_pkl(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.model, file_path)

    def get_params_clean(self):
        params = self.model.get_params()
        clean_params = {}
        for k,v in params.items():
            if isinstance(v, np.int64): clean_params[k] = int(v)
            elif isinstance(v, np.float64): clean_params[k] = float(v) 
            else: clean_params[k] = v
        return clean_params
    
    @classmethod
    def load(cls, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No model found at {file_path}")
        model = joblib.load(file_path)
        m_type = 'svc' if isinstance(model, SVC) else 'svr'
        instance = cls(m_type)
        instance.model = model
        instance.params = model.get_params()
        return instance
    
    def plot_learning_curve(self, X, y, cv=5, scoring=None):

        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5)
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        if train_scores_mean[0] < 0:
            train_scores_mean = -train_scores_mean
            test_scores_mean = -test_scores_mean

        if scoring is None:
            ylabel = 'Accuracy'
        elif isinstance(scoring, str) and 'mean_squared' in scoring:
            ylabel = 'Mean Squared Error'
        else:
            ylabel = 'Mean Euclidean Error'

        plt.figure(figsize=(8, 5))
        plt.title(f'Learning Curve ({self.model_type.upper()})')
        plt.xlabel('Training Examples')
        plt.ylabel(ylabel)
        plt.grid(True)

        plt.fill_between(train_sizes, 
                         train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, 
                         alpha=0.1, color="r")
        plt.fill_between(train_sizes, 
                         test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, 
                         alpha=0.1, color="g")

        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
        
        plt.legend(loc='best')
        plt.show()
        


class SVCModel(BaseSVMWrapper):
    """
    Wrapper for Support Vector Classifier (SVC) model.
    """
    def __init__(self, **kwargs):
        super().__init__(model_type='svc', **kwargs)
    
    def plot_classification_analysis(self, X_test, y_test):
        y_pred = self.predict(X_test)

        try:
            y_scores = self.decision_function(X_test)
        except AttributeError:
            y_scores = y_pred

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax[0], cmap='Blues')
        ax[0].set_title('Confusion Matrix')

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr) * 100

        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax[1])
        ax[1].set_title(f'ROC Curve (AUC = {roc_auc:.2f}%)')
        ax[1].plot([0, 1], [0, 1], 'r--', label='Random Guessing')
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

class SVRModel(BaseSVMWrapper):
    """
    Wrapper for Support Vector Regressor (SVR) model.
    """
    def __init__(self, **kwargs):
      super().__init__(model_type='svr', **kwargs)
    
    def plot_regression_analysis(self, X_test, y_test, title_suffix=''):
        y_pred = self.predict(X_test)

        plt.figure(figsize=(7, 7))

        plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='k', label='Predictions')
        
        all_vals = np.concatenate([y_test, y_pred])
        min_val, max_val = all_vals.min(), all_vals.max()

        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

        plt.title(f'True vs Predicted Values {title_suffix}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()