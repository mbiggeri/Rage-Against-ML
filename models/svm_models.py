from sklearn.svm import SVC, SVR

class SVCModel:
    """
    Wrapper for Support Vector Classifier (SVC) model.
    """
    def __init__(self, **kwargs):
        # Sets up the classifier with the passed parameters
        self.model = SVC(**kwargs)
        # Training (self.model.fit) and evaluation will be done in main.py

class SVRModel:
    """
    Wrapper for Support Vector Regressor (SVR) model.
    """
    def __init__(self, **kwargs):
        # Sets up the regressor with the passed parameters
        self.model = SVR(**kwargs)
        # Training (self.model.fit) and evaluation will be done in main.py