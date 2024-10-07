from .V_JEPA import VJEPA
from .utils import model_cleanup, feature_aggregator
import os
import numpy as np

class JEDiMetric:
    def __init__(self, feature_path=None, model_dir=None):
        self.feature_path = feature_path
        if feature_path is not None:
            import warnings
            warnings.warn("feature_path is not provided, will not save computed features.")
        self.model_dir = model_dir if model_dir is not None else os.getcwd()
    
    def compute_metric(self):
        assert hasattr(self, 'train_features'), "train_features is not loaded"
        assert hasattr(self, 'test_features'), "test_features is not loaded"
        from .mmd_polynomial import mmd_poly
        return mmd_poly(self.train_features, self.test_features, degree=2, coef0=0)*100

    def load_features(self, train_loader=None, test_loader=None, num_samples=5000):
        if os.path.exists(f'{self.feature_path}/train.npy'):
            self.train_features = np.load(f'{self.feature_path}/train.npy')
        else:
            if not hasattr(self, 'vjepa'):
                self.vjepa = VJEPA(model_dir=self.model_dir)
            print("Computing features for training set")
            assert train_loader is not None, "train_loader is not provided"
            self.train_features = feature_aggregator(self.vjepa, train_loader, num_samples=num_samples,
                    filename=f'{self.feature_path}/train.npy' if self.feature_path is not None else None)
        
        if os.path.exists(f'{self.feature_path}/test.npy'):
            self.test_features = np.load(f'{self.feature_path}/test.npy')
        else:
            if not hasattr(self, 'vjepa'):
                self.vjepa = VJEPA(model_dir=self.model_dir)
            print("Computing features for testing set")
            assert test_loader is not None, "test_loader is not provided"
            self.test_features = feature_aggregator(self.vjepa, test_loader, num_samples=num_samples,
                    filename=f'{self.feature_path}/test.npy' if self.feature_path is not None else None)
        
        if hasattr(self, 'vjepa'):
            model_cleanup(self.vjepa)
        return self.train_features, self.test_features
        