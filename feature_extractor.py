import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureExtractor:
    def __init__(self):
        self.selector = SelectKBest(f_classif, k=10)
        self.feature_names = []
    
    def extract_statistical_features(self, data):
        features = {}
        features['mean'] = np.mean(data, axis=1)
        features['std'] = np.std(data, axis=1)
        features['median'] = np.median(data, axis=1)
        features['max'] = np.max(data, axis=1)
        features['min'] = np.min(data, axis=1)
        return pd.DataFrame(features)
    
    def create_interaction_features(self, X):
        n_features = X.shape[1]
        interactions = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                interactions.append(X[:, i] * X[:, j])
        return np.column_stack(interactions)
    
    def select_best_features(self, X, y, k=10):
        self.selector.k = k
        X_selected = self.selector.fit_transform(X, y)
        return X_selected, self.selector.get_support()