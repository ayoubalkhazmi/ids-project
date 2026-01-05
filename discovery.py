import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from .preprocessing import DataPreprocessor
import os

class BehavioralDiscovery:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        
    def get_discovery_map(self, data_list):
        """
        Takes a list of flow records and returns:
        - PCA (2D) coordinates
        - DBSCAN cluster labels
        - Original status (from labels if available)
        """
        if not data_list:
            return []
            
        df = pd.DataFrame(data_list)
        
        # 1. Preprocess
        try:
            X_scaled = self.preprocessor.transform(df)
        except:
            # If not fitted, we might need a fallback or just return empty
            return []
            
        # 2. PCA for 2D projection
        # We need at least 2 samples and 2 features
        n_samples = X_scaled.shape[0]
        if n_samples < 2:
            return []
            
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # 3. DBSCAN for Density Analysis
        # eps and min_samples might need tuning, but let's use defaults/sensible starting points
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(X_scaled)
        
        results = []
        for i in range(n_samples):
            results.append({
                "x": float(X_pca[i, 0]),
                "y": float(X_pca[i, 1]),
                "cluster": int(clusters[i]),
                "is_outlier": bool(clusters[i] == -1),
                "label": df.iloc[i].get('Label', 'Unknown')
            })
            
        return results

# Singleton instance
discovery_engine = BehavioralDiscovery()
