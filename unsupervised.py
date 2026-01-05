import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from .preprocessing import DataPreprocessor
import joblib
import os

class UnsupervisedIDS:
    def __init__(self):
        # contamination='auto' allows the model to estimate % of outliers
        self.model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        self.preprocessor = DataPreprocessor()
        self.model_path = "iso_forest.joblib"
        self.stats = {}

    def train(self, data_path):
        """
        Trains Isolation Forest (Unsupervised).
        Ignores labels, looks for anomalies in feature space.
        """
        print(f"Loading data from {data_path} for Unsupervised Training...")
        df = pd.read_csv(data_path)
        
        # Preprocess (Ignore labels inside fit_transform logic for X)
        X, _ = self.preprocessor.fit_transform(df)
        
        # Train
        print("Training Isolation Forest...")
        self.model.fit(X)
        
        # Calculate some stats for the dashboard (e.g. how many anomalies found in training set)
        preds = self.model.predict(X)
        n_anomalies = np.sum(preds == -1)
        n_normal = np.sum(preds == 1)
        
        self.stats = {
            "total_samples": int(len(X)),
            "detected_anomalies_in_training": int(n_anomalies),
            "detected_normal_in_training": int(n_normal),
            "anomaly_percentage": float(n_anomalies / len(X) * 100)
        }
        
        # Save model
        joblib.dump(self.model, self.model_path)
        print("Unsupervised Training complete.")
        return self.stats

    def detect(self, input_data):
        """
        Detects anomalies.
        Returns: 
          - Prediction (-1: Anomaly, 1: Normal)
          - Anomaly Score (lower is more anomalous)
        """
        df = pd.DataFrame(input_data)
        X = self.preprocessor.transform(df)
        
        if not hasattr(self.model, "estimators_"):
             if os.path.exists(self.model_path):
                 self.model = joblib.load(self.model_path)
             else:
                 raise Exception("Model not trained.")
        
        preds = self.model.predict(X)
        scores = self.model.decision_function(X) # raw anomaly scores
        
        results = []
        for p, s in zip(preds, scores):
            results.append({
                "prediction": "Anomaly" if p == -1 else "Normal",
                "score": float(s)
            })
            
        return results
