import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .preprocessing import DataPreprocessor
import joblib
import os

class SupervisedIDS:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=50, 
            random_state=42
            # class_weight removed - default performs best (110 errors vs 199 with 'balanced')
        )
        self.preprocessor = DataPreprocessor()
        self.model_path = "rf_model.joblib"
        self.metrics = {}

    def train(self, data_path):
        """
        Trains Random Forest on the dataset using 'Cat' column for detailed classification.
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Preprocess using 'Cat' as target
        X, y = self.preprocessor.fit_transform(df, target_col='Cat')
        
        # Split with stratification to preserve label distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # K-Fold Cross-Validation for more reliable evaluation
        print("Running 5-fold cross-validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=skf, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train
        print("Training Random Forest on full training set...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "cv_scores": cv_scores.tolist(),
            "confusion_matrix": cm.tolist(),
            "classes": self.model.classes_.tolist()
        }
        
        # Save model
        joblib.dump(self.model, self.model_path)
        print("Training complete.")
        return self.metrics

    def predict(self, input_data, return_proba=False):
        """
        Predicts detailed labels for new data. 
        input_data: List of dicts or DataFrame
        """
        df = pd.DataFrame(input_data)
        X = self.preprocessor.transform(df)
        
        # Load model if not in memory
        if not hasattr(self.model, "estimators_"):
             if os.path.exists(self.model_path):
                 self.model = joblib.load(self.model_path)
             else:
                 raise Exception("Model not trained.")
                 
        preds = self.model.predict(X)
        
        if return_proba:
            probas = self.model.predict_proba(X)
            max_probas = np.max(probas, axis=1)
            return preds.tolist(), max_probas.tolist()
            
        return preds.tolist()
