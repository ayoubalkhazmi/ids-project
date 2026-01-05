import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = None
        self.scaler_path = "scaler.joblib"
        
    def fit_transform(self, df, target_col='Cat'):
        """
        Prepares data for training:
        1. Drops non-numeric/label columns not needed for training features
        2. Imputes missing values
        3. Scales features
        Returns: Processed numpy array, Target series (if available)
        """
        # identify label columns to drop from features
        exclude_cols = ['Label', 'Cat', 'Sub_Cat', 'Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp', 'Src_Port', 'Dst_Port']
        
        # separate features and targets (if exist)
        y_target = df[target_col] if target_col in df.columns else None
        
        # drop excluded columns to keep only numerical features
        X_df = df.drop([c for c in exclude_cols if c in df.columns], axis=1)
        
        # Filter only numeric columns just in case
        X_df = X_df.select_dtypes(include=[np.number])
        
        # Replace Infinity with NaN to allow Imputer to handle it
        X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Save feature names for later use
        self.feature_columns = X_df.columns.tolist()
        
        # Impute
        X_imputed = self.imputer.fit_transform(X_df)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Check for Infinity or NaNs after scaling
        X_scaled = np.nan_to_num(X_scaled)
        
        # Save scaler for inference
        joblib.dump((self.scaler, self.imputer, self.feature_columns), self.scaler_path)
        
        return X_scaled, y_target

    def transform(self, df):
        """
        Prepares specific input data for prediction using saved scaler.
        """
        # separate features
        exclude_cols = ['Label', 'Cat', 'Sub_Cat', 'Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp', 'Src_Port', 'Dst_Port']
        X_df = df.drop([c for c in exclude_cols if c in df.columns], axis=1)
        
        # Ensure we have the same columns as training
        if self.feature_columns is None:
             # Try to load
             if os.path.exists(self.scaler_path):
                 self.scaler, self.imputer, self.feature_columns = joblib.load(self.scaler_path)
             else:
                 raise Exception("Preprocessor not fitted!")
        
        # Replace Infinity
        X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
         
        # fill missing cols with 0 if necessary (alignment)
        for col in self.feature_columns:
            if col not in X_df.columns:
                X_df[col] = 0
                
        # Reorder to match training
        X_df = X_df[self.feature_columns]
        
        # Impute
        X_imputed = self.imputer.transform(X_df)
        
        # Scale
        X_scaled = self.scaler.transform(X_imputed)
        
        X_scaled = np.nan_to_num(X_scaled)
        
        return X_scaled
