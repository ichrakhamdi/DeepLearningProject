import pandas as pd
import numpy as np
import warnings
import collections
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import ADASYN
from config.settings import LSTMSettings

class DataPreprocessor:
    def __init__(self):
        self.config = LSTMSettings()
        self.scaler = None
        self.encoder = None
        self.class_distributions = {}
        self.is_fitted = False

    def preprocess_train(self, train_df, class_type, augmentation=False):
        """Process training data:
           1. Validate input  
           2. Fit scaler/encoder  
           3. Optionally apply augmentation  
           4. Return scaled (and augmented) training data."""
        self._validate_input(train_df, class_type)
        
        # Extract features and labels
        X_train = train_df[self.config.FEATURE_COLS].copy()
        y_train = self._get_labels(train_df, class_type, fit_encoder=True)
        
        # Fit scaler on training data
        self.scaler = StandardScaler().fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Apply augmentation if requested (only for training)
        if augmentation:
            X_train_scaled, y_train = self._safe_apply_augmentation(X_train_scaled, y_train, class_type)
        
        self.is_fitted = True
        return X_train_scaled, y_train, (self.encoder if class_type != 'binary' else None)

    def preprocess_validation_test(self, df, class_type):
        """Process validation/test data using scaler/encoder from training."""
        if not self.is_fitted:
            raise RuntimeError("Training data must be processed first")
            
        self._validate_input(df, class_type)
        X = df[self.config.FEATURE_COLS].copy()
        y = self._get_labels(df, class_type, fit_encoder=False)
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, y, (self.encoder if class_type != 'binary' else None)

    def _get_labels(self, df, class_type, fit_encoder):
        """Encode labels appropriately for binary or multi-class classification."""
        label_col = self.config.CLASS_MAP[class_type]
        
        if class_type == 'binary':
            unique_labels = np.unique(df[label_col])
            benign_candidates = [label for label in unique_labels if str(label).lower() == 'benign']
            if benign_candidates:
                benign_label = benign_candidates[0]
            elif len(unique_labels) == 2:
                benign_label = sorted(unique_labels)[0]
                warnings.warn(f"'Benign' class not found; assuming '{benign_label}' as benign for binary classification.")
            else:
                raise ValueError("Binary classification requires exactly two classes.")
            return (df[label_col].apply(lambda x: str(x).lower()) != benign_label.lower()).astype(int).values
        else:
            if fit_encoder:
                self.encoder = LabelEncoder()
                self.encoder.fit(df[label_col])
            return self.encoder.transform(df[label_col])

    def _safe_apply_augmentation(self, X, y, class_type):
        """Apply ADASYN augmentation only if there are at least two classes."""
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"[DEBUG] Entering _safe_apply_augmentation for class_type='{class_type}'")
        
        if len(unique_classes) < 2:
            warnings.warn(f"Skipping augmentation for {class_type} - only one class present")
            return X, y
        
        try:
            print(f"[DEBUG] Before ADASYN: X.shape={X.shape}, class_distribution={dict(collections.Counter(y))}")
            ada = ADASYN(sampling_strategy='minority', random_state=42)
            print("[DEBUG] Fitting ADASYN... (this may take a while for large datasets)")
            X_res, y_res = ada.fit_resample(X, y)
            print(f"[DEBUG] After ADASYN: X_res.shape={X_res.shape}, new_class_distribution={dict(collections.Counter(y_res))}")
            return X_res, y_res
        
        except Exception as e:
            warnings.warn(f"Augmentation failed: {str(e)}")
            return X, y

    def _validate_input(self, df, class_type):
        """Ensure required columns exist in the dataframe."""
        if df.empty:
            raise ValueError("Empty DataFrame provided")
        required_columns = self.config.FEATURE_COLS + [self.config.CLASS_MAP[class_type]]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def create_sequences(self, X, y, window_size):
        """Convert data into time-series sequences for LSTM input."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if window_size <= 0 or window_size > len(X):
            raise ValueError(f"Invalid window size: {window_size}")
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size + 1):
            X_seq.append(X[i:i+window_size])
            y_seq.append(y[i+window_size-1])
        return np.array(X_seq), np.array(y_seq)

    def get_class_weights(self, y):
        """Compute class weights for imbalanced data."""
        unique, counts = np.unique(y, return_counts=True)
        return {cls: sum(counts)/(len(counts)*count) for cls, count in zip(unique, counts)}
