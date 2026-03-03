"""
Model training module for LightGBM ranking model.
"""
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score
from typing import Tuple
import pickle
from pathlib import Path


class RankingModelTrainer:
    """Train LightGBM ranking model for recommendation."""
    
    def __init__(self):
        """Initialize trainer."""
        self.model = None
        self.best_params = None
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict = None
    ) -> lgb.Booster:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            params: Model hyperparameters
        
        Returns:
            Trained LightGBM model
        """
        if params is None:
            params = self._get_default_params()
        
        print("Training LightGBM model...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Positive rate (train): {y_train.mean():.3f}, Positive rate (val): {y_val.mean():.3f}")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        
        self.best_params = params
        
        # Evaluate
        train_auc = self._evaluate_auc(X_train, y_train)
        val_auc = self._evaluate_auc(X_val, y_val)
        
        print(f"\nTraining complete!")
        print(f"Train AUC: {train_auc:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")
        
        return self.model
    
    def _get_default_params(self) -> dict:
        """Get default LightGBM parameters optimized for ranking."""
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_data_in_leaf': 100,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'scale_pos_weight': 5.0  # Handle class imbalance (more negatives than positives)
        }
    
    def _evaluate_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate AUC score."""
        if self.model is None:
            return 0.0
        
        y_pred = self.model.predict(X)
        return roc_auc_score(y, y_pred)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict acceptance probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def save_model(self, path: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        self.model = lgb.Booster(model_file=path)
        print(f"Model loaded from {path}")
    
    def get_feature_importance(self, feature_names: list = None) -> dict:
        """Get feature importance scores."""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importance(importance_type='gain')
        
        if feature_names:
            return dict(zip(feature_names, importance))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importance)}
