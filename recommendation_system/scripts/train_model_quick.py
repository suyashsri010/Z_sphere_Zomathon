"""
Quick training script for fast testing (fewer iterations).
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from pathlib import Path
from src.feature_engineering import FeatureExtractor
from src.llm_embeddings import LLMEmbeddingGenerator
from src.model_training import RankingModelTrainer


def main():
    """Train the recommendation model quickly."""
    print("="*60)
    print("QUICK MODEL TRAINING (for testing)")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading data...")
    data_dir = Path('data/generated')
    
    users = pd.read_csv(data_dir / 'users.csv')
    restaurants = pd.read_csv(data_dir / 'restaurants.csv')
    items = pd.read_csv(data_dir / 'items.csv')
    train_sessions = pd.read_csv(data_dir / 'train_sessions.csv')
    val_sessions = pd.read_csv(data_dir / 'val_sessions.csv')
    
    print(f"Loaded: {len(users)} users, {len(restaurants)} restaurants, {len(items)} items")
    print(f"Train sessions: {len(train_sessions)}, Val sessions: {len(val_sessions)}")
    
    # Generate LLM embeddings
    print("\n[2/6] Generating LLM embeddings (AI Edge)...")
    embedding_path = Path('models/embeddings.pkl')
    
    if embedding_path.exists():
        print("Loading pre-computed embeddings...")
        llm_gen = LLMEmbeddingGenerator()
        llm_gen.load_embeddings(str(embedding_path))
    else:
        llm_gen = LLMEmbeddingGenerator()
        llm_gen.generate_item_embeddings(items)
        llm_gen.generate_user_embeddings(users, train_sessions, items)
        llm_gen.save_embeddings(str(embedding_path))
    
    # Extract features
    print("\n[3/6] Extracting features...")
    feature_extractor = FeatureExtractor()
    
    # Fit on training data
    feature_extractor.fit(users, restaurants, items, train_sessions)
    
    # Extract training features (use subset for speed)
    print("Extracting training features (using subset for speed)...")
    train_subset = train_sessions.sample(min(10000, len(train_sessions)), random_state=42)
    val_subset = val_sessions.sample(min(2000, len(val_sessions)), random_state=42)
    
    X_train, y_train = feature_extractor.extract_features_for_training(
        train_subset, users, restaurants, items
    )
    
    print("Extracting validation features...")
    X_val, y_val = feature_extractor.extract_features_for_training(
        val_subset, users, restaurants, items
    )
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Save feature extractor
    feature_extractor.save('models/feature_extractor.pkl')
    
    # Train model with QUICK settings
    print("\n[4/6] Training LightGBM model (quick mode)...")
    trainer = RankingModelTrainer()
    
    # Quick training params
    quick_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # Reduced from 31
        'learning_rate': 0.1,  # Increased from 0.05
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 4,  # Reduced from 6
        'min_data_in_leaf': 50,  # Reduced from 100
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'scale_pos_weight': 5.0
    }
    
    # Train with fewer rounds
    model = trainer.train(X_train, y_train, X_val, y_val, params=quick_params)
    
    # Save model
    print("\n[5/6] Saving model...")
    trainer.save_model('models/lightgbm_model.txt')
    
    # Feature importance
    print("\n[6/6] Feature importance (top 10):")
    importance = trainer.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_importance[:10]:
        print(f"  {feat}: {imp:.2f}")
    
    print("\n" + "="*60)
    print("✅ QUICK MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: models/lightgbm_model.txt")
    print(f"Feature extractor saved to: models/feature_extractor.pkl")
    print(f"Embeddings saved to: models/embeddings.pkl")


if __name__ == '__main__':
    main()
