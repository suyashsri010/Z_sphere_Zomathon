"""
Script to evaluate the trained model.
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from src.feature_engineering import FeatureExtractor
from src.model_training import RankingModelTrainer


def compute_ndcg_at_k(y_true, y_pred, k=10):
    """Compute NDCG@K."""
    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)[::-1][:k]
    
    # Get relevance scores (1 for accepted, 0 for rejected)
    relevance = y_true[sorted_indices]
    
    # Compute DCG
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
    
    # Compute IDCG (ideal DCG)
    ideal_relevance = np.sort(y_true)[::-1][:k]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
    
    # NDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_precision_at_k(y_true, y_pred, k=10):
    """Compute Precision@K."""
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    return y_true[top_k_indices].mean()


def compute_recall_at_k(y_true, y_pred, k=10):
    """Compute Recall@K."""
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    return y_true[top_k_indices].sum() / max(y_true.sum(), 1)


def evaluate_by_segment(y_true, y_pred, segment_labels, segment_name="segment"):
    """Evaluate performance by segment."""
    results = []
    
    for segment in np.unique(segment_labels):
        mask = segment_labels == segment
        if mask.sum() == 0:
            continue
        
        y_true_seg = y_true[mask]
        y_pred_seg = y_pred[mask]
        
        auc = roc_auc_score(y_true_seg, y_pred_seg) if len(np.unique(y_true_seg)) > 1 else 0.0
        precision_10 = compute_precision_at_k(y_true_seg, y_pred_seg, k=10)
        recall_10 = compute_recall_at_k(y_true_seg, y_pred_seg, k=10)
        ndcg_10 = compute_ndcg_at_k(y_true_seg, y_pred_seg, k=10)
        
        results.append({
            'segment_type': segment_name,
            'segment_value': segment,
            'count': mask.sum(),
            'auc': auc,
            'precision@10': precision_10,
            'recall@10': recall_10,
            'ndcg@10': ndcg_10
        })
    
    return pd.DataFrame(results)


def main():
    """Evaluate the trained model."""
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load data
    print("\n[1/4] Loading data...")
    data_dir = Path('data/generated')
    
    users = pd.read_csv(data_dir / 'users.csv')
    restaurants = pd.read_csv(data_dir / 'restaurants.csv')
    items = pd.read_csv(data_dir / 'items.csv')
    test_sessions = pd.read_csv(data_dir / 'test_sessions.csv')
    
    print(f"Test sessions: {len(test_sessions)}")
    
    # Load model and feature extractor
    print("\n[2/4] Loading model...")
    trainer = RankingModelTrainer()
    trainer.load_model('models/lightgbm_model.txt')
    
    feature_extractor = FeatureExtractor()
    feature_extractor.load('models/feature_extractor.pkl')
    
    # Extract test features
    print("\n[3/4] Extracting test features...")
    X_test, y_test = feature_extractor.extract_features_for_training(
        test_sessions, users, restaurants, items
    )
    
    print(f"Test samples: {len(X_test)}")
    
    # Predict
    print("\n[4/4] Evaluating...")
    y_pred = trainer.predict(X_test)
    
    # Overall metrics
    print("\n" + "="*60)
    print("OVERALL PERFORMANCE")
    print("="*60)
    
    auc = roc_auc_score(y_test, y_pred)
    print(f"AUC: {auc:.4f}")
    
    for k in [5, 10, 20]:
        precision_k = compute_precision_at_k(y_test, y_pred, k=k)
        recall_k = compute_recall_at_k(y_test, y_pred, k=k)
        ndcg_k = compute_ndcg_at_k(y_test, y_pred, k=k)
        
        print(f"\nTop-{k} Metrics:")
        print(f"  Precision@{k}: {precision_k:.4f}")
        print(f"  Recall@{k}: {recall_k:.4f}")
        print(f"  NDCG@{k}: {ndcg_k:.4f}")
    
    # Business metrics
    print("\n" + "="*60)
    print("BUSINESS IMPACT METRICS")
    print("="*60)
    
    acceptance_rate = y_test.mean()
    predicted_acceptance_rate = (y_pred > 0.5).mean()
    
    print(f"Actual Acceptance Rate: {acceptance_rate:.2%}")
    print(f"Predicted Acceptance Rate: {predicted_acceptance_rate:.2%}")
    
    # Estimate AOV lift (simplified)
    # Assuming average item price is 150 and acceptance rate improvement
    avg_item_price = items['price'].mean()
    baseline_acceptance = 0.10  # Baseline (random/popularity)
    model_acceptance = acceptance_rate
    
    aov_lift = (model_acceptance - baseline_acceptance) * avg_item_price
    aov_lift_pct = aov_lift / (3 * avg_item_price) * 100  # Assuming avg cart value is 3 items
    
    print(f"\nEstimated AOV Lift: ₹{aov_lift:.2f} ({aov_lift_pct:.1f}%)")
    
    # Segment-level analysis
    print("\n" + "="*60)
    print("SEGMENT-LEVEL ANALYSIS")
    print("="*60)
    
    # Merge with session data for segment info
    test_sessions_subset = test_sessions.iloc[:len(y_test)].copy()
    test_sessions_subset['y_pred'] = y_pred
    test_sessions_subset['y_true'] = y_test
    
    # By meal time
    print("\nBy Meal Time:")
    meal_time_results = evaluate_by_segment(
        y_test, y_pred, 
        test_sessions_subset['meal_time'].values,
        "meal_time"
    )
    print(meal_time_results.to_string(index=False))
    
    # By city
    print("\nBy City:")
    city_results = evaluate_by_segment(
        y_test, y_pred,
        test_sessions_subset['city'].values,
        "city"
    )
    print(city_results.to_string(index=False))
    
    # Baseline comparison
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    
    # Random baseline
    y_pred_random = np.random.random(len(y_test))
    auc_random = roc_auc_score(y_test, y_pred_random)
    
    # Popularity baseline (use item popularity scores)
    # For simplicity, use uniform scores
    y_pred_popularity = np.ones(len(y_test)) * 0.5
    
    print(f"Model AUC: {auc:.4f}")
    print(f"Random Baseline AUC: {auc_random:.4f}")
    print(f"Improvement over Random: {(auc - auc_random) / auc_random * 100:.1f}%")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
