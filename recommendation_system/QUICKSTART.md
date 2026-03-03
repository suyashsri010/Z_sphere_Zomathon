# Quick Start Guide

Get the Cart Super Add-On Recommendation System running in 5 steps!

## Prerequisites

- Python 3.8+
- 4GB RAM minimum
- 2GB disk space

## Step-by-Step Setup

### 1. Install Dependencies (2 minutes)

```bash
cd recommendation_system
pip install -r requirements.txt
```

This installs:
- FastAPI, pandas, numpy, scikit-learn
- LightGBM (gradient boosting)
- sentence-transformers (LLM embeddings)
- hypothesis, pytest (testing)

### 2. Generate Synthetic Data (1 minute)

```bash
python scripts/generate_data.py
```

**Output:**
```
Generated 5000 users, 500 restaurants, 10000+ items, 50000 session interactions
Train: 35000, Val: 7500, Test: 7500
```

**What it creates:**
- Realistic user profiles with order history
- Restaurants with menus across multiple cuisines
- Cart sessions with temporal patterns (peak hours, meal times)
- Train/val/test splits (chronological, no leakage)

### 3. Train the Model (5-10 minutes)

```bash
python scripts/train_model.py
```

**What happens:**
1. Generates LLM embeddings for all items (AI Edge feature)
2. Extracts 50+ features per example
3. Trains LightGBM ranking model
4. Saves model, feature extractor, and embeddings

**Expected output:**
```
Train AUC: 0.82
Validation AUC: 0.78
Model saved to: models/lightgbm_model.txt
```

### 4. Evaluate Performance (1 minute)

```bash
python scripts/evaluate_model.py
```

**Metrics you'll see:**
- AUC, Precision@K, Recall@K, NDCG@K
- Business metrics (acceptance rate, AOV lift)
- Segment-level performance (by meal time, city)
- Baseline comparisons

### 5. Start the API (instant)

```bash
uvicorn app.main:app --reload
```

API runs at: `http://localhost:8000`

## Test the API

### Health Check

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "status": "healthy",
  "service": "Cart Super Add-On Recommendation System"
}
```

### Get Recommendations

```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_0",
    "restaurant_id": "rest_0",
    "cart_items": [
      {"item_id": "item_0", "name": "Chicken Biryani", "price": 250}
    ],
    "top_n": 5
  }'
```

Response:
```json
{
  "recommendations": [
    {
      "item_id": "item_123",
      "name": "Indian Raita",
      "category": "side",
      "price": 50,
      "score": 0.78,
      "latency_ms": 45.2
    },
    {
      "item_id": "item_456",
      "name": "Indian Coke",
      "category": "beverage",
      "price": 40,
      "score": 0.65,
      "latency_ms": 45.2
    }
    // ... more recommendations
  ],
  "count": 5
}
```

## Run Tests

```bash
# Run property-based tests
pytest tests/test_properties.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## What's Next?

### Explore the Data

```python
import pandas as pd

# Load generated data
users = pd.read_csv('data/generated/users.csv')
items = pd.read_csv('data/generated/items.csv')
sessions = pd.read_csv('data/generated/train_sessions.csv')

# Explore patterns
print(users['segment'].value_counts())
print(sessions['meal_time'].value_counts())
print(f"Acceptance rate: {sessions['accepted'].mean():.2%}")
```

### Experiment with Parameters

Edit `src/model_training.py` to tune hyperparameters:
- `num_leaves`: Tree complexity (default: 31)
- `learning_rate`: Learning speed (default: 0.05)
- `max_depth`: Tree depth (default: 6)

### Add More Data

Increase dataset size in `scripts/generate_data.py`:
```python
users = generator.generate_users(n_users=10000)  # More users
sessions = generator.generate_cart_sessions(n_sessions=100000)  # More sessions
```

### Customize Features

Add new features in `src/feature_engineering.py`:
```python
def _construct_feature_vector(self, ...):
    # Add your custom features here
    features.append(my_custom_feature)
```

## Troubleshooting

### Issue: "Model not found"

**Solution:** Run `python scripts/train_model.py` first

### Issue: "Data not found"

**Solution:** Run `python scripts/generate_data.py` first

### Issue: "Out of memory"

**Solution:** Reduce dataset size in `generate_data.py`:
```python
n_users=1000, n_sessions=10000
```

### Issue: "Slow training"

**Solution:** Reduce `num_boost_round` in `model_training.py`:
```python
num_boost_round=500  # Instead of 1000
```

## Performance Tips

1. **Pre-compute embeddings**: Embeddings are cached after first run
2. **Use smaller datasets**: For development, use 1K users, 10K sessions
3. **Batch predictions**: API supports batch requests
4. **Cache features**: User/restaurant features are cached in memory


## Need Help?

- Check the logs for detailed error messages
- Review test output: `pytest tests/ -v`
- Verify data generation: `ls -lh data/generated/`
- Check model files: `ls -lh models/`

Happy recommending! 
