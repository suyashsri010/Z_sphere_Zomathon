# Cart Super Add-On Recommendation System

A production-ready ML-powered recommendation system for food delivery cart add-ons, built for the hackathon challenge. This system uses LightGBM for fast ranking and LLM embeddings for semantic understanding.

## 🎯 Key Features

- **Realistic Data Generation**: Synthetic data with temporal patterns, city-specific behaviors, and cold-start scenarios
- **LLM-Powered Embeddings**: Semantic understanding using sentence transformers (AI Edge)
- **Fast Ranking Model**: LightGBM for sub-300ms inference latency
- **Comprehensive Evaluation**: AUC, NDCG, Precision@K, Recall@K metrics
- **Property-Based Testing**: Hypothesis framework for correctness validation
- **Production-Ready API**: FastAPI service with request validation

## 📊 System Performance

- **Latency**: < 300ms end-to-end
- **AUC**: > 0.75 (target)
- **NDCG@10**: > 0.65 (target)
- **Acceptance Rate**: 15-20% (realistic)

## 🏗️ Architecture

```
┌─────────────┐
│   FastAPI   │  ← REST API
└──────┬──────┘
       │
┌──────▼──────────────────────────┐
│  Recommendation Engine          │
│  ┌────────────────────────────┐ │
│  │ 1. Candidate Retrieval     │ │
│  │ 2. LightGBM Ranking        │ │
│  │ 3. Post-Processing         │ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
       │
┌──────▼──────────────────────────┐
│  Feature Engineering            │
│  + LLM Embeddings (AI Edge)     │
└─────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data

```bash
# Generate synthetic data (5K users, 500 restaurants, 50K sessions)
python scripts/generate_data.py
```

This creates:
- `data/generated/users.csv` - User profiles
- `data/generated/restaurants.csv` - Restaurant profiles
- `data/generated/items.csv` - Menu items
- `data/generated/train_sessions.csv` - Training data
- `data/generated/val_sessions.csv` - Validation data
- `data/generated/test_sessions.csv` - Test data

### 3. Train Model

```bash
# Train LightGBM model with LLM embeddings
python scripts/train_model.py
```

This creates:
- `models/lightgbm_model.txt` - Trained model
- `models/feature_extractor.pkl` - Feature pipeline
- `models/embeddings.pkl` - LLM embeddings

Training takes ~5-10 minutes on CPU.

### 4. Evaluate Model

```bash
# Evaluate on test set
python scripts/evaluate_model.py
```

Outputs:
- Overall metrics (AUC, Precision@K, Recall@K, NDCG@K)
- Business impact metrics (acceptance rate, AOV lift)
- Segment-level analysis (by meal time, city)
- Baseline comparisons

### 5. Run API

```bash
# Start FastAPI server
uvicorn app.main:app --reload
```

API available at: `http://localhost:8000`

### 6. Test API

```bash
# Example request
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_0",
    "restaurant_id": "rest_0",
    "cart_items": [
      {"item_id": "item_0", "name": "Biryani", "price": 250}
    ],
    "top_n": 10
  }'
```

## 📁 Project Structure

```
recommendation_system/
├── app/
│   ├── main.py              # FastAPI application
│   └── logic.py             # Business logic
├── src/
│   ├── data_generator.py    # Synthetic data generation
│   ├── feature_engineering.py  # Feature extraction
│   ├── llm_embeddings.py    # LLM embeddings (AI Edge)
│   ├── model_training.py    # LightGBM training
│   └── recommendation_engine.py  # End-to-end engine
├── scripts/
│   ├── generate_data.py     # Data generation script
│   ├── train_model.py       # Model training script
│   └── evaluate_model.py    # Evaluation script
├── tests/
│   └── test_properties.py   # Property-based tests
├── data/
│   └── generated/           # Generated datasets
├── models/                  # Trained models
└── requirements.txt
```

## 🧪 Testing

### Run Property-Based Tests

```bash
# Run all property tests
pytest tests/test_properties.py -v

# Run with more examples (CI mode)
HYPOTHESIS_PROFILE=ci pytest tests/test_properties.py -v
```

Tests include:
- Data generation validity
- Temporal split correctness
- Feature completeness
- Recommendation exclusion
- Diversity validation

## 📈 Evaluation Criteria Coverage

### Data Preparation & Feature Engineering (20%)
✅ Realistic synthetic data with temporal, geographic, behavioral patterns  
✅ Comprehensive feature pipeline (user, cart, item, context, interaction)  
✅ Cold-start handling with fallback strategies  
✅ Feature caching for real-time inference  

### Ideation & Problem Formulation (15%)
✅ Ranking problem formulation with two-stage pipeline  
✅ Sequential cart update handling  
✅ Cold-start strategies for users, restaurants, items  

### Model Architecture & AI Edge (20%)
✅ **LLM embeddings** for semantic understanding (sentence transformers)  
✅ LightGBM for fast, accurate ranking  
✅ Complementarity scoring using cosine similarity  
✅ Cold-start handling with semantic representations  

### Model Evaluation & Fine-Tuning (15%)
✅ Temporal train/val/test splits  
✅ Comprehensive metrics (AUC, NDCG, Precision@K, Recall@K)  
✅ Segment-level analysis  
✅ Baseline comparisons  

### System Design & Production Readiness (15%)
✅ Sub-300ms latency target  
✅ FastAPI service with request validation  
✅ Error handling and fallbacks  
✅ Monitoring and logging  

### Business Impact & A/B Testing (15%)
✅ Business metrics estimation (AOV lift, acceptance rate)  
✅ A/B test design documented  
✅ Success criteria defined  

## 🎓 Key Design Decisions

### Why LightGBM?
- **Speed**: Sub-millisecond inference per item
- **Accuracy**: Competitive with deep learning
- **Interpretability**: Feature importance analysis
- **Production-ready**: Mature deployment patterns

### Why LLM Embeddings?
- **Cold-start**: Rich semantic representations for new items
- **Complementarity**: Captures "biryani + raita" patterns
- **Zero-shot**: Works for new menu items immediately
- **Lightweight**: 80MB model, 10ms inference

### Why Two-Stage Pipeline?
- **Scalability**: Reduces ranking workload
- **Latency**: Fast retrieval + precise ranking
- **Flexibility**: Independent optimization

## 📊 Sample Results

```
=== OVERALL PERFORMANCE ===
AUC: 0.7823
Precision@10: 0.2145
Recall@10: 0.4521
NDCG@10: 0.6734

=== BUSINESS IMPACT ===
Acceptance Rate: 18.2%
Estimated AOV Lift: ₹27.30 (6.1%)

=== BY MEAL TIME ===
lunch:    AUC=0.7891, NDCG@10=0.6812
dinner:   AUC=0.7798, NDCG@10=0.6701
breakfast: AUC=0.7654, NDCG@10=0.6543
```

## 🔧 Configuration

Key parameters in `src/model_training.py`:
- `num_leaves`: 31 (tree complexity)
- `learning_rate`: 0.05
- `max_depth`: 6
- `scale_pos_weight`: 5.0 (class imbalance)

Embedding model: `all-MiniLM-L6-v2` (384-dim, 80MB)

## 📝 A/B Test Design

**Metrics**:
- Primary: AOV lift, acceptance rate, C2O ratio
- Guardrails: Cart abandonment, order completion

**Sample Size**: ~50K sessions per group  
**Duration**: 7-14 days  
**Success Criteria**: AOV lift > 3%, acceptance > 18%

## 🤝 Contributing

This is a hackathon project. For questions or improvements, please open an issue.

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- LightGBM team for the excellent gradient boosting library
- Sentence Transformers for semantic embeddings
- Hypothesis for property-based testing framework
