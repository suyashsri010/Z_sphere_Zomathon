"""
Complete recommendation engine with retrieval, ranking, and post-processing.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import time


class RecommendationEngine:
    """End-to-end recommendation engine."""
    
    def __init__(
        self,
        model_path: str = 'models/lightgbm_model.txt',
        feature_extractor_path: str = 'models/feature_extractor.pkl',
        embeddings_path: str = 'models/embeddings.pkl'
    ):
        """
        Initialize recommendation engine.
        
        Args:
            model_path: Path to trained LightGBM model
            feature_extractor_path: Path to feature extractor
            embeddings_path: Path to LLM embeddings
        """
        from src.model_training import RankingModelTrainer
        from src.feature_engineering import FeatureExtractor
        from src.llm_embeddings import LLMEmbeddingGenerator
        
        print("Loading recommendation engine...")
        
        # Load model
        self.trainer = RankingModelTrainer()
        if Path(model_path).exists():
            self.trainer.load_model(model_path)
        else:
            print(f"Warning: Model not found at {model_path}")
            self.trainer = None
        
        # Load feature extractor
        self.feature_extractor = FeatureExtractor()
        if Path(feature_extractor_path).exists():
            self.feature_extractor.load(feature_extractor_path)
        else:
            print(f"Warning: Feature extractor not found at {feature_extractor_path}")
        
        # Load embeddings
        self.llm_gen = LLMEmbeddingGenerator()
        if Path(embeddings_path).exists():
            self.llm_gen.load_embeddings(embeddings_path)
        else:
            print(f"Warning: Embeddings not found at {embeddings_path}")
        
        # Load data (cache in memory for fast access)
        self._load_data()
        
        print("Recommendation engine ready!")
    
    def _load_data(self):
        """Load and cache data."""
        data_dir = Path('data/generated')
        
        if data_dir.exists():
            self.users = pd.read_csv(data_dir / 'users.csv').set_index('user_id').to_dict('index')
            self.restaurants = pd.read_csv(data_dir / 'restaurants.csv').set_index('restaurant_id').to_dict('index')
            self.items = pd.read_csv(data_dir / 'items.csv')
            self.items_dict = self.items.set_index('item_id').to_dict('index')
            print(f"Loaded data: {len(self.users)} users, {len(self.restaurants)} restaurants, {len(self.items)} items")
        else:
            print("Warning: Data not found. Using empty cache.")
            self.users = {}
            self.restaurants = {}
            self.items = pd.DataFrame()
            self.items_dict = {}
    
    def get_recommendations(
        self,
        user_id: str,
        restaurant_id: str,
        cart_items: List[Dict[str, Any]],
        top_n: int = 10,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top-N recommendations for a cart.
        
        Args:
            user_id: User ID
            restaurant_id: Restaurant ID
            cart_items: List of items currently in cart
            top_n: Number of recommendations to return
            context: Optional context (meal_time, timestamp, etc.)
        
        Returns:
            List of recommended items with scores
        """
        start_time = time.time()
        
        # Step 1: Retrieve candidates
        candidates = self._retrieve_candidates(restaurant_id, cart_items)
        
        if len(candidates) == 0:
            return []
        
        # Step 2: Rank candidates
        ranked_candidates = self._rank_candidates(
            user_id, restaurant_id, cart_items, candidates, context
        )
        
        # Step 3: Post-process (diversity, business rules)
        final_recommendations = self._post_process(ranked_candidates, cart_items, top_n)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Add metadata
        for rec in final_recommendations:
            rec['latency_ms'] = latency_ms
        
        return final_recommendations
    
    def _retrieve_candidates(
        self,
        restaurant_id: str,
        cart_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidate items from restaurant menu.
        
        Strategy: Return all available items from restaurant that are NOT in cart.
        """
        # Get cart item IDs
        cart_item_ids = {item.get('item_id', item.get('id', '')) for item in cart_items}
        
        # Get restaurant items
        restaurant_items = self.items[self.items['restaurant_id'] == restaurant_id]
        
        # Filter out cart items and unavailable items
        candidates = restaurant_items[
            (~restaurant_items['item_id'].isin(cart_item_ids)) &
            (restaurant_items['is_available'] == True)
        ]
        
        return candidates.to_dict('records')
    
    def _rank_candidates(
        self,
        user_id: str,
        restaurant_id: str,
        cart_items: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Rank candidates using trained model."""
        if self.trainer is None or self.trainer.model is None:
            # Fallback: rank by popularity
            return sorted(candidates, key=lambda x: x.get('popularity_score', 0), reverse=True)
        
        # Get user and restaurant data
        user = self.users.get(user_id, self._get_default_user())
        restaurant = self.restaurants.get(restaurant_id, self._get_default_restaurant())
        
        # Prepare context
        if context is None:
            context = self._get_default_context()
        
        # Convert cart items to dict format
        cart_item_dicts = []
        for item in cart_items:
            item_id = item.get('item_id', item.get('id', ''))
            if item_id in self.items_dict:
                cart_item_dicts.append(self.items_dict[item_id])
        
        # Extract features for each candidate
        features_list = []
        for candidate in candidates:
            try:
                feature_vector = self.feature_extractor._construct_feature_vector(
                    user, restaurant, cart_item_dicts, candidate, context
                )
                features_list.append(feature_vector)
            except Exception as e:
                print(f"Error extracting features: {e}")
                features_list.append(np.zeros(50))  # Fallback
        
        if len(features_list) == 0:
            return candidates
        
        # Predict scores
        X = np.array(features_list)
        scores = self.trainer.predict(X)
        
        # Combine candidates with scores
        ranked = [
            {**candidate, 'score': float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        
        # Sort by score
        ranked.sort(key=lambda x: x['score'], reverse=True)
        
        return ranked
    
    def _post_process(
        self,
        ranked_candidates: List[Dict[str, Any]],
        cart_items: List[Dict[str, Any]],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """Apply diversity filtering and business rules."""
        final = []
        category_counts = {}
        
        for candidate in ranked_candidates:
            if len(final) >= top_n:
                break
            
            category = candidate.get('category', 'main')
            
            # Diversity: max 3 items per category
            if category_counts.get(category, 0) >= 3:
                continue
            
            # Business rule: check margin
            if candidate.get('margin', 0) < 0.1:
                continue
            
            final.append(candidate)
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return final
    
    def _get_default_user(self) -> Dict:
        """Get default user for cold-start."""
        return {
            'user_id': 'unknown',
            'city': 'Mumbai',
            'segment': 'frequent',
            'order_count_30d': 5,
            'order_count_90d': 15,
            'avg_order_value': 300,
            'std_order_value': 100,
            'days_since_last_order': 7,
            'avg_items_per_order': 3,
            'veg_preference_ratio': 0.5,
            'price_sensitivity_score': 0.5
        }
    
    def _get_default_restaurant(self) -> Dict:
        """Get default restaurant."""
        return {
            'restaurant_id': 'unknown',
            'cuisine_type': 'Indian',
            'price_range': 2,
            'rating': 4.0,
            'popularity_score': 0.5,
            'is_chain': False
        }
    
    def _get_default_context(self) -> Dict:
        """Get default context."""
        from datetime import datetime
        now = datetime.now()
        hour = now.hour
        
        if 6 <= hour < 11:
            meal_time = 'breakfast'
        elif 11 <= hour < 16:
            meal_time = 'lunch'
        elif 16 <= hour < 23:
            meal_time = 'dinner'
        else:
            meal_time = 'late_night'
        
        return {
            'timestamp': now.isoformat(),
            'meal_time': meal_time,
            'day_of_week': now.weekday(),
            'is_weekend': now.weekday() >= 5,
            'is_peak_hour': (12 <= hour <= 14) or (19 <= hour <= 22)
        }
