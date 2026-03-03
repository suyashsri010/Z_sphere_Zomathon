"""
Feature engineering pipeline for recommendation system.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from pathlib import Path


class FeatureExtractor:
    """Extract and engineer features for recommendation model."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        self.label_encoders = {}
        self.fitted = False
        
    def fit(self, users: pd.DataFrame, restaurants: pd.DataFrame, 
            items: pd.DataFrame, sessions: pd.DataFrame):
        """
        Fit scalers and encoders on training data.
        
        Args:
            users: User profiles
            restaurants: Restaurant profiles
            items: Menu items
            sessions: Training sessions
        """
        # Fit user feature scaler
        user_numeric_cols = ['order_count_30d', 'order_count_90d', 'avg_order_value', 
                            'std_order_value', 'days_since_last_order', 'avg_items_per_order',
                            'veg_preference_ratio', 'price_sensitivity_score']
        self.user_scaler.fit(users[user_numeric_cols].fillna(0))
        
        # Fit item feature scaler
        item_numeric_cols = ['price', 'popularity_score', 'rating', 'num_ratings', 'margin']
        self.item_scaler.fit(items[item_numeric_cols].fillna(0))
        
        # Fit label encoders for categorical features
        for col in ['city', 'segment', 'cuisine_type', 'category', 'meal_time']:
            le = LabelEncoder()
            if col in users.columns:
                le.fit(users[col].astype(str))
            elif col in restaurants.columns:
                le.fit(restaurants[col].astype(str))
            elif col in items.columns:
                le.fit(items[col].astype(str))
            elif col in sessions.columns:
                le.fit(sessions[col].astype(str))
            self.label_encoders[col] = le
        
        self.fitted = True
    
    def extract_features_for_training(
        self,
        sessions: pd.DataFrame,
        users: pd.DataFrame,
        restaurants: pd.DataFrame,
        items: pd.DataFrame
    ) -> tuple:
        """
        Extract features for model training.
        
        Args:
            sessions: Session data with labels
            users: User profiles
            restaurants: Restaurant profiles
            items: Menu items
        
        Returns:
            Tuple of (X, y) where X is feature matrix and y is labels
        """
        features_list = []
        labels = []
        
        # Create lookups for fast access
        user_dict = users.set_index('user_id').to_dict('index')
        restaurant_dict = restaurants.set_index('restaurant_id').to_dict('index')
        item_dict = items.set_index('item_id').to_dict('index')
        
        for _, session in sessions.iterrows():
            try:
                user = user_dict[session['user_id']]
                restaurant = restaurant_dict[session['restaurant_id']]
                candidate_item = item_dict[session['candidate_item_id']]
                
                # Get cart items
                cart_item_ids = session['cart_items'].split(',')
                cart_items = [item_dict[iid] for iid in cart_item_ids if iid in item_dict]
                
                # Extract features
                feature_vector = self._construct_feature_vector(
                    user, restaurant, cart_items, candidate_item, session
                )
                
                features_list.append(feature_vector)
                labels.append(session['accepted'])
            except KeyError:
                continue
        
        X = np.array(features_list)
        y = np.array(labels)
        
        return X, y
    
    def _construct_feature_vector(
        self,
        user: Dict,
        restaurant: Dict,
        cart_items: List[Dict],
        candidate_item: Dict,
        session: Dict
    ) -> np.ndarray:
        """Construct feature vector for a single example."""
        features = []
        
        # User features (8 features)
        user_numeric = [
            user.get('order_count_30d', 0),
            user.get('order_count_90d', 0),
            user.get('avg_order_value', 0),
            user.get('std_order_value', 0),
            user.get('days_since_last_order', 0),
            user.get('avg_items_per_order', 0),
            user.get('veg_preference_ratio', 0),
            user.get('price_sensitivity_score', 0)
        ]
        if self.fitted:
            user_numeric = self.user_scaler.transform([user_numeric])[0]
        features.extend(user_numeric)
        
        # User segment (encoded)
        segment_encoded = self.label_encoders['segment'].transform([user.get('segment', 'budget')])[0]
        features.append(segment_encoded)
        
        # Restaurant features (4 features)
        features.extend([
            restaurant.get('price_range', 2),
            restaurant.get('rating', 4.0),
            restaurant.get('popularity_score', 0.5),
            1 if restaurant.get('is_chain', False) else 0
        ])
        
        # Restaurant cuisine (encoded)
        cuisine_encoded = self.label_encoders['cuisine_type'].transform([restaurant.get('cuisine_type', 'Indian')])[0]
        features.append(cuisine_encoded)
        
        # Cart features (10 features)
        num_items_in_cart = len(cart_items)
        total_cart_value = sum(item.get('price', 0) for item in cart_items)
        avg_item_price = total_cart_value / max(num_items_in_cart, 1)
        
        cart_categories = [item.get('category', '') for item in cart_items]
        has_main = 1 if 'main' in cart_categories else 0
        has_side = 1 if 'side' in cart_categories else 0
        has_beverage = 1 if 'beverage' in cart_categories else 0
        has_dessert = 1 if 'dessert' in cart_categories else 0
        has_appetizer = 1 if 'appetizer' in cart_categories else 0
        
        veg_count = sum(1 for item in cart_items if item.get('is_veg', False))
        veg_ratio = veg_count / max(num_items_in_cart, 1)
        
        features.extend([
            num_items_in_cart,
            total_cart_value,
            avg_item_price,
            has_main, has_side, has_beverage, has_dessert, has_appetizer,
            veg_ratio,
            1 if num_items_in_cart == 1 else 0  # single item cart flag
        ])
        
        # Candidate item features (5 features)
        item_numeric = [
            candidate_item.get('price', 0),
            candidate_item.get('popularity_score', 0),
            candidate_item.get('rating', 4.0) if candidate_item.get('rating') else 4.0,
            candidate_item.get('num_ratings', 0),
            candidate_item.get('margin', 0.3)
        ]
        if self.fitted:
            item_numeric = self.item_scaler.transform([item_numeric])[0]
        features.extend(item_numeric)
        
        features.append(1 if candidate_item.get('is_veg', False) else 0)
        
        # Candidate category (encoded)
        category_encoded = self.label_encoders['category'].transform([candidate_item.get('category', 'main')])[0]
        features.append(category_encoded)
        
        # Context features (6 features)
        features.extend([
            session.get('day_of_week', 0),
            1 if session.get('is_weekend', False) else 0,
            1 if session.get('is_peak_hour', False) else 0,
        ])
        
        # Meal time (encoded)
        meal_time_encoded = self.label_encoders['meal_time'].transform([session.get('meal_time', 'lunch')])[0]
        features.append(meal_time_encoded)
        
        # City (encoded)
        city_encoded = self.label_encoders['city'].transform([user.get('city', 'Mumbai')])[0]
        features.append(city_encoded)
        
        # Hour of day
        try:
            from datetime import datetime
            timestamp = datetime.fromisoformat(session.get('timestamp', '2024-01-01T12:00:00'))
            hour = timestamp.hour
        except:
            hour = 12
        features.append(hour)
        
        # Interaction features (5 features)
        # Price compatibility
        price_diff = abs(candidate_item.get('price', 0) - avg_item_price)
        price_ratio = candidate_item.get('price', 0) / max(avg_item_price, 1)
        features.extend([price_diff, price_ratio])
        
        # Category complementarity
        candidate_category = candidate_item.get('category', '')
        category_complement_score = 0
        if candidate_category == 'beverage' and not has_beverage:
            category_complement_score = 1.0
        elif candidate_category == 'side' and has_main and not has_side:
            category_complement_score = 0.8
        elif candidate_category == 'dessert' and has_main and not has_dessert:
            category_complement_score = 0.6
        features.append(category_complement_score)
        
        # Veg compatibility
        veg_compatible = 1 if (candidate_item.get('is_veg', False) and veg_ratio > 0.5) else 0
        features.append(veg_compatible)
        
        # Item already in cart (should always be 0 for candidates)
        features.append(0)
        
        return np.array(features)
    
    def save(self, path: str):
        """Save fitted feature extractor."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'user_scaler': self.user_scaler,
                'item_scaler': self.item_scaler,
                'label_encoders': self.label_encoders,
                'fitted': self.fitted
            }, f)
    
    def load(self, path: str):
        """Load fitted feature extractor."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.user_scaler = data['user_scaler']
            self.item_scaler = data['item_scaler']
            self.label_encoders = data['label_encoders']
            self.fitted = data['fitted']
