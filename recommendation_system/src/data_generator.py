"""
Data generation module for creating realistic synthetic food delivery data.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
import random


class DataGenerator:
    """Generate synthetic food delivery data with realistic patterns."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        
        # Define realistic data distributions
        self.cuisines = ['Indian', 'Chinese', 'Italian', 'Mexican', 'Thai', 'American', 'Japanese']
        self.cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai']
        self.segments = ['budget', 'premium', 'frequent']
        self.categories = ['main', 'side', 'beverage', 'dessert', 'appetizer']
        self.meal_times = ['breakfast', 'lunch', 'dinner', 'late_night']
        
        # City-specific cuisine preferences
        self.city_cuisine_prefs = {
            'Mumbai': {'Indian': 0.4, 'Chinese': 0.2, 'Italian': 0.15, 'Mexican': 0.1, 'Thai': 0.1, 'American': 0.03, 'Japanese': 0.02},
            'Delhi': {'Indian': 0.5, 'Chinese': 0.15, 'Italian': 0.1, 'Mexican': 0.1, 'Thai': 0.08, 'American': 0.05, 'Japanese': 0.02},
            'Bangalore': {'Indian': 0.3, 'Chinese': 0.2, 'Italian': 0.2, 'Mexican': 0.1, 'Thai': 0.1, 'American': 0.07, 'Japanese': 0.03},
            'Hyderabad': {'Indian': 0.45, 'Chinese': 0.2, 'Italian': 0.12, 'Mexican': 0.08, 'Thai': 0.08, 'American': 0.05, 'Japanese': 0.02},
            'Chennai': {'Indian': 0.5, 'Chinese': 0.18, 'Italian': 0.1, 'Mexican': 0.08, 'Thai': 0.08, 'American': 0.04, 'Japanese': 0.02}
        }

    
    def generate_users(self, n_users: int, cities: List[str] = None) -> pd.DataFrame:
        """
        Generate synthetic user profiles with realistic ordering patterns.
        
        Args:
            n_users: Number of users to generate
            cities: List of cities (uses default if None)
        
        Returns:
            DataFrame with user profiles
        """
        if cities is None:
            cities = self.cities
        
        users = []
        
        for i in range(n_users):
            # Power-law distribution: 20% users generate 80% orders
            if i < n_users * 0.2:
                order_count_90d = np.random.randint(20, 100)
            else:
                order_count_90d = np.random.randint(1, 20)
            
            order_count_30d = min(order_count_90d, np.random.randint(0, order_count_90d + 1))
            
            # 15% users with sparse history (cold-start)
            if i < n_users * 0.15:
                order_count_90d = np.random.randint(0, 5)
                order_count_30d = min(order_count_90d, np.random.randint(0, order_count_90d + 1))
            
            # Segment based on order frequency
            if order_count_90d >= 30:
                segment = 'frequent'
            elif order_count_90d >= 10:
                segment = 'premium'
            else:
                segment = 'budget'
            
            city = np.random.choice(cities)
            
            # Preferred cuisines based on city
            city_prefs = self.city_cuisine_prefs.get(city, self.city_cuisine_prefs['Mumbai'])
            preferred_cuisines = np.random.choice(
                list(city_prefs.keys()),
                size=min(3, len(city_prefs)),
                p=list(city_prefs.values()),
                replace=False
            ).tolist()
            
            user = {
                'user_id': f'user_{i}',
                'city': city,
                'segment': segment,
                'order_count_30d': order_count_30d,
                'order_count_90d': order_count_90d,
                'avg_order_value': np.random.uniform(200, 800) if segment == 'premium' else np.random.uniform(100, 400),
                'std_order_value': np.random.uniform(50, 150),
                'days_since_last_order': np.random.randint(0, 30),
                'preferred_cuisines': ','.join(preferred_cuisines),
                'avg_items_per_order': np.random.uniform(2, 5),
                'veg_preference_ratio': np.random.uniform(0, 1),
                'price_sensitivity_score': np.random.uniform(0, 1),
                'registration_date': (datetime.now() - timedelta(days=np.random.randint(30, 365))).isoformat()
            }
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_restaurants(self, n_restaurants: int, cities: List[str] = None) -> pd.DataFrame:
        """
        Generate synthetic restaurant profiles.
        
        Args:
            n_restaurants: Number of restaurants to generate
            cities: List of cities (uses default if None)
        
        Returns:
            DataFrame with restaurant profiles
        """
        if cities is None:
            cities = self.cities
        
        restaurants = []
        
        for i in range(n_restaurants):
            city = np.random.choice(cities)
            cuisine_type = np.random.choice(self.cuisines)
            
            # 10% newly listed restaurants (cold-start)
            is_new = i < n_restaurants * 0.1
            
            restaurant = {
                'restaurant_id': f'rest_{i}',
                'name': f'{cuisine_type} Restaurant {i}',
                'city': city,
                'cuisine_type': cuisine_type,
                'price_range': np.random.randint(1, 5),  # 1-4
                'rating': np.random.uniform(3.5, 5.0) if not is_new else np.random.uniform(3.0, 4.0),
                'num_ratings': np.random.randint(10, 1000) if not is_new else np.random.randint(0, 50),
                'is_chain': np.random.choice([True, False], p=[0.3, 0.7]),
                'avg_delivery_time': np.random.randint(20, 60),
                'popularity_score': np.random.uniform(0.5, 1.0) if not is_new else np.random.uniform(0.1, 0.5),
                'menu_size': np.random.randint(20, 100),
                'is_new': is_new
            }
            restaurants.append(restaurant)
        
        return pd.DataFrame(restaurants)

    
    def generate_menu_items(self, restaurants: pd.DataFrame) -> pd.DataFrame:
        """
        Generate menu items for each restaurant.
        
        Args:
            restaurants: DataFrame of restaurants
        
        Returns:
            DataFrame with menu items
        """
        items = []
        item_id = 0
        
        # Common item names by category
        item_names = {
            'main': ['Biryani', 'Pizza', 'Burger', 'Pasta', 'Noodles', 'Curry', 'Tacos', 'Sushi'],
            'side': ['Fries', 'Salad', 'Raita', 'Garlic Bread', 'Spring Rolls', 'Nachos'],
            'beverage': ['Coke', 'Pepsi', 'Water', 'Juice', 'Lassi', 'Coffee', 'Tea'],
            'dessert': ['Ice Cream', 'Brownie', 'Gulab Jamun', 'Tiramisu', 'Cheesecake'],
            'appetizer': ['Samosa', 'Pakora', 'Chicken Wings', 'Soup', 'Bruschetta']
        }
        
        for _, restaurant in restaurants.iterrows():
            # Generate 15-30 items per restaurant
            n_items = np.random.randint(15, 31)
            
            for _ in range(n_items):
                category = np.random.choice(self.categories, p=[0.35, 0.25, 0.2, 0.1, 0.1])
                name = np.random.choice(item_names[category])
                
                # 20% new items (cold-start)
                is_new = item_id < n_items * 0.2
                
                # Price based on restaurant price range and category
                base_price = restaurant['price_range'] * 50
                if category == 'main':
                    price = base_price * np.random.uniform(1.5, 3.0)
                elif category == 'side':
                    price = base_price * np.random.uniform(0.3, 0.8)
                elif category == 'beverage':
                    price = base_price * np.random.uniform(0.2, 0.5)
                elif category == 'dessert':
                    price = base_price * np.random.uniform(0.5, 1.2)
                else:  # appetizer
                    price = base_price * np.random.uniform(0.4, 1.0)
                
                item = {
                    'item_id': f'item_{item_id}',
                    'restaurant_id': restaurant['restaurant_id'],
                    'name': f'{restaurant["cuisine_type"]} {name}',
                    'category': category,
                    'price': round(price, 2),
                    'is_veg': np.random.choice([True, False], p=[0.4, 0.6]),
                    'popularity_score': np.random.uniform(0.5, 1.0) if not is_new else np.random.uniform(0.1, 0.4),
                    'rating': np.random.uniform(3.5, 5.0) if not is_new else None,
                    'num_ratings': np.random.randint(10, 500) if not is_new else 0,
                    'is_available': np.random.choice([True, False], p=[0.95, 0.05]),
                    'margin': np.random.uniform(0.2, 0.5),
                    'is_promotional': np.random.choice([True, False], p=[0.1, 0.9]),
                    'is_new': is_new
                }
                items.append(item)
                item_id += 1
        
        return pd.DataFrame(items)

    
    def generate_cart_sessions(
        self,
        users: pd.DataFrame,
        restaurants: pd.DataFrame,
        items: pd.DataFrame,
        n_sessions: int,
        start_date: str = '2024-01-01',
        end_date: str = '2024-03-01'
    ) -> pd.DataFrame:
        """
        Generate cart sessions with temporal and contextual patterns.
        
        Args:
            users: DataFrame of users
            restaurants: DataFrame of restaurants
            items: DataFrame of menu items
            n_sessions: Number of sessions to generate
            start_date: Start date for sessions
            end_date: End date for sessions
        
        Returns:
            DataFrame with cart sessions and interaction labels
        """
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        date_range = (end - start).days
        
        sessions = []
        
        # Create item lookup by restaurant
        items_by_restaurant = items.groupby('restaurant_id')['item_id'].apply(list).to_dict()
        
        for session_id in range(n_sessions):
            # Select random user and restaurant
            user = users.sample(1).iloc[0]
            restaurant = restaurants[restaurants['city'] == user['city']].sample(1).iloc[0]
            
            # Generate timestamp with realistic patterns
            day_offset = np.random.randint(0, date_range)
            timestamp = start + timedelta(days=day_offset)
            
            # Add time of day with peak hour patterns
            hour = self._sample_hour_with_peaks()
            timestamp = timestamp.replace(hour=hour, minute=np.random.randint(0, 60))
            
            # Determine meal time
            if 6 <= hour < 11:
                meal_time = 'breakfast'
            elif 11 <= hour < 16:
                meal_time = 'lunch'
            elif 16 <= hour < 23:
                meal_time = 'dinner'
            else:
                meal_time = 'late_night'
            
            day_of_week = timestamp.weekday()
            is_weekend = day_of_week >= 5
            is_peak_hour = (12 <= hour <= 14) or (19 <= hour <= 22)
            
            # Build cart incrementally (1-5 items)
            cart_size = np.random.randint(1, 6)
            restaurant_items = items[items['restaurant_id'] == restaurant['restaurant_id']]
            
            if len(restaurant_items) == 0:
                continue
            
            # Start with main dish
            main_items = restaurant_items[restaurant_items['category'] == 'main']
            if len(main_items) > 0:
                cart_items = [main_items.sample(1).iloc[0]['item_id']]
            else:
                cart_items = [restaurant_items.sample(1).iloc[0]['item_id']]
            
            # Add more items sequentially
            for _ in range(cart_size - 1):
                available_items = restaurant_items[~restaurant_items['item_id'].isin(cart_items)]
                if len(available_items) > 0:
                    # Prefer complementary categories
                    cart_categories = restaurant_items[restaurant_items['item_id'].isin(cart_items)]['category'].values
                    if 'main' in cart_categories and 'side' not in cart_categories:
                        side_items = available_items[available_items['category'] == 'side']
                        if len(side_items) > 0:
                            cart_items.append(side_items.sample(1).iloc[0]['item_id'])
                            continue
                    
                    cart_items.append(available_items.sample(1).iloc[0]['item_id'])
            
            # Generate candidate items (items NOT in cart)
            candidate_items = restaurant_items[~restaurant_items['item_id'].isin(cart_items)]
            
            # Sample 5-10 candidates
            n_candidates = min(10, len(candidate_items))
            if n_candidates == 0:
                continue
            
            sampled_candidates = candidate_items.sample(n=n_candidates)
            
            # Generate acceptance labels (10-30% acceptance rate)
            base_acceptance_rate = 0.15
            
            for _, candidate in sampled_candidates.iterrows():
                # Higher acceptance for complementary items
                acceptance_prob = base_acceptance_rate
                
                cart_categories = restaurant_items[restaurant_items['item_id'].isin(cart_items)]['category'].values
                
                # Boost beverages if no beverage in cart
                if candidate['category'] == 'beverage' and 'beverage' not in cart_categories:
                    acceptance_prob *= 2.0
                
                # Boost desserts if main+side in cart
                if candidate['category'] == 'dessert' and 'main' in cart_categories and 'side' in cart_categories:
                    acceptance_prob *= 1.5
                
                # Boost sides if only main in cart
                if candidate['category'] == 'side' and 'main' in cart_categories and 'side' not in cart_categories:
                    acceptance_prob *= 1.8
                
                # Cap at reasonable rate
                acceptance_prob = min(acceptance_prob, 0.4)
                
                accepted = np.random.random() < acceptance_prob
                
                session = {
                    'session_id': f'session_{session_id}',
                    'user_id': user['user_id'],
                    'restaurant_id': restaurant['restaurant_id'],
                    'timestamp': timestamp.isoformat(),
                    'cart_items': ','.join(cart_items),
                    'candidate_item_id': candidate['item_id'],
                    'accepted': accepted,
                    'meal_time': meal_time,
                    'day_of_week': day_of_week,
                    'is_weekend': is_weekend,
                    'is_peak_hour': is_peak_hour,
                    'city': user['city']
                }
                sessions.append(session)
        
        return pd.DataFrame(sessions)
    
    def _sample_hour_with_peaks(self) -> int:
        """Sample hour of day with peak patterns (lunch: 12-2pm, dinner: 7-10pm)."""
        # Define hour probabilities (peak hours have 3x probability)
        hour_probs = np.ones(24)
        hour_probs[12:15] *= 3  # Lunch peak
        hour_probs[19:23] *= 3  # Dinner peak
        hour_probs = hour_probs / hour_probs.sum()
        
        return np.random.choice(24, p=hour_probs)
    
    def create_temporal_split(
        self,
        sessions: pd.DataFrame,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/val/test splits.
        
        Args:
            sessions: DataFrame of sessions
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Sort by timestamp
        sessions = sessions.sort_values('timestamp').reset_index(drop=True)
        
        n = len(sessions)
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))
        
        train_df = sessions.iloc[:train_end].copy()
        val_df = sessions.iloc[train_end:val_end].copy()
        test_df = sessions.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
