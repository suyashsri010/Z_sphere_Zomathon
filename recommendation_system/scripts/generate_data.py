"""
Script to generate synthetic data for the recommendation system.
"""
import sys
sys.path.append('.')

from src.data_generator import DataGenerator
import pandas as pd
from pathlib import Path

def main():
    """Generate and save synthetic data."""
    print("Generating synthetic data...")
    
    # Initialize generator
    generator = DataGenerator(seed=42)
    
    # Generate data (MVP sizes for fast iteration)
    print("Generating users...")
    users = generator.generate_users(n_users=5000)
    
    print("Generating restaurants...")
    restaurants = generator.generate_restaurants(n_restaurants=500)
    
    print("Generating menu items...")
    items = generator.generate_menu_items(restaurants)
    
    print("Generating cart sessions...")
    sessions = generator.generate_cart_sessions(
        users=users,
        restaurants=restaurants,
        items=items,
        n_sessions=50000,
        start_date='2024-01-01',
        end_date='2024-03-01'
    )
    
    print(f"Generated {len(users)} users, {len(restaurants)} restaurants, "
          f"{len(items)} items, {len(sessions)} session interactions")
    
    # Create temporal splits
    print("Creating temporal splits...")
    train_df, val_df, test_df = generator.create_temporal_split(sessions)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save data
    data_dir = Path('data/generated')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving data...")
    users.to_csv(data_dir / 'users.csv', index=False)
    restaurants.to_csv(data_dir / 'restaurants.csv', index=False)
    items.to_csv(data_dir / 'items.csv', index=False)
    train_df.to_csv(data_dir / 'train_sessions.csv', index=False)
    val_df.to_csv(data_dir / 'val_sessions.csv', index=False)
    test_df.to_csv(data_dir / 'test_sessions.csv', index=False)
    
    print("Data generation complete!")
    
    # Print statistics
    print("\n=== Data Statistics ===")
    print(f"Users: {len(users)}")
    print(f"  - Cold-start users (<5 orders): {len(users[users['order_count_90d'] < 5])} ({len(users[users['order_count_90d'] < 5])/len(users)*100:.1f}%)")
    print(f"  - Segments: {users['segment'].value_counts().to_dict()}")
    print(f"\nRestaurants: {len(restaurants)}")
    print(f"  - New restaurants: {len(restaurants[restaurants['is_new']])} ({len(restaurants[restaurants['is_new']])/len(restaurants)*100:.1f}%)")
    print(f"  - Cities: {restaurants['city'].value_counts().to_dict()}")
    print(f"\nMenu Items: {len(items)}")
    print(f"  - Categories: {items['category'].value_counts().to_dict()}")
    print(f"  - New items: {len(items[items['is_new']])} ({len(items[items['is_new']])/len(items)*100:.1f}%)")
    print(f"\nSessions: {len(sessions)}")
    print(f"  - Acceptance rate: {sessions['accepted'].mean()*100:.1f}%")
    print(f"  - Meal times: {sessions['meal_time'].value_counts().to_dict()}")
    print(f"  - Peak hours: {sessions['is_peak_hour'].sum()} ({sessions['is_peak_hour'].mean()*100:.1f}%)")

if __name__ == '__main__':
    main()
