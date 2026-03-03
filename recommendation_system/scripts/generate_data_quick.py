"""
Quick data generation script for fast testing (small dataset).
"""
import sys
sys.path.append('.')

from src.data_generator import DataGenerator
import pandas as pd
from pathlib import Path

def main():
    """Generate small synthetic data for quick testing."""
    print("Generating SMALL synthetic data for quick testing...")
    
    # Initialize generator
    generator = DataGenerator(seed=42)
    
    # Generate SMALL data (10x smaller for speed)
    print("Generating users...")
    users = generator.generate_users(n_users=500)  # Instead of 5000
    
    print("Generating restaurants...")
    restaurants = generator.generate_restaurants(n_restaurants=50)  # Instead of 500
    
    print("Generating menu items...")
    items = generator.generate_menu_items(restaurants)
    
    print("Generating cart sessions...")
    sessions = generator.generate_cart_sessions(
        users=users,
        restaurants=restaurants,
        items=items,
        n_sessions=5000,  # Instead of 50000
        start_date='2024-01-01',
        end_date='2024-02-01'  # Shorter time range
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
    
    print("✅ Quick data generation complete!")
    
    # Print statistics
    print("\n=== Data Statistics ===")
    print(f"Users: {len(users)}")
    print(f"  - Cold-start users (<5 orders): {len(users[users['order_count_90d'] < 5])}")
    print(f"Restaurants: {len(restaurants)}")
    print(f"Menu Items: {len(items)}")
    print(f"Sessions: {len(sessions)}")
    print(f"  - Acceptance rate: {sessions['accepted'].mean()*100:.1f}%")

if __name__ == '__main__':
    main()
