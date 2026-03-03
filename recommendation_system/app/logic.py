from typing import List, Dict, Any

try:
    # When installed as a package
    from recommendation_system.src.recommendation_engine import RecommendationEngine
except ModuleNotFoundError:
    # When running locally from the project root
    from src.recommendation_engine import RecommendationEngine

# Initialize recommendation engine (singleton)
_engine = None

def get_engine() -> RecommendationEngine:
    """Get or create recommendation engine instance."""
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine

def get_recommendations(
    user_id: str,
    restaurant_id: str,
    cart_items: List[Dict[str, Any]],
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Get personalized add-on recommendations.
    
    Args:
        user_id: User ID
        restaurant_id: Restaurant ID
        cart_items: List of items currently in cart
        top_n: Number of recommendations to return
    
    Returns:
        List of recommended items with scores
    """
    engine = get_engine()
    
    try:
        recommendations = engine.get_recommendations(
            user_id=user_id,
            restaurant_id=restaurant_id,
            cart_items=cart_items,
            top_n=top_n
        )
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        # Fallback: return empty list
        return []
