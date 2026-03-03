from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from .logic import get_recommendations

app = FastAPI(title="Cart Super Add-On Recommendation System")

class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    user_id: str
    restaurant_id: str
    cart_items: List[Dict[str, Any]]
    top_n: int = 10

class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommendations: List[Dict[str, Any]]
    count: int

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Cart Super Add-On Recommendation System"}

@app.post("/recommendations", response_model=RecommendationResponse)
def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Get addon recommendations for the given cart.
    
    Args:
        request: Recommendation request with user_id, restaurant_id, cart_items
    
    Returns:
        Recommended items with scores
    """
    try:
        recommended_items = get_recommendations(
            user_id=request.user_id,
            restaurant_id=request.restaurant_id,
            cart_items=request.cart_items,
            top_n=request.top_n
        )
        
        return RecommendationResponse(
            recommendations=recommended_items,
            count=len(recommended_items)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
