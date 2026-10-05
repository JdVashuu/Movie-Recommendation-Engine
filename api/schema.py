from typing import List, Optional

from pydantic import BaseModel, Field


class MovieItem(BaseModel):
    movie_id: int
    title: Optional[str] = None
    genres: Optional[List[str]] = None


class RecommendationRequest(BaseModel):
    user_id: int
    n: int = Field(default=5, ge=1, le=50, description="Number of recommendations to return")
    model: Optional[str] = Field(
        default="dqn",
        description="Recommendation model: 'dqn', 'bandit', 'linucb', 'popular', 'random'",
    )
    diversity_weight: Optional[float] = Field(
        default=0.2, ge=0.0, le=1.0, description="Penalty for genre repetition"
    )


class RecommendationResponse(BaseModel):
    user_id: int
    model_used: str
    recommendations: List[int]
    items: Optional[List[MovieItem]] = None


class FeedbackRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating between 1.0 and 5.0")


class FeedbackResponse(BaseModel):
    status: str
    user_id: int
    movie_id: int
    rating: float
    reward_applied: float
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models_available: List[str]
    dqn_weights_loaded: bool
    bandit_weights_loaded: bool
    device: str
