from typing import List

from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    user_id: int
    n: int = 5


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[int]


class FeedbackRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float
