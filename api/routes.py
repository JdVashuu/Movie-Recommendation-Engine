from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.schema import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MovieItem,
    RecommendationRequest,
    RecommendationResponse,
)
from api.service import get_recommendation_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    service = get_recommendation_service()
    return service.get_health()


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    try:
        service = get_recommendation_service()
        model_used, rec_ids, items = service.get_recommendation(
            user_id=request.user_id,
            n=request.n,
            model=request.model or "dqn",
            diversity_weight=request.diversity_weight if request.diversity_weight is not None else 0.2,
        )
        movie_items = [
            MovieItem(
                movie_id=it["movie_id"],
                title=it.get("title"),
                genres=it.get("genres"),
            )
            for it in items
        ]
        return RecommendationResponse(
            user_id=request.user_id,
            model_used=model_used,
            recommendations=rec_ids,
            items=movie_items,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    try:
        service = get_recommendation_service()
        result = service.process_feedback(
            user_id=request.user_id,
            movie_id=request.movie_id,
            rating=request.rating,
        )
        return FeedbackResponse(
            status=result["status"],
            user_id=result["user_id"],
            movie_id=result["movie_id"],
            rating=result["rating"],
            reward_applied=result["reward_applied"],
            message=result.get("message"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/movies/{movie_id}")
async def get_movie(movie_id: int):
    service = get_recommendation_service()
    info = service.get_movie(movie_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    return info


@router.get("/movies")
async def search_movies(
    q: Optional[str] = Query(None, description="Search keyword in movie title"),
    limit: int = Query(10, ge=1, le=100),
):
    service = get_recommendation_service()
    if q:
        return service.search_movies(q, limit=limit)
    # Default to top popular movies
    popular_ids = service.env.loader.get_popular_movies(limit=limit)
    return [service.get_movie(mid) for mid in popular_ids]
