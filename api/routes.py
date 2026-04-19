from fastapi import APIRouter, HTTPException

from api.schema import FeedbackRequest, RecommendationRequest, RecommendationResponse
from api.service import RecommendationService

router = APIRouter()
service = RecommendationService()


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    try:
        recs = service.get_recommendation(user_id=request.user_id, n=request.n)
        return RecommendationResponse(user_id=request.user_id, recommendations=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def feedback(request: FeedbackRequest):
    try:
        result = service.process_feedback(
            user_id=request.user_id, movie_id=request.movie_id, rating=request.rating
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
