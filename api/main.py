from fastapi import FastAPI

from api.routes import router
from api.schema import HealthResponse
from api.service import get_recommendation_service

app = FastAPI(
    title="Movie Recommendation Engine (RL & Bandits)",
    description=(
        "Production-ready reinforcement learning movie recommendation service. "
        "Supports Dueling DQN with diversity reranking, contextual LinUCB, "
        "multi-armed bandits, and online continuous learning from user feedback."
    ),
    version="0.2.0",
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "message": "Welcome to Movie Recommendation Engine",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def root_health():
    service = get_recommendation_service()
    return service.get_health()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
