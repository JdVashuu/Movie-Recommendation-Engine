from fastapi import FastAPI

from api.routes import router

app = FastAPI(title="Movie Recommendation using RL")

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Welcome to Movie Recommendation system"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
