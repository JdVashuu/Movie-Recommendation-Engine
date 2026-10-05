from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "dqn" in data["models_available"]
    assert "bandit" in data["models_available"]


def test_recommend_dqn():
    payload = {"user_id": 1, "n": 5, "model": "dqn"}
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert len(data["recommendations"]) == 5
    assert len(data["items"]) == 5
    for item in data["items"]:
        assert "movie_id" in item
        assert "title" in item


def test_recommend_bandit():
    payload = {"user_id": 1, "n": 3, "model": "bandit"}
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 3
    assert data["model_used"] == "bandit"


def test_recommend_linucb():
    payload = {"user_id": 2, "n": 4, "model": "linucb"}
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 4
    assert data["model_used"] == "linucb"


def test_recommend_cold_start():
    # User 999999 doesn't exist in MovieLens 100k; should succeed gracefully
    payload = {"user_id": 999999, "n": 5, "model": "dqn"}
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 5


def test_feedback_endpoint():
    payload = {"user_id": 1, "movie_id": 1, "rating": 5.0}
    response = client.post("/api/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["reward_applied"] == 1.0


def test_get_movie_details():
    response = client.get("/api/movies/1")
    assert response.status_code == 200
    data = response.json()
    assert data["movie_id"] == 1
    assert "Toy Story" in data["title"]


def test_search_movies():
    response = client.get("/api/movies?q=Star Wars")
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) > 0
    assert any("Star Wars" in item["title"] for item in results)
