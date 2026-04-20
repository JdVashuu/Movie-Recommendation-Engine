import numpy as np
import torch
from env.simulator import MovieRecommendEnv
from models.dqn import DQNAgent

class RecommendationService:
    def __init__(self):
        self.env = MovieRecommendEnv(top_n=5)
        n_movies = self.env.loader.movies_df["movie_id"].max()
        
        # Build the genre matrix for diversity reranking
        genre_df = self.env.loader.get_movie_genres()
        genre_matrix = np.zeros((n_movies, 19))
        for mid, row in genre_df.iterrows():
            if mid <= n_movies:
                genre_matrix[mid-1] = row.values
        
        # Initialize DQNAgent (state_dim=42, action_dim=n_movies)
        self.agent = DQNAgent(
            state_dim=42, 
            action_dim=n_movies, 
            epsilon=0.0, # 0.0 for inference
            genre_matrix=genre_matrix
        )
        
        # Load pre-trained weights
        try:
            weights_path = "weights/dqn_model.pth"
            self.agent.policy_net.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))
            self.agent.policy_net.eval()
            print(f"✅ Successfully loaded DQN weights from {weights_path}")
        except Exception as e:
            print(f"⚠️ Could not load DQN weights: {e}. Using untrained agent.")

    def get_recommendation(self, user_id: int, n: int):
        # Env reset returns the 42-dim state vector
        state = self.env.reset(user_id=user_id)
        # Use predict with diversity weight
        movie_ids = self.agent.predict(state, n_to_recommend=n, diversity_weight=0.2)
        return [int(mid) for mid in movie_ids]

    def process_feedback(self, user_id: int, movie_id: int, rating: float):
        reward = 1.0 if rating >= 4 else 0.0
        return {"status": "success", "reward_applied": reward}

recommendation_service = RecommendationService()
