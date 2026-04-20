import numpy as np
from env.simulator import MovieRecommendEnv
from models.bandit import EpsilonGreedyBandit

def train():
    print("Initializing Environment and Agent...")
    env = MovieRecommendEnv(top_n=5)
    
    # Get the number of movies to initialize the bandit
    n_movies = env.loader.movies_df['movie_id'].max()
    agent = EpsilonGreedyBandit(n_movies=n_movies, epsilon=0.2)
    
    episodes = 2000
    rewards_history = []
    
    print(f"Starting training for {episodes} episodes...")
    
    for i in range(episodes):
        # 1. Reset env for a new user session
        state = env.reset()
        
        # 2. Agent predicts top-N movies
        action_movie_ids = agent.predict(n_to_recommend=5)
        
        # 3. Env returns feedback
        next_state, total_reward, done, info = env.step(action_movie_ids)
        
        # 4. Update Agent knowledge
        # Update with individual feedback for each recommended movie
        individual_rewards = info.get("individual_rewards", {})
        for movie_id in action_movie_ids:
            # If the movie was in the user's ratings, we have a reward (0 or 1)
            # If not rated by the user, we can treat it as 0 (neutral/uninterested)
            reward = individual_rewards.get(movie_id, 0)
            agent.update(movie_id, reward)
            
        rewards_history.append(total_reward)
        
        if (i + 1) % 200 == 0:
            avg_reward = np.mean(rewards_history[-200:])
            print(f"Episode {i+1}/{episodes} - Average Reward (Last 200): {avg_reward:.4f}")

    print("\nTraining Complete!")
    
    # Final Evaluation: Compare first 200 vs last 200
    first_200 = np.mean(rewards_history[:200])
    last_200 = np.mean(rewards_history[-200:])
    print(f"Initial Avg Reward: {first_200:.4f}")
    print(f"Final Avg Reward: {last_200:.4f}")
    print(f"Improvement: {((last_200 - first_200) / (first_200 + 1e-9) * 100):.2f}%")

if __name__ == "__main__":
    train()
