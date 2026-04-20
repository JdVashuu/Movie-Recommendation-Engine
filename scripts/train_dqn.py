import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.simulator import MovieRecommendEnv
from models.dqn import DQNAgent

def train_dqn():
    print("Init Env and DQN Agent...")
    env = MovieRecommendEnv(top_n=5)

    n_movies = env.loader.movies_df['movie_id'].max()
    state_dim = 42
    action_dim = n_movies

    # Initialize genre matrix for diversity
    genre_df = env.loader.get_movie_genres()
    genre_matrix = np.zeros((n_movies, 19))
    for mid, row in genre_df.iterrows():
        if mid <= n_movies:
            genre_matrix[mid-1] = row.values

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, genre_matrix=genre_matrix)

    episodes = 5000
    batch_size = 128
    reward_history = []

    print(f"Starting DQN training for {episodes} episodes...")

    for i in range(episodes):
        state = env.reset()

        action_movie_ids = agent.predict(state, n_to_recommend=5)
        next_state, total_reward, done, info = env.step(action_movie_ids)

        indv_rewards = info.get("individual_rewards", {})
        for movie_id in action_movie_ids:
            reward = indv_rewards.get(movie_id, 0)
            agent.memory.push(state, movie_id, reward, next_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        # slower target update
        if (i + 1) % 100 == 0:
            agent.update_target_network()

        reward_history.append(total_reward)

        if (i + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"Episode {i + 1}/{episodes} | Avg reward (last 100) : {avg_reward:.4f} | Epsilon : {agent.epsilon:.4f}")
    
    print("DQN Trainging complete")

    first_100 = np.mean(reward_history[:100])                                                                                               
    last_100 = np.mean(reward_history[-100:])                                                                                               
    print(f"Initial Avg Reward: {first_100:.4f}")                                                                                         
    print(f"Final Avg Reward: {last_100:.4f}")                                                                                            
    
    # Save the model                                                                                                                         
    os.makedirs("weights", exist_ok=True)                                                                                                    
    torch.save(agent.policy_net.state_dict(), "weights/dqn_model.pth")                                                                       
    print(f"Model weights saved to weights/dqn_model.pth")                                                                                

if __name__ == "__main__":                                                                                                                   
    train_dqn()        

        
