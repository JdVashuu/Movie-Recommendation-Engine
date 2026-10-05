import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.config import (
    DQN_BATCH_SIZE,
    DQN_LR,
    DQN_TARGET_UPDATE_FREQ,
    DQN_WEIGHTS_PATH,
    NUM_GENRES,
    STATE_DIM,
    WEIGHTS_DIR,
)
from env.simulator import MovieRecommendEnv
from models.dqn import DQNAgent


def train_dqn(
    episodes: int = 3000,
    batch_size: int = DQN_BATCH_SIZE,
    lr: float = DQN_LR,
    save_path: str = str(DQN_WEIGHTS_PATH),
):
    print("Initializing MovieRecommendEnv and DQNAgent...")
    env = MovieRecommendEnv(top_n=5)

    n_movies = int(env.loader.movies_df["movie_id"].max())
    state_dim = STATE_DIM
    action_dim = n_movies

    # Initialize genre matrix for diversity
    genre_df = env.loader.get_movie_genres()
    genre_matrix = np.zeros((n_movies, NUM_GENRES), dtype=np.float32)
    for mid, row in genre_df.iterrows():
        if mid <= n_movies:
            genre_matrix[mid - 1] = row.values

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        genre_matrix=genre_matrix,
    )

    reward_history = []
    print(f"Device: {agent.device}")
    print(f"Starting DQN training for {episodes} episodes...")

    for i in range(episodes):
        state = env.reset()

        action_movie_ids = agent.predict(state, n_to_recommend=5)
        next_state, total_reward, done, info = env.step(action_movie_ids)

        indv_rewards = info.get("individual_rewards", {})
        for movie_id in action_movie_ids:
            reward = indv_rewards.get(movie_id, 0.0)
            agent.memory.push(state, movie_id, reward, next_state, done)

        if len(agent.memory) >= batch_size:
            agent.update(batch_size)

        # Target network update
        if (i + 1) % DQN_TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        reward_history.append(total_reward)

        if (i + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(
                f"Episode {i + 1}/{episodes} | Avg reward (last 100): {avg_reward:.4f} | Epsilon: {agent.epsilon:.4f}"
            )

    print("\nDQN Training complete!")

    first_100 = float(np.mean(reward_history[:100]))
    last_100 = float(np.mean(reward_history[-100:]))
    print(f"Initial Avg Reward: {first_100:.4f}")
    print(f"Final Avg Reward: {last_100:.4f}")

    # Save model weights
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    agent.save(save_path)
    print(f"Model weights saved to {save_path}")

    return agent, reward_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dueling DQN Recommender")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=DQN_BATCH_SIZE, help="Replay buffer batch size")
    parser.add_argument("--lr", type=float, default=DQN_LR, help="Adam learning rate")
    parser.add_argument("--save-path", type=str, default=str(DQN_WEIGHTS_PATH), help="Path to save weights")
    args = parser.parse_args()

    train_dqn(
        episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )
