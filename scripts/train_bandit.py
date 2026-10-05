import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.config import BANDIT_WEIGHTS_PATH, STATE_DIM, WEIGHTS_DIR
from env.simulator import MovieRecommendEnv
from models.bandit import EpsilonGreedyBandit, LinUCBBandit, UCBBandit


def train_bandit(
    bandit_type: str = "epsilon_greedy",
    episodes: int = 2000,
    epsilon: float = 0.2,
    c: float = 1.0,
    alpha: float = 0.5,
    save_path: str = str(BANDIT_WEIGHTS_PATH),
):
    print(f"Initializing Environment and Agent ({bandit_type})...")
    env = MovieRecommendEnv(top_n=5)

    n_movies = int(env.loader.movies_df["movie_id"].max())

    if bandit_type == "epsilon_greedy":
        agent = EpsilonGreedyBandit(n_movies=n_movies, epsilon=epsilon)
    elif bandit_type == "ucb":
        agent = UCBBandit(n_movies=n_movies, c=c)
    elif bandit_type == "linucb":
        agent = LinUCBBandit(n_movies=n_movies, state_dim=STATE_DIM, alpha=alpha)
    else:
        raise ValueError(f"Unknown bandit_type: {bandit_type}")

    rewards_history = []
    print(f"Starting training for {episodes} episodes...")

    for i in range(episodes):
        state = env.reset()

        if bandit_type == "linucb":
            action_movie_ids = agent.predict(state, n_to_recommend=5)
        else:
            action_movie_ids = agent.predict(n_to_recommend=5)

        next_state, total_reward, done, info = env.step(action_movie_ids)

        individual_rewards = info.get("individual_rewards", {})
        for movie_id in action_movie_ids:
            reward = individual_rewards.get(movie_id, 0.0)
            if bandit_type == "linucb":
                agent.update(movie_id, reward, state)
            else:
                agent.update(movie_id, reward)

        rewards_history.append(total_reward)

        if (i + 1) % 200 == 0:
            avg_reward = np.mean(rewards_history[-200:])
            print(f"Episode {i+1}/{episodes} - Average Reward (Last 200): {avg_reward:.4f}")

    print("\nTraining Complete!")
    first_200 = float(np.mean(rewards_history[:200]))
    last_200 = float(np.mean(rewards_history[-200:]))
    print(f"Initial Avg Reward: {first_200:.4f}")
    print(f"Final Avg Reward: {last_200:.4f}")
    improvement = ((last_200 - first_200) / (abs(first_200) + 1e-9)) * 100
    print(f"Improvement: {improvement:.2f}%")

    if save_path:
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        agent.save(save_path)
        print(f"Model saved to {save_path}")

    return agent, rewards_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Armed or Contextual Bandit Recommender")
    parser.add_argument(
        "--type",
        choices=["epsilon_greedy", "ucb", "linucb"],
        default="epsilon_greedy",
        help="Type of bandit algorithm",
    )
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Epsilon for exploration")
    parser.add_argument("--save-path", type=str, default=str(BANDIT_WEIGHTS_PATH), help="Path to save model")
    args = parser.parse_args()

    train_bandit(
        bandit_type=args.type,
        episodes=args.episodes,
        epsilon=args.epsilon,
        save_path=args.save_path,
    )
