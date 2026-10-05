import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.config import (
    BANDIT_WEIGHTS_PATH,
    DQN_WEIGHTS_PATH,
    NUM_GENRES,
    STATE_DIM,
)
from env.simulator import MovieRecommendEnv
from models.bandit import EpsilonGreedyBandit, LinUCBBandit, UCBBandit
from models.dqn import DQNAgent


def compute_intra_list_diversity(movie_ids: List[int], genre_matrix: np.ndarray) -> float:
    """Computes average pairwise cosine distance (1 - cosine_similarity) between recommended items."""
    k = len(movie_ids)
    if k <= 1:
        return 0.0

    vectors = []
    for mid in movie_ids:
        idx = mid - 1
        if 0 <= idx < len(genre_matrix):
            vec = genre_matrix[idx]
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm > 0 else vec)
        else:
            vectors.append(np.zeros(genre_matrix.shape[1]))

    distances = []
    for i in range(k):
        for j in range(i + 1, k):
            sim = float(np.dot(vectors[i], vectors[j]))
            distances.append(1.0 - sim)

    return float(np.mean(distances)) if distances else 0.0


def run_benchmark(n_episodes: int = 200, top_k: int = 5):
    print(f"Loading environment for benchmark across {n_episodes} test episodes (top-{top_k})...\n")
    env = MovieRecommendEnv(top_n=top_k)
    n_movies = int(env.loader.movies_df["movie_id"].max())

    # Build genre matrix
    genre_df = env.loader.get_movie_genres()
    genre_matrix = np.zeros((n_movies, NUM_GENRES), dtype=np.float32)
    for mid, row in genre_df.iterrows():
        if mid <= n_movies:
            genre_matrix[mid - 1] = row.values

    # Initialize / Load Agents
    # 1. Random
    # 2. Popular
    popular_movies = env.loader.get_popular_movies(limit=top_k)

    # 3. Bandit
    bandit = EpsilonGreedyBandit(n_movies=n_movies, epsilon=0.0)
    if os.path.exists(BANDIT_WEIGHTS_PATH):
        try:
            bandit = EpsilonGreedyBandit.load(str(BANDIT_WEIGHTS_PATH))
            bandit.epsilon = 0.0
            print(f"Loaded trained Bandit from {BANDIT_WEIGHTS_PATH}")
        except Exception:
            pass

    # 4. LinUCB
    linucb = LinUCBBandit(n_movies=n_movies, state_dim=STATE_DIM, alpha=0.2)

    # 5. DQN (with diversity)
    dqn_diverse = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=n_movies,
        epsilon=0.0,
        genre_matrix=genre_matrix,
    )
    # 6. DQN (standard greedy)
    dqn_greedy = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=n_movies,
        epsilon=0.0,
        genre_matrix=None,
    )

    if os.path.exists(DQN_WEIGHTS_PATH):
        try:
            dqn_diverse.load(str(DQN_WEIGHTS_PATH))
            dqn_greedy.load(str(DQN_WEIGHTS_PATH))
            print(f"Loaded trained DQN from {DQN_WEIGHTS_PATH}")
        except Exception as e:
            print(f"Note: Pretrained DQN weights not loaded ({e}), evaluating initialized model.")

    models = {
        "Random": lambda state: np.random.choice(np.arange(1, n_movies + 1), top_k, replace=False),
        "Popularity": lambda state: np.array(popular_movies),
        "Eps-Bandit": lambda state: bandit.predict(top_k),
        "LinUCB": lambda state: linucb.predict(state, top_k),
        "Dueling-DQN (Greedy)": lambda state: dqn_greedy.predict(state, top_k, diversity_weight=0.0, deterministic=True),
        "Dueling-DQN (Diverse)": lambda state: dqn_diverse.predict(state, top_k, diversity_weight=0.3, deterministic=True),
    }

    # Generate fixed set of test user IDs for fair evaluation
    all_users = env.loader.ratings_df["user_id"].unique()
    test_users = np.random.choice(all_users, size=min(n_episodes, len(all_users)), replace=False)

    results: Dict[str, Dict[str, float]] = {}

    for name, policy in models.items():
        rewards: List[float] = []
        hits: List[int] = []
        positive_hits: List[int] = []
        diversities: List[float] = []
        recommended_all = set()

        for user_id in test_users:
            state = env.reset(user_id=int(user_id))
            action_movie_ids = [int(x) for x in policy(state)]
            recommended_all.update(action_movie_ids)

            _, reward, _, info = env.step(action_movie_ids)
            rewards.append(reward)

            ind_rewards = info.get("individual_rewards", {})
            hits.append(len(ind_rewards))
            pos_hits = sum(1 for r in ind_rewards.values() if r >= 0.6)
            positive_hits.append(pos_hits)

            div = compute_intra_list_diversity(action_movie_ids, genre_matrix)
            diversities.append(div)

        results[name] = {
            "Avg Reward": float(np.mean(rewards)),
            "Hit Rate": float(np.mean(hits)) / top_k,
            "Positive Precision": float(np.mean(positive_hits)) / top_k,
            "Intra-List Diversity": float(np.mean(diversities)),
            "Catalog Coverage (%)": (len(recommended_all) / n_movies) * 100.0,
        }

    # Print Table
    print("\n" + "=" * 95)
    print(f"{'Model':<24} | {'Avg Reward':<12} | {'Hit Rate@5':<12} | {'Pos Precision':<14} | {'Diversity':<10} | {'Catalog %':<10}")
    print("=" * 95)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<24} | "
            f"{metrics['Avg Reward']:>12.4f} | "
            f"{metrics['Hit Rate']:>12.4f} | "
            f"{metrics['Positive Precision']:>14.4f} | "
            f"{metrics['Intra-List Diversity']:>10.4f} | "
            f"{metrics['Catalog Coverage (%)']:>9.2f}%"
        )
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark recommendation models on MovieLens-100k")
    parser.add_argument("--episodes", type=int, default=200, help="Number of test evaluation episodes")
    parser.add_argument("--top-k", type=int, default=5, help="Slate recommendation size")
    args = parser.parse_args()

    run_benchmark(n_episodes=args.episodes, top_k=args.top_k)
