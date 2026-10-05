import os
from pathlib import Path
from typing import Dict, List

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = DATA_DIR / "ml-100k.zip"
ML_100K_DIR = DATA_DIR / "ml-100k"

WEIGHTS_DIR = BASE_DIR / "weights"
DQN_WEIGHTS_PATH = WEIGHTS_DIR / "dqn_model.pth"
BANDIT_WEIGHTS_PATH = WEIGHTS_DIR / "bandit_model.pkl"

# MovieLens 100k Genre Names
GENRE_NAMES: List[str] = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

NUM_GENRES: int = len(GENRE_NAMES)  # 19
STATE_DIM: int = 42  # 19 genre affinity + 2 demographic (age, gender) + 21 occupation dummy features
NUM_MOVIES: int = 1682

# Granular Reward Scheme
REWARD_MAPPING: Dict[int, float] = {
    5: 1.0,
    4: 0.6,
    3: 0.1,
    2: -0.5,
    1: -0.5,
}

# Hyperparameters
DEFAULT_TOP_N: int = 5
DEFAULT_DIVERSITY_WEIGHT: float = 0.2

# DQN Hyperparameters
DQN_LR: float = 1e-4
DQN_GAMMA: float = 0.99
DQN_EPSILON_START: float = 1.0
DQN_EPSILON_MIN: float = 0.01
DQN_EPSILON_DECAY: float = 0.997
DQN_BATCH_SIZE: int = 128
DQN_TARGET_UPDATE_FREQ: int = 100
REPLAY_BUFFER_CAPACITY: int = 50000

# Bandit Hyperparameters
BANDIT_EPSILON: float = 0.1
