from data.loader import MovieLensLoader


def test_loader():
    print("Initializing loader...")
    loader = MovieLensLoader()

    print("Loading data into DataFrames...")
    loader.load_data()

    # Check 1: Did the ratings load correctly?
    print("\n--- Ratings Data (u.data) ---")
    print(f"Total ratings loaded: {len(loader.ratings_df)}")
    print("First 3 rows:")
    print(loader.ratings_df.head(3))

    # Check 2: Did the movies load correctly?
    print("\n--- Movies Data (u.item) ---")
    print(f"Total movies loaded: {len(loader.movies_df)}")
    print("First 3 rows (Movie ID, Title, and Genre flags):")
    print(loader.movies_df.head(3))

    # Check 3: Is the implicit feedback logic working?
    print("\n--- Implicit Feedback (Reward Mapping) ---")
    feedback_df = loader.get_implicit_feedback(threshold=4)
    print("First 5 mapped rewards (Rating >= 4 is 1, else 0):")
    print(feedback_df[["user_id", "movie_id", "rating", "reward"]].head(5))

    # Check 4: Can we fetch a user's history?
    test_user_id = 1
    print(f"\n--- User History for User ID {test_user_id} ---")
    history = loader.get_user_history(user_id=test_user_id, limit=5)
    print(f"Last 5 movie IDs watched by User {test_user_id}: {history}")


if __name__ == "__main__":
    test_loader()
