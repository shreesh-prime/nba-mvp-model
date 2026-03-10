"""
Load the trained MVP model (Ridge regression on key correlating factors) and predict this season's top candidates.
Run mvp_model.py first to train and save the model.
"""
import os
import json
import joblib

from data.fetch_data import fetch_current_season_advanced


MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def load_model():
    """Load Ridge model, scaler, and feature columns from disk."""
    model = joblib.load(os.path.join(MODEL_DIR, "ridge_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols


def predict(season: str = "2025-26", min_games: int = 20, top_n: int = 10):
    """Predict top MVP candidates for the given season (ranked by Ridge MVP score)."""
    model, scaler, feature_cols = load_model()
    current = fetch_current_season_advanced(season=season, min_games=min_games)
    if current.empty:
        print(f"No data for {season}. Try a different season or check Basketball Reference.")
        return None
    missing = [c for c in feature_cols if c not in current.columns]
    if missing:
        print(f"Current data missing columns: {missing}. Re-run mvp_model.py to retrain.")
        return None
    X = current[feature_cols].fillna(0)
    X_s = scaler.transform(X)
    current["mvp_score"] = model.predict(X_s)
    current["mvp_rank"] = current["mvp_score"].rank(ascending=False).astype(int)
    top = current.nsmallest(top_n, "mvp_rank")[
        ["player", "team_abbr", "mvp_rank", "mvp_score"] + feature_cols
    ]
    print(f"Top {top_n} MVP candidates for {season}")
    print(top.to_string(index=False))
    return top


if __name__ == "__main__":
    predict(season="2025-26", min_games=20, top_n=10)
