"""
Refresh current-season stats and update the current-season MVP prediction.
Run daily (e.g. via cron) so the app always shows up-to-date predictions.

Usage:
  python3 update_data.py

Cron example (every day at 6 AM):
  0 6 * * * cd /path/to/nba-mvp-model && python3 update_data.py
"""
import os
import json
import joblib
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "model")
OUTPUT_PATH = os.path.join(DATA_DIR, "current_season_prediction.json")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

from mvp_model import apply_within_season_ranks
from season_utils import current_season


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "ridge_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    features_path = os.path.join(MODEL_DIR, "feature_cols.json")

    if not os.path.isfile(model_path) or not os.path.isfile(scaler_path) or not os.path.isfile(features_path):
        print("Model not found. Run first: python3 mvp_model.py")
        return

    from data.fetch_data import fetch_current_season_advanced

    season = current_season()
    print(f"Fetching {season} advanced stats from Basketball Reference...")
    current = fetch_current_season_advanced(season=season, min_games=20)
    if current.empty:
        print(f"No data for {season}. Check Basketball Reference or try again later.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path) as f:
        feature_cols = json.load(f)

    use_within_season_ranks = False
    if os.path.isfile(MODEL_METADATA_PATH):
        try:
            with open(MODEL_METADATA_PATH) as f:
                use_within_season_ranks = json.load(f).get("use_within_season_ranks", False)
        except Exception:
            use_within_season_ranks = False

    missing = [c for c in feature_cols if c not in current.columns]
    if missing:
        print("Current data missing columns:", missing)
        return

    if use_within_season_ranks:
        current = apply_within_season_ranks(current, feature_cols)

    mask = current[feature_cols].notna().all(axis=1)
    current = current[mask].copy()
    if current.empty:
        print("No players had complete feature values for prediction.")
        return
    X = current[feature_cols].astype(float)
    X_s = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        current["mvp_score"] = model.predict_proba(X_s)[:, 1]
    else:
        current["mvp_score"] = model.predict(X_s)
    current["mvp_rank"] = current["mvp_score"].rank(ascending=False).astype(int)
    top10_df = current.nsmallest(10, "mvp_rank")

    top10 = []
    for i, (_, row) in enumerate(top10_df.iterrows(), 1):
        top10.append({
            "rank": int(row["mvp_rank"]),
            "player": str(row.get("player", "")),
            "team": str(row.get("team_abbr", "")),
            "score": float(row["mvp_score"]),
        })

    top1 = top10[0] if top10 else {}
    result = {
        "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "season": season,
        "top1_name": top1.get("player", "—"),
        "top1_team": top1.get("team", ""),
        "top10": top10,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved prediction to", OUTPUT_PATH)
    print("Predicted MVP:", result["top1_name"], "(" + result["top1_team"] + ")")


if __name__ == "__main__":
    main()
