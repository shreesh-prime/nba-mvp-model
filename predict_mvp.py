"""
Load a trained MVP model and predict this season's top candidates.
Run mvp_model.py first to train and save the model.

  python3 predict_mvp.py                   # legacy model in model/ (what app.py uses)
  python3 predict_mvp.py --variant logtop  # Ridge vote-share variant in model/logtop/
"""
from data.fetch_data import fetch_current_season_advanced
from mvp_model import apply_within_season_ranks, load_variant_artifacts, score_players, MODEL_VARIANTS
from season_utils import current_season

# Variant used by predict(); None = legacy un-suffixed model files. Set from --variant on the CLI.
PREDICT_VARIANT: str | None = None


def load_model(variant: str | None = None):
    """Load (model, scaler, feature_cols, info) from model/<variant>/, or the legacy model/ files if None."""
    return load_variant_artifacts(variant)


def predict(season: str | None = None, min_games: int = 20, top_n: int = 10):
    """Predict top MVP candidates for the given season (default: current) with PREDICT_VARIANT's model."""
    season = season or current_season()
    model, scaler, feature_cols, info = load_model(PREDICT_VARIANT)
    use_within_season_ranks = info["use_within_season_ranks"]
    current = fetch_current_season_advanced(season=season, min_games=min_games)
    if current.empty:
        print(f"No data for {season}. Try a different season or check Basketball Reference.")
        return None
    missing = [c for c in feature_cols if c not in current.columns]
    if missing:
        print(f"Current data missing columns: {missing}. Re-run mvp_model.py to retrain.")
        return None
    if "vorp" in current.columns:
        current["_raw_vorp"] = current["vorp"]
    if use_within_season_ranks:
        current = apply_within_season_ranks(current, feature_cols, transform=info["rank_transform"])
    mask = current[feature_cols].notna().all(axis=1)
    current = current[mask].copy()
    if current.empty:
        print("No players had complete feature values for prediction.")
        return None
    current["mvp_score"] = score_players(current, model, scaler, feature_cols)
    current["mvp_rank"] = current["mvp_score"].rank(ascending=False, method="first").astype(int)
    top = current.nsmallest(top_n, "mvp_rank")[
        ["player", "team_abbr", "mvp_rank", "mvp_score"] + feature_cols
    ]
    print(f"Top {top_n} MVP candidates for {season} (model: {info['variant'] or 'legacy'})")
    print(top.to_string(index=False))
    return top


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict this season's MVP candidates.")
    parser.add_argument("--variant", default=None, choices=list(MODEL_VARIANTS))
    parser.add_argument("--season", default=None, help="e.g. 2025-26 (default: current season)")
    args = parser.parse_args()
    PREDICT_VARIANT = args.variant
    predict(season=args.season, min_games=20, top_n=10)
