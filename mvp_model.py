"""
NBA MVP prediction: correlation analysis on advanced stats, then train a model
and predict this season's MVP. Uses Basketball Reference advanced stats only.
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from data.fetch_data import (
    build_advanced_historical_dataset,
    fetch_current_season_advanced,
    ADVANCED_STAT_COLUMNS,
    FULL_FEATURE_COLUMNS,
    REGULAR_AND_WINS_COLUMNS,
)

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "historical_advanced.csv")
FEATURE_RANK_PATH = os.path.join(DATA_DIR, "mvp_feature_correlations.json")
BACKTEST_REPORT_PATH = os.path.join(DATA_DIR, "backtest_results.csv")
BACKTEST_DETAILS_PATH = os.path.join(DATA_DIR, "backtest_season_details.json")
FORMULA_PATH = os.path.join(DATA_DIR, "mvp_formula.txt")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def get_historical_data(use_cache: bool = True, start_year: int = 2000, end_year: int = 2025) -> pd.DataFrame:
    """Load or build historical player-season data with advanced stats and is_mvp."""
    if use_cache and os.path.isfile(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH)
        print(f"Loaded cached data: {len(df)} rows from {CACHE_PATH}")
        return df
    print("Building historical dataset from Basketball Reference (this may take a few minutes)...")
    try:
        df = build_advanced_historical_dataset(start_year=start_year, end_year=end_year)
    except (OSError, Exception) as e:
        err_msg = str(e).lower()
        use_cache_on_error = (
            "nodename nor servname" in err_msg or "gaierror" in err_msg or "urlopen" in err_msg
            or "certificate" in err_msg or "could not fetch mvp" in err_msg or "rate limited" in err_msg
        )
        if use_cache_on_error and use_cache and os.path.isfile(CACHE_PATH):
            print("Fetch failed (network/rate limit). Using cached data.")
            return pd.read_csv(CACHE_PATH)
        if "could not fetch mvp" in err_msg or "rate limited" in err_msg:
            raise RuntimeError(
                "Could not fetch MVP history from Basketball Reference (often 429 rate limit).\n"
                "  - Wait 5–10 minutes and run again, or keep data/historical_advanced.csv to use cache."
            ) from e
        if use_cache_on_error:
            raise RuntimeError(
                "Could not reach Basketball Reference (check internet and DNS).\n"
                "  - Try opening https://www.basketball-reference.com in your browser.\n"
                "  - If you have data/historical_advanced.csv from a previous run, the script will use it next time."
            ) from e
        raise
    df.to_csv(CACHE_PATH, index=False)
    print(f"Saved to {CACHE_PATH}")
    return df


def compute_correlations(df: pd.DataFrame) -> pd.Series:
    """
    Compute correlation of each feature with is_mvp.
    Uses full feature set: advanced + ppg, rpg, apg, team_wins (points and wins emphasized).
    """
    candidate_cols = [c for c in FULL_FEATURE_COLUMNS if c in df.columns]
    if not candidate_cols:
        candidate_cols = [c for c in ADVANCED_STAT_COLUMNS if c in df.columns]
    df_clean = df[candidate_cols + ["is_mvp"]].dropna(how="all", subset=candidate_cols)
    correlations = df_clean[candidate_cols].corrwith(df_clean["is_mvp"]).abs().sort_values(ascending=False)
    # Emphasize criteria that match MVP voting: team success (strongest), then points
    for col in ("top_team_3", "team_win_rank", "team_wins", "ppg"):
        if col in correlations.index and not pd.isna(correlations[col]):
            boost = 1.6 if col in ("top_team_3", "team_win_rank") else 1.4
            correlations[col] = min(1.0, correlations[col] * boost)
    return correlations.sort_values(ascending=False)


def analyze_and_save_correlations(df: pd.DataFrame, plot: bool = True) -> pd.Series:
    """Compute correlations, save to JSON, print top stats, and optionally plot."""
    correlations = compute_correlations(df)
    with open(FEATURE_RANK_PATH, "w") as f:
        json.dump(correlations.round(4).to_dict(), f, indent=2)
    print("\n--- Correlation with MVP (higher = more predictive) ---")
    print(correlations.head(15).to_string())
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            corr_plot = correlations.head(15)
            corr_plot.plot(kind="barh", ax=ax, color="steelblue", edgecolor="navy", alpha=0.8)
            ax.set_xlabel("|Correlation with MVP|")
            ax.set_title("NBA Advanced Stats: Correlation with Winning MVP")
            ax.invert_yaxis()
            plt.tight_layout()
            plot_path = os.path.join(DATA_DIR, "mvp_correlation_plot.png")
            plt.savefig(plot_path, dpi=120)
            plt.close()
            print(f"Saved correlation plot to {plot_path}")
        except Exception as e:
            print(f"Could not save plot: {e}")
    return correlations


def get_top_features(correlations: pd.Series, top_k: int = 12) -> list:
    """Top-k features; prioritize team success (top_team_3, team_win_rank) and ppg in the score, plus impactful advanced stats."""
    top = correlations.head(top_k).index.tolist()
    for must in ("top_team_3", "team_win_rank", "team_wins", "ppg"):
        if must in correlations.index and must not in top:
            top = [must] + [f for f in top if f != must][: top_k - 1]
    return top[:top_k]


def prepare_xy(df: pd.DataFrame, feature_cols: list):
    """Prepare X (numeric features) and y (is_mvp). Drop rows with missing feature values."""
    X = df[feature_cols].copy()
    y = df["is_mvp"].copy()
    mask = X.notna().all(axis=1)
    X = X[mask].astype(float)
    y = y[mask].astype(int)
    return X, y, df[mask]


def train_model(
    df: pd.DataFrame,
    feature_cols: list,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train Ridge regression (primary) and Random Forest on correlated features.
    Ridge ranks by predicted MVP score (linear combo of key factors).
    Returns (ridge, scaler, feature_cols) for production.
    """
    X, y, df_xy = prepare_xy(df, feature_cols)
    if y.sum() < 2:
        raise ValueError("Need at least 2 MVP seasons in the data to train.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0, random_state=random_state)
    ridge.fit(X_train_s, y_train)
    y_score = ridge.predict(X_test_s)
    y_pred_ridge = (y_score >= np.percentile(y_score, 100 * (1 - 1 / len(y_score)))) if len(y_score) else np.array([])
    print("\n--- Ridge regression (MVP score) — test set ---")
    if len(y_pred_ridge):
        print(classification_report(y_test, y_pred_ridge, zero_division=0))

    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=random_state, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    print("\n--- Random Forest (reference) — test set ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n--- Feature importance (Random Forest) ---")
    print(imp.to_string())
    return ridge, scaler, feature_cols


def backtest_historical(
    df: pd.DataFrame,
    feature_cols: list,
    min_seasons: int = 5,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Leave-one-season-out backtest using Ridge (linear combo of key factors).
    Saves backtest_results.csv and backtest_season_details.json with top_n per season.
    """
    seasons = df["season"].dropna().unique().tolist()
    seasons = sorted([s for s in seasons if str(s).strip() != ""])
    if len(seasons) < min_seasons:
        raise ValueError(f"Need at least {min_seasons} seasons, got {len(seasons)}")

    results = []
    details_by_season = {}

    for test_season in seasons:
        train_df = df[df["season"] != test_season].copy()
        test_df = df[df["season"] == test_season].copy()
        X_train, y_train, train_xy = prepare_xy(train_df, feature_cols)
        X_test, y_test, test_xy = prepare_xy(test_df, feature_cols)
        if X_train.empty or X_test.empty or y_train.sum() < 1:
            actual_mvp = _get_actual_mvp(df, test_season)
            results.append({"season": test_season, "actual_mvp": actual_mvp, "predicted_mvp": None, "correct": False, "note": "insufficient data"})
            details_by_season[str(test_season)] = {"actual_mvp": actual_mvp, "predicted_mvp": None, "correct": False, "top10": []}
            continue
        if y_test.sum() < 1:
            actual_mvp = _get_actual_mvp(df, test_season)
            results.append({"season": test_season, "actual_mvp": actual_mvp, "predicted_mvp": None, "correct": False, "note": "no MVP in test set"})
            details_by_season[str(test_season)] = {"actual_mvp": actual_mvp, "predicted_mvp": None, "correct": False, "top10": []}
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train_s, y_train)
        scores = ridge.predict(X_test_s)
        test_xy = test_xy.copy()
        test_xy["_mvp_score"] = scores
        # Rank all players by MVP score (team success is in the formula but not required; standout stats can win)
        candidates = test_xy.sort_values("_mvp_score", ascending=False).reset_index(drop=True)
        top_n_df = candidates.head(top_n)
        predicted_row = top_n_df.iloc[0]
        predicted_mvp = predicted_row["player"] if "player" in predicted_row else predicted_row.get("player_clean", "?")
        actual_mvp = _get_actual_mvp(df, test_season)
        correct = _names_match(actual_mvp, str(predicted_mvp))

        top10_list = []
        for i, (_, row) in enumerate(top_n_df.iterrows(), 1):
            top10_list.append({
                "rank": i,
                "player": str(row.get("player", row.get("player_clean", "?"))),
                "team": str(row.get("team_abbr", "")),
                "score": float(row["_mvp_score"]),
            })

        results.append({"season": test_season, "actual_mvp": actual_mvp, "predicted_mvp": predicted_mvp, "correct": correct, "note": ""})
        details_by_season[str(test_season)] = {"actual_mvp": actual_mvp, "predicted_mvp": predicted_mvp, "correct": correct, "top10": top10_list}

    report = pd.DataFrame(results)
    report.to_csv(BACKTEST_REPORT_PATH, index=False)
    with open(BACKTEST_DETAILS_PATH, "w") as f:
        json.dump(details_by_season, f, indent=2)
    n = report["correct"].sum()
    total = len(report[report["predicted_mvp"].notna()])
    if total:
        print(f"\n--- Backtest (Ridge, leave-one-season-out): {n}/{total} seasons correct ({100 * n / total:.1f}%) ---")
        print(report.to_string(index=False))
        print(f"\nSaved to {BACKTEST_REPORT_PATH} and {BACKTEST_DETAILS_PATH}")
    return report


def _get_actual_mvp(df: pd.DataFrame, season: str) -> str:
    """Return actual MVP player name for a season (from is_mvp=1)."""
    sub = df[(df["season"] == season) & (df["is_mvp"] == 1)]
    if sub.empty:
        return ""
    name = sub.iloc[0].get("player_clean") or sub.iloc[0].get("player") or ""
    return str(name).strip()


def _names_match(a: str, b: str) -> bool:
    """Compare player names (strip, case-insensitive; normalize accents e.g. Jokić->Jokic)."""
    import unicodedata
    if not a or not b or pd.isna(a) or pd.isna(b):
        return False
    def norm(s):
        s = str(s).strip().lower().replace(".", "").replace("'", "")
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        return s
    a, b = norm(a), norm(b)
    return a == b or (len(a) > 2 and len(b) > 2 and (a in b or b in a))


def get_mvp_formula(
    df: pd.DataFrame,
    feature_cols: list,
    save_path: str | None = None,
) -> tuple:
    """
    Fit logistic regression on the same features to get an interpretable formula.
    P(MVP) = 1 / (1 + exp(-z)) where z = intercept + coef1*feat1 + ...
    Returns (fitted model, scaler, formula_string).
    """
    X, y, _ = prepare_xy(df, feature_cols)
    if y.sum() < 2:
        return None, None, "Insufficient MVP samples for formula."
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        C=0.5,
    )
    lr.fit(X_s, y)
    # Build formula string: z = intercept + c1*feat1 + c2*feat2 + ...
    parts = [f"{lr.intercept_[0]:.4f}"]
    for name, coef in zip(feature_cols, lr.coef_[0]):
        sign = "+" if coef >= 0 else ""
        parts.append(f"{sign}{coef:.4f}*{name}")
    z_expr = " ".join(parts)
    formula = (
        "MVP formula (logistic regression, after standardizing features):\n"
        "  P(MVP) = 1 / (1 + exp(-z))\n"
        "  z = " + z_expr + "\n\n"
        "Higher z => higher MVP probability. Coefficients (in z) show how much each "
        "standardized stat pushes toward MVP (positive = helps, negative = hurts).\n"
        "Features are standardized (mean=0, std=1) before plugging in."
    )
    if save_path is None:
        save_path = FORMULA_PATH
    with open(save_path, "w") as f:
        f.write(formula)
    print("\n--- MVP formula (interpretable model) ---")
    print(formula)
    print(f"Saved to {save_path}")
    return lr, scaler, formula


def predict_current_season(
    model,
    scaler,
    feature_cols: list,
    season: str = "2025-26",
    min_games: int = 20,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Fetch current season advanced stats, score with Ridge model (linear combo of key factors),
    return top_n candidates ranked by MVP score.
    """
    current = fetch_current_season_advanced(season=season, min_games=min_games)
    if current.empty:
        raise FileNotFoundError(f"No data for season {season}. Check Basketball Reference.")
    missing = [c for c in feature_cols if c not in current.columns]
    if missing:
        raise ValueError(f"Current season data missing columns: {missing}")
    X = current[feature_cols].fillna(0)
    X_s = scaler.transform(X)
    current["mvp_score"] = model.predict(X_s)
    # Rank all players by MVP score (top-3 team helps in the formula but not required)
    current["mvp_rank"] = current["mvp_score"].rank(ascending=False).astype(int)
    top = current.nsmallest(top_n, "mvp_rank")[["player", "team_abbr", "mvp_rank", "mvp_score"] + [c for c in feature_cols if c in current.columns]]
    return top


def run_full_pipeline(
    start_year: int = 2000,
    end_year: int = 2025,
    top_k_features: int = 12,
    current_season: str = "2025-26",
    use_cache: bool = True,
    run_backtest: bool = True,
):
    """
    1) Load historical data
    2) Compute and save correlations (which stats correlate most with MVP)
    3) Optionally run backtest (leave-one-season-out) and print MVP formula
    4) Train model on top-k correlated stats
    5) Predict current season and print top 10
    """
    df = get_historical_data(use_cache=use_cache, start_year=start_year, end_year=end_year)
    correlations = analyze_and_save_correlations(df)
    feature_cols = get_top_features(correlations, top_k=top_k_features)
    print(f"\nUsing top-{top_k_features} stats for model: {feature_cols}")

    if run_backtest:
        backtest_historical(df, feature_cols)
        get_mvp_formula(df, feature_cols, save_path=FORMULA_PATH)
        total = pd.read_csv(BACKTEST_REPORT_PATH)
        n_correct = total["correct"].sum()
        n_total = total["predicted_mvp"].notna().sum()
        if n_total:
            print(f"\n>>> Backtest accuracy: {n_correct}/{n_total} seasons ({100 * n_correct / n_total:.1f}%)")
            if n_correct / n_total < 0.6:
                print(">>> Tip: See 'Ideas to improve accuracy' in README or run: python mvp_model.py backtest")

    clf, scaler, feature_cols = train_model(df, feature_cols)
    joblib.dump(clf, os.path.join(MODEL_DIR, "ridge_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    print(f"\nRidge model and scaler saved to {MODEL_DIR}")

    top = None
    try:
        top = predict_current_season(clf, scaler, feature_cols, season=current_season, top_n=10)
        print(f"\n--- Top 10 MVP candidates for {current_season} (by model) ---")
        print(top.to_string(index=False))
    except (FileNotFoundError, ValueError) as e:
        print(f"\n--- Could not fetch {current_season} data: {e}")
        print(">>> Model is saved. Run 'python3 update_data.py' later to fetch current stats and generate 2025-26 prediction.")

    return df, correlations, clf, scaler, top


def run_backtest_and_formula(
    start_year: int = 2000,
    end_year: int = 2025,
    top_k_features: int = 12,
    use_cache: bool = True,
) -> tuple:
    """
    Run backtest (leave-one-season-out), print MVP formula from logistic regression,
    and print improvement suggestions if accuracy is low.
    """
    df = get_historical_data(use_cache=use_cache, start_year=start_year, end_year=end_year)
    correlations = compute_correlations(df)
    feature_cols = get_top_features(correlations, top_k=top_k_features)
    print(f"Using top-{top_k_features} features for backtest: {feature_cols}")

    # Backtest
    report = backtest_historical(df, feature_cols)
    total = report["predicted_mvp"].notna().sum()
    correct = report["correct"].sum()
    accuracy = (correct / total * 100) if total else 0

    # Formula (interpretable model)
    get_mvp_formula(df, feature_cols, save_path=FORMULA_PATH)

    # Improvement suggestions
    suggestions = []
    if total and accuracy < 50:
        suggestions = [
            "• Try more features (top_k_features=15) or fewer (e.g. 5–6) to reduce noise.",
            "• Add team wins (merge standings): MVPs usually come from top seeds.",
            "• Use a longer training window (start_year=1990) for more MVP examples.",
            "• Try logistic regression for prediction (simpler; sometimes generalizes better).",
        ]
    elif total and accuracy < 70:
        suggestions = [
            "• Add team wins or conference rank; MVPs are often on top-2 seeds.",
            "• Try top_k_features between 6 and 12; check backtest again.",
        ]
    if suggestions:
        print("\n--- Ideas to improve accuracy ---")
        for s in suggestions:
            print(s)
    else:
        print("\n--- Backtest accuracy is solid; you can still try team wins or other features. ---")

    return report, accuracy, suggestions


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].lower() == "backtest":
        run_backtest_and_formula(
            start_year=2000,
            end_year=2025,
            top_k_features=12,
            use_cache=True,
        )
    else:
        run_full_pipeline(
        start_year=2000,
        end_year=2025,
        top_k_features=10,
        current_season="2025-26",
        use_cache=True,
    )
