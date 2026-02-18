"""
NBA MVP prediction: correlation analysis on advanced stats, then train a model
and predict this season's MVP. Uses Basketball Reference advanced stats only.
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from data.fetch_data import (
    build_advanced_historical_dataset,
    fetch_current_season_advanced,
    ADVANCED_STAT_COLUMNS,
)

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "historical_advanced.csv")
FEATURE_RANK_PATH = os.path.join(DATA_DIR, "mvp_feature_correlations.json")
BACKTEST_REPORT_PATH = os.path.join(DATA_DIR, "backtest_results.csv")
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
    df = build_advanced_historical_dataset(start_year=start_year, end_year=end_year)
    df.to_csv(CACHE_PATH, index=False)
    print(f"Saved to {CACHE_PATH}")
    return df


def compute_correlations(df: pd.DataFrame) -> pd.Series:
    """
    Compute correlation of each advanced stat with is_mvp.
    Returns series of (stat -> correlation with is_mvp), sorted by absolute value.
    """
    feature_cols = [c for c in ADVANCED_STAT_COLUMNS if c in df.columns]
    df_clean = df[feature_cols + ["is_mvp"]].dropna(how="all", subset=feature_cols)
    correlations = df_clean[feature_cols].corrwith(df_clean["is_mvp"]).abs().sort_values(ascending=False)
    return correlations


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


def get_top_features(correlations: pd.Series, top_k: int = 10) -> list:
    """Return list of top-k feature names by correlation with is_mvp."""
    return correlations.head(top_k).index.tolist()


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
    Train RandomForest to predict is_mvp. Returns fitted model, scaler, and feature list.
    """
    X, y, df_xy = prepare_xy(df, feature_cols)
    if y.sum() < 2:
        raise ValueError("Need at least 2 MVP seasons in the data to train.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=random_state, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    print("\n--- Model performance (test set) ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    # Feature importance (from tree model)
    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n--- Feature importance (from Random Forest) ---")
    print(imp.to_string())
    return clf, scaler, feature_cols


def backtest_historical(
    df: pd.DataFrame,
    feature_cols: list,
    min_seasons: int = 5,
) -> pd.DataFrame:
    """
    Leave-one-season-out backtest: for each season, train on all other seasons,
    predict that season's MVP (top-1 by probability), compare to actual winner.
    Returns DataFrame with season, actual_mvp, predicted_mvp, correct.
    """
    seasons = df["season"].dropna().unique().tolist()
    seasons = sorted([s for s in seasons if str(s).strip() != ""])
    if len(seasons) < min_seasons:
        raise ValueError(f"Need at least {min_seasons} seasons, got {len(seasons)}")

    results = []
    for test_season in seasons:
        train_df = df[df["season"] != test_season].copy()
        test_df = df[df["season"] == test_season].copy()
        X_train, y_train, train_xy = prepare_xy(train_df, feature_cols)
        X_test, y_test, test_xy = prepare_xy(test_df, feature_cols)
        if X_train.empty or X_test.empty or y_train.sum() < 1:
            results.append({
                "season": test_season,
                "actual_mvp": _get_actual_mvp(df, test_season),
                "predicted_mvp": None,
                "correct": False,
                "note": "insufficient data",
            })
            continue
        if y_test.sum() < 1:
            results.append({
                "season": test_season,
                "actual_mvp": _get_actual_mvp(df, test_season),
                "predicted_mvp": None,
                "correct": False,
                "note": "no MVP in test set",
            })
            continue
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
        )
        clf.fit(X_train_s, y_train)
        proba = clf.predict_proba(X_test_s)[:, 1]
        test_xy = test_xy.copy()
        test_xy["_mvp_prob"] = proba
        predicted_row = test_xy.loc[test_xy["_mvp_prob"].idxmax()]
        predicted_mvp = predicted_row["player"] if "player" in predicted_row else predicted_row.get("player_clean", "?")
        actual_mvp = _get_actual_mvp(df, test_season)
        correct = _names_match(actual_mvp, str(predicted_mvp))
        results.append({
            "season": test_season,
            "actual_mvp": actual_mvp,
            "predicted_mvp": predicted_mvp,
            "correct": correct,
            "note": "",
        })

    report = pd.DataFrame(results)
    report.to_csv(BACKTEST_REPORT_PATH, index=False)
    n = report["correct"].sum()
    total = len(report[report["predicted_mvp"].notna()])
    if total:
        print(f"\n--- Backtest (leave-one-season-out): {n}/{total} seasons correct ({100 * n / total:.1f}%) ---")
        print(report.to_string(index=False))
        print(f"\nSaved to {BACKTEST_REPORT_PATH}")
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
    clf,
    scaler,
    feature_cols: list,
    season: str = "2025-26",
    min_games: int = 20,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Fetch current season advanced stats, score with model, return top_n candidates.
    """
    current = fetch_current_season_advanced(season=season, min_games=min_games)
    if current.empty:
        raise FileNotFoundError(f"No data for season {season}. Check Basketball Reference.")
    missing = [c for c in feature_cols if c not in current.columns]
    if missing:
        raise ValueError(f"Current season data missing columns: {missing}")
    X = current[feature_cols].fillna(0)  # fill rare missing with 0
    X_s = scaler.transform(X)
    current["mvp_prob"] = clf.predict_proba(X_s)[:, 1]
    current["mvp_rank"] = current["mvp_prob"].rank(ascending=False).astype(int)
    top = current.nsmallest(top_n, "mvp_rank")[["player", "team_abbr", "mvp_rank", "mvp_prob"] + feature_cols]
    return top


def run_full_pipeline(
    start_year: int = 2000,
    end_year: int = 2025,
    top_k_features: int = 10,
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
    top = predict_current_season(clf, scaler, feature_cols, season=current_season, top_n=10)
    print(f"\n--- Top 10 MVP candidates for {current_season} (by model) ---")
    print(top.to_string(index=False))
    # Save model, scaler, and feature list for reuse
    joblib.dump(clf, os.path.join(MODEL_DIR, "mvp_classifier.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    print(f"\nModel and scaler saved to {MODEL_DIR}")
    return df, correlations, clf, scaler, top


def run_backtest_and_formula(
    start_year: int = 2000,
    end_year: int = 2025,
    top_k_features: int = 10,
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
            top_k_features=10,
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
