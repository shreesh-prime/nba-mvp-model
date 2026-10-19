"""
NBA MVP prediction: Ridge regression on MVP award share (from Basketball Reference voting),
using within-season percentile ranks of advanced stats + team success + year-over-year BPM change.

Variants (see MODEL_VARIANTS): "pct" and "logtop" are Ridge on vote share and differ only in how the
within-season rank is transformed; "logistic" is LogisticRegression on binary is_mvp. All are backtested
leave-one-season-out with the same metrics and saved side by side.
"""
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import joblib

from season_utils import current_season as get_current_season

from data.fetch_data import (
    build_advanced_historical_dataset,
    build_mvp_voting_dataset,
    merge_mvp_voting,
    add_team_rank,
    add_yoy_features,
    fetch_current_season_advanced,
    ADVANCED_STAT_COLUMNS,
    FULL_FEATURE_COLUMNS,
    YOY_FEATURE_COLUMNS,
)

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "historical_advanced.csv")
# Cache is parameterized by the training window so different year ranges don't get mixed up.
# (Kept compatible with older cached filenames by falling back if the parameterized cache doesn't exist.)
CACHE_PATH_TEMPLATE = os.path.join(DATA_DIR, "historical_advanced_{start_year}_{end_year}.csv")
VOTING_CACHE_TEMPLATE = os.path.join(DATA_DIR, "mvp_voting_{first_year_end}_{end_year}.csv")
FEATURE_RANK_PATH = os.path.join(DATA_DIR, "mvp_feature_correlations.json")
CORRELATION_PLOT_PATH = os.path.join(DATA_DIR, "mvp_correlation_plot.png")
BACKTEST_REPORT_PATH = os.path.join(DATA_DIR, "backtest_results.csv")
BACKTEST_DETAILS_PATH = os.path.join(DATA_DIR, "backtest_season_details.json")
BACKTEST_SUMMARY_PATH = os.path.join(DATA_DIR, "backtest_summary.json")
FORMULA_PATH = os.path.join(DATA_DIR, "mvp_formula.txt")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "mvp_share"
RIDGE_ALPHA = 1.0
# Features left as-is by apply_within_season_ranks (already binary).
BINARY_FEATURES = {"top_team_3"}
# Features where smaller raw values are better.
LOWER_IS_BETTER = {"team_win_rank"}
RANK_TRANSFORMS = ("pct", "logtop")
RANK_TRANSFORM_DESCRIPTIONS = {
    "pct": "percentile rank: rank/n, 1/n = league best (smaller is better)",
    "logtop": "-log(pct): larger is better, top of the league spread out",
}

BASE_FEATURES = [
    "top_team_3", "ppg", "bpm", "ws", "vorp", "obpm", "per", "ows",
    "ws_per_48", "usg_pct", "team_win_rank", "team_wins",
]
LOGISTIC_C = 0.8
MODEL_TYPES = ("ridge", "logistic", "ensemble")
# Within-season z(logistic P) + z(Ridge share) + z(raw VORP). Best honest LOSO top-1 of the variants.
ENSEMBLE_COMPONENTS = ("logistic", "ridge", "vorp")


class EnsembleModel:
    """Blend of logistic P(MVP), Ridge vote-share, and raw VORP, each z-scored within the scored season."""

    is_ensemble = True

    def __init__(self, logistic, ridge):
        self.logistic = logistic
        self.ridge = ridge

    def score(self, df: pd.DataFrame, scaler, feature_cols: list) -> pd.Series:
        X_s = scaler.transform(df[feature_cols].astype(float))
        lr = self.logistic.predict_proba(X_s)[:, 1]
        rg = self.ridge.predict(X_s)
        vorp = pd.to_numeric(df.get("_raw_vorp", df.get("vorp")), errors="coerce")
        vorp = vorp.fillna(vorp.median() if vorp.notna().any() else 0.0)

        def _z(a: np.ndarray) -> np.ndarray:
            a = np.asarray(a, dtype=float)
            sd = float(a.std())
            return (a - a.mean()) / (sd if sd > 1e-9 else 1.0)

        return pd.Series(_z(lr) + _z(rg) + _z(vorp.to_numpy()), index=df.index)


MODEL_VARIANTS = {
    "pct": {
        "model": "ridge",
        "target": "mvp_share",
        "rank_transform": "pct",
        "features": BASE_FEATURES + ["bpm_change"],
        "description": "Ridge on vote share; within-season percentile rank: rank/n, 1/n = league best (smaller is better).",
    },
    "logtop": {
        "model": "ridge",
        "target": "mvp_share",
        "rank_transform": "logtop",
        "features": BASE_FEATURES + ["bpm_change"],
        "description": "Ridge on vote share; -log(pct): larger is better, and the top of the league is spread out "
                       "(#1 vs #5 differ far more than #100 vs #105).",
    },
    "logistic": {
        "model": "logistic",
        "target": "is_mvp",
        "rank_transform": "pct",
        "features": BASE_FEATURES,
        "description": f"LogisticRegression(class_weight='balanced', C={LOGISTIC_C}) on binary is_mvp; "
                       "within-season pct ranks; score = P(MVP).",
    },
    "ensemble": {
        "model": "ensemble",
        "target": "mvp_share",
        "rank_transform": "pct",
        "features": BASE_FEATURES,
        "description": "Within-season z-score blend of logistic P(MVP), Ridge vote-share, and raw VORP.",
    },
}
DEFAULT_VARIANT = "pct"
VARIANT_CHOICES = tuple(MODEL_VARIANTS) + ("all",)

# Historical reference only: leave-one-season-out result of the pre-fix shipped model (model/ridge_model.joblib:
# the "logistic" configuration, but trained with team_win_rank computed over player rows). Its top-1 picks
# match the shipped data/backtest_results.csv. The "logistic" variant is the reproducible, fixed version.
BASELINE_METRICS = {
    "description": "Historical reference (not recomputed): pre-fix shipped model, LogisticRegression("
                   "class_weight='balanced', C=0.8) on is_mvp, 12 features, team_win_rank ranked over player rows",
    "n_seasons": 30,
    "top1_correct": 20,
    "top1_accuracy": round(20 / 30, 4),
    "top3_correct": 26,
    "top3_accuracy": round(26 / 30, 4),
    "mrr": 0.783,
    "mean_actual_rank": 1.97,
    "mean_spearman_vote_share": 0.665,
}


def get_historical_data(use_cache: bool = True, start_year: int = 2000, end_year: int = 2025) -> pd.DataFrame:
    """Load or build historical player-season data with advanced stats and is_mvp."""
    cache_path = CACHE_PATH_TEMPLATE.format(start_year=start_year, end_year=end_year)
    legacy_cache_path = CACHE_PATH
    if use_cache and os.path.isfile(CACHE_PATH):
        df = pd.read_csv(legacy_cache_path)
        print(f"Loaded cached data: {len(df)} rows from {legacy_cache_path}")
        return add_team_rank(df)
    if use_cache and os.path.isfile(cache_path):
        df = pd.read_csv(cache_path)
        print(f"Loaded cached data: {len(df)} rows from {cache_path}")
        return add_team_rank(df)
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
            return add_team_rank(pd.read_csv(CACHE_PATH))
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
    df.to_csv(cache_path, index=False)
    print(f"Saved to {cache_path}")
    return df


def get_mvp_voting(use_cache: bool = True, start_year: int = 1995, end_year: int = 2025) -> pd.DataFrame:
    """MVP voting (season, player, team, share, rank, player_id), cached to data/mvp_voting_*.csv."""
    cache_path = VOTING_CACHE_TEMPLATE.format(first_year_end=start_year + 1, end_year=end_year)
    return build_mvp_voting_dataset(start_year=start_year, end_year=end_year, cache_path=cache_path, use_cache=use_cache)


def build_model_frame(df: pd.DataFrame, voting: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw player-seasons -> modeling frame: fixed team rank, YoY features, and mvp_share/mvp_vote_rank."""
    out = add_yoy_features(add_team_rank(df))
    if voting is not None and not voting.empty:
        out = merge_mvp_voting(out, voting)
    for src, dest in (("vorp", "_raw_vorp"), ("ws", "_raw_ws"), ("ppg", "_raw_ppg"), ("rpg", "_raw_rpg"), ("apg", "_raw_apg")):
        if src in out.columns:
            out[dest] = pd.to_numeric(out[src], errors="coerce")
    return out


def apply_within_season_ranks(df: pd.DataFrame, feature_cols: list[str], transform: str = "pct") -> pd.DataFrame:
    """
    Replace each numeric feature with its within-season rank.

    transform="pct":    percentile of the descending rank, in (0, 1]; 1/n = league best that season,
                        1.0 = worst (the original orientation, which the shipped model expects).
    transform="logtop": -log(pct) — larger = better, and the gap between #1 and #5 is much larger
                        than between #100 and #105 (percentiles compress the top of the league).

    Rationale: MVP voters compare players against their peers each season, so
    rank-normalizing features reduces scale-era effects. Binary features are left unchanged.
    """
    if "season" not in df.columns:
        raise ValueError("DataFrame must include a 'season' column for within-season ranking.")
    if transform not in RANK_TRANSFORMS:
        raise ValueError(f"Unknown rank transform {transform!r}; expected one of {RANK_TRANSFORMS}")

    out = df.copy()
    for col in feature_cols:
        if col not in out.columns or col in BINARY_FEATURES:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        if col in LOWER_IS_BETTER:
            values = -values
        ranked = values.groupby(out["season"]).rank(pct=True, ascending=False)
        if transform == "logtop":
            ranked = -np.log(ranked)
        out[col] = ranked
    return out


def compute_correlations(df: pd.DataFrame, target: str = TARGET_COL) -> pd.Series:
    """|Correlation| of each candidate feature (FULL_FEATURE_COLUMNS + YoY) with the target column."""
    candidate_cols = [c for c in FULL_FEATURE_COLUMNS + YOY_FEATURE_COLUMNS if c in df.columns]
    if not candidate_cols:
        candidate_cols = [c for c in ADVANCED_STAT_COLUMNS if c in df.columns]
    df_clean = df[candidate_cols + [target]].dropna(how="all", subset=candidate_cols)
    correlations = df_clean[candidate_cols].corrwith(df_clean[target]).abs().sort_values(ascending=False)
    return correlations


def analyze_and_save_correlations(
    df: pd.DataFrame,
    plot: bool = True,
    use_within_season_ranks: bool = True,
    rank_transform: str = "pct",
    target: str = TARGET_COL,
    json_path: str | None = None,
    plot_path: str | None = None,
) -> pd.Series:
    """Compute correlations with the target, save to JSON, print top stats, and optionally plot."""
    json_path = json_path or FEATURE_RANK_PATH
    plot_path = plot_path or CORRELATION_PLOT_PATH
    data = df
    if use_within_season_ranks:
        data = apply_within_season_ranks(df, FULL_FEATURE_COLUMNS + YOY_FEATURE_COLUMNS, transform=rank_transform)
    correlations = compute_correlations(data, target=target)
    with open(json_path, "w") as f:
        json.dump(correlations.round(4).to_dict(), f, indent=2)
    target_label = "MVP vote share" if target == TARGET_COL else target
    print(f"\n--- Correlation with {target_label} (higher = more predictive) ---")
    print(correlations.head(15).to_string())
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            correlations.head(15).plot(kind="barh", ax=ax, color="steelblue", edgecolor="navy", alpha=0.8)
            ax.set_xlabel(f"|Correlation with {target_label}| (within-season ranks)")
            ax.set_title(f"NBA Stats: Correlation with {target_label}")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(plot_path, dpi=120)
            plt.close()
            print(f"Saved correlation plot to {plot_path}")
        except Exception as e:
            print(f"Could not save plot: {e}")
    return correlations


def prepare_xy(df: pd.DataFrame, feature_cols: list, target: str = TARGET_COL):
    """Prepare X (numeric features) and y (target). Drop rows with missing feature values."""
    X = df[feature_cols].copy()
    y = df[target].copy()
    mask = X.notna().all(axis=1)
    return X[mask].astype(float), y[mask].astype(float), df[mask]


def fit_model(
    df: pd.DataFrame,
    feature_cols: list,
    model_type: str = "ridge",
    target: str | None = None,
    alpha: float = RIDGE_ALPHA,
):
    """
    Fit StandardScaler + model on (already rank-transformed) features. Returns (model, scaler).
    model_type "ridge": Ridge(alpha) regressing target (default mvp_share).
    model_type "logistic": LogisticRegression(class_weight='balanced', C=LOGISTIC_C) on target (default is_mvp).
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type {model_type!r}; expected one of {MODEL_TYPES}")
    if model_type == "ensemble":
        X, _, kept = prepare_xy(df, feature_cols, TARGET_COL)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        ridge = Ridge(alpha=alpha).fit(Xs, kept[TARGET_COL].astype(float))
        logistic = LogisticRegression(
            max_iter=2000, random_state=42, class_weight="balanced", C=LOGISTIC_C, solver="lbfgs"
        ).fit(Xs, kept["is_mvp"].astype(int))
        return EnsembleModel(logistic, ridge), scaler
    target = target or (TARGET_COL if model_type == "ridge" else "is_mvp")
    X, y, _ = prepare_xy(df, feature_cols, target)
    scaler = StandardScaler()
    if model_type == "ridge":
        model = Ridge(alpha=alpha)
    else:
        model = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced", C=LOGISTIC_C, solver="lbfgs")
        y = y.astype(int)
    model.fit(scaler.fit_transform(X), y)
    return model, scaler


def fit_ridge(df: pd.DataFrame, feature_cols: list, target: str = TARGET_COL, alpha: float = RIDGE_ALPHA):
    """Fit StandardScaler + Ridge on (already rank-transformed) features. Returns (model, scaler)."""
    return fit_model(df, feature_cols, "ridge", target, alpha)


def model_coefficients(model) -> tuple[float, np.ndarray]:
    """(intercept, coefficient vector) for a fitted Ridge or LogisticRegression."""
    if getattr(model, "is_ensemble", False):
        return 0.0, np.zeros(1)
    return float(np.ravel(model.intercept_)[0]), np.ravel(model.coef_)


def score_players(df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.Series:
    """Predicted MVP score (≈ award share) for rows of an already rank-transformed frame."""
    if getattr(model, "is_ensemble", False):
        return model.score(df, scaler, feature_cols)
    X_s = scaler.transform(df[feature_cols].astype(float))
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(X_s)[:, 1], index=df.index)
    return pd.Series(model.predict(X_s), index=df.index)


def _none_if_nan(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def backtest_historical(
    df: pd.DataFrame,
    feature_cols: list,
    min_seasons: int = 5,
    top_n: int = 10,
    target: str = TARGET_COL,
    alpha: float = RIDGE_ALPHA,
    report_path: str | None = None,
    details_path: str | None = None,
    model_type: str = "ridge",
) -> pd.DataFrame:
    """
    Leave-one-season-out backtest: for each season, fit the model (see fit_model; `target` is what it
    is trained on) on all other seasons and rank that season's players by predicted score.
    df must already be rank-transformed. Vote-share metrics use mvp_share when present.

    Per-season columns: season, actual_mvp, predicted_mvp, correct, note, actual_rank_in_model,
    in_top3, reciprocal_rank, actual_share, predicted_score, spearman_vote_share, n_vote_getters.
    actual_rank_in_model is NaN (reciprocal_rank 0) when the winner lacks complete features.
    Saves the CSV and a per-season JSON with the top_n players.
    """
    report_path = report_path or BACKTEST_REPORT_PATH
    details_path = details_path or BACKTEST_DETAILS_PATH
    seasons = sorted(s for s in df["season"].dropna().unique().tolist() if str(s).strip() != "")
    if len(seasons) < min_seasons:
        raise ValueError(f"Need at least {min_seasons} seasons, got {len(seasons)}")
    has_votes = "mvp_vote_rank" in df.columns
    share_col = TARGET_COL if TARGET_COL in df.columns else target

    results = []
    details_by_season = {}
    for test_season in seasons:
        actual_mvp = _get_actual_mvp(df, test_season)
        train_df = df[df["season"] != test_season]
        _, _, test_xy = prepare_xy(df[df["season"] == test_season], feature_cols, target)
        if test_xy.empty or train_df[target].sum() <= 0:
            results.append({"season": test_season, "actual_mvp": actual_mvp, "predicted_mvp": None,
                             "correct": False, "note": "insufficient data"})
            details_by_season[str(test_season)] = {"actual_mvp": actual_mvp, "predicted_mvp": None,
                                                   "correct": False, "top10": []}
            continue

        model, scaler = fit_model(train_df, feature_cols, model_type, target, alpha)
        test_xy = test_xy.copy()
        test_xy["_mvp_score"] = score_players(test_xy, model, scaler, feature_cols)
        candidates = test_xy.sort_values("_mvp_score", ascending=False).reset_index(drop=True)
        candidates["_model_rank"] = np.arange(1, len(candidates) + 1)
        predicted_mvp = str(candidates.iloc[0].get("player", candidates.iloc[0].get("player_clean", "?")))
        correct = _names_match(actual_mvp, predicted_mvp)

        winner = candidates[candidates["is_mvp"] == 1]
        actual_rank = float(winner["_model_rank"].iloc[0]) if not winner.empty else float("nan")
        actual_share = float(winner[share_col].iloc[0]) if not winner.empty else float("nan")
        voters = candidates[candidates[share_col] > 0]
        rho = float("nan")
        if len(voters) >= 3 and voters[share_col].nunique() > 1:
            rho = float(spearmanr(voters["_mvp_score"], voters[share_col]).correlation)

        note = "" if not winner.empty else "actual MVP missing features"
        results.append({
            "season": test_season,
            "actual_mvp": actual_mvp,
            "predicted_mvp": predicted_mvp,
            "correct": correct,
            "note": note,
            "actual_rank_in_model": actual_rank,
            "in_top3": bool(actual_rank <= 3) if not np.isnan(actual_rank) else False,
            "reciprocal_rank": 1.0 / actual_rank if not np.isnan(actual_rank) else 0.0,
            "actual_share": actual_share,
            "predicted_score": float(candidates.iloc[0]["_mvp_score"]),
            "spearman_vote_share": rho,
            "n_vote_getters": int(len(voters)),
        })

        top10_list = []
        for _, row in candidates.head(top_n).iterrows():
            entry = {
                "rank": int(row["_model_rank"]),
                "player": str(row.get("player", row.get("player_clean", "?"))),
                "team": str(row.get("team_abbr", "")),
                "score": float(row["_mvp_score"]),
                "actual_share": float(row[share_col]),
            }
            if has_votes:
                entry["actual_vote_rank"] = _none_if_nan(float(row["mvp_vote_rank"]))
            top10_list.append(entry)
        details_by_season[str(test_season)] = {
            "actual_mvp": actual_mvp,
            "predicted_mvp": predicted_mvp,
            "correct": correct,
            "actual_rank_in_model": _none_if_nan(actual_rank),
            "spearman_vote_share": _none_if_nan(rho),
            "top10": top10_list,
        }

    report = pd.DataFrame(results)
    report.to_csv(report_path, index=False)
    with open(details_path, "w") as f:
        json.dump(details_by_season, f, indent=2)
    print(f"Saved {report_path} and {details_path}")
    return report


def summarize_backtest(report: pd.DataFrame) -> dict:
    """Aggregate a backtest_historical report: top-1/top-3 accuracy, MRR, winner rank, mean Spearman."""
    scored = report[report["predicted_mvp"].notna()]
    n = len(scored)
    if n == 0:
        return {"n_seasons": 0}
    ranks = scored["actual_rank_in_model"]
    return {
        "n_seasons": int(n),
        "top1_correct": int(scored["correct"].astype(bool).sum()),
        "top1_accuracy": round(float(scored["correct"].astype(bool).mean()), 4),
        "top3_correct": int(scored["in_top3"].astype(bool).sum()),
        "top3_accuracy": round(float(scored["in_top3"].astype(bool).mean()), 4),
        "mrr": round(float(scored["reciprocal_rank"].mean()), 4),
        "mean_actual_rank": round(float(ranks.mean()), 3),
        "median_actual_rank": float(ranks.median()),
        "worst_actual_rank": float(ranks.max()),
        "mean_spearman_vote_share": round(float(scored["spearman_vote_share"].mean()), 4),
        "misses": scored.loc[~scored["correct"].astype(bool), "season"].tolist(),
    }


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
    alpha: float = RIDGE_ALPHA,
    rank_transform: str = "pct",
    target: str | None = None,
    model_type: str = "ridge",
) -> tuple:
    """
    Fit the model on all seasons and write a human-readable formula, z_i = standardized within-season rank:
      ridge:    predicted_share = intercept + sum(coef_i * z_i)
      logistic: P(MVP) = 1 / (1 + exp(-(intercept + sum(coef_i * z_i))))
    Returns (fitted model, scaler, formula_string).
    """
    model, scaler = fit_model(df, feature_cols, model_type, target, alpha)
    if model_type == "ensemble":
        formula = (
            "MVP formula (ensemble):\n"
            "  score = z(P_logistic) + z(share_ridge) + z(raw VORP)\n"
            "Each term is standardized within the season being scored (mean 0, std 1),\n"
            "then added. Logistic and Ridge use the same within-season percentile-ranked\n"
            "feature set as the logistic / pct variants; VORP is the raw Basketball Reference value.\n"
            "Highest ensemble score in a season = predicted MVP.\n"
        )
        with open(save_path or FORMULA_PATH, "w") as f:
            f.write(formula)
        print("\n--- MVP formula (interpretable model) ---")
        print(formula)
        print(f"Saved to {save_path or FORMULA_PATH}")
        return model, scaler, formula
    intercept, coef_vec = model_coefficients(model)
    coefs = pd.Series(coef_vec, index=feature_cols)
    z_expr = " ".join([f"{intercept:.4f}"] + [f"{'+' if c >= 0 else ''}{c:.4f}*{n}" for n, c in coefs.items()])
    by_size = "\n".join(f"    {n:<14} {c:+.4f}" for n, c in coefs.reindex(coefs.abs().sort_values(ascending=False).index).items())
    rank_desc = RANK_TRANSFORM_DESCRIPTIONS.get(rank_transform, rank_transform)
    direction = (
        "For ranked stats a NEGATIVE coefficient helps (smaller pct = better rank); for top_team_3 positive helps."
        if rank_transform == "pct" else "Positive coefficient = helps."
    )
    if model_type == "logistic":
        header = (f"MVP formula (LogisticRegression on is_mvp, class_weight='balanced', C={LOGISTIC_C}, "
                  f"rank transform '{rank_transform}'):\n"
                  "  P(MVP) = 1 / (1 + exp(-z))\n  z = " + z_expr + "\n\n")
        units = "Coefficients are in log-odds per standard deviation."
        pick = "Highest P(MVP) in a season = predicted MVP."
    else:
        header = (f"MVP formula (Ridge regression on MVP award share, alpha={alpha}, rank transform '{rank_transform}'):\n"
                  "  predicted_share = " + z_expr + "\n\n")
        units = "Coefficients are in award-share units per standard deviation."
        pick = "Highest predicted_share in a season = predicted MVP."
    formula = (
        header
        + f"Each stat is first converted to a within-season rank ({rank_desc}),\n"
        "then standardized (mean=0, std=1) across all player-seasons. top_team_3 is a 0/1 flag.\n"
        f"{units} {direction}\n"
        f"{pick}\n\n"
        "Coefficients by magnitude:\n" + by_size + "\n"
    )
    with open(save_path or FORMULA_PATH, "w") as f:
        f.write(formula)
    print("\n--- MVP formula (interpretable model) ---")
    print(formula)
    print(f"Saved to {save_path or FORMULA_PATH}")
    return model, scaler, formula


def resolve_variants(variant: str) -> list[str]:
    """'pct' -> ['pct'], 'all' -> every MODEL_VARIANTS key."""
    if variant == "all":
        return list(MODEL_VARIANTS)
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANT_CHOICES}")
    return [variant]


def variant_artifact_paths(variant: str) -> dict:
    """Namespaced output locations for a variant (coexist with the legacy un-suffixed files)."""
    model_dir = os.path.join(MODEL_DIR, variant)
    return {
        "model_dir": model_dir,
        "model": os.path.join(model_dir, "ridge_model.joblib"),
        "scaler": os.path.join(model_dir, "scaler.joblib"),
        "feature_cols": os.path.join(model_dir, "feature_cols.json"),
        "metadata": os.path.join(model_dir, "model_metadata.json"),
        "backtest_results": os.path.join(DATA_DIR, f"backtest_results_{variant}.csv"),
        "backtest_details": os.path.join(DATA_DIR, f"backtest_season_details_{variant}.json"),
        "formula": os.path.join(DATA_DIR, f"mvp_formula_{variant}.txt"),
        "correlations": os.path.join(DATA_DIR, f"mvp_feature_correlations_{variant}.json"),
        "correlation_plot": os.path.join(DATA_DIR, f"mvp_correlation_plot_{variant}.png"),
    }


def load_variant_artifacts(variant: str | None = None) -> tuple:
    """
    Load (model, scaler, feature_cols, info) from model/<variant>/, or from the legacy un-suffixed
    model/ files when variant is None. info has keys variant, rank_transform, use_within_season_ranks.
    """
    model_dir = MODEL_DIR if variant is None else variant_artifact_paths(variant)["model_dir"]
    meta = {}
    meta_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    model = joblib.load(os.path.join(model_dir, "ridge_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    with open(os.path.join(model_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    info = {
        "variant": variant,
        "rank_transform": meta.get("rank_transform", "pct"),
        "use_within_season_ranks": meta.get("use_within_season_ranks", False),
    }
    return model, scaler, feature_cols, info


def save_variant_artifacts(variant: str, model, scaler, feature_cols: list, alpha: float = RIDGE_ALPHA) -> dict:
    """Save a variant's model, scaler, features and metadata to model/<variant>/. Returns the metadata."""
    cfg = MODEL_VARIANTS[variant]
    paths = variant_artifact_paths(variant)
    os.makedirs(paths["model_dir"], exist_ok=True)
    joblib.dump(model, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    with open(paths["feature_cols"], "w") as f:
        json.dump(feature_cols, f)
    meta = {
        "variant": variant,
        "use_within_season_ranks": True,
        "rank_transform": cfg["rank_transform"],
        "model_type": type(model).__name__,
        "target": cfg["target"],
        "feature_cols": feature_cols,
    }
    if cfg["model"] == "ridge":
        meta.update({"alpha": alpha})
    elif cfg["model"] == "logistic":
        meta.update({"C": LOGISTIC_C, "class_weight": "balanced"})
    else:
        meta.update({"blend": "z(logistic)+z(ridge)+z(raw_vorp)"})
    with open(paths["metadata"], "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def predict_current_season(
    model,
    scaler,
    feature_cols: list,
    season: str | None = None,
    min_games: int = 20,
    top_n: int = 10,
    use_within_season_ranks: bool = True,
    rank_transform: str = "pct",
) -> pd.DataFrame:
    """
    Fetch current season stats and score candidates by predicted MVP vote share.
    """
    season = season or get_current_season()
    current = fetch_current_season_advanced(season=season, min_games=min_games)
    if current.empty:
        raise FileNotFoundError(f"No data for season {season}. Check Basketball Reference.")
    missing = [c for c in feature_cols if c not in current.columns]
    if missing:
        raise ValueError(f"Current season data missing columns: {missing}")
    if "vorp" in current.columns:
        current["_raw_vorp"] = pd.to_numeric(current["vorp"], errors="coerce")
    if use_within_season_ranks:
        current = apply_within_season_ranks(current, feature_cols, transform=rank_transform)
    mask = current[feature_cols].notna().all(axis=1)
    current = current[mask].copy()
    if current.empty:
        raise ValueError("No players had complete feature values for prediction.")
    current["mvp_score"] = score_players(current, model, scaler, feature_cols)
    current["mvp_rank"] = current["mvp_score"].rank(ascending=False, method="first").astype(int)
    top = current.nsmallest(top_n, "mvp_rank")[["player", "team_abbr", "mvp_rank", "mvp_score"] + [c for c in feature_cols if c in current.columns]]
    return top


def _load_model_frame(start_year: int, end_year: int, use_cache: bool) -> pd.DataFrame:
    df = get_historical_data(use_cache=use_cache, start_year=start_year, end_year=end_year)
    voting = get_mvp_voting(use_cache=use_cache, start_year=start_year, end_year=end_year)
    return build_model_frame(df, voting)


def _print_comparison(summary: dict) -> None:
    rows = [("pre-fix shipped", summary["historical_reference_preshipped"])]
    rows += [(v, s["metrics"]) for v, s in summary["variants"].items()]
    print("\n--- Backtest comparison (leave-one-season-out) ---")
    print(f"{'model':<16}{'top-1':>8}{'top-3':>8}{'MRR':>8}{'mean rank':>11}{'spearman':>10}")
    for name, m in rows:
        print(f"{name:<16}{m['top1_correct']:>5}/{m['n_seasons']:<2}{m['top3_correct']:>5}/{m['n_seasons']:<2}"
              f"{m['mrr']:>8.3f}{m['mean_actual_rank']:>11.2f}{m['mean_spearman_vote_share']:>10.3f}")


def _backtest_variant(df_model: pd.DataFrame, variant: str) -> dict:
    """Backtest one variant and write its namespaced results, details, formula and correlations."""
    cfg = MODEL_VARIANTS[variant]
    paths = variant_artifact_paths(variant)
    print(f"\n=== Variant '{variant}' ({cfg['description']}) ===")
    analyze_and_save_correlations(df_model, rank_transform=cfg["rank_transform"], target=cfg["target"],
                                  json_path=paths["correlations"], plot_path=paths["correlation_plot"])
    ranked = apply_within_season_ranks(df_model, cfg["features"], transform=cfg["rank_transform"])
    report = backtest_historical(ranked, cfg["features"], target=cfg["target"], model_type=cfg["model"],
                                 report_path=paths["backtest_results"], details_path=paths["backtest_details"])
    print(report[["season", "actual_mvp", "predicted_mvp", "correct", "actual_rank_in_model"]].to_string(index=False))
    model, _, _ = get_mvp_formula(ranked, cfg["features"], save_path=paths["formula"], target=cfg["target"],
                                  rank_transform=cfg["rank_transform"], model_type=cfg["model"])
    intercept, coefs = model_coefficients(model)
    if cfg["model"] == "ensemble":
        hyper = {"blend": "z(logistic)+z(ridge)+z(raw_vorp)"}
        coef_map = {}
        intercept_out = 0.0
    elif cfg["model"] == "ridge":
        hyper = {"alpha": RIDGE_ALPHA}
        coef_map = {f: round(float(c), 5) for f, c in zip(cfg["features"], coefs)}
        intercept_out = round(intercept, 5)
    else:
        hyper = {"C": LOGISTIC_C, "class_weight": "balanced"}
        coef_map = {f: round(float(c), 5) for f, c in zip(cfg["features"], coefs)}
        intercept_out = round(intercept, 5)
    return {
        "model_type": type(model).__name__,
        "target": cfg["target"],
        "hyperparameters": hyper,
        "rank_transform": cfg["rank_transform"],
        "description": cfg["description"],
        "features": cfg["features"],
        "intercept": intercept_out,
        "coefficients": coef_map,
        "metrics": summarize_backtest(report),
        "backtest_results": os.path.relpath(paths["backtest_results"], os.path.dirname(__file__)),
        "backtest_details": os.path.relpath(paths["backtest_details"], os.path.dirname(__file__)),
        "formula": os.path.relpath(paths["formula"], os.path.dirname(__file__)),
        "model_dir": os.path.relpath(paths["model_dir"], os.path.dirname(__file__)),
    }


def run_backtest_and_formula(
    start_year: int = 1995,
    end_year: int = 2025,
    use_cache: bool = True,
    variant: str = "all",
    df_model: pd.DataFrame | None = None,
) -> dict:
    """
    Leave-one-season-out backtest for `variant` (a MODEL_VARIANTS key or "all"). Writes each variant's
    namespaced outputs and merges its entry into data/backtest_summary.json (other variants' entries
    are kept). Returns the full summary dict.
    """
    if df_model is None:
        df_model = _load_model_frame(start_year, end_year, use_cache)
    summary = {}
    if os.path.isfile(BACKTEST_SUMMARY_PATH):
        with open(BACKTEST_SUMMARY_PATH) as f:
            summary = json.load(f)
    for stale in ("target", "model_type", "alpha", "baseline"):
        summary.pop(stale, None)
    summary.update({
        "seasons": sorted(df_model["season"].unique().tolist()),
        "historical_reference_preshipped": BASELINE_METRICS,
    })
    summary.setdefault("variants", {})
    for v in resolve_variants(variant):
        summary["variants"][v] = _backtest_variant(df_model, v)
    summary["variants"] = {v: summary["variants"][v] for v in MODEL_VARIANTS if v in summary["variants"]}
    with open(BACKTEST_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    _print_comparison(summary)
    print(f"\nSaved summary to {BACKTEST_SUMMARY_PATH}")
    return summary


def run_full_pipeline(
    start_year: int = 1995,
    end_year: int = 2025,
    current_season: str | None = None,
    use_cache: bool = True,
    run_backtest: bool = True,
    variant: str = DEFAULT_VARIANT,
):
    """
    For `variant` (a MODEL_VARIANTS key or "all"):
    1) Load historical data + MVP voting; build features (fixed team rank, YoY BPM change)
    2) Optionally backtest and write namespaced results/formula/correlations + backtest_summary.json
    3) Train the variant's model on all seasons and save model/<variant>/ artifacts
    4) Predict the current season and print the top 10
    Returns {variant: (model, scaler, top10 DataFrame or None)} plus the modeling frame under "_df".
    """
    current_season = current_season or get_current_season()
    df_model = _load_model_frame(start_year, end_year, use_cache)
    if run_backtest:
        run_backtest_and_formula(variant=variant, df_model=df_model)

    results = {"_df": df_model}
    for v in resolve_variants(variant):
        cfg = MODEL_VARIANTS[v]
        ranked = apply_within_season_ranks(df_model, cfg["features"], transform=cfg["rank_transform"])
        model, scaler = fit_model(ranked, cfg["features"], cfg["model"], cfg["target"])
        save_variant_artifacts(v, model, scaler, cfg["features"])
        print(f"\n{type(model).__name__} for variant '{v}' saved to {variant_artifact_paths(v)['model_dir']}")
        top = None
        try:
            top = predict_current_season(model, scaler, cfg["features"], season=current_season,
                                         top_n=10, rank_transform=cfg["rank_transform"])
            print(f"\n--- Top 10 MVP candidates for {current_season} (variant '{v}') ---")
            print(top.to_string(index=False))
        except (FileNotFoundError, ValueError) as e:
            print(f"\n--- Could not fetch {current_season} data: {e}")
            print(f">>> Model is saved. Run 'python3 update_data.py' later to fetch current stats and generate the {current_season} prediction.")
        results[v] = (model, scaler, top)
    return results


def _parse_cli(argv: list[str]):
    import argparse
    parser = argparse.ArgumentParser(description="Train/backtest the NBA MVP vote-share model.")
    parser.add_argument("command", nargs="?", default="train", choices=["train", "backtest"])
    parser.add_argument("--variant", default="all", choices=VARIANT_CHOICES)
    parser.add_argument("--no-backtest", action="store_true", help="train: skip the backtest step")
    return parser.parse_args(argv)


if __name__ == "__main__":
    import sys
    args = _parse_cli(sys.argv[1:])
    if args.command == "backtest":
        run_backtest_and_formula(
            start_year=1995,
            end_year=2025,
            use_cache=True,
            variant=args.variant,
        )
    else:
        run_full_pipeline(
        start_year=1995,
        end_year=2025,
        use_cache=True,
        run_backtest=not args.no_backtest,
        variant=args.variant,
    )
