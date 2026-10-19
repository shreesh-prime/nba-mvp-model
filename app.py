"""
NBA MVP predictor: prediction from key correlating factors (Ridge regression).
Dropdown to test each year and see what the predictor came up with.
Run update_data.py to refresh current-season prediction.
"""
import os
import json
import csv
import threading
import time
from datetime import datetime

from flask import Flask, render_template_string, request, jsonify
from mvp_model import _names_match, apply_within_season_ranks
from season_utils import current_season as get_current_season, season_from_start_year, season_start_year

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "model")
BACKTEST_PATH = os.path.join(DATA_DIR, "backtest_results.csv")
BACKTEST_DETAILS_PATH = os.path.join(DATA_DIR, "backtest_season_details.json")
CURRENT_PREDICTION_PATH = os.path.join(DATA_DIR, "current_season_prediction.json")
FEATURE_CORR_PATH = os.path.join(DATA_DIR, "mvp_feature_correlations.json")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.json")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
REPO_URL = "https://github.com/shreesh-prime/nba-mvp-model"
BACKTEST_SUMMARY_PATH = os.path.join(DATA_DIR, "backtest_summary.json")

# "shipped" is the un-suffixed model in model/ that drives live predictions; the rest are mvp_model.py variants.
MODEL_CHOICES = {
    "shipped": "Shipped model (logistic, pre team-rank fix)",
    "logistic": "Logistic regression",
    "pct": "Vote-share Ridge (percentile ranks)",
    "logtop": "Vote-share Ridge (log-top ranks)",
    "ensemble": "Ensemble (logistic + Ridge + VORP)",
}


def get_season_list():
    years = range(1995, season_start_year(get_current_season()) + 1)
    return [season_from_start_year(y) for y in years]


def read_backtest():
    if not os.path.isfile(BACKTEST_PATH):
        return []
    rows = []
    with open(BACKTEST_PATH) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def get_backtest_for_season(season):
    rows = read_backtest()
    for r in rows:
        if str(r.get("season", "")).strip() == str(season).strip():
            return r
    return None


def read_backtest_details(model="shipped"):
    path = BACKTEST_DETAILS_PATH
    if model != "shipped":
        path = os.path.join(DATA_DIR, f"backtest_season_details_{model}.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def read_model_comparison():
    """Backtest metrics for every model in MODEL_CHOICES that has them."""
    if not os.path.isfile(BACKTEST_SUMMARY_PATH):
        return []
    try:
        with open(BACKTEST_SUMMARY_PATH) as f:
            summary = json.load(f)
    except Exception:
        return []
    metrics_by_model = {k: v.get("metrics", {}) for k, v in summary.get("variants", {}).items()}
    metrics_by_model["shipped"] = summary.get("historical_reference_preshipped", {})
    rows = []
    for key, label in MODEL_CHOICES.items():
        m = metrics_by_model.get(key)
        if not m:
            continue
        n = m.get("n_seasons", 0)
        rows.append({
            "key": key,
            "label": label,
            "top1": f"{m.get('top1_correct', 0)}/{n}",
            "top3": f"{m.get('top3_correct', 0)}/{n}",
            "mrr": m.get("mrr"),
            "mean_rank": m.get("mean_actual_rank"),
            "spearman": m.get("mean_spearman_vote_share"),
        })
    return rows


def get_season_picks(season):
    """Each model's backtest pick for one past season, plus where it ranked the actual MVP."""
    picks = []
    for key, label in MODEL_CHOICES.items():
        d = read_backtest_details(key).get(season)
        if not d:
            continue
        rank = d.get("actual_rank_in_model")
        if rank is None:
            rank = next((p.get("rank") for p in d.get("top10", []) if _names_match(p.get("player"), d.get("actual_mvp"))), None)
        picks.append({
            "label": label,
            "predicted_mvp": d.get("predicted_mvp"),
            "correct": bool(d.get("correct")),
            "actual_rank": int(rank) if rank is not None else None,
        })
    return picks


def get_key_factors():
    if os.path.isfile(FEATURE_COLS_PATH):
        with open(FEATURE_COLS_PATH) as f:
            return json.load(f)
    if os.path.isfile(FEATURE_CORR_PATH):
        with open(FEATURE_CORR_PATH) as f:
            data = json.load(f)
        return list(data.keys())[:10] if isinstance(data, dict) else []
    return []


def read_current_season_prediction():
    if not os.path.isfile(CURRENT_PREDICTION_PATH):
        return None
    try:
        with open(CURRENT_PREDICTION_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def is_correct_val(r):
    v = r.get("correct")
    return v is True or str(v).lower() == "true"


def compute_current_season_prediction():
    """Generate current-season prediction on the fly. Returns dict or dict with _error key."""
    season = get_current_season()
    model_path = os.path.join(MODEL_DIR, "ridge_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    features_path = os.path.join(MODEL_DIR, "feature_cols.json")
    if not os.path.isfile(model_path) or not os.path.isfile(scaler_path) or not os.path.isfile(features_path):
        return {"_error": "Model not trained. From the project folder run: python3 mvp_model.py"}
    try:
        import joblib
        from data.fetch_data import fetch_current_season_advanced
        with open(features_path) as f:
            feature_cols = json.load(f)
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        use_within_season_ranks = False
        if os.path.isfile(MODEL_METADATA_PATH):
            try:
                with open(MODEL_METADATA_PATH) as f:
                    use_within_season_ranks = json.load(f).get("use_within_season_ranks", False)
            except Exception:
                use_within_season_ranks = False
        current = fetch_current_season_advanced(season=season, min_games=20)
        if current.empty:
            return {"_error": f"Could not fetch {season} stats from Basketball Reference. Try again later."}
        missing = [c for c in feature_cols if c not in current.columns]
        if missing:
            return {"_error": f"Data missing columns: {missing}. Re-run python3 mvp_model.py to retrain."}
        if use_within_season_ranks:
            current = apply_within_season_ranks(current, feature_cols)
        mask = current[feature_cols].notna().all(axis=1)
        current = current[mask].copy()
        if current.empty:
            return {"_error": "No players had complete feature values for prediction."}
        X = current[feature_cols].astype(float)
        X_s = scaler.transform(X)
        if hasattr(model, "predict_proba"):
            current["mvp_score"] = model.predict_proba(X_s)[:, 1]
        else:
            current["mvp_score"] = model.predict(X_s)
        current["mvp_rank"] = current["mvp_score"].rank(ascending=False).astype(int)
        top10_df = current.nsmallest(10, "mvp_rank")
        top10 = [{"rank": int(row["mvp_rank"]), "player": str(row.get("player", "")), "team": str(row.get("team_abbr", "")), "score": float(row["mvp_score"])} for _, row in top10_df.iterrows()]
        top1 = top10[0] if top10 else {}
        os.makedirs(DATA_DIR, exist_ok=True)
        result = {"last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "season": season, "top1_name": top1.get("player", "—"), "top1_team": top1.get("team", ""), "top10": top10}
        with open(CURRENT_PREDICTION_PATH, "w") as f:
            json.dump(result, f, indent=2)
        return result
    except Exception as e:
        return {"_error": str(e)}


# Scraping is slow and rate-limited, so on-the-fly results (including errors)
# are kept in memory; the JSON file written by update_data.py always wins.
_LIVE_CACHE_TTL = int(os.environ.get("LIVE_PREDICTION_TTL", "21600"))
_live_cache = {"value": None, "at": 0.0}
_live_lock = threading.Lock()


def get_cached_current_season_prediction():
    cached = read_current_season_prediction()
    if cached is not None and cached.get("season") == get_current_season():
        return cached
    with _live_lock:
        if _live_cache["value"] is not None and time.time() - _live_cache["at"] < _LIVE_CACHE_TTL:
            return _live_cache["value"]
        _live_cache["value"] = compute_current_season_prediction()
        _live_cache["at"] = time.time()
        return _live_cache["value"]


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NBA MVP Predictor</title>
  <style>
    * { box-sizing: border-box; }
    html, body { margin: 0; }
    body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #0d1117; color: #e6edf3; line-height: 1.45; }
    .container { max-width: 960px; margin: 0 auto; padding: 1.5rem 1rem 0; }
    header { display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
    h1 { color: #58a6ff; font-size: 1.5rem; margin: 0 0 0.25rem; }
    h2 { font-size: 1.1rem; margin: 0 0 0.75rem; }
    .subtitle { color: #8b949e; font-size: 0.9rem; margin: 0; }
    label { display: block; margin-bottom: 0.35rem; color: #8b949e; font-size: 0.9rem; }
    select { padding: 0.5rem 0.75rem; font-size: 1rem; background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; min-width: 140px; cursor: pointer; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .grid .card { margin: 0; }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
    .row { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin: 0.5rem 0; }
    .label { color: #8b949e; min-width: 100px; }
    .value { color: #e6edf3; font-weight: 500; }
    .match-yes { color: #3fb950; }
    .match-no { color: #f85149; }
    .factors { color: #8b949e; font-size: 0.85rem; margin: 0.75rem 0 0; }
    .factors span { display: inline-block; margin: 0.15rem 0.3rem 0.15rem 0; padding: 0.05rem 0.45rem; background: #21262d; border-radius: 999px; }
    .updated { color: #8b949e; font-size: 0.85rem; }
    .bars { list-style: none; margin: 0; padding: 0; }
    .bar-row { display: grid; grid-template-columns: minmax(7rem, 11rem) 1fr 3.2rem; gap: 0.5rem; align-items: center; margin: 0.3rem 0; font-size: 0.85rem; }
    .bar-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bar-track { background: #21262d; border-radius: 4px; height: 0.9rem; overflow: hidden; }
    .bar-fill { display: block; background: linear-gradient(90deg, #1f6feb, #58a6ff); height: 100%; border-radius: 4px; }
    .bar-row.top .bar-fill { background: linear-gradient(90deg, #bb8009, #e3b341); }
    .bar-val { text-align: right; color: #8b949e; font-variant-numeric: tabular-nums; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { padding: 0.45rem 0.6rem; text-align: left; border-bottom: 1px solid #30363d; }
    th { color: #58a6ff; font-weight: 600; }
    td.num { font-variant-numeric: tabular-nums; }
    .meta { color: #8b949e; font-size: 0.85rem; margin-top: 1rem; }
    footer { border-top: 1px solid #30363d; margin-top: 2rem; padding: 1rem 0 1.5rem; color: #8b949e; font-size: 0.85rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; }
    a { color: #58a6ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
<div class="container">
  {% macro top10_view(top10, title) %}
  {% set scores = top10|map(attribute='score')|select('number')|list %}
  {% set max_score = (scores|max if scores else 0) %}
  <div class="grid">
    <div class="card">
      <h2>{{ title }} — model score</h2>
      <ul class="bars">
        {% for p in top10 %}
        {% set sc = p.get('score', p.get('prob', 0)) or 0 %}
        {% set pct = ((sc if sc > 0 else 0) / max_score * 100) if max_score > 0 else 0 %}
        <li class="bar-row {{ 'top' if loop.first else '' }}" title="{{ p.player }} ({{ p.team }}): {{ '%.3f'|format(sc) }}">
          <span class="bar-name">{{ p.rank }}. {{ p.player }}</span>
          <span class="bar-track"><span class="bar-fill" style="width: {{ '%.1f'|format(pct) }}%"></span></span>
          <span class="bar-val">{{ "%.3f"|format(sc) }}</span>
        </li>
        {% endfor %}
      </ul>
    </div>
    <div class="card table-wrap">
      <h2>{{ title }}</h2>
      <table>
        <thead><tr><th>Rank</th><th>Player</th><th>Team</th><th>Score</th></tr></thead>
        <tbody>
          {% for p in top10 %}
          <tr><td>{{ p.rank }}</td><td>{{ p.player }}</td><td>{{ p.team }}</td><td class="num">{{ "%.3f"|format(p.get('score', p.get('prob', 0))) }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
  {% endmacro %}

  <header>
    <div>
      <h1>NBA MVP Predictor</h1>
      <p class="subtitle">Predictions from within-season ranks of advanced stats and team success. Test any season below.</p>
    </div>
    <div class="row">
      <div>
        <label for="season">Season</label>
        <select id="season" name="season">
          {% for s in seasons %}
          <option value="{{ s }}" {{ 'selected' if s == selected_season else '' }}>{{ s }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label for="model">Model (past seasons)</label>
        <select id="model" name="model">
          {% for key, label in models.items() %}
          <option value="{{ key }}" {{ 'selected' if key == selected_model else '' }}>{{ label }}</option>
          {% endfor %}
        </select>
      </div>
    </div>
  </header>

  {% if key_factors %}
  <p class="factors">Key factors: {% for f in key_factors %}<span>{{ f }}</span>{% endfor %}</p>
  {% endif %}

  <div id="result">
    {% if selected_season == current_season %}
      {% if current_pred %}
      <div class="card">
        <h2>Predicted MVP — {{ selected_season }}</h2>
        <div class="row"><span class="label">Predicted winner</span><span class="value">{{ current_pred.get('top1_name', '—') }} ({{ current_pred.get('top1_team', '') }})</span></div>
        <div class="row"><span class="label">Last updated</span><span class="updated">{{ current_pred.get('last_updated', 'N/A') }}</span></div>
      </div>
      {{ top10_view(current_pred.get('top10', []), 'Top 10') }}
      {% else %}
      <div class="card">
        <h2>Predicted MVP — {{ selected_season }}</h2>
        {% if current_pred_error %}
        <p class="meta" style="color:#f85149;">{{ current_pred_error }}</p>
        {% else %}
        <p class="meta">No prediction yet. Run <code>python3 mvp_model.py</code> then <code>python3 update_data.py</code>.</p>
        {% endif %}
      </div>
      {% endif %}
    {% else %}
      {% if details %}
      <div class="card">
        <h2>Predictor result — {{ selected_season }}</h2>
        <div class="row"><span class="label">Predicted MVP</span><span class="value">{{ details.get('predicted_mvp') or '—' }}</span></div>
        <div class="row"><span class="label">Actual MVP</span><span class="value">{{ details.get('actual_mvp') or '—' }}</span></div>
        <div class="row"><span class="label">Match</span><span class="value {{ 'match-yes' if details.get('correct') else 'match-no' }}">{{ 'Yes' if details.get('correct') else 'No' }}</span></div>
      </div>
      {{ top10_view(details.get('top10', []), 'Top 10 predicted that year') }}
      {% if season_picks %}
      <div class="card table-wrap">
        <h2>Every model's pick — {{ selected_season }}</h2>
        <table>
          <thead><tr><th>Model</th><th>Predicted MVP</th><th>Match</th><th>Actual MVP's rank</th></tr></thead>
          <tbody>
            {% for p in season_picks %}
            <tr>
              <td>{{ p.label }}</td>
              <td>{{ p.predicted_mvp or '—' }}</td>
              <td class="{{ 'match-yes' if p.correct else 'match-no' }}">{{ 'Yes' if p.correct else 'No' }}</td>
              <td class="num">{{ p.actual_rank if p.actual_rank is not none else '>10' }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% endif %}
      {% else %}
      <div class="card">
        <h2>Predictor result — {{ selected_season }}</h2>
        <p class="meta">No backtest data. Run <code>python3 mvp_model.py</code> to generate results.</p>
      </div>
      {% endif %}
    {% endif %}
  </div>

  {% if model_comparison %}
  <div class="card table-wrap">
    <h2>Model comparison — leave-one-season-out backtest</h2>
    <table>
      <thead><tr><th>Model</th><th>Exact MVP</th><th>MVP in top 3</th><th>MRR</th><th>Avg. MVP rank</th><th>Vote-order match</th></tr></thead>
      <tbody>
        {% for m in model_comparison %}
        <tr{% if m.key == selected_model %} style="background:#1f2937;"{% endif %}>
          <td>{{ m.label }}</td>
          <td class="num">{{ m.top1 }}</td>
          <td class="num">{{ m.top3 }}</td>
          <td class="num">{{ '%.3f'|format(m.mrr) if m.mrr is not none else '—' }}</td>
          <td class="num">{{ '%.2f'|format(m.mean_rank) if m.mean_rank is not none else '—' }}</td>
          <td class="num">{{ '%.3f'|format(m.spearman) if m.spearman is not none else '—' }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    <p class="meta">MRR = mean of 1 / (rank the model gave the actual MVP). Vote-order match = Spearman correlation with actual vote shares among vote-getters. Live predictions use the shipped model.</p>
  </div>
  {% endif %}

  <footer>
    <span>NBA MVP Predictor · Ridge regression on advanced stats{% if current_pred and current_pred.get('last_updated') %} · Last updated {{ current_pred.get('last_updated') }}{% endif %}</span>
    <a href="{{ repo_url }}" target="_blank" rel="noopener">View source on GitHub</a>
  </footer>
</div>

  <script>
    function reload() {
      const params = new URLSearchParams({
        season: document.getElementById('season').value,
        model: document.getElementById('model').value,
      });
      window.location.search = params.toString();
    }
    document.getElementById('season').addEventListener('change', reload);
    document.getElementById('model').addEventListener('change', reload);
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    seasons = get_season_list()
    saved_pred = read_current_season_prediction()
    # Until the new season has stats, the latest saved prediction (last season's) stays the default view.
    live_season = (saved_pred or {}).get("season") or get_current_season()
    selected = request.args.get("season", live_season)
    if selected not in seasons:
        selected = live_season
    selected_model = request.args.get("model", "shipped")
    if selected_model not in MODEL_CHOICES:
        selected_model = "shipped"
    live_view = selected in (live_season, get_current_season())
    details = None if live_view else read_backtest_details(selected_model).get(selected)
    season_picks = [] if live_view else get_season_picks(selected)
    current_pred = None
    if live_view:
        if selected == live_season and saved_pred is not None:
            current_pred = saved_pred
        else:
            current_pred = get_cached_current_season_prediction()
    current_pred_error = None
    if current_pred and current_pred.get("_error"):
        current_pred_error = current_pred.get("_error")
        current_pred = None
    key_factors = get_key_factors()
    return render_template_string(
        INDEX_HTML,
        seasons=seasons,
        selected_season=selected,
        current_season=selected if live_view else live_season,
        details=details,
        current_pred=current_pred,
        current_pred_error=current_pred_error,
        key_factors=key_factors,
        repo_url=REPO_URL,
        models=MODEL_CHOICES,
        selected_model=selected_model,
        model_comparison=read_model_comparison(),
        season_picks=season_picks,
    )


@app.route("/api/<season>")
@app.route("/api/season/<season>")
def api_season(season):
    """JSON for a given season: predicted, actual, match, top10 (or the current-season prediction)."""
    seasons = get_season_list()
    if season not in seasons:
        return jsonify({"error": "Unknown season"}), 404
    saved_pred = read_current_season_prediction()
    if saved_pred is not None and saved_pred.get("season") == season:
        return jsonify(saved_pred)
    if season == get_current_season():
        return jsonify({"error": "No current season prediction. Run update_data.py."})
    details = read_backtest_details().get(season)
    if not details:
        row = get_backtest_for_season(season)
        if not row:
            return jsonify({"error": "No backtest data. Run mvp_model.py."}), 404
        return jsonify({
            "season": season,
            "predicted_mvp": row.get("predicted_mvp"),
            "actual_mvp": row.get("actual_mvp"),
            "correct": (row.get("correct") or "").lower() == "true",
            "top10": [],
        })
    return jsonify({
        "season": season,
        "predicted_mvp": details.get("predicted_mvp"),
        "actual_mvp": details.get("actual_mvp"),
        "correct": details.get("correct", False),
        "top10": details.get("top10", []),
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
