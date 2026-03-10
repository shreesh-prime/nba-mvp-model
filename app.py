"""
NBA MVP predictor: prediction from key correlating factors (Ridge regression).
Dropdown to test each year and see what the predictor came up with.
Run update_data.py to refresh current-season prediction.
"""
import os
import json
import csv
from datetime import datetime

from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "model")
BACKTEST_PATH = os.path.join(DATA_DIR, "backtest_results.csv")
BACKTEST_DETAILS_PATH = os.path.join(DATA_DIR, "backtest_season_details.json")
CURRENT_PREDICTION_PATH = os.path.join(DATA_DIR, "current_season_prediction.json")
FEATURE_CORR_PATH = os.path.join(DATA_DIR, "mvp_feature_correlations.json")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.json")


def get_season_list():
    years = list(range(2000, 2026))
    return [f"{y}-{str(y+1)[-2:]}" for y in years]


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


def read_backtest_details():
    if not os.path.isfile(BACKTEST_DETAILS_PATH):
        return {}
    try:
        with open(BACKTEST_DETAILS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


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
    """Generate 2025-26 prediction on the fly. Returns dict or dict with _error key."""
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
        current = fetch_current_season_advanced(season="2025-26", min_games=20)
        if current.empty:
            return {"_error": "Could not fetch 2025-26 stats from Basketball Reference. Try again later."}
        missing = [c for c in feature_cols if c not in current.columns]
        if missing:
            return {"_error": f"Data missing columns: {missing}. Re-run python3 mvp_model.py to retrain."}
        X = current[feature_cols].fillna(0)
        X_s = scaler.transform(X)
        current = current.copy()
        current["mvp_score"] = model.predict(X_s)
        current["mvp_rank"] = current["mvp_score"].rank(ascending=False).astype(int)
        top10_df = current.nsmallest(10, "mvp_rank")
        top10 = [{"rank": int(row["mvp_rank"]), "player": str(row.get("player", "")), "team": str(row.get("team_abbr", "")), "score": float(row["mvp_score"])} for _, row in top10_df.iterrows()]
        top1 = top10[0] if top10 else {}
        os.makedirs(DATA_DIR, exist_ok=True)
        result = {"last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "season": "2025-26", "top1_name": top1.get("player", "—"), "top1_team": top1.get("team", ""), "top10": top10}
        with open(CURRENT_PREDICTION_PATH, "w") as f:
            json.dump(result, f, indent=2)
        return result
    except Exception as e:
        return {"_error": str(e)}


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NBA MVP Predictor</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 680px; margin: 0 auto; padding: 1.5rem; background: #0d1117; color: #e6edf3; }
    h1 { color: #58a6ff; font-size: 1.35rem; margin-bottom: 0.5rem; }
    .subtitle { color: #8b949e; font-size: 0.9rem; margin-bottom: 1.25rem; }
    label { display: block; margin-bottom: 0.35rem; color: #8b949e; font-size: 0.9rem; }
    select { padding: 0.5rem 0.75rem; font-size: 1rem; background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; min-width: 140px; cursor: pointer; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
    .row { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin: 0.5rem 0; }
    .label { color: #8b949e; min-width: 100px; }
    .value { color: #e6edf3; font-weight: 500; }
    .match-yes { color: #3fb950; }
    .match-no { color: #f85149; }
    .factors { color: #8b949e; font-size: 0.85rem; margin-top: 0.5rem; }
    .factors span { margin-right: 0.5rem; }
    table { width: 100%; border-collapse: collapse; margin: 0.75rem 0; font-size: 0.9rem; }
    th, td { padding: 0.45rem 0.6rem; text-align: left; border: 1px solid #30363d; }
    th { background: #21262d; color: #58a6ff; }
    .meta { color: #8b949e; font-size: 0.8rem; margin-top: 1rem; }
  </style>
</head>
<body>
  <h1>NBA MVP Predictor</h1>
  <p class="subtitle">Predictions from key correlating factors (Ridge regression). Test any season below.</p>

  <div class="card">
    <label for="season">Season</label>
    <select id="season" name="season">
      {% for s in seasons %}
      <option value="{{ s }}" {{ 'selected' if s == selected_season else '' }}>{{ s }}</option>
      {% endfor %}
    </select>
    {% if key_factors %}
    <p class="factors">Key factors: {% for f in key_factors %}<span>{{ f }}</span>{% endfor %}</p>
    {% endif %}
  </div>

  <div id="result">
    {% if selected_season == current_season %}
      <h2>Predicted MVP — {{ selected_season }}</h2>
      {% if current_pred %}
      <p class="meta">Updated: {{ current_pred.get('last_updated', 'N/A') }}</p>
      <div class="card">
        <div class="row"><span class="label">Predicted winner</span><span class="value">{{ current_pred.get('top1_name', '—') }} ({{ current_pred.get('top1_team', '') }})</span></div>
      </div>
      <h2>Top 10 (by model score)</h2>
      <table>
        <thead><tr><th>Rank</th><th>Player</th><th>Team</th><th>Score</th></tr></thead>
        <tbody>
          {% for p in current_pred.get('top10', []) %}
          <tr><td>{{ p.rank }}</td><td>{{ p.player }}</td><td>{{ p.team }}</td><td>{{ "%.3f"|format(p.get('score', p.get('prob', 0))) }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
      {% if current_pred_error %}
      <p class="meta" style="color:#f85149;">{{ current_pred_error }}</p>
      {% else %}
      <p class="meta">No prediction yet. Run <code>python3 mvp_model.py</code> then <code>python3 update_data.py</code>.</p>
      {% endif %}
      {% endif %}
    {% else %}
      <h2>Predictor result — {{ selected_season }}</h2>
      {% if details %}
      <div class="card">
        <div class="row"><span class="label">Predicted MVP</span><span class="value">{{ details.get('predicted_mvp') or '—' }}</span></div>
        <div class="row"><span class="label">Actual MVP</span><span class="value">{{ details.get('actual_mvp') or '—' }}</span></div>
        <div class="row"><span class="label">Match</span><span class="value {{ 'match-yes' if details.get('correct') else 'match-no' }}">{{ 'Yes' if details.get('correct') else 'No' }}</span></div>
      </div>
      <h2>Top 10 predicted that year</h2>
      <table>
        <thead><tr><th>Rank</th><th>Player</th><th>Team</th><th>Score</th></tr></thead>
        <tbody>
          {% for p in details.get('top10', []) %}
          <tr><td>{{ p.rank }}</td><td>{{ p.player }}</td><td>{{ p.team }}</td><td>{{ "%.3f"|format(p.get('score', 0)) }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
      <p class="meta">No backtest data. Run <code>python3 mvp_model.py</code> to generate results.</p>
      {% endif %}
    {% endif %}
  </div>

  <script>
    document.getElementById('season').addEventListener('change', function() {
      window.location.search = 'season=' + encodeURIComponent(this.value);
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    seasons = get_season_list()
    selected = request.args.get("season", "2025-26")
    if selected not in seasons:
        selected = "2025-26"
    current_season = "2025-26"
    backtest_details = read_backtest_details()
    details = backtest_details.get(selected) if selected != current_season else None
    current_pred = read_current_season_prediction()
    if current_pred is None and selected == current_season:
        current_pred = compute_current_season_prediction()
    current_pred_error = None
    if current_pred and current_pred.get("_error"):
        current_pred_error = current_pred.get("_error")
        current_pred = None
    key_factors = get_key_factors()
    return render_template_string(
        INDEX_HTML,
        seasons=seasons,
        selected_season=selected,
        current_season=current_season,
        details=details,
        current_pred=current_pred,
        current_pred_error=current_pred_error,
        key_factors=key_factors,
    )


@app.route("/api/season/<season>")
def api_season(season):
    """JSON for a given season: predicted, actual, match, top10 (or 2025-26 current prediction)."""
    seasons = get_season_list()
    if season not in seasons:
        return jsonify({"error": "Unknown season"}), 404
    if season == "2025-26":
        data = read_current_season_prediction()
        return jsonify(data or {"error": "No current season prediction. Run update_data.py."})
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
    app.run(host="0.0.0.0", port=8080, debug=False)
