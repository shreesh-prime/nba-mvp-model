"""
NBA MVP predictor: dropdown by year (predicted vs actual), 2025-26 current-season prediction.
No formula on the main UI. Run update_data.py daily to refresh stats and predictions.
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
CURRENT_PREDICTION_PATH = os.path.join(DATA_DIR, "current_season_prediction.json")


def get_season_list():
    """Historical seasons + current."""
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


def read_current_season_prediction():
    """Read 2025-26 prediction and last_updated from daily script."""
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


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NBA MVP Predictor</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 0 auto; padding: 1.5rem; background: #0d1117; color: #e6edf3; }
    h1 { color: #58a6ff; }
    h2 { color: #79c0ff; margin-top: 1.5rem; font-size: 1.1rem; }
    label { display: block; margin-bottom: 0.35rem; color: #8b949e; }
    select { padding: 0.5rem 0.75rem; font-size: 1rem; background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; min-width: 140px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
    .row { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin: 0.5rem 0; }
    .label { color: #8b949e; min-width: 100px; }
    .value { color: #e6edf3; font-weight: 500; }
    .match-yes { color: #3fb950; }
    .match-no { color: #f85149; }
    .meta { color: #8b949e; font-size: 0.85rem; margin-top: 1.5rem; }
    table { width: 100%; border-collapse: collapse; margin: 0.75rem 0; font-size: 0.95rem; }
    th, td { padding: 0.4rem 0.6rem; text-align: left; border: 1px solid #30363d; }
    th { background: #21262d; color: #58a6ff; }
    a { color: #58a6ff; }
  </style>
</head>
<body>
  <h1>NBA MVP Predictor</h1>

  <h2>Pick a season</h2>
  <div class="card">
    <label for="season">Season</label>
    <select id="season" name="season">
      {% for s in seasons %}
      <option value="{{ s }}" {{ 'selected' if s == selected_season else '' }}>{{ s }}</option>
      {% endfor %}
    </select>
  </div>

  <div id="result">
    {% if selected_season == current_season %}
      <!-- 2025-26: current season prediction -->
      <h2>Predicted MVP for {{ selected_season }} (based on current stats)</h2>
      {% if current_pred %}
      <p class="meta">Data last updated: {{ current_pred.get('last_updated', 'N/A') }}</p>
      <div class="card">
        <div class="row"><span class="label">Predicted winner</span><span class="value">{{ current_pred.get('top1_name', '—') }} ({{ current_pred.get('top1_team', '') }})</span></div>
      </div>
      <h2>Top 10 candidates</h2>
      <table>
        <thead><tr><th>Rank</th><th>Player</th><th>Team</th><th>MVP prob</th></tr></thead>
        <tbody>
          {% for p in current_pred.get('top10', []) %}
          <tr><td>{{ p.rank }}</td><td>{{ p.player }}</td><td>{{ p.team }}</td><td>{{ "%.1f"|format(p.prob * 100) }}%</td></tr>
          {% endfor %}
        </tbody>
      </table>
      <p class="meta">Actual MVP: TBD at end of season.</p>
      {% else %}
      <p class="meta">No prediction yet. Run <code>python3 update_data.py</code> to fetch current stats and predict (or run <code>python3 mvp_model.py</code> once to train the model first).</p>
      {% endif %}
    {% else %}
      <!-- Historical: compare predicted vs actual -->
      <h2>Predicted vs actual MVP — {{ selected_season }}</h2>
      {% if backtest_row %}
      <div class="card">
        <div class="row"><span class="label">Predicted MVP</span><span class="value">{{ backtest_row.get('predicted_mvp') or '—' }}</span></div>
        <div class="row"><span class="label">Actual MVP</span><span class="value">{{ backtest_row.get('actual_mvp') or '—' }}</span></div>
        <div class="row"><span class="label">Match</span><span class="value {{ 'match-yes' if ((backtest_row.get('correct') or '')|lower == 'true') else 'match-no' }}">{{ 'Yes' if ((backtest_row.get('correct') or '')|lower == 'true') else 'No' }}</span></div>
      </div>
      {% else %}
      <p class="meta">No backtest data for this season. Run <code>python3 mvp_model.py</code> to generate backtest results.</p>
      {% endif %}
    {% endif %}
  </div>

  <p class="meta">To refresh 2025–26 stats and predictions daily, run <code>python3 update_data.py</code> (e.g. via cron).</p>
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
    backtest_row = None if selected == current_season else get_backtest_for_season(selected)
    current_pred = read_current_season_prediction()
    return render_template_string(
        INDEX_HTML,
        seasons=seasons,
        selected_season=selected,
        current_season=current_season,
        backtest_row=backtest_row,
        current_pred=current_pred,
    )


@app.route("/api/season/<season>")
def api_season(season):
    """JSON for a given season: predicted, actual, match (or 2025-26 top 10)."""
    seasons = get_season_list()
    if season not in seasons:
        return jsonify({"error": "Unknown season"}), 404
    if season == "2025-26":
        data = read_current_season_prediction()
        return jsonify(data or {"error": "No current season prediction. Run update_data.py."})
    row = get_backtest_for_season(season)
    if not row:
        return jsonify({"error": "No backtest data. Run mvp_model.py."})
    return jsonify({
        "season": season,
        "predicted_mvp": row.get("predicted_mvp"),
        "actual_mvp": row.get("actual_mvp"),
        "correct": (row.get("correct") or "").lower() == "true",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
