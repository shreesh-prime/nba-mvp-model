"""
Simple web app for the NBA MVP model: formula, backtest results, and top predictions.
Run: python app.py
Open: http://localhost:8080
"""
import os
import json
import csv

from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

BASE = os.path.join(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data")
FORMULA_PATH = os.path.join(DATA_DIR, "mvp_formula.txt")
BACKTEST_PATH = os.path.join(DATA_DIR, "backtest_results.csv")
CORRELATIONS_PATH = os.path.join(DATA_DIR, "mvp_feature_correlations.json")


def read_formula():
    if os.path.isfile(FORMULA_PATH):
        with open(FORMULA_PATH) as f:
            return f.read()
    return "Run 'python mvp_model.py' or 'python mvp_model.py backtest' to generate the formula."


def read_backtest():
    if not os.path.isfile(BACKTEST_PATH):
        return None
    rows = []
    with open(BACKTEST_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def read_correlations():
    if not os.path.isfile(CORRELATIONS_PATH):
        return None
    with open(CORRELATIONS_PATH) as f:
        return json.load(f)


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NBA MVP Model</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 1.5rem; background: #0d1117; color: #e6edf3; }
    h1 { color: #58a6ff; }
    h2 { color: #79c0ff; margin-top: 2rem; }
    a { color: #58a6ff; }
    pre { background: #161b22; padding: 1rem; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; }
    table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    th, td { padding: 0.5rem 0.75rem; text-align: left; border: 1px solid #30363d; }
    th { background: #21262d; color: #58a6ff; }
    tr:nth-child(even) { background: #161b22; }
    .correct { color: #3fb950; }
    .wrong { color: #f85149; }
    .meta { color: #8b949e; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>NBA MVP prediction model</h1>
  <p class="meta">Formula, backtest results, and correlations. Run the pipeline to refresh data.</p>
  <nav>
    <a href="#formula">Formula</a> &middot;
    <a href="#backtest">Backtest</a> &middot;
    <a href="#correlations">Correlations</a>
  </nav>

  <h2 id="formula">Formula to win MVP</h2>
  <pre>{{ formula }}</pre>

  <h2 id="backtest">Backtest: predicted vs actual MVP</h2>
  {% if backtest %}
  <p class="meta">Leave-one-season-out: model predicted the correct MVP in <strong>{{ correct_count }}/{{ total_count }}</strong> seasons.</p>
  <table>
    <thead>
      <tr>
        <th>Season</th>
        <th>Actual MVP</th>
        <th>Predicted MVP</th>
        <th>Match</th>
      </tr>
    </thead>
    <tbody>
      {% for row in backtest %}
      <tr>
        <td>{{ row.get('season', '') }}</td>
        <td>{{ row.get('actual_mvp', '') }}</td>
        <td>{{ row.get('predicted_mvp', '') or '—' }}</td>
        <td class="{{ 'correct' if ((row.get('correct') or '')|lower == 'true') else 'wrong' }}">{{ 'Yes' if ((row.get('correct') or '')|lower == 'true') else 'No' }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <p>No backtest results yet. Run <code>python mvp_model.py</code> or <code>python mvp_model.py backtest</code>.</p>
  {% endif %}

  <h2 id="correlations">Stat correlations with MVP</h2>
  {% if correlations %}
  <p class="meta">Higher value = stronger association with winning MVP.</p>
  <table>
    <thead>
      <tr><th>Stat</th><th>|Correlation|</th></tr>
    </thead>
    <tbody>
      {% for stat, val in correlations.items() %}
      <tr><td>{{ stat }}</td><td>{{ "%.4f"|format(val) }}</td></tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <p>No correlations file yet. Run the pipeline to generate <code>data/mvp_feature_correlations.json</code>.</p>
  {% endif %}
</body>
</html>
"""


@app.route("/")
def index():
    formula = read_formula()
    backtest = read_backtest()
    correlations = read_correlations()
    def is_correct(r):
        v = r.get("correct")
        return v is True or str(v).lower() == "true"
    correct_count = sum(1 for r in (backtest or []) if is_correct(r))
    total_count = len([r for r in (backtest or []) if r.get("predicted_mvp")])
    return render_template_string(
        INDEX_HTML,
        formula=formula,
        backtest=backtest,
        correlations=correlations or {},
        correct_count=correct_count,
        total_count=total_count,
    )


@app.route("/api/formula")
def api_formula():
    return jsonify({"formula": read_formula()})


@app.route("/api/backtest")
def api_backtest():
    rows = read_backtest()
    return jsonify({"backtest": rows or []})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
