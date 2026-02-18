# NBA MVP Prediction Model

Predict the NBA MVP using **advanced stats** from Basketball Reference. The pipeline:

1. **Fetch historical data** – Advanced stats (PER, WS, BPM, VORP, WS/48, etc.) for each player-season.
2. **Correlation analysis** – See which stats correlate most with winning MVP (saved to `data/mvp_feature_correlations.json` and `data/mvp_correlation_plot.png`).
3. **Train a model** – Random Forest trained on the top correlated stats to predict MVP (binary: winner vs non-winner).
4. **Predict this season** – Score current-season players and rank top MVP candidates.

## Setup and run

Run everything from the **folder that contains** `mvp_model.py`, `app.py`, and `requirements.txt` (the repo root). For example, if you have:

```
nba-model/
  nba-mvp-model/    ← go here (this is the repo root)
    mvp_model.py
    app.py
    requirements.txt
    data/
    ...
```

then:

```bash
cd nba-model/nba-mvp-model
pip install -r requirements.txt
```

All commands below are run from that same folder (where the Python files live).

## Run the pipeline

**Full pipeline** (fetch data, backtest, formula, train, predict):

```bash
python mvp_model.py
```

**Backtest and formula only:**

```bash
python mvp_model.py backtest
```

**Predict current season** (after training once):

```bash
python predict_mvp.py
```

## Web app on port 8080

From the folder that contains `app.py`:

```bash
pip install -r requirements.txt
python3 app.py
```

Then open **[http://localhost:8080](http://localhost:8080)**. You get:

- **Dropdown by year** — Pick a season (2000–01 through 2025–26).
- **Historical seasons** — Shows **predicted MVP** vs **actual MVP** and whether they match.
- **2025–26** — Shows **predicted winner** (and top 10) based on current stats; actual MVP TBD at end of season.

Run the pipeline once so the app has backtest and model data:

```bash
python3 mvp_model.py
python3 app.py
```

## Keep 2025–26 prediction updated daily

To refresh current-season stats and update the model’s prediction every day:

```bash
python3 update_data.py
```

This fetches the latest 2025–26 advanced stats from Basketball Reference and overwrites `data/current_season_prediction.json`. The app reads that file for the “2025–26” tab.

**Cron example** (run every day at 6 AM):

```bash
0 6 * * * cd /path/to/nba-mvp-model && python3 update_data.py
```

Replace `/path/to/nba-mvp-model` with the folder that contains `update_data.py`.

## GitHub

- **Clone:** `git clone https://github.com/YOUR_USERNAME/nba-mvp-model.git && cd nba-mvp-model` (then run commands from that folder; if you put it inside another folder like `nba-model`, use `cd nba-model/nba-mvp-model`)
- **Push changes:** `git add . && git commit -m "..." && git push`
- **`.gitignore`** keeps data and model files out of the repo; anyone who clones can run the pipeline to regenerate them.

## Usage

### Full pipeline (recommended first run)

Build historical data, compute correlations, train the model, and predict this year’s top 10:

```bash
python mvp_model.py
```

- **Historical data** is cached at `data/historical_advanced.csv` (so later runs are fast).
- **Correlations** are printed, saved to `data/mvp_feature_correlations.json`, and plotted to `data/mvp_correlation_plot.png`.
- **Model** is saved under `model/` (classifier, scaler, feature list).
- **Top 10** for the current season (e.g. 2025-26) are printed.

### Predict only (after training)

After you’ve run `mvp_model.py` once, you can refresh the prediction using the latest stats without retraining:

```bash
python predict_mvp.py
```

Edit `predict_mvp.py` to change `season`, `min_games`, or `top_n` if needed.

### Customization

In `mvp_model.py` you can change:

- **`run_full_pipeline(..., start_year=2000, end_year=2025)`** – Which seasons to use for training.
- **`top_k_features=10`** – How many of the top-correlated stats to use in the model.
- **`current_season="2025-26"`** – Which season to predict.

In `data/fetch_data.py`:

- **`build_advanced_historical_dataset(..., min_games=41)`** – Minimum games to count a player-season (41 ≈ half season).

## Backtest: did the model predict past MVPs?

Each run of the full pipeline **backtests** the model with **leave-one-season-out**:

- For each past season (e.g. 2019–20), the model is trained on **all other seasons**, then asked to predict that season’s MVP (the player with highest MVP probability).
- Results are compared to the **actual MVP** and written to `data/backtest_results.csv` (columns: `season`, `actual_mvp`, `predicted_mvp`, `correct`).
- Summary: **“Model predicted correct MVP in X out of Y seasons”** is printed.

To run **only** the backtest and formula (no current-season prediction):

```bash
python mvp_model.py backtest
```

## The “formula” to win MVP

The script fits a **logistic regression** on the same advanced stats so you get an interpretable equation:

- **P(MVP) = 1 / (1 + exp(-z))**, where **z** is a linear combination of (standardized) stats.
- The equation is printed and saved to **`data/mvp_formula.txt`**, e.g.  
  **z = intercept + c1*vorp + c2*bpm + c3*ws + ...**
- **Positive coefficient** = higher that stat → higher MVP chance; **negative** = hurts.

So the “formula” is: **maximize that combination of stats** (e.g. VORP, BPM, WS) subject to the weights in `mvp_formula.txt`. The Random Forest is used for the actual predictions; the logistic regression is for understanding what drives the model.

## If accuracy is low

If the backtest shows the model often picks the wrong MVP:

- **Add team success**: merge in team wins (e.g. from standings); MVPs are usually on top seeds.
- **Change features**: try `top_k_features=6` or `12` in `mvp_model.py`.
- **More history**: use `start_year=1990` for more MVP examples.
- **Try logistic regression for prediction**: in code, swap the classifier to `LogisticRegression` and re-run backtest.

---

## Advanced stats used

From Basketball Reference’s **Advanced** table:

- **PER** – Player Efficiency Rating  
- **TS%** – True Shooting  
- **WS, WS/48** – Win Shares (total and per 48)  
- **BPM, VORP** – Box Plus/Minus, Value over Replacement Player  
- **OWS, DWS** – Offensive/Defensive Win Shares  
- **USG%** – Usage  
- **OBPM, DBPM** – Offensive/Defensive BPM  
- Plus rate stats: **ORB%, DRB%, TRB%, AST%, STL%, BLK%, TOV%**, etc.

The script **ranks these by correlation with MVP** and uses the top ones as model inputs, so the “determining factors” are data-driven from past MVP results.
