# NBA MVP Prediction Model

Predict the NBA MVP using **advanced stats** from Basketball Reference. The pipeline:

1. **Fetch historical data** – Advanced stats (PER, WS, BPM, VORP, WS/48, etc.) for each player-season.
2. **Correlation analysis** – See which stats correlate most with winning MVP (saved to `data/mvp_feature_correlations.json` and `data/mvp_correlation_plot.png`).
3. **Train a model** – Random Forest trained on the top correlated stats to predict MVP (binary: winner vs non-winner).
4. **Predict this season** – Score current-season players and rank top MVP candidates.

## Setup

```bash
cd "nba model"
pip install -r requirements.txt
```

## GitHub and local web app (port 8080)

### Put the project on GitHub

1. **Create a repo** on GitHub (e.g. `nba-mvp-model`). Do not add a README/license if you already have them locally.

2. **From the project folder:**

   ```bash
   cd "/path/to/nba model"
   git init
   git add .
   git commit -m "Initial commit: NBA MVP prediction model"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/nba-mvp-model.git
   git push -u origin main
   ```

3. **`.gitignore`** is set so data files and model artifacts (e.g. `data/historical_advanced.csv`, `model/*.joblib`) are not committed. Others can clone and run the pipeline to regenerate them.

### Run the web app locally on port 8080

A small Flask app serves the **formula**, **backtest results**, and **correlations** in the browser.

1. **Install and run:**

   ```bash
   pip install -r requirements.txt
   python app.py
   ```

2. **Open:** [http://localhost:8080](http://localhost:8080)

3. **Optional:** Generate data first so the page has content:

   ```bash
   python mvp_model.py
   python app.py
   ```

   Then refresh [http://localhost:8080](http://localhost:8080) to see the formula, backtest table, and stat correlations.

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
