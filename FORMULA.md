# MVP formula (according to the model)

The model uses two views of “how to win MVP”:

---

## 1. Interpretable formula (logistic regression)

After you run `python mvp_model.py` or `python mvp_model.py backtest`, the **exact equation** is saved to **`data/mvp_formula.txt`**. It looks like this:

**Probability of being MVP:**

```
P(MVP) = 1 / (1 + exp(-z))
```

**Score z (linear combination of standardized stats):**

```
z = intercept + c1*vorp + c2*bpm + c3*ws + c4*ws_per_48 + c5*per + ...
```

- **Features** (e.g. `vorp`, `bpm`, `ws`, `ws_per_48`, `per`) are the top advanced stats by correlation with MVP. They are **standardized** (mean 0, std 1) before being used.
- **Coefficients** `c1`, `c2`, ...: **positive** = higher that stat → higher MVP probability; **negative** = hurts.
- **Intercept**: baseline when all stats are at league average.

So in words: **the formula to win MVP is to maximize z** — i.e. have high values in the stats that have positive coefficients (usually VORP, BPM, WS, WS/48, PER) and avoid being low in them.

---

## 2. Which stats matter most?

The **weights** (coefficients) and **order of importance** depend on your run; they’re in `data/mvp_formula.txt` and in the “Feature importance” output from the Random Forest. In practice, the model tends to weight:

- **VORP** (value over replacement)
- **BPM** (box plus/minus)
- **WS** (win shares)
- **WS/48** (win shares per 48 minutes)
- **PER** (player efficiency rating)

So the short version of the formula: **maximize a weighted combination of these advanced stats** (and any others in your `mvp_formula.txt`). The exact numbers are in that file after you run the pipeline.
