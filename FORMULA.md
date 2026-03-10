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

- **Advanced stats**: VORP, BPM, WS, WS/48, PER, ts_pct, usage, etc.
- **Regular stats**: **Points per game (ppg)** — emphasized; rebounds (rpg), assists (apg).
- **Team context**: **top_team_3** (1 if team is top-3 in wins), **team_win_rank** (1 = most wins), **team_wins** — being on a top-3 team is **prioritized** in the score (strong positive signal) but **not required**; a player whose stats greatly outweigh everyone else can still win.

The model produces an **MVP score** for every player: a weighted combination of these regular stats, impactful advanced stats, and team success. The winner is whoever has the highest score—no hard filter by team. So: **maximize the MVP score** (points + advanced impact + team success); top-3 team helps a lot but standout stats can overcome a lower seed. The exact coefficients are in `mvp_formula.txt` after you run the pipeline.
