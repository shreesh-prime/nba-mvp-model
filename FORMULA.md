# The MVP "formula"

Every model scores players with a linear combination of their stats, so each one doubles as an interpretable formula. Running `python3 mvp_model.py` writes the exact equations, with fitted coefficients, to:

- `data/mvp_formula.txt`: shipped model
- `data/mvp_formula_logistic.txt`, `data/mvp_formula_pct.txt`, `data/mvp_formula_logtop.txt`: the three variants

## Logistic models (shipped and `logistic`)

```
P(MVP) = 1 / (1 + exp(-z))
z = intercept + c1*top_team_3 + c2*ppg + c3*bpm + c4*ws + ... + c12*team_wins
```

The player with the highest P(MVP) in a season is the predicted MVP.

## Ensemble (`ensemble`)

```
score = z(P_logistic) + z(share_ridge) + z(raw VORP)
```

Each term is standardized within the season being scored, then added. This is the strongest exact-winner model (22/30 leave-one-season-out).

## Vote-share Ridge models (`pct` and `logtop`)

```
predicted award share = intercept + c1*top_team_3 + c2*ppg + ... + c13*bpm_change
```

Each model is fit to the share of MVP voting points each player actually received.

## Reading the coefficients

- Each stat is first converted to a **within-season percentile rank**, then standardized. `top_team_3` is a 0/1 flag for a top-3 team by wins.
- With percentile ranks (`shipped`, `logistic`, `pct`), **1/n is the league's best**. A *negative* coefficient means "ranking near the top of the league helps".
- With log-top ranks (`logtop`), larger is better, so a *positive* coefficient helps.
- `team_wins` and `team_win_rank` carry the same information, so the models split their weight between them. Several advanced stats are strongly correlated (WS, OWS, OBPM, BPM). Read the coefficients as a group rather than one by one.

In the logistic models, the biggest weights are on offensive impact (OBPM, OWS), PER, WS/48, BPM and team success. In short, the formula for winning MVP is to be among the league's very best in offensive impact and efficiency, on one of its best teams.
