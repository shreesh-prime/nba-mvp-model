import numpy as np
import pandas as pd
import pytest

from mvp_model import MODEL_VARIANTS, apply_within_season_ranks, load_variant_artifacts, score_players


@pytest.mark.parametrize("variant", [None, *MODEL_VARIANTS])
def test_saved_models_score_a_synthetic_season(variant):
    model, scaler, feature_cols, info = load_variant_artifacts(variant)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.uniform(0, 30, size=(40, len(feature_cols))), columns=feature_cols)
    df["season"] = "2030-31"
    df["_raw_vorp"] = rng.uniform(-2, 12, size=len(df))
    if "top_team_3" in feature_cols:
        df["top_team_3"] = (np.arange(40) < 3).astype(int)
    ranked = apply_within_season_ranks(df, feature_cols, transform=info["rank_transform"])

    scores = score_players(ranked, model, scaler, feature_cols)

    assert len(scores) == 40
    assert scores.notna().all()
    assert scores.nunique() > 1
