import numpy as np
import pandas as pd
import pytest

from data.fetch_data import add_team_rank
from mvp_model import _names_match, apply_within_season_ranks


@pytest.mark.parametrize(
    "a, b",
    [
        ("Nikola Jokić", "Nikola Jokic"),
        ("Luka Dončić", "luka doncic"),
        ("Shaquille O'Neal", "Shaquille ONeal"),
        ("  Kevin Durant ", "Kevin Durant"),
    ],
)
def test_names_match_normalizes_accents_case_and_punctuation(a, b):
    assert _names_match(a, b)


@pytest.mark.parametrize("a, b", [("Kevin Durant", "Kevin Garnett"), ("", "Kevin Durant"), (None, "Kevin Durant")])
def test_names_match_rejects_different_or_missing_names(a, b):
    assert not _names_match(a, b)


def _two_seasons():
    return pd.DataFrame(
        {
            "season": ["2000-01"] * 4 + ["2001-02"] * 2,
            "ppg": [30.0, 20.0, 10.0, 5.0, 1.0, 2.0],
            "top_team_3": [1, 0, 0, 0, 1, 0],
        }
    )


def test_pct_ranks_are_computed_within_each_season():
    ranked = apply_within_season_ranks(_two_seasons(), ["ppg", "top_team_3"])
    # Best player in a season gets 1/n, worst gets 1.0, independent of other seasons.
    assert ranked["ppg"].tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0, 1.0, 0.5])
    assert ranked["top_team_3"].tolist() == [1, 0, 0, 0, 1, 0]


def test_logtop_ranks_reward_the_top_of_the_league():
    ranked = apply_within_season_ranks(_two_seasons(), ["ppg"], transform="logtop")
    first = ranked[ranked["season"] == "2000-01"]["ppg"].tolist()
    assert first == sorted(first, reverse=True)
    assert first[-1] == pytest.approx(0.0)
    assert first[0] == pytest.approx(-np.log(0.25))


def test_ranks_require_season_column_and_known_transform():
    with pytest.raises(ValueError):
        apply_within_season_ranks(pd.DataFrame({"ppg": [1.0]}), ["ppg"])
    with pytest.raises(ValueError):
        apply_within_season_ranks(_two_seasons(), ["ppg"], transform="bogus")


def test_team_rank_ranks_teams_not_player_rows():
    # Many players on a 64-win team must not push its rank down (the old bug ranked player rows).
    df = pd.DataFrame(
        {
            "season": ["1996-97"] * 6,
            "team_abbr": ["UTA", "UTA", "UTA", "CHI", "MIA", "TOT"],
            "team_wins": [64, 64, 64, 69, 61, np.nan],
        }
    )
    out = add_team_rank(df)
    ranks = dict(zip(out["team_abbr"], out["team_win_rank"], strict=True))
    assert ranks["CHI"] == 1
    assert ranks["UTA"] == 2
    assert ranks["MIA"] == 3
    assert np.isnan(ranks["TOT"])
    assert out.loc[out["team_abbr"] == "TOT", "top_team_3"].item() == 0
