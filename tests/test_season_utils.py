from datetime import date

import pytest

from season_utils import (
    current_season,
    last_completed_season_end_year,
    previous_season,
    season_end_year,
    season_from_start_year,
    season_start_year,
)


@pytest.mark.parametrize(
    "today, expected",
    [
        (date(2026, 10, 1), "2026-27"),
        (date(2026, 9, 30), "2025-26"),
        (date(2026, 3, 15), "2025-26"),
        (date(1999, 12, 31), "1999-00"),
        (date(2000, 1, 1), "1999-00"),
    ],
)
def test_current_season_rolls_over_on_october_1(today, expected):
    assert current_season(today) == expected


def test_season_string_round_trip():
    assert season_from_start_year(2009) == "2009-10"
    assert season_start_year("2009-10") == 2009
    assert season_end_year("2009-10") == 2010
    assert previous_season("2000-01") == "1999-00"


def test_last_completed_season_end_year():
    assert last_completed_season_end_year(date(2026, 10, 9)) == 2026
    assert last_completed_season_end_year(date(2026, 4, 1)) == 2025
