"""
NBA season helpers. Seasons are strings like "2025-26"; a new season starts on Oct 1.
"""
from datetime import date


def season_from_start_year(start_year: int) -> str:
    """2025 => '2025-26'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_start_year(season: str) -> int:
    """'2025-26' => 2025."""
    return int(str(season).split("-")[0])


def season_end_year(season: str) -> int:
    """'2025-26' => 2026."""
    return season_start_year(season) + 1


def current_season(today: date | None = None) -> str:
    """Season in progress (or about to start) on `today`. Oct 2026 => '2026-27'; Mar 2026 => '2025-26'."""
    today = today or date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return season_from_start_year(start_year)


def previous_season(season: str | None = None) -> str:
    """'2026-27' => '2025-26'. Defaults to the season before the current one."""
    season = season or current_season()
    return season_from_start_year(season_start_year(season) - 1)


def last_completed_season_end_year(today: date | None = None) -> int:
    """
    Ending year of the last completed season, matching the `end_year` convention used by
    the historical dataset builders (range(start_year, end_year) of season start years).
    Oct 2026 => 2026 (last completed season is 2025-26).
    """
    return season_end_year(previous_season(current_season(today)))
