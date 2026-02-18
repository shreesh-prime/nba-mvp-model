"""
Fetch NBA data: MVP history (Basketball Reference), player stats & standings (nba_api).
"""
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup


def fetch_mvp_history() -> pd.DataFrame:
    """
    Scrape NBA MVP winners from Basketball Reference.
    Returns DataFrame with columns: season, player, team_abbr, age, etc.
    """
    url = "https://www.basketball-reference.com/awards/mvp.html"
    # pandas read_html works well on BR tables
    tables = pd.read_html(url)
    # First table is NBA winners (second might be ABA)
    df = tables[0]
    # Rename columns for consistency (BR uses 'Season', 'Player', 'Tm', etc.)
    df = df.rename(columns={
        "Season": "season",
        "Player": "player",
        "Tm": "team_abbr",
        "Age": "age",
        "G": "games",
        "MP": "mpg",
        "PTS": "ppg",
        "TRB": "rpg",
        "AST": "apg",
        "STL": "spg",
        "BLK": "bpg",
        "FG%": "fg_pct",
        "3P%": "fg3_pct",
        "FT%": "ft_pct",
        "WS": "win_shares",
        "WS/48": "ws_per_48",
    })
    # Filter to NBA only (drop ABA rows if present) and recent decades for modeling
    if "Lg" in df.columns:
        df = df[df["Lg"] == "NBA"].copy()
    # Season format: "2023-24" -> keep as is for joining
    df["season"] = df["season"].astype(str)
    return df


def fetch_player_stats_nba_api(season: str) -> pd.DataFrame:
    """
    Fetch per-game player stats for a season using nba_api.
    season format: "2023-24"
    """
    from nba_api.stats.endpoints import leaguedashplayerstats

    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
        df = stats.get_data_frames()[0]
        df["season"] = season
        return df
    except Exception as e:
        print(f"nba_api failed for {season}: {e}")
        return pd.DataFrame()


def fetch_standings_nba_api(season: str) -> pd.DataFrame:
    """Fetch team standings (wins, losses) for a season."""
    from nba_api.stats.endpoints import leaguestandings

    try:
        stand = leaguestandings.LeagueStandings(
            league_id="00", season=season, season_type="Regular Season"
        )
        df = stand.get_data_frames()[0]
        df["season"] = season
        return df
    except Exception as e:
        print(f"Standings failed for {season}: {e}")
        return pd.DataFrame()


def fetch_advanced_stats_br(season: str) -> pd.DataFrame:
    """
    Fetch advanced stats (PER, WS, WS/48, BPM, VORP, TS%, USG%, etc.) from Basketball Reference.
    URL: .../leagues/NBA_{year}_advanced.html
    """
    year = season.split("-")[1]
    full_year = "20" + year
    url = f"https://www.basketball-reference.com/leagues/NBA_{full_year}_advanced.html"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        # Remove repeated header rows
        df = df[df["Rk"] != "Rk"].copy()
        df["season"] = season
        df["Player"] = df["Player"].str.replace(r"\*", "", regex=True)
        # Standardize column names (BR uses PER, TS%, WS, WS/48, BPM, VORP, etc.)
        rename = {
            "Player": "player",
            "Tm": "team_abbr",
            "Team": "team_abbr",
            "G": "games",
            "MP": "mp",
            "PER": "per",
            "TS%": "ts_pct",
            "3PAr": "fg3ar",
            "FTr": "ftr",
            "ORB%": "orb_pct",
            "DRB%": "drb_pct",
            "TRB%": "trb_pct",
            "AST%": "ast_pct",
            "STL%": "stl_pct",
            "BLK%": "blk_pct",
            "TOV%": "tov_pct",
            "USG%": "usg_pct",
            "OWS": "ows",
            "DWS": "dws",
            "WS": "ws",
            "WS/48": "ws_per_48",
            "OBPM": "obpm",
            "DBPM": "dbpm",
            "BPM": "bpm",
            "VORP": "vorp",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        return df
    except Exception as e:
        print(f"BR advanced failed for {season}: {e}")
        return pd.DataFrame()


def get_season_list(start_year: int = 2000, end_year: int = 2025) -> list[str]:
    """Generate list of season strings: 2000-01, 2001-02, ..., 2024-25."""
    return [f"{y}-{str(y+1)[-2:]}" for y in range(start_year, end_year)]


# Advanced stat columns used for correlation and modeling (numeric only).
ADVANCED_STAT_COLUMNS = [
    "per", "ts_pct", "fg3ar", "ftr", "orb_pct", "drb_pct", "trb_pct",
    "ast_pct", "stl_pct", "blk_pct", "tov_pct", "usg_pct",
    "ows", "dws", "ws", "ws_per_48", "obpm", "dbpm", "bpm", "vorp",
]


def build_historical_dataset(
    start_year: int = 2000,
    end_year: int = 2025,
    use_nba_api: bool = True,
    use_br_advanced: bool = True,
) -> pd.DataFrame:
    """
    Build a single DataFrame: one row per player-season with stats and is_mvp label.
    If use_nba_api: use nba_api for stats (no advanced). If use_br_advanced: use BR for advanced (slower).
    """
    mvp_df = fetch_mvp_history()
    seasons = get_season_list(start_year=start_year, end_year=end_year)
    # Restrict MVP to our season range
    mvp_df = mvp_df[mvp_df["season"].isin(seasons)].copy()
    mvp_df["is_mvp"] = 1

    all_player_seasons = []
    for i, season in enumerate(seasons):
        if use_nba_api:
            df = fetch_player_stats_nba_api(season)
            if df.empty:
                continue
            # nba_api columns: PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, GP, MIN, PTS, REB, AST, ...
            df = df.rename(columns={
                "PLAYER_NAME": "player",
                "TEAM_ABBREVIATION": "team_abbr",
                "GP": "games",
                "MIN": "mpg",
                "PTS": "ppg",
                "REB": "rpg",
                "AST": "apg",
                "STL": "spg",
                "BLK": "bpg",
                "FG_PCT": "fg_pct",
                "FG3_PCT": "fg3_pct",
                "FT_PCT": "ft_pct",
            })
        else:
            df = fetch_advanced_stats_br(season)
            if df.empty:
                continue
            # BR per_game has Player, Tm, G, GS, MP, FG%, 3P%, 2P%, eFG%, FT%, TRB, AST, STL, BLK, PTS
            df = df.rename(columns={"Tm": "team_abbr", "G": "games", "MP": "mpg", "PTS": "ppg", "TRB": "rpg", "AST": "apg", "STL": "spg", "BLK": "bpg"})
        # Ensure numeric
        for col in ["ppg", "rpg", "apg", "games", "mpg", "fg_pct", "ft_pct", "fg3_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Minimum games to be a candidate
        df = df[df["games"].fillna(0) >= 41].copy()  # ~half season
        all_player_seasons.append(df)
        time.sleep(0.6)  # rate limit

    player_df = pd.concat(all_player_seasons, ignore_index=True)

    # Add team wins
    all_standings = []
    for season in seasons:
        st = fetch_standings_nba_api(season)
        if not st.empty:
            all_standings.append(st)
        time.sleep(0.5)
    if all_standings:
        standings = pd.concat(all_standings, ignore_index=True)
        # nba_api standings: TEAM_ID, W, L, etc. Need team abbrev
        if "TeamName" in standings.columns and "W" in standings.columns:
            # Some endpoints use TeamName; we need abbreviation. Use team_abbr from player data and merge by season+team
            pass
        # LeagueStandings might have Abbreviation or TeamName; check common column for wins
        wins_col = "W" if "W" in standings.columns else "WINS"
        if "TEAM" in standings.columns:
            standings["team_abbr"] = standings["TEAM"].str[-3:]  # often "TeamName (XXX)"
        # Simpler: merge on team + season. Player df has team_abbr.
        abbrev_col = [c for c in standings.columns if "ABBREV" in c.upper() or c == "TeamCity"]
        if not abbrev_col and "TeamName" in standings.columns:
            # Parse "City Name" -> we need abbreviation; nba_api standings may have Conference/TeamId
            pass
        # Use nba_api convention: LeagueStandings returns TeamID; we need to map to abbreviation
        # For robustness, add wins by matching team name from player data
        team_wins = standings[["season", wins_col]].copy()
        if "TEAM" in standings.columns:
            team_wins["team_abbr"] = standings["TEAM"].str.split().str[-1].str.strip("()")
        elif "Abbreviation" in standings.columns:
            team_wins["team_abbr"] = standings["Abbreviation"]
        if "team_abbr" in team_wins.columns:
            player_df = player_df.merge(
                team_wins.rename(columns={wins_col: "team_wins"}),
                on=["season", "team_abbr"],
                how="left",
            )
    else:
        player_df["team_wins"] = None

    # Label MVP
    mvp_merge = mvp_df[["season", "player", "is_mvp"]].drop_duplicates()
    # Normalize names: BR might have "Stephen Curry" vs nba_api "Stephen Curry" (same) but sometimes suffixes
    player_df["player_clean"] = player_df["player"].str.strip()
    mvp_merge["player_clean"] = mvp_merge["player"].str.strip()
    player_df = player_df.merge(
        mvp_merge[["season", "player_clean", "is_mvp"]],
        left_on=["season", "player_clean"],
        right_on=["season", "player_clean"],
        how="left",
    )
    player_df["is_mvp"] = player_df["is_mvp"].fillna(0).astype(int)
    return player_df


def build_advanced_historical_dataset(
    start_year: int = 2000,
    end_year: int = 2025,
    min_games: int = 41,
) -> pd.DataFrame:
    """
    Build historical dataset using only Basketball Reference advanced stats.
    One row per player-season; columns include PER, WS, BPM, VORP, etc., plus is_mvp.
    """
    mvp_df = fetch_mvp_history()
    seasons = get_season_list(start_year=start_year, end_year=end_year)
    mvp_df = mvp_df[mvp_df["season"].isin(seasons)].copy()
    mvp_df["is_mvp"] = 1

    all_dfs = []
    for season in seasons:
        df = fetch_advanced_stats_br(season)
        if df.empty:
            continue
        # Minimum games
        if "games" in df.columns:
            df["games"] = pd.to_numeric(df["games"], errors="coerce")
            df = df[df["games"].fillna(0) >= min_games].copy()
        # Keep one row per player (primary team if traded)
        if "team_abbr" in df.columns and df["player"].duplicated().any():
            df = df.sort_values("mp", ascending=False).drop_duplicates(subset=["player", "season"], keep="first")
        all_dfs.append(df)
        time.sleep(0.6)

    player_df = pd.concat(all_dfs, ignore_index=True)

    # Coerce advanced stats to numeric
    for col in ADVANCED_STAT_COLUMNS:
        if col in player_df.columns:
            player_df[col] = pd.to_numeric(player_df[col], errors="coerce")

    # Label MVP
    mvp_merge = mvp_df[["season", "player", "is_mvp"]].drop_duplicates()
    player_df["player_clean"] = player_df["player"].str.strip()
    mvp_merge["player_clean"] = mvp_merge["player"].str.strip()
    player_df = player_df.merge(
        mvp_merge[["season", "player_clean", "is_mvp"]],
        on=["season", "player_clean"],
        how="left",
    )
    player_df["is_mvp"] = player_df["is_mvp"].fillna(0).astype(int)
    return player_df


def fetch_current_season_advanced(season: str = "2025-26", min_games: int = 20) -> pd.DataFrame:
    """Fetch advanced stats for current season for MVP prediction."""
    df = fetch_advanced_stats_br(season)
    if df.empty:
        return df
    if "games" in df.columns:
        df["games"] = pd.to_numeric(df["games"], errors="coerce")
        df = df[df["games"].fillna(0) >= min_games].copy()
    if "team_abbr" in df.columns and df["player"].duplicated().any():
        df = df.sort_values("mp", ascending=False).drop_duplicates(subset=["player", "season"], keep="first")
    for col in ADVANCED_STAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
