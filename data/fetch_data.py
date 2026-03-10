"""
Fetch NBA data: MVP history (Basketball Reference), player stats & standings (nba_api).
"""
import ssl
import certifi

# Fix SSL certificate verification on macOS (Python.org installs don't use system certs)
_ctx = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: _ctx

import io
import time
import random
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Browser-like User-Agent to reduce 403/429 from Basketball Reference
BR_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

# Minimum delay between BR requests (seconds) to avoid 429
BR_DELAY_MIN = 2.0
BR_DELAY_MAX = 4.0


def _br_delay():
    """Sleep between BR requests to avoid rate limiting."""
    time.sleep(random.uniform(BR_DELAY_MIN, BR_DELAY_MAX))


def _fetch_br_html(url: str, max_retries: int = 4) -> str | None:
    """Fetch HTML from Basketball Reference with retries and backoff on 429."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=BR_HEADERS, timeout=20)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 90))
                wait = min(wait, 120)
                print(f"  Rate limited (429). Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.text
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait = 90
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            if e.response.status_code == 403:
                return None
            raise
        except Exception:
            time.sleep(BR_DELAY_MAX)
            continue
    return None


def fetch_mvp_history() -> pd.DataFrame:
    """
    Scrape NBA MVP winners from Basketball Reference.
    Returns DataFrame with columns: season, player, team_abbr, age, etc.
    """
    url = "https://www.basketball-reference.com/awards/mvp.html"
    html = _fetch_br_html(url)
    if not html:
        return pd.DataFrame()
    tables = pd.read_html(io.StringIO(html))
    df = tables[0]
    # BR MVP table has multi-level columns: level 0 = group names, level 1 = "Season", "Player", "Tm"
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels - 1, -1, -1):
            lev = df.columns.get_level_values(level)
            if "Season" in lev.values or "Player" in lev.values:
                df.columns = lev
                break
        else:
            df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.strip()
    # Map BR column names to our names (only columns that exist)
    rename_map = {
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
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # Ensure we have a season column (BR may use different header or multi-index)
    if "season" not in df.columns:
        for cand in ("Season", "season", "Year", "Yr"):
            if cand in df.columns:
                df["season"] = df[cand].astype(str)
                break
        if "season" not in df.columns and len(df.columns):
            df["season"] = df.iloc[:, 0].astype(str)
    # Ensure we have a player column - must be actual player names, not league (Lg = "NBA")
    if "player" not in df.columns:
        for cand in ("Player", "player", "Name", "NAME"):
            if cand in df.columns:
                df["player"] = df[cand].astype(str).str.strip()
                break
        if "player" not in df.columns and len(df.columns) >= 2:
            for i in range(1, min(6, len(df.columns))):
                col = df.iloc[:, i].astype(str).str.strip()
                sample = col.iloc[0] if len(col) else ""
                if sample and sample not in ("NBA", "ABA") and not sample.isdigit() and len(sample) > 2:
                    df["player"] = col
                    break
            if "player" not in df.columns:
                df["player"] = df.iloc[:, 1].astype(str).str.strip()
    if "Lg" in df.columns:
        df = df[df["Lg"] == "NBA"].copy()
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
        html = _fetch_br_html(url)
        if not html:
            raise FileNotFoundError("No response")
        tables = pd.read_html(io.StringIO(html))
        df = tables[0]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = df.columns.str.strip()
        # Remove repeated header rows
        if "Rk" in df.columns:
            df = df[df["Rk"] != "Rk"].copy()
        df["season"] = season
        if "Player" in df.columns:
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
        if "player" not in df.columns and len(df.columns) >= 2:
            df["player"] = df.iloc[:, 1].astype(str).str.strip()
        _br_delay()
        return df
    except Exception as e:
        print(f"BR advanced failed for {season}: {e}")
        return pd.DataFrame()


def fetch_per_game_stats_br(season: str, min_games: int = 41) -> pd.DataFrame:
    """Fetch per-game PTS, TRB, AST from BR per_game table. One row per player (primary team)."""
    year = season.split("-")[1]
    full_year = "20" + year
    url = f"https://www.basketball-reference.com/leagues/NBA_{full_year}_per_game.html"
    try:
        html = _fetch_br_html(url)
        if not html:
            raise FileNotFoundError("No response")
        tables = pd.read_html(io.StringIO(html))
        df = tables[0]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = df.columns.str.strip()
        if "Rk" in df.columns:
            df = df[df["Rk"] != "Rk"].copy()
        if "Player" in df.columns:
            df["Player"] = df["Player"].str.replace(r"\*", "", regex=True)
        rename = {"Player": "player", "Tm": "team_abbr", "G": "games", "PTS": "ppg", "TRB": "rpg", "AST": "apg"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "player" not in df.columns and len(df.columns) >= 2:
            df["player"] = df.iloc[:, 1].astype(str).str.strip()
        for c in ["games", "ppg", "rpg", "apg"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df["games"].fillna(0) >= min_games].copy()
        df["season"] = season
        if "player" in df.columns and df["player"].duplicated().any():
            df = df.sort_values("games", ascending=False).drop_duplicates(subset=["player", "season"], keep="first")
        cols = [c for c in ["player", "team_abbr", "season", "ppg", "rpg", "apg", "games"] if c in df.columns]
        _br_delay()
        return df[cols].copy() if "player" in cols and "ppg" in cols else pd.DataFrame()
    except Exception as e:
        print(f"BR per_game failed for {season}: {e}")
        return pd.DataFrame()


def fetch_standings_br(season: str) -> pd.DataFrame:
    """Fetch team wins from BR standings. Returns team_abbr, season, team_wins."""
    year = season.split("-")[1]
    full_year = "20" + year
    url = f"https://www.basketball-reference.com/leagues/NBA_{full_year}_standings.html"
    try:
        html = _fetch_br_html(url)
        if not html:
            return pd.DataFrame()
        tables = pd.read_html(io.StringIO(html))
        if not tables:
            return pd.DataFrame()
        df = pd.concat([tables[0], tables[1]], ignore_index=True) if len(tables) >= 2 else tables[0]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = df.columns.str.strip()
        if "W" not in df.columns:
            return pd.DataFrame()
        df["W"] = pd.to_numeric(df["W"], errors="coerce")
        soup = BeautifulSoup(html, "lxml")
        team_abbrs = []
        for a in soup.select("a[href*='/teams/']"):
            href = a.get("href", "")
            parts = href.split("/")
            if len(parts) >= 3 and len(parts[2]) == 3 and parts[2] not in team_abbrs:
                team_abbrs.append(parts[2])
        if len(team_abbrs) >= len(df):
            df = df.head(len(team_abbrs)).copy()
            df["team_abbr"] = team_abbrs[: len(df)]
        elif "Team" in df.columns:
            df["team_abbr"] = df["Team"].astype(str).str[:3].str.upper()
        else:
            return pd.DataFrame()
        df["season"] = season
        df = df.rename(columns={"W": "team_wins"})
        _br_delay()
        return df[["team_abbr", "season", "team_wins"]].dropna(subset=["team_wins"])
    except Exception as e:
        print(f"BR standings failed for {season}: {e}")
        return pd.DataFrame()


def get_season_list(start_year: int = 2000, end_year: int = 2025) -> list[str]:
    """Generate list of season strings: 2000-01, 2001-02, ..., 2024-25."""
    return [f"{y}-{str(y+1)[-2:]}" for y in range(start_year, end_year)]


def _normalize_season(s: str) -> str:
    """Normalize season to YYYY-YY (e.g. 2000-01)."""
    s = str(s).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = a.strip(), b.strip()
        if len(b) == 1 and b.isdigit():
            b = "0" + b
        return f"{a}-{b}"
    return s


def _normalize_player_name(s: str) -> str:
    """Normalize for matching: strip, remove *, unicode to ASCII."""
    import unicodedata
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip().replace("*", "").replace(".", "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().strip()
    return s


# Advanced stat columns used for correlation and modeling (numeric only).
ADVANCED_STAT_COLUMNS = [
    "per", "ts_pct", "fg3ar", "ftr", "orb_pct", "drb_pct", "trb_pct",
    "ast_pct", "stl_pct", "blk_pct", "tov_pct", "usg_pct",
    "ows", "dws", "ws", "ws_per_48", "obpm", "dbpm", "bpm", "vorp",
]
# Regular box-score and team context. Team success is the strongest MVP predictor (top-3 seed).
REGULAR_AND_WINS_COLUMNS = ["ppg", "rpg", "apg", "team_wins", "team_win_rank", "top_team_3", "games"]
# Full feature set: advanced + regular + wins + rank.
FULL_FEATURE_COLUMNS = ADVANCED_STAT_COLUMNS + REGULAR_AND_WINS_COLUMNS


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
    Build historical dataset: advanced stats + per-game (ppg, rpg, apg) + team_wins.
    One row per player-season; combines regular stats, advanced stats, and wins for MVP prediction.
    """
    mvp_df = fetch_mvp_history()
    _br_delay()
    if mvp_df.empty or "season" not in mvp_df.columns:
        raise RuntimeError(
            "Could not fetch MVP history from Basketball Reference (rate limited, 403, or page changed). "
            "Wait a few minutes and try again, or use cached data: keep data/historical_advanced.csv and re-run."
        )
    seasons = get_season_list(start_year=start_year, end_year=end_year)
    mvp_df["season"] = mvp_df["season"].apply(_normalize_season)
    mvp_df = mvp_df[mvp_df["season"].isin(seasons)].copy()
    mvp_df["is_mvp"] = 1

    all_dfs = []
    for season in seasons:
        df = fetch_advanced_stats_br(season)
        if df.empty:
            continue
        if "games" in df.columns:
            df["games"] = pd.to_numeric(df["games"], errors="coerce")
            df = df[df["games"].fillna(0) >= min_games].copy()
        if "team_abbr" in df.columns and "player" in df.columns and df["player"].duplicated().any():
            df = df.sort_values("mp", ascending=False).drop_duplicates(subset=["player", "season"], keep="first")

        per_game = fetch_per_game_stats_br(season, min_games=min_games)
        if not per_game.empty and "ppg" in per_game.columns:
            pg = per_game[["player", "season", "ppg", "rpg", "apg"]].copy()
            pg["_pclean"] = pg["player"].apply(_normalize_player_name)
            df["_pclean"] = df["player"].apply(_normalize_player_name)
            df = df.merge(pg[["season", "_pclean", "ppg", "rpg", "apg"]], on=["season", "_pclean"], how="left")
            df = df.drop(columns=["_pclean"], errors="ignore")
        time.sleep(0.3)

        standings = fetch_standings_br(season)
        if not standings.empty and "team_abbr" in df.columns:
            df = df.merge(standings, on=["team_abbr", "season"], how="left")
        time.sleep(0.3)

        all_dfs.append(df)
        time.sleep(0.6)

    player_df = pd.concat(all_dfs, ignore_index=True)

    # Team wins rank (1 = most wins in league that season). MVP voters strongly favor top-3 seeds.
    if "team_wins" in player_df.columns:
        rank_df = (
            player_df[["season", "team_abbr", "team_wins"]]
            .drop_duplicates()
            .sort_values(["season", "team_wins"], ascending=[True, False])
        )
        rank_df["team_win_rank"] = rank_df.groupby("season")["team_wins"].rank(method="min", ascending=False).astype(int)
        player_df = player_df.merge(rank_df[["season", "team_abbr", "team_win_rank"]], on=["season", "team_abbr"], how="left")
        player_df["top_team_3"] = (player_df["team_win_rank"] <= 3).astype(int)

    for col in ADVANCED_STAT_COLUMNS:
        if col in player_df.columns:
            player_df[col] = pd.to_numeric(player_df[col], errors="coerce")
    for col in REGULAR_AND_WINS_COLUMNS:
        if col in player_df.columns:
            player_df[col] = pd.to_numeric(player_df[col], errors="coerce")

    mvp_merge = mvp_df[["season", "player", "is_mvp"]].drop_duplicates()
    mvp_merge["player_clean"] = mvp_merge["player"].apply(_normalize_player_name)
    mvp_merge["season"] = mvp_merge["season"].apply(_normalize_season)
    player_df["player_clean"] = player_df["player"].apply(_normalize_player_name)
    player_df["season"] = player_df["season"].astype(str).apply(_normalize_season)
    player_df = player_df.merge(
        mvp_merge[["season", "player_clean", "is_mvp"]],
        on=["season", "player_clean"],
        how="left",
    )
    player_df["is_mvp"] = player_df["is_mvp"].fillna(0).astype(int)
    return player_df


def fetch_current_season_advanced(season: str = "2025-26", min_games: int = 20) -> pd.DataFrame:
    """Fetch advanced + per-game stats + team wins for current season (same formula as historical)."""
    df = fetch_advanced_stats_br(season)
    if df.empty:
        return df
    if "games" in df.columns:
        df["games"] = pd.to_numeric(df["games"], errors="coerce")
        df = df[df["games"].fillna(0) >= min_games].copy()
    if "team_abbr" in df.columns and df["player"].duplicated().any():
        df = df.sort_values("mp", ascending=False).drop_duplicates(subset=["player", "season"], keep="first")
    per_game = fetch_per_game_stats_br(season, min_games=min_games)
    if not per_game.empty and "ppg" in per_game.columns:
        pg = per_game[["player", "season", "ppg", "rpg", "apg"]].copy()
        pg["_pclean"] = pg["player"].apply(_normalize_player_name)
        df["_pclean"] = df["player"].apply(_normalize_player_name)
        df = df.merge(pg[["season", "_pclean", "ppg", "rpg", "apg"]], on=["season", "_pclean"], how="left")
        df = df.drop(columns=["_pclean"], errors="ignore")
    standings = fetch_standings_br(season)
    if not standings.empty and "team_abbr" in df.columns:
        df = df.merge(standings, on=["team_abbr", "season"], how="left")
        # Team wins rank and top-3 flag (MVP almost always from top-3 team)
        if "team_wins" in df.columns:
            rank_df = df[["season", "team_abbr", "team_wins"]].drop_duplicates().sort_values("team_wins", ascending=False)
            rank_df["team_win_rank"] = rank_df["team_wins"].rank(method="min", ascending=False).astype(int)
            df = df.merge(rank_df[["season", "team_abbr", "team_win_rank"]], on=["season", "team_abbr"], how="left")
            df["top_team_3"] = (df["team_win_rank"] <= 3).astype(int)
    for col in ADVANCED_STAT_COLUMNS + REGULAR_AND_WINS_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
