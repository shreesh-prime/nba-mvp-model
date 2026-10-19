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

from season_utils import current_season

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
            # BR serves UTF-8 but omits the charset header, so requests falls back to latin-1
            # and mangles accented names (Jokić -> JokiÄ\x87).
            r.encoding = "utf-8"
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


def _find_br_table(html: str, table_id: str):
    """Return the BeautifulSoup <table id=table_id>, also searching inside HTML comments (BR hides some tables)."""
    from bs4 import Comment
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=table_id)
    if table is not None:
        return table
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if f'id="{table_id}"' in comment:
            table = BeautifulSoup(comment, "lxml").find("table", id=table_id)
            if table is not None:
                return table
    return None


MVP_VOTING_COLUMNS = ["season", "player", "team", "share", "rank", "player_id"]


def fetch_mvp_voting_br(season: str) -> pd.DataFrame:
    """
    Scrape MVP voting for one season ("2023-24") from BR awards_YYYY.html (table id "mvp").
    Returns MVP_VOTING_COLUMNS; rank is the integer voting place (ties like "5T" -> 5).
    """
    year_end = int(str(season).split("-")[0]) + 1
    url = f"https://www.basketball-reference.com/awards/awards_{year_end}.html"
    html = _fetch_br_html(url)
    if not html:
        return pd.DataFrame(columns=MVP_VOTING_COLUMNS)
    table = _find_br_table(html, "mvp")
    if table is None or table.find("tbody") is None:
        return pd.DataFrame(columns=MVP_VOTING_COLUMNS)
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        cells = {c.get("data-stat"): c for c in tr.find_all(["th", "td"])}
        if "player" not in cells or "award_share" not in cells:
            continue
        rank_txt = cells["rank"].get_text(strip=True).rstrip("T") if "rank" in cells else ""
        rows.append({
            "season": season,
            "player": cells["player"].get_text(strip=True),
            "team": cells["team_id"].get_text(strip=True) if "team_id" in cells else "",
            "share": pd.to_numeric(cells["award_share"].get_text(strip=True), errors="coerce"),
            "rank": pd.to_numeric(rank_txt, errors="coerce"),
            "player_id": cells["player"].get("data-append-csv", ""),
        })
    return pd.DataFrame(rows, columns=MVP_VOTING_COLUMNS)


def _fetch_mvp_voting_external(start_year: int, end_year: int) -> pd.DataFrame:
    """Fallback: MVP award shares from the GitHub dataset derived from BR."""
    awards = _load_external_csv(EXTERNAL_AWARD_SHARES_URL)
    awards = awards[awards["award"].astype(str).str.lower() == "nba mvp"].copy()
    awards["season"] = pd.to_numeric(awards["season"], errors="coerce")
    awards = awards[(awards["season"] >= start_year + 1) & (awards["season"] <= end_year)]
    out = pd.DataFrame({
        "season": awards["season"].astype(int).apply(_external_year_end_to_season_str),
        "player": awards["player"],
        "team": awards["tm"],
        "share": pd.to_numeric(awards["share"], errors="coerce"),
        "player_id": awards["player_id"],
    })
    out["rank"] = out.groupby("season")["share"].rank(method="min", ascending=False).astype(int)
    return out[MVP_VOTING_COLUMNS]


def build_mvp_voting_dataset(
    start_year: int = 1995,
    end_year: int = 2025,
    cache_path: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    MVP voting (award share + voting rank) for seasons start_year-(start_year+1) .. (end_year-1)-end_year.
    Scrapes BR awards pages politely (one request per season); falls back to the external dataset.
    Cached to data/mvp_voting_{start_year+1}_{end_year}.csv by default.
    """
    import os
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(__file__), f"mvp_voting_{start_year + 1}_{end_year}.csv")
    if use_cache and os.path.isfile(cache_path):
        return pd.read_csv(cache_path)
    frames = []
    for season in get_season_list(start_year=start_year, end_year=end_year):
        part = fetch_mvp_voting_br(season)
        if part.empty:
            print(f"  MVP voting unavailable from BR for {season}")
        else:
            frames.append(part)
        _br_delay()
    voting = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MVP_VOTING_COLUMNS)
    expected = len(get_season_list(start_year=start_year, end_year=end_year))
    if voting["season"].nunique() < expected:
        print("BR MVP voting incomplete; using external award-share dataset.")
        voting = _fetch_mvp_voting_external(start_year, end_year)
    voting.to_csv(cache_path, index=False)
    print(f"Saved MVP voting ({len(voting)} rows) to {cache_path}")
    return voting


def merge_mvp_voting(df: pd.DataFrame, voting: pd.DataFrame) -> pd.DataFrame:
    """
    Add mvp_share (0 for non-receivers) and mvp_vote_rank (NaN for non-receivers) to player-seasons.
    Matches on season + player_id when available, else season + accent-insensitive normalized name.
    """
    out = df.copy()
    v = voting.copy()
    v["_pclean"] = v["player"].apply(_normalize_player_name).str.lower()
    out["_pclean"] = out["player"].apply(_normalize_player_name).str.lower()
    out["mvp_share"] = float("nan")
    out["mvp_vote_rank"] = float("nan")
    if "player_id" in out.columns and "player_id" in v.columns:
        by_id = v.dropna(subset=["player_id"]).set_index(["season", "player_id"])
        idx = pd.MultiIndex.from_frame(out[["season", "player_id"]].astype(str))
        out["mvp_share"] = by_id["share"].reindex(idx).to_numpy()
        out["mvp_vote_rank"] = by_id["rank"].reindex(idx).to_numpy()
    missing = out["mvp_share"].isna()
    if missing.any():
        by_name = v.drop_duplicates(["season", "_pclean"]).set_index(["season", "_pclean"])
        idx = pd.MultiIndex.from_frame(out.loc[missing, ["season", "_pclean"]])
        out.loc[missing, "mvp_share"] = by_name["share"].reindex(idx).to_numpy()
        out.loc[missing, "mvp_vote_rank"] = by_name["rank"].reindex(idx).to_numpy()
    out["mvp_share"] = out["mvp_share"].fillna(0.0)
    return out.drop(columns=["_pclean"])


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
    # Basketball-Reference uses the *ending* year in the URL.
    # Example: season "1995-96" => ending year 1996 => NBA_1996_advanced.html.
    start_year = int(str(season).split("-")[0])
    full_year = start_year + 1
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
    # Basketball-Reference uses the *ending* year in the URL.
    start_year = int(str(season).split("-")[0])
    full_year = start_year + 1
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
    # Basketball-Reference uses the *ending* year in the URL.
    start_year = int(str(season).split("-")[0])
    full_year = start_year + 1
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
# Year-over-year improvement features: "<stat>_change" = this season minus the player's previous season.
YOY_FEATURE_SOURCES = ["bpm"]
YOY_FEATURE_COLUMNS = [f"{c}_change" for c in YOY_FEATURE_SOURCES]
# Combined/traded-player rows have no single team record.
MULTI_TEAM_ABBRS = {"TOT", "2TM", "3TM", "4TM", "5TM"}


def add_team_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Re)compute team_win_rank (1 = most wins among *teams* that season, ties share the best rank)
    and top_team_3 (team_win_rank <= 3) from team_wins. Multi-team rows get NaN rank and top_team_3 = 0.
    """
    out = df.drop(columns=["team_win_rank", "top_team_3"], errors="ignore")
    if "team_wins" not in out.columns or "team_abbr" not in out.columns:
        return out
    out["team_wins"] = pd.to_numeric(out["team_wins"], errors="coerce")
    teams = (
        out.loc[~out["team_abbr"].isin(MULTI_TEAM_ABBRS), ["season", "team_abbr", "team_wins"]]
        .dropna()
        .drop_duplicates(subset=["season", "team_abbr"])
    )
    teams["team_win_rank"] = teams.groupby("season")["team_wins"].rank(method="min", ascending=False)
    out = out.merge(teams[["season", "team_abbr", "team_win_rank"]], on=["season", "team_abbr"], how="left")
    out["top_team_3"] = (out["team_win_rank"] <= 3).astype(int)
    return out


def _prev_season(season: str) -> str:
    """'2024-25' -> '2023-24'."""
    y = int(str(season).split("-")[0])
    return f"{y - 1}-{str(y)[-2:]}"


def add_yoy_features(df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Add YOY_FEATURE_COLUMNS (e.g. bpm_change) = stat this season minus the same player's stat in the
    immediately preceding season, looked up in df itself plus optional `history`. Players without a
    qualifying previous season get 0. Matches on player_id when available, else normalized name.
    """
    out = df.copy()
    prior = pd.concat([history, df], ignore_index=True) if history is not None and not history.empty else df
    use_id = (
        "player_id" in out.columns and "player_id" in prior.columns
        and out["player_id"].notna().all() and prior["player_id"].notna().all()
    )
    key = "player_id" if use_id else "_pkey"
    if not use_id:
        out["_pkey"] = out["player"].apply(_normalize_player_name).str.lower()
        prior = prior.assign(_pkey=prior["player"].apply(_normalize_player_name).str.lower())
    out["_prev_season"] = out["season"].apply(_prev_season)
    for stat in YOY_FEATURE_SOURCES:
        if stat not in out.columns or stat not in prior.columns:
            continue
        p = prior[["season", key, stat]].copy()
        p[stat] = pd.to_numeric(p[stat], errors="coerce")
        p = p.dropna().drop_duplicates(subset=["season", key])
        p = p.rename(columns={"season": "_prev_season", stat: "_prev_val"})
        out = out.merge(p, on=["_prev_season", key], how="left")
        out[f"{stat}_change"] = (pd.to_numeric(out[stat], errors="coerce") - out["_prev_val"]).fillna(0.0)
        out = out.drop(columns=["_prev_val"])
    return out.drop(columns=["_prev_season", "_pkey"], errors="ignore")


def _load_prior_season_stats(season: str, min_games: int = 41) -> pd.DataFrame:
    """Previous season's player stats for YoY features: from the cached historical CSVs if present, else BR."""
    import glob
    import os
    prev = _prev_season(season)
    for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "historical_advanced*.csv"))):
        cached = pd.read_csv(path)
        cached = cached[cached["season"] == prev]
        if not cached.empty:
            return cached
    prior = fetch_advanced_stats_br(prev)
    if prior.empty:
        return prior
    prior["games"] = pd.to_numeric(prior.get("games"), errors="coerce")
    return prior[prior["games"].fillna(0) >= min_games].copy()


# ----------------------------
# External-data fallback (no BR scraping)
# ----------------------------
# Basketball-Reference blocks automated requests here (Cloudflare 403). To keep the
# MVP modeling pipeline working, we fall back to a public GitHub-hosted dataset
# derived from Basketball-Reference tables.
#
# Dataset:
#   https://github.com/sumitrodatta/bball-reference-datasets
#
# We only use season-level aggregates, so the behavior matches the original model.
EXTERNAL_BASE = "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/Data"
EXTERNAL_ADVANCED_URL = f"{EXTERNAL_BASE}/Advanced.csv"
EXTERNAL_PER_GAME_URL = f"{EXTERNAL_BASE}/Player%20Per%20Game.csv"
EXTERNAL_TEAM_SUMMARIES_URL = f"{EXTERNAL_BASE}/Team%20Summaries.csv"
EXTERNAL_AWARD_SHARES_URL = f"{EXTERNAL_BASE}/Player%20Award%20Shares.csv"


def _external_year_end_to_season_str(year_end: int) -> str:
    """Convert ending year (e.g. 2025) => '2024-25'."""
    start_year = int(year_end) - 1
    end_short = str(int(year_end))[-2:]
    return f"{start_year}-{end_short}"


def _season_str_to_external_year_end(season: str) -> int:
    """Convert '2024-25' => ending year 2025."""
    start_year = int(str(season).split("-")[0])
    return start_year + 1


def _load_external_csv(url: str) -> pd.DataFrame:
    """Load external CSV with pandas, relying on ssl context setup above."""
    return pd.read_csv(url)


def build_historical_dataset_external(
    start_year: int = 1995,
    end_year: int = 2025,
    min_games: int = 41,
) -> pd.DataFrame:
    """
    Build the same training dataset as `build_advanced_historical_dataset`, but using
    external CSVs instead of scraping Basketball-Reference.

    The external dataset uses `season` as the ending year (e.g. 2025 for 2024-25).
    Our `start_year/end_year` inputs follow the same convention as the original code:
      - start_year=1995,end_year=2025 => seasons '1995-96'..'2024-25' => ending years 1996..2025
    """
    year_end_start = int(start_year) + 1
    year_end_end = int(end_year)

    adv = _load_external_csv(EXTERNAL_ADVANCED_URL)
    per = _load_external_csv(EXTERNAL_PER_GAME_URL)
    teams = _load_external_csv(EXTERNAL_TEAM_SUMMARIES_URL)
    awards = _load_external_csv(EXTERNAL_AWARD_SHARES_URL)

    # Keep NBA only; dataset includes other leagues/competitions.
    adv = adv[adv["lg"] == "NBA"].copy()
    per = per[per["lg"] == "NBA"].copy()
    teams = teams[teams["lg"] == "NBA"].copy()
    awards = awards[awards["award"].astype(str).str.lower() == "nba mvp"].copy()

    # Filter to training window by ending-year season.
    adv["season"] = pd.to_numeric(adv["season"], errors="coerce")
    per["season"] = pd.to_numeric(per["season"], errors="coerce")
    teams["season"] = pd.to_numeric(teams["season"], errors="coerce")
    awards["season"] = pd.to_numeric(awards["season"], errors="coerce")

    adv = adv[(adv["season"] >= year_end_start) & (adv["season"] <= year_end_end)]
    per = per[(per["season"] >= year_end_start) & (per["season"] <= year_end_end)]
    teams = teams[(teams["season"] >= year_end_start) & (teams["season"] <= year_end_end)]
    awards = awards[(awards["season"] >= year_end_start) & (awards["season"] <= year_end_end)]

    # --- Advanced stats (one row per player-season) ---
    adv = adv.rename(
        columns={
            "player": "player",
            "team": "team_abbr",
            "g": "games",
            "ts_percent": "ts_pct",
            "x3p_ar": "fg3ar",
            "f_tr": "ftr",
            "orb_percent": "orb_pct",
            "drb_percent": "drb_pct",
            "trb_percent": "trb_pct",
            "ast_percent": "ast_pct",
            "stl_percent": "stl_pct",
            "blk_percent": "blk_pct",
            "tov_percent": "tov_pct",
            "usg_percent": "usg_pct",
            "ws_48": "ws_per_48",
        }
    )
    adv["games"] = pd.to_numeric(adv["games"], errors="coerce")
    if "mp" in adv.columns:
        adv["mp"] = pd.to_numeric(adv["mp"], errors="coerce")
    # Drop duplicated player-season rows (trades) by max MP.
    sort_cols = [c for c in ["season", "player_id", "mp", "games"] if c in adv.columns]
    if "mp" in adv.columns:
        adv = adv.sort_values(["season", "player_id", "mp"], ascending=[True, True, False])
    else:
        adv = adv.sort_values(["season", "player_id", "games"], ascending=[True, True, False])
    adv = adv.drop_duplicates(subset=["season", "player_id"], keep="first")
    adv = adv[adv["games"].fillna(0) >= min_games].copy()

    # --- Per-game stats ---
    per = per.rename(
        columns={
            "team": "team_abbr",
            "g": "games",
            "pts_per_game": "ppg",
            "trb_per_game": "rpg",
            "ast_per_game": "apg",
        }
    )
    per["games"] = pd.to_numeric(per["games"], errors="coerce")
    if "mp_per_game" in per.columns:
        per["mp_per_game"] = pd.to_numeric(per["mp_per_game"], errors="coerce")
        per = per.sort_values(["season", "player_id", "mp_per_game"], ascending=[True, True, False])
    else:
        per = per.sort_values(["season", "player_id", "games"], ascending=[True, True, False])
    per = per.drop_duplicates(subset=["season", "player_id"], keep="first")
    per = per[per["games"].fillna(0) >= min_games].copy()
    per_small = per[["season", "player_id", "ppg", "rpg", "apg"]].copy()

    player_df = adv.merge(per_small, on=["season", "player_id"], how="left")
    # Some players may have advanced stats without enough per-game games.
    # Keep the row and let modeling drop NaNs.

    # --- Team wins (for contextual features) ---
    teams = teams.rename(columns={"abbreviation": "team_abbr", "w": "team_wins"})
    teams["team_wins"] = pd.to_numeric(teams["team_wins"], errors="coerce")
    teams = teams[["season", "team_abbr", "team_wins"]].copy()
    player_df = player_df.merge(teams, on=["season", "team_abbr"], how="left")

    player_df = add_team_rank(player_df)

    # --- MVP labels ---
    awards = awards[awards["winner"] == True].copy()
    awards["player_id"] = awards["player_id"].astype(str)
    mvp_df = awards[["season", "player_id"]].drop_duplicates().copy()
    mvp_df["is_mvp"] = 1

    player_df["player_clean"] = player_df["player"].apply(_normalize_player_name)
    mvp_df["player_clean"] = mvp_df["player_id"].apply(lambda _: "")  # placeholder; not needed for merges

    player_df = player_df.merge(mvp_df[["season", "player_id", "is_mvp"]], on=["season", "player_id"], how="left")
    player_df["is_mvp"] = player_df["is_mvp"].fillna(0).astype(int)

    # Convert external ending-year season into our 'YYYY-YY' format.
    player_df["season"] = player_df["season"].apply(_external_year_end_to_season_str)

    # Keep only columns we care about (plus identifiers used by the model/app).
    cols_needed = set(FULL_FEATURE_COLUMNS) | {"season", "player", "player_id", "player_clean", "team_abbr", "is_mvp"}
    keep_cols = [c for c in player_df.columns if c in cols_needed]
    player_df = player_df[keep_cols].copy()

    return player_df


def fetch_current_season_advanced_external(
    season: str | None = None,
    min_games: int = 20,
) -> pd.DataFrame:
    """Fetch current season advanced+per-game+team_wins using external CSVs."""
    season = season or current_season()
    year_end = _season_str_to_external_year_end(season)

    adv = _load_external_csv(EXTERNAL_ADVANCED_URL)
    per = _load_external_csv(EXTERNAL_PER_GAME_URL)
    teams = _load_external_csv(EXTERNAL_TEAM_SUMMARIES_URL)

    adv = adv[adv["lg"] == "NBA"].copy()
    per = per[per["lg"] == "NBA"].copy()
    teams = teams[teams["lg"] == "NBA"].copy()

    adv["season"] = pd.to_numeric(adv["season"], errors="coerce")
    per["season"] = pd.to_numeric(per["season"], errors="coerce")
    teams["season"] = pd.to_numeric(teams["season"], errors="coerce")

    adv = adv[adv["season"] == year_end].copy()
    per = per[per["season"] == year_end].copy()
    teams = teams[teams["season"] == year_end].copy()

    adv = adv.rename(
        columns={
            "team": "team_abbr",
            "g": "games",
            "ts_percent": "ts_pct",
            "x3p_ar": "fg3ar",
            "f_tr": "ftr",
            "orb_percent": "orb_pct",
            "drb_percent": "drb_pct",
            "trb_percent": "trb_pct",
            "ast_percent": "ast_pct",
            "stl_percent": "stl_pct",
            "blk_percent": "blk_pct",
            "tov_percent": "tov_pct",
            "usg_percent": "usg_pct",
            "ws_48": "ws_per_48",
        }
    )
    adv["games"] = pd.to_numeric(adv["games"], errors="coerce")
    if "mp" in adv.columns:
        adv["mp"] = pd.to_numeric(adv["mp"], errors="coerce")
        adv = adv.sort_values(["player_id", "mp"], ascending=[True, False])
    adv = adv.drop_duplicates(subset=["player_id"], keep="first")
    adv = adv[adv["games"].fillna(0) >= min_games].copy()

    per = per.rename(
        columns={
            "team": "team_abbr",
            "g": "games",
            "pts_per_game": "ppg",
            "trb_per_game": "rpg",
            "ast_per_game": "apg",
        }
    )
    per["games"] = pd.to_numeric(per["games"], errors="coerce")
    if "mp_per_game" in per.columns:
        per["mp_per_game"] = pd.to_numeric(per["mp_per_game"], errors="coerce")
        per = per.sort_values(["player_id", "mp_per_game"], ascending=[True, False])
    per = per.drop_duplicates(subset=["player_id"], keep="first")
    per = per[per["games"].fillna(0) >= min_games].copy()
    per_small = per[["player_id", "ppg", "rpg", "apg"]].copy()

    player_df = adv.merge(per_small, on=["player_id"], how="left")

    teams = teams.rename(columns={"abbreviation": "team_abbr", "w": "team_wins"})
    teams["team_wins"] = pd.to_numeric(teams["team_wins"], errors="coerce")
    teams = teams[["team_abbr", "team_wins"]].copy()
    player_df = player_df.merge(teams, on=["team_abbr"], how="left")

    player_df["player_clean"] = player_df["player"].apply(_normalize_player_name)
    player_df["season"] = _external_year_end_to_season_str(year_end)
    player_df = add_team_rank(player_df)

    cols_needed = set(FULL_FEATURE_COLUMNS) | {"season", "player", "player_id", "player_clean", "team_abbr"}
    keep_cols = [c for c in player_df.columns if c in cols_needed]
    player_df = player_df[keep_cols].copy()
    return player_df


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
        # Basketball-Reference is blocked here (Cloudflare 403), so fall back to
        # a public dataset derived from the same tables.
        print("Basketball-Reference scraping failed; using external dataset fallback.")
        return build_historical_dataset_external(start_year=start_year, end_year=end_year, min_games=min_games)
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
    player_df = add_team_rank(player_df)

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


def fetch_current_season_advanced(season: str | None = None, min_games: int = 20) -> pd.DataFrame:
    """Fetch advanced + per-game stats + team wins for current season (same formula as historical)."""
    season = season or current_season()
    df = fetch_advanced_stats_br(season)
    if df.empty:
        # Basketball-Reference is blocked here; use the external dataset instead.
        df = fetch_current_season_advanced_external(season=season, min_games=min_games)
        return add_yoy_features(df, _load_prior_season_stats(season)) if not df.empty else df
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
        df = add_team_rank(df)
    for col in ADVANCED_STAT_COLUMNS + REGULAR_AND_WINS_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return add_yoy_features(df, _load_prior_season_stats(season))
