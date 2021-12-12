from json import load
from numpy.lib.function_base import median
import pandas as pd
import numpy as np
import audl_advanced_stats as audl
from settings import get_config
from pandas.core.frame import DataFrame

from typing import Tuple

CONFIG = get_config()


def get_all_events(team_abbrev: str = None) -> pd.DataFrame:
    s = audl.Season()
    all_games = s.get_game_info()

    dfs = []
    if team_abbrev:
        home_game_urls = all_games[all_games.home_team == team_abbrev]["url"]
        away_game_urls = all_games[all_games.away_team == team_abbrev]["url"]

        for url in home_game_urls:
            g = audl.Game(url)
            dfs.append(g.get_home_events())

        for url in away_game_urls:
            g = audl.Game(url)
            dfs.append(g.get_away_events())
    else:
        for url in all_games["url"]:
            g = audl.Game(url)
            dfs.append(g.get_home_events())
            dfs.append(g.get_away_events())

    return pd.concat(dfs, axis=0)


def make_grid(x_max, x_min, y_max, y_min, n_grids_x, n_grids_y):
    """ """
    xdims = np.linspace(x_min, x_max, n_grids_x + 1)
    ydims = np.linspace(y_min, y_max, n_grids_y + 1)

    zones = []
    zone_id = 0
    for x_idx in range(n_grids_x):
        for y_idx in range(n_grids_y):
            zones.append(
                {
                    "zone_id": zone_id,
                    "x_min": xdims[x_idx],
                    "x_max": xdims[x_idx + 1],
                    "y_min": ydims[y_idx],
                    "y_max": ydims[y_idx + 1],
                    "x_zone": x_idx,
                    "y_zone": y_idx,
                }
            )
            zone_id += 1

    # Manually add endzone
    zones.append(
        {
            "zone_id": CONFIG.score_zone_id,
            "x_min": xdims[0],
            "x_max": xdims[-1],
            "y_min": 100.01,  # Stoppages often result in disc on the goal line
            "y_max": 125,
            "x_zone": n_grids_x // 2,
            "y_zone": n_grids_y,
        }
    )

    zones.append({"zone_id": CONFIG.turnover_zone_id})

    return pd.DataFrame().from_dict(zones).set_index("zone_id", drop=True)


def add_grid_labels(df: pd.DataFrame, zone_df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits field into an n_x X n_y grid and adds labels to event df.
    """
    for zone_id, zone in zone_df.iterrows():
        df.loc[
            (df.x >= zone.x_min)
            & (df.x < zone.x_max)
            & (df.y >= zone.y_min)
            & (df.y < zone.y_max),
            ["zone_id", "x_zone", "y_zone"],
        ] = (zone_id, zone.x_zone, zone.y_zone)

        df.loc[
            (df.x_after >= zone.x_min)
            & (df.x_after < zone.x_max)
            & (df.y_after >= zone.y_min)
            & (df.y_after < zone.y_max),
            "zone_id_dest",
        ] = zone_id

    return df


def load_data(
    team_abbrev: str = None, keep_all: bool = False
) -> Tuple[DataFrame, DataFrame]:
    cols = [
        "t",
        "r",
        "x",
        "y",
        "game_id",
        "team_id",
        "event_number",
        "event_name",
        "playoffs",
        "period",
        "o_point",
        "possession_number",
        "point_outcome",
        "possession_outcome",
        "possession_outcome_general",
        "r_after",
        "x_after",
        "y_after",
        "t_after",
        "event_name_after",
        "throw_outcome",
        "centering_pass",
        "throw_type",
    ]
    df = get_all_events(team_abbrev)
    df = df[cols]

    # Apply grid labels
    x_max = max(25, df.x.max())
    x_min = min(-25, df.x.min())
    y_max = 100
    y_min = min(0, df.y.min())

    # Make a df of zones
    zone_df = make_grid(
        x_max, x_min, y_max, y_min, CONFIG.grid_size["x"], CONFIG.grid_size["y"]
    )
    df = add_grid_labels(df, zone_df)

    df.loc[df.throw_outcome == "Turnover", "zone_id_dest"] = CONFIG.turnover_zone_id

    # Possession value as int
    df["possession_value"] = (df["possession_outcome_general"] == "Score").astype(
        "int8"
    )

    # Label received pass types
    df.loc[df.centering_pass == True, "throw_type"] = "Throw"
    df["received_throw_type"] = df["throw_type"].shift(1).fillna("Stoppage")
    df.loc[:, "outcome"] = df["throw_outcome"] == "Completion"

    # Drop possessions that are ended by end of quarter
    eoq_events = [
        "End of 1st Quarter",
        "End of 2nd Quarter",
        "End of 3rd Quarter",
        "End of 4th Quarter",
        "End of 1st Overtime",
    ]
    df = df[~df.possession_outcome.isin(eoq_events)]
    if not keep_all:
        df.dropna(
            subset=["zone_id", "zone_id_dest", "r", "throw_outcome"], inplace=True
        )

    return df.sort_values(["game_id", "event_number"]), zone_df


if __name__ == "__main__":
    data, zone_df = load_data()
    print(data.zone_id_dest.value_counts())
