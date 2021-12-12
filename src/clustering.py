import numpy as np
from matplotlib.figure import Figure
from numpy.random.mtrand import f
from pandas.core.frame import DataFrame
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib.pyplot import Figure

import audl_advanced_stats as audl

from settings import get_config

CONFIG = get_config()

FEATS = [
    # "playerid",
    # "name",
    # "team",
    # "opponent",
    # "game_date",
    # "year",
    # "playoffs",
    # "games",
    # "total_points",
    # "o_points",
    # "o_point_scores",
    # "o_point_noturns",
    # "d_points",
    # "d_point_scores",
    # "d_point_turns",
    # "points_team",
    # "o_points_team",
    # "d_points_team",
    # "total_possessions",
    # "o_possessions",
    # "o_possession_scores",
    # "d_possessions",
    # "d_possession_scores_allowed",
    # "possessions_team",
    # "o_possessions_team",
    # "d_possessions_team",
    # "minutes_played",
    # "throwaways",
    # "completions",
    # "receptions",
    # "drops",
    # "stalls",
    # "assists",
    # "goals",
    # "blocks",
    # "callahans",
    # "throwaways_team",
    # "drops_team",
    # "completions_team",
    # "stalls_team",
    # "goals_team",
    # "blocks_team",
    # "callahans_team",
    # "turnovers",
    # "plus_minus",
    # "throw_attempts",
    # "catch_attempts",
    # "completions_dish",
    # "completions_dump",
    # "completions_huck",
    # "completions_swing",
    # "completions_throw",
    # "throwaways_dish",
    # "throwaways_dump",
    # "throwaways_huck",
    # "throwaways_swing",
    # "throwaways_throw",
    # "receptions_dish",
    # "receptions_dump",
    # "receptions_huck",
    # "receptions_swing",
    # "receptions_throw",
    # "attempts_dish",
    # "attempts_dump",
    # "attempts_huck",
    # "attempts_swing",
    # "attempts_throw",
    # "xyards_throwing",
    # "xyards_throwing_center",
    # "yyards_throwing",
    # "yyards_throwing_center",
    # "yyards_raw_throwing",
    # "yyards_raw_throwing_center",
    # "yards_throwing",
    # "yards_throwing_center",
    # "yards_raw_throwing",
    # "yards_raw_throwing_center",
    "x_after_throwing",
    # "x_after_throwing_center",
    "y_after_throwing",
    # "y_after_throwing_center",
    "x_throwing",
    # "x_throwing_center",
    "y_throwing",
    # "y_throwing_center",
    # "xyards_throwing_total",
    # "yyards_throwing_total",
    # "yyards_raw_throwing_total",
    # "yards_throwing_total",
    # "yards_raw_throwing_total",
    # "xyards_receiving",
    # "xyards_receiving_center",
    # "yyards_receiving",
    # "yyards_receiving_center",
    # "yyards_raw_receiving",
    # "yyards_raw_receiving_center",
    # "yards_receiving",
    # "yards_receiving_center",
    # "yards_raw_receiving",
    # "yards_raw_receiving_center",
    # "xyards_receiving_total",
    # "yyards_receiving_total",
    # "yyards_raw_receiving_total",
    # "yards_receiving_total",
    # "yards_raw_receiving_total",
    # "xyards_team",
    # "yyards_team",
    # "yyards_raw_team",
    # "yards_team",
    # "yards_raw_team",
    # "xyards_throwaway",
    # "yyards_throwaway",
    # "yyards_raw_throwaway",
    "yards_throwaway",
    # "yards_raw_throwaway",
    # "xyards_total",
    # "yyards_total",
    # "yards_total",
    # "yyards_raw_total",
    # "yards_raw_total",
    # "xyards_center",
    # "yyards_center",
    # "yards_center",
    # "yyards_raw_center",
    # "yards_raw_center",
    # "xyards",
    # "yyards",
    # "yards",
    # "yyards_raw",
    # "yards_raw",
    # "throwaways_pp",
    # "drops_pp",
    # "stalls_pp",
    # "completions_pp",
    # "receptions_pp",
    # "turnovers_pp",
    # "assists_pp",
    # "goals_pp",
    # "blocks_pp",
    "yx_ratio_throwing",
    # "yx_ratio_receiving",
    # "xyards_throwing_pp",
    # "yyards_throwing_pp",
    # "yards_throwing_pp",
    # "yyards_raw_throwing_pp",
    # "yards_raw_throwing_pp",
    # "xyards_receiving_pp",
    # "yyards_receiving_pp",
    # "yards_receiving_pp",
    # "yyards_raw_receiving_pp",
    # "yards_raw_receiving_pp",
    # "xyards_pp",
    # "yyards_pp",
    # "yards_pp",
    # "yyards_raw_pp",
    # "yards_raw_pp",
    "xyards_throwing_percompletion",
    "yyards_throwing_percompletion",
    # "yards_throwing_percompletion",
    # "yyards_raw_throwing_percompletion",
    # "yards_raw_throwing_percompletion",
    # "xyards_receiving_perreception",
    # "yyards_receiving_perreception",
    # "yards_receiving_perreception",
    # "yyards_raw_receiving_perreception",
    # "yards_raw_receiving_perreception",
    # "xyards_throwing_perthrowaway",
    # "yyards_throwing_perthrowaway",
    # "yards_throwing_perthrowaway",
    # "yyards_raw_throwing_perthrowaway",
    # "yards_raw_throwing_perthrowaway",
    # "o_point_score_pct",
    # "d_point_score_pct",
    # "o_point_noturn_pct",
    # "d_point_turn_pct",
    # "xyards_total_pct",
    # "yyards_total_pct",
    # "yards_total_pct",
    # "yyards_raw_total_pct",
    # "yards_raw_total_pct",
    # "xyards_throwing_total_pct",
    # "yyards_throwing_total_pct",
    # "yards_throwing_total_pct",
    # "yyards_raw_throwing_total_pct",
    # "yards_raw_throwing_total_pct",
    # "xyards_receiving_total_pct",
    # "yyards_receiving_total_pct",
    # "yards_receiving_total_pct",
    # "yyards_raw_receiving_total_pct",
    # "yards_raw_receiving_total_pct",
    "completion_pct",
    # "reception_pct",
    # "assists_perthrowattempt",
    # "goals_percatchattempt",
    # "points_team_pct",
    # "o_points_team_pct",
    # "d_points_team_pct",
    # "possessions_team_pct",
    # "o_possessions_team_pct",
    # "d_possessions_team_pct",
    # "throwaways_team_pct",
    # "drops_team_pct",
    # "receptions_team_pct",
    # "completions_team_pct",
    # "stalls_team_pct",
    # "assists_team_pct",
    # "goals_team_pct",
    # "blocks_team_pct",
    # "callahans_team_pct",
    # "o_possession_score_pct",
    # "d_possession_score_allowed_pct",
    "completion_dish_pct",
    "attempts_dish_pct",
    # "receptions_dish_pct",
    "completion_dump_pct",
    "attempts_dump_pct",
    # "receptions_dump_pct",
    "completion_huck_pct",
    "attempts_huck_pct",
    # "receptions_huck_pct",
    "completion_swing_pct",
    "attempts_swing_pct",
    # "receptions_swing_pct",
    "completion_throw_pct",
    "attempts_throw_pct",
    # "receptions_throw_pct",
]


def add_cluster_labels(
    data: DataFrame,
    stats_data=DataFrame,
    estimator=KMeans,
    n_clusters: int = CONFIG.n_clusters,
    player_id: str = "r",
):

    feats = stats_data.columns[7:-1]
    feats = FEATS
    data_prep_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(verbose=2)),
            ("scaler", StandardScaler()),
        ]
    )

    X = (
        stats_data.groupby("playerid")[feats]
        .mean()
        .replace(np.inf, np.nan)
        .replace(np.NINF, np.nan)
    )
    pipe = Pipeline(
        steps=[
            ("pca", PCA(CONFIG.pca_components)),
            (
                "cluster",
                estimator(n_clusters=n_clusters, random_state=CONFIG.random_state),
            ),
        ]
    )
    X["cluster_id"] = pipe.fit_predict(data_prep_pipe.fit_transform(X))

    X.set_index(X.index.astype("float64"), inplace=True)
    data = data.join(X["cluster_id"], on="r")

    # Join on destination as well
    X.loc[:, "cluster_id_dest"] = X["cluster_id"]
    data = data.join(X["cluster_id_dest"], on="r_after")
    return data


def k_elbow_plot(
    data: DataFrame, estimator: KMeans, min_max_k: Tuple[int, int]
) -> Figure:
    """
    Make an elbow plot
    """

    scores = []
    k_range = range(min_max_k[0], min_max_k[1])
    for k in k_range:
        m = estimator(n_clusters=k, random_state=88)
        X = data.copy()
        labels = m.fit_predict(data)
        scores.append(silhouette_score(X=X, labels=labels))

    plt.plot(k_range, scores)
    plt.xlabel("k clusters")
    plt.ylabel("silhoutte coefficient")
    plt.show()


def eps_elbow_plot(data: DataFrame, estimator: DBSCAN, eps_vals: List[float]) -> Figure:
    """
    Make an elbow plot
    """

    scores = []
    for eps in eps_vals:
        m = estimator(eps=eps)
        labels = m.fit_predict(X=data)
        scores.append(silhouette_score(X=data, labels=labels))

    plt.plot(eps_vals, scores)
    plt.xlabel("k clusters")
    plt.ylabel("silhoutte coefficient")


if __name__ == "__main__":
    s = audl.Season()
    df = s.get_player_stats_by_game()

    cols = df.columns
    data_prep_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(verbose=2)),
            ("scaler", StandardScaler()),
            ("pca", PCA(CONFIG.pca_components)),
        ]
    )

    X = (
        df.groupby("playerid")[FEATS]  # [cols[7:-1]]
        .mean()
        .replace(np.inf, np.nan)
        .replace(np.NINF, np.nan)
    )

    fig = k_elbow_plot(
        data_prep_pipe.fit_transform(X),
        KMeans,
        (2, 20),
    )
