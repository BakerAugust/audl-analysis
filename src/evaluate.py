"""
Model evaluation for distance (grid) and predictive power.
"""

import itertools
import numpy as np
import pandas as pd
from pandas.core.indexes.multi import MultiIndex
import plotly.express as px
import plotly.graph_objects as go
import audl_advanced_stats as audl
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from plotly.subplots import make_subplots
from scipy.special import kl_div
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from clustering import add_cluster_labels
from models import Model1, MarkovModel
from load_data import load_data, make_grid
from sklearn.model_selection import GroupKFold
from scipy.spatial.distance import euclidean
from typing import List, Any
import time

from settings import get_config
from visualize import add_field_boundaries, plot_zones, plot_results

CONFIG = get_config()


def calc_param_distance(
    arr_1: np.ndarray, arr_2: np.ndarray, smooth: bool = True
) -> float:
    """
    Calculates kl_div  between 1-d arrays of parameters.
    """
    # Improve this smoothing to take away probability mass from non-adjusted entries...
    if smooth == True:
        arr_1 = np.where(arr_1 == 0, 0.01, arr_1)
        arr_2 = np.where(arr_2 == 0, 0.01, arr_2)
    # out = np.sum(np.where(kl_div(arr_2, arr_1) == np.inf, 0, 1)) / arr_1.shape[0]
    out = np.sum(kl_div(arr_2, arr_1)) / arr_1.shape[0]
    # if out < 1:
    #     print(arr_1, arr_2)
    #     print(kl_div(arr_1, arr_2))
    #     raise ValueError
    return out


def calc_mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Mean-squared error
    """
    # return np.sum(np.square(y - y_hat)) / y.shape[0]
    return np.sum(np.abs(y - y_hat)) / y.shape[0]


def calc_mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Mean absolute error
    """
    return np.sum(np.abs(y - y_hat)) / y.shape[0]


def evaluate(
    data: DataFrame,
    x_cols: List[str],
    y_col: str,
    states: Any,
    n_folds: int = None,
    naive: bool = False,
):
    if not n_folds:
        n_folds = data.game_id.value_counts().shape[0]

    gpkfold = GroupKFold(n_splits=n_folds)

    maes = []
    param_distances = []

    for train_index, test_index in gpkfold.split(
        data[x_cols], data[y_col], data.game_id
    ):
        train_data, test_data = data.iloc[train_index, :], data.iloc[test_index, :]

        y = test_data[y_col]
        if naive:
            y_hat = np.ones(y.shape) * train_data[y_col].mean()
            param_distance = calc_param_distance([train_data[y_col].mean()], [y.mean()])
        else:
            m1 = Model1()
            m1.fit(train_data[x_cols + [y_col]], x_cols, y_col, states=states)
            train_params = m1.params

            y_hat = m1.predict(test_data).values

            m1.fit(test_data[x_cols + [y_col]], x_cols, y_col, states=states)
            test_params = m1.params

            param_distance = calc_param_distance(train_params, test_params)
        param_distances.append(param_distance)

        mae = calc_mae(y.values, y_hat)
        maes.append(mae)

    return (maes, param_distances)


def evaluate_mm(
    data: DataFrame,
    y_col: str,
    zone_df: pd.DataFrame,
    state_origin_cols: List[str] = ["zone_id"],
    state_destination_cols: List[str] = ["zone_id_dest"],
    states: Any = None,
    priors: Any = None,
    n_folds: int = None,
):
    if not n_folds:
        n_folds = data.game_id.value_counts().shape[0]

    gpkfold = GroupKFold(n_splits=n_folds)

    maes = []
    param_distances = []

    for train_index, test_index in gpkfold.split(data, data[y_col], data.game_id):
        train_data, test_data = data.iloc[train_index, :], data.iloc[test_index, :]

        y = test_data[y_col]

        mm = MarkovModel(zone_df)
        mm.fit(
            train_data,
            state_origin_cols,
            state_destination_cols,
            "outcome",
            states=states,
            priors=priors,
        )
        mm.set_epv_params()
        train_params = mm.epv_params
        y_hat = mm.predict(test_data).values
        mae = calc_mae(y.values, y_hat)
        maes.append(mae)

        mm_test = MarkovModel(zone_df)
        mm_test.fit(
            test_data,
            state_origin_cols,
            state_destination_cols,
            "outcome",
            states=states,
            priors=priors,
        )
        mm_test.set_epv_params()
        test_params = mm_test.epv_params
        param_distances.append(
            calc_param_distance(train_params["zone_epv"], test_params["zone_epv"])
        )
    return (maes, param_distances)


def make_classifier_pipe(
    classifier: Any, num_feats: List[str], cat_feats: List[str]
) -> Pipeline:
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_feats),
            ("cat", categorical_transformer, cat_feats),
        ]
    )

    return Pipeline(steps=[("pre", preprocessor), ("clf", classifier)])


def evaluate_classifer(
    classifier,
    data: DataFrame,
    y_col: str,
    features: List[str],
    n_folds: int = None,
):
    if not n_folds:
        n_folds = data.game_id.value_counts().shape[0]

    gpkfold = GroupKFold(n_splits=n_folds)

    maes = []
    param_distances = []

    for train_index, test_index in gpkfold.split(data, data[y_col], data.game_id):
        train_data, test_data = data.iloc[train_index, :], data.iloc[test_index, :]

        y = test_data[y_col]
        classifier.fit(X=train_data[features], y=train_data[y_col].values)
        y_hat = classifier.predict_proba(test_data[features])[:, 1]
        mae = calc_mae(y.values, y_hat)
        maes.append(mae)

    return (maes, param_distances)


def evaluate_on_team(team_abbrev: str = "NY") -> None:
    data = load_data(team_abbrev)
    data.dropna(subset=["x", "y"], inplace=True)

    game_ids = data.game_id.unique()
    results_ZON = {"maes": [], "param_distances": []}
    results_ZRT = {"maes": [], "param_distances": []}
    results_ZCL = {"maes": [], "param_distances": []}

    s = audl.Season()
    stats_df = s.get_player_stats_by_game()

    n_folds = None

    for i in range(2, len(game_ids)):
        # choose game_ids
        df = data[data.game_id.isin(game_ids[:i])].copy()
        ZON = evaluate(df, ["zone_id"], "possession_value", n_folds=n_folds)
        results_ZON["maes"].append(np.mean(ZON[0]))
        results_ZON["param_distances"].append(np.mean(ZON[1]))

        ZRT = evaluate(
            df, ["zone_id", "received_throw_type"], "possession_value", n_folds=n_folds
        )
        results_ZRT["maes"].append(np.mean(ZRT[0]))
        results_ZRT["param_distances"].append(np.mean(ZRT[1]))

        stats_data = stats_df[stats_df.game_id.isin(game_ids[:i])]
        df = add_cluster_labels(df, stats_data, n_clusters=CONFIG.n_clusters)
        ZCL = evaluate(
            df, ["zone_id", "cluster_id"], "possession_value", n_folds=n_folds
        )
        results_ZCL["maes"].append(np.mean(ZCL[0]))
        results_ZCL["param_distances"].append(np.mean(ZCL[1]))

    fig = make_subplots(1, 2)
    fig.add_trace(
        go.Scatter(
            x=np.arange(2, len(game_ids) - 1),
            y=results_ZON["maes"],
            mode="lines+markers",
            name="ZON",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(2, len(game_ids) - 1),
            y=results_ZON["param_distances"],
            mode="lines+markers",
            name="ZON",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(2, len(game_ids) - 1),
            y=results_ZRT["maes"],
            mode="lines+markers",
            name="ZRT",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(2, len(game_ids) - 1),
            y=results_ZRT["param_distances"],
            mode="lines+markers",
            name="ZRT",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(2, len(game_ids) - 1),
            y=results_ZCL["maes"],
            mode="lines+markers",
            name="ZCL",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(2, len(game_ids) - 1),
            y=results_ZCL["param_distances"],
            mode="lines+markers",
            name="ZCL",
        ),
        row=1,
        col=2,
    )
    fig.show()


def evaluate_on_all_teams() -> None:
    """
    Big loop to evaluate all methods on
    """
    all_data, zone_df = load_data()
    all_data.dropna(subset=["x", "y", "r"], inplace=True)
    ZCL_STATES = MultiIndex.from_product([list(zone_df.index), [0.0, 1.0]])
    ZRT_STATES = MultiIndex.from_product(
        [list(zone_df.index), all_data.received_throw_type.unique()]
    )
    s = audl.Season()
    stats_df = s.get_player_stats_by_game()
    n_folds = 2
    dfs = []

    prior_data = add_cluster_labels(all_data, stats_df)

    for team_id, data in all_data.groupby("team_id"):

        # Calculate some priors
        mmzc_priors = MarkovModel(zone_df)
        zc_priors = mmzc_priors.fit(
            prior_data[prior_data.team_id != team_id],
            ["zone_id", "cluster_id"],
            ["zone_id_dest", "cluster_id_dest"],
            "outcome",
            states=ZCL_STATES,
        )
        zc_priors = (zc_priors[0] / 300, zc_priors[1] / 300)

        mmz_priors = MarkovModel(zone_df)
        z_priors = mmz_priors.fit(
            prior_data[prior_data.team_id != team_id],
            ["zone_id"],
            ["zone_id_dest"],
            "outcome",
            states=zone_df.index.values,
        )
        z_priors = (z_priors[0] / 300, z_priors[1] / 300)

        game_ids = data.game_id.unique()
        results = {
            "maes": [],
            "param_distances": [],
            "treatment": [],
            "games_train": [],
        }

        for i in range(6, len(game_ids)):
            # choose game_ids
            df = data[data.game_id.isin(game_ids[:i])].copy()
            ZON = evaluate(
                df,
                ["zone_id"],
                "possession_value",
                n_folds=n_folds,
                states=zone_df.index,
            )
            results["maes"].append(ZON[0])
            results["param_distances"].append(ZON[1])
            results["treatment"].append("ZON")
            results["games_train"].append(i - 1)

            ZRT = evaluate(
                data=df,
                x_cols=["zone_id", "received_throw_type"],
                y_col="possession_value",
                n_folds=n_folds,
                states=ZRT_STATES,
            )
            results["maes"].append(ZRT[0])
            results["param_distances"].append(ZRT[1])
            results["treatment"].append("ZRT")
            results["games_train"].append(i - 1)

            stats_data = stats_df[stats_df.game_id.isin(game_ids[:i])]
            df = add_cluster_labels(df, stats_data, n_clusters=CONFIG.n_clusters)
            ZCL = evaluate(
                df,
                ["zone_id", "cluster_id"],
                "possession_value",
                n_folds=n_folds,
                states=ZCL_STATES,
            )
            results["maes"].append(ZCL[0])
            results["param_distances"].append(ZCL[1])
            results["treatment"].append("ZCL")
            results["games_train"].append(i - 1)

            naive = evaluate(
                df,
                ["cluster_id"],
                "possession_value",
                n_folds=n_folds,
                naive=True,
                states=None,
            )
            results["maes"].append(naive[0])
            results["param_distances"].append(naive[1])
            results["treatment"].append("naive")
            results["games_train"].append(i - 1)

            MMZ = evaluate_mm(
                df,
                "possession_value",
                state_origin_cols=["zone_id"],
                state_destination_cols=["zone_id_dest"],
                zone_df=zone_df,
                n_folds=n_folds,
                priors=z_priors,
                states=zone_df.index.values,
            )
            results["maes"].append(MMZ[0])
            results["param_distances"].append(MMZ[1])
            results["treatment"].append("MMZ")
            results["games_train"].append(i - 1)

            MMZC = evaluate_mm(
                df,
                "possession_value",
                state_origin_cols=["zone_id", "cluster_id"],
                state_destination_cols=["zone_id_dest", "cluster_id_dest"],
                zone_df=zone_df,
                n_folds=n_folds,
                priors=zc_priors,
                states=ZCL_STATES,
            )
            results["maes"].append(MMZC[0])
            results["param_distances"].append(MMZC[1])
            results["treatment"].append("MMZC")
            results["games_train"].append(i - 1)

        df = pd.DataFrame.from_dict(results)
        df.loc[:, "team_id"] = team_id
        dfs.append(df)
    return pd.concat(dfs, axis=0).reset_index()


def visualize_EPV(game_id: int, possession_number: int, team_abbrev: str):
    prior_data, zone_df = load_data()
    s = audl.Season()
    stats_df = s.get_player_stats_by_game()
    prior_data = add_cluster_labels(prior_data, stats_df)
    priors_mm = MarkovModel(zone_df)
    ZCL_STATES = MultiIndex.from_product([list(zone_df.index), [0.0, 1.0]])
    priors = priors_mm.fit(
        prior_data,
        ["zone_id", "cluster_id"],
        ["zone_id_dest", "cluster_id_dest"],
        "outcome",
        states=ZCL_STATES,
    )
    priors = (priors[0] / 300, priors[1] / 300)

    df, zone_df = load_data(team_abbrev, keep_all=True)

    # keep full df for plotting
    data = df.dropna(subset=["zone_id", "zone_id_dest", "r", "throw_outcome"])
    data = add_cluster_labels(data, stats_df)
    train_data = data[data.game_id != game_id]
    possession_data = data[
        (data.game_id == game_id) & (data.possession_number == possession_number)
    ]
    poss_plot_data = df[
        (df.game_id == game_id) & (df.possession_number == possession_number)
    ].dropna(subset=["x", "y", "r"])

    ZRT_STATES = MultiIndex.from_product(
        [list(zone_df.index), data.received_throw_type.unique()]
    )

    y_col = "possession_value"
    ZON = Model1()
    zon_feats = ["zone_id"]
    ZON.fit(
        train_data[zon_feats + [y_col]],
        x_cols=zon_feats,
        y_col=y_col,
        states=zone_df.index,
    )
    ZON_y_hat = ZON.predict(possession_data[zon_feats])

    ZRT = Model1()
    zrt_feats = ["zone_id", "received_throw_type"]
    ZRT.fit(
        train_data[zrt_feats + [y_col]],
        x_cols=zrt_feats,
        y_col=y_col,
        states=ZRT_STATES,
    )
    ZRT_y_hat = ZRT.predict(possession_data[zrt_feats])

    ZCL = Model1()
    zcl_feats = ["zone_id", "cluster_id"]
    ZCL.fit(
        train_data[zcl_feats + [y_col]],
        x_cols=zcl_feats,
        y_col=y_col,
        states=ZCL_STATES,
    )
    ZCL_y_hat = ZCL.predict(possession_data[zcl_feats])

    MMZ = MarkovModel(zone_df)
    MMZ.fit(
        train_data,
        ["zone_id"],
        ["zone_id_dest"],
        "outcome",
        states=zone_df.index.values,
        # priors=priors,
    )
    MMZ.set_epv_params()
    MMZ_y_hat = MMZ.predict(possession_data["zone_id"])

    MMZC = MarkovModel(zone_df)
    MMZC.fit(
        train_data,
        ["zone_id", "cluster_id"],
        ["zone_id_dest", "cluster_id_dest"],
        "outcome",
        states=ZCL_STATES,
        priors=priors,
    )
    MMZC.set_epv_params()
    MMZC_y_hat = MMZC.predict(possession_data[["zone_id", "cluster_id"]])

    results = {
        "CP-MZ": ZON_y_hat,
        "CP-MZC": ZRT_y_hat,
        "CP-MZR": ZCL_y_hat,
        "ST-MZ": MMZ_y_hat,
        "ST-MZC": MMZC_y_hat,
    }

    fig = make_subplots(
        2, 1, row_heights=[0.75, 0.25], subplot_titles=("Possession Map", "EPV")
    )
    for name, result in results.items():
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, possession_data.shape[0]),
                y=result,
                mode="lines+markers",
                name=name,
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            y=poss_plot_data["x"],
            x=poss_plot_data["y"],
            mode="lines+markers",
            name="possession",
        ),
        row=1,
        col=1,
    )
    fig.layout.update(
        shapes=[
            {
                "type": "line",
                "x0": 100,
                "y0": -25,
                "x1": 100,
                "y1": 25,
                "xref": "x1",
                "yref": "y1",
            },
        ]
    )

    fig.layout.xaxis.update(range=(0, 125))
    fig.layout.yaxis.update(range=(-25, 25))
    fig.update_xaxes(title_text="Throw Number", row=2, col=1)
    fig.update_yaxes(title_text="EPV", row=2, col=1)

    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.95,
        y=0.95,
        text="endzone",
        showarrow=False,
        row=1,
        col=1,
    )

    fig.write_image(f"{team_abbrev}-{game_id}-{possession_number}.png")


def evaluate_on_all() -> None:
    data, zone_df = load_data()
    data.dropna(subset=["x", "y"], inplace=True)
    s = audl.Season()
    stats_df = s.get_player_stats_by_game()
    n_folds = None
    dfs = []
    game_ids = sorted(data.game_id.unique())

    n_folds = 2

    n_games = np.arange(start=30, step=10, stop=130)
    for i in n_games:
        results = {
            "mses": [],
            "param_distances": [],
            "treatment": [],
            "games_train": [],
        }
        # choose game_ids
        df = data[data.game_id.isin(game_ids[:i])].copy()
        ZON = evaluate(df, ["zone_id"], "possession_value", n_folds=n_folds)
        results["mses"].append(np.mean(ZON[0]))
        results["param_distances"].append(np.mean(ZON[1]))
        results["treatment"].append("ZON")
        results["games_train"].append(i - 1)

        ZRT = evaluate(
            data=df,
            x_cols=["received_throw_type"],
            y_col="possession_value",
            n_folds=n_folds,
        )
        results["mses"].append(np.mean(ZRT[0]))
        results["param_distances"].append(np.mean(ZRT[1]))
        results["treatment"].append("ZRT")
        results["games_train"].append(i - 1)

        stats_data = stats_df[stats_df.game_id.isin(game_ids[:i])]
        df = add_cluster_labels(df, stats_data, n_clusters=CONFIG.n_clusters)
        ZCL = evaluate(df, ["cluster_id"], "possession_value", n_folds=n_folds)
        results["mses"].append(np.mean(ZCL[0]))
        results["param_distances"].append(np.mean(ZCL[1]))
        results["treatment"].append("ZCL")
        results["games_train"].append(i - 1)

        naive = evaluate(
            df,
            ["cluster_id"],
            "possession_value",
            n_folds=n_folds,
            naive=True,
        )
        results["mses"].append(np.mean(naive[0]))
        results["param_distances"].append(np.mean(naive[1]))
        results["treatment"].append("naive")
        results["games_train"].append(i - 1)

        # ZMM = evaluate_mm(data, y_col="possession_value", zone_df=zone_df)
        # results["mses"].append(np.mean(ZMM[0]))
        # results["param_distances"].append(np.mean(ZMM[1]))
        # results["treatment"].append("ZMM")
        # results["games_train"].append(i - 1)

        dfs.append(pd.DataFrame.from_dict(results))

    return pd.concat(dfs, axis=0).reset_index()



if __name__ == "__main__":
    # Main evaluation loop
    # df = evaluate_on_all_teams()
    # df.to_csv("temp.csv")
    # print(df.head(20))
    # fig = plot_results(df.groupby(["games_train", "treatment"], as_index=False).mean())
    # fig.show()

    # Some EPV tests
    visualize_EPV(2796, 8, "RAL")  # https://youtu.be/75IUVUbd0Ds?t=1227
    visualize_EPV(2796, 12, "RAL")  # https://youtu.be/75IUVUbd0Ds?t=1739
    visualize_EPV(
        2796, 10, "RAL"
    )  # https://www.youtube.com/watch?v=75IUVUbd0Ds punctuated by timeout
    visualize_EPV(
        2796, 15, "RAL"
    )  # https://youtu.be/75IUVUbd0Ds?t=2095 # tracking not great
    visualize_EPV(2796, 17, "RAL")  # https://youtu.be/75IUVUbd0Ds?t=2289 quick turn
    visualize_EPV(2796, 19, "RAL")  # https://www.youtube.com/watch?v=75IUVUbd0Ds
    visualize_EPV(2796, 21, "RAL")  # https://www.youtube.com/watch?v=75IUVUbd0Ds
