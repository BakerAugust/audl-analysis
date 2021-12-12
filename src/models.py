"""
Simplest model

input --> load_data.py
output --> class with predict method
"""
from pandas import DataFrame, merge, concat, MultiIndex
from pandas.core.indexes.datetimes import date_range
from pandas.core.series import Series
from typing import List, Union, Any, Optional, Tuple
import numpy as np
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import _figure as Figure
from plotly.subplots import make_subplots
from settings import get_config
from visualize import add_field_boundaries

CONFIG = get_config()

random.seed(CONFIG.random_state)


class Model1:
    """
    MLE "model" for learning the probability of an eventual score.
    """

    def __init__(self) -> None:
        self.random_seed = None
        pass

    def fit(self, data: DataFrame, x_cols, y_col, states) -> None:
        data = data.copy()[x_cols + [y_col]]
        n = data.shape[0]
        y_count = data[y_col].sum()

        # Calculate P(label)
        p_y = y_count / n

        # Calculate P(data)
        p_data = data.groupby(x_cols).count() / n
        p_data.rename({y_col: "p_data"}, inplace=True, axis=1)

        # Calculate P(data | label)
        p_conditional = data[data[y_col] == 1].groupby(x_cols).sum() / y_count
        p_conditional.rename({y_col: "p_conditional"}, inplace=True, axis=1)

        # Calculate (P(label) * P(data | label)) / P(data)
        merged = p_data.join(p_conditional)
        merged["p"] = p_y * merged["p_conditional"].fillna(0) / merged["p_data"]

        self.x_cols = x_cols
        self.y_col = y_col
        self.params = merged["p"].reindex(states, fill_value=0)

    def predict(self, data: DataFrame) -> Series:
        y_hat = data.join(self.params, on=self.x_cols)
        return y_hat["p"].fillna(value=self.params.mean())


class MarkovModel:
    """
    Markov Model -- estimates attempted transition probabilities and probability of a
    successful transition (completion) for all combinations of states.
    """

    def __init__(self, zone_df: Optional[DataFrame] = None) -> None:
        # Parameters for each source state, includes p_trans and p_success
        self.origin_dfs: dict = {}
        self.origin_dfs_priors: dict = {}
        self.zone_df: DataFrame = zone_df  # Dataframe of zone id lookups
        self.epv_params: DataFrame = DataFrame()
        pass

    def get_neighbors(self, zone_id: float) -> List[float]:
        """
        Returns neighbors of provided state.
        """
        x_zone, y_zone = self.zone_df.loc[zone_id, ["x_zone", "y_zone"]]

        neighbors = list(
            self.zone_df[
                (
                    (self.zone_df.x_zone - x_zone).abs()
                    + (self.zone_df.y_zone - y_zone).abs()
                )
                <= 1
            ].index
        )

        neighbors.remove(zone_id)

        return neighbors

    def fit(
        self,
        data: DataFrame,
        state_origin_cols: List[str],
        state_destination_cols: List[str],
        outcome_col: str,
        states: Any,
        priors: Tuple[np.ndarray, np.ndarray] = (
            0,
            0,
        ),  # tuple of success_count, attempts_count
    ) -> None:
        """
        Estimate transition probabilites with MLE
        """
        data = data.copy()
        self.x_cols = state_origin_cols
        self.states = {}
        for i, state in enumerate(states):
            self.states[state] = i

        if isinstance(list(self.states.keys())[0], tuple):
            self.scoring_states = [
                self.states[x] for x in self.states.keys() if CONFIG.score_zone_id in x
            ]
            self.turnover_states = [
                self.states[x]
                for x in self.states.keys()
                if CONFIG.turnover_zone_id in x
            ]
        else:
            self.scoring_states = [self.states[CONFIG.score_zone_id]]
            self.turnover_states = [self.states[CONFIG.turnover_zone_id]]

        self.attempt_counts = np.zeros((len(self.states), len(self.states)))

        for state_origin, df in data.groupby(state_origin_cols):
            grouped = df.groupby(state_destination_cols)
            df_out = grouped.agg(
                attempt_counts=(outcome_col, "count"),
                success_counts=(outcome_col, "sum"),
            )
            # Reindexing to fill out the array for all states
            if isinstance(list(self.states.keys())[0], tuple):
                idx = MultiIndex.from_tuples(self.states.keys())
            else:
                idx = self.states.keys()
            df_out = df_out[["attempt_counts", "success_counts"]].reindex(
                idx, fill_value=0
            )

            self.attempt_counts[self.states[state_origin], :] = df_out[
                "attempt_counts"
            ].values

        prior = np.broadcast_to(priors[1], self.attempt_counts.shape)
        if prior.sum() > 0:
            print(f"Prior ratio: {prior.sum()/(prior.sum()+self.attempt_counts.sum())}")
        self.p_trans = np.nan_to_num(
            np.divide(
                self.attempt_counts + prior,
                np.broadcast_to(
                    (self.attempt_counts + prior).sum(1), self.attempt_counts.shape
                ).T,
            )
        )

        return (self.p_trans, self.attempt_counts)

    def visualize_from_origin(
        self, state: Any, zone_df: DataFrame = DataFrame()
    ) -> Figure:
        if not zone_df.empty:
            self.zone_df = zone_df
        elif self.zone_df.empty:
            raise ValueError("zone_df not set!")
        d = zone_df.join(self.origin_dfs[state], on="zone_id")
        fig = make_subplots(rows=2, cols=1, subplot_titles=["P_trans", "P_success"])
        fig.add_trace(
            trace=go.Heatmap(
                z=d.p_trans,
                y=(d.x_zone - 2) * 10,
                x=((d.y_zone + 0.5) * 8.33),
                colorscale="RdBu",
                zsmooth="best",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            trace=go.Heatmap(
                z=d.p_success,
                y=(d.x_zone - 2) * 10,
                x=((d.y_zone + 0.5) * 8.33),
                colorscale="RdBu",
                zmin=0,
                zmax=1,
                zsmooth="best",
            ),
            row=2,
            col=1,
        )

        # fig = add_field_boundaries(fig, "horizontal")
        fig.show()

    def get_next_state(self, current_state):
        next_state = self.origin_dfs[current_state].sample(1, weights="p_trans")
        return next_state.index[0], next_state["p_success"].values[0]

    def sample_possession(self, starting_state: Any, n: int = 1) -> np.ndarray:

        rng = np.random.default_rng()

        MAX_ITER = 100
        outcomes = np.full((n), np.nan)

        # Using 9999 instead of np.nan to keep arr as int
        state_paths = np.full((n, MAX_ITER), 9999)
        state_paths[:, 0] = self.states[starting_state]
        i = 0
        state_list = list(self.states.values())

        while np.any(np.isnan(outcomes)) and i < MAX_ITER - 1:
            mask = np.isnan(outcomes)  # Masks to only consider live plays
            next_states = np.array(
                [
                    rng.choice(a=state_list, p=self.p_trans[x])
                    for x in state_paths[mask, i]
                ]
            )

            state_paths[mask, i + 1] = next_states

            # Is it a score?
            outcomes[mask] = np.where(
                np.isin(next_states, self.scoring_states), 1, outcomes[mask]
            )

            # Is it a turnover?
            outcomes[mask] = np.where(
                np.isin(next_states, self.turnover_states), 0, outcomes[mask]
            )
            i += 1

        return (np.where(np.isnan(outcomes), 0, outcomes), state_paths)

    def epv(self, starting_state: Any, iterations: int = 1000):
        if starting_state in [
            CONFIG.turnover_zone_id,
            CONFIG.score_zone_id,
        ]:
            return 0
        if self.epv_params.empty:
            result, _ = self.sample_possession(starting_state, n=iterations)
            return np.sum(result) / iterations
        else:
            return self.epv_params.loc[starting_state, "zone_epv"]

    def plot_possession(self, possessions: List[list]) -> Figure:
        fig = go.Figure()
        for possession in possessions:
            p = possession[~(possession == 9999)]  # fill value
            x = np.array([self.zone_df.loc[zone_id]["x_zone"] for zone_id in p])
            y = np.array([self.zone_df.loc[zone_id]["y_zone"] for zone_id in p])

            # add some noise
            x = x + np.random.random(len(x)) - 0.5
            y = y + np.random.random(len(y)) - 0.5

            fig.add_trace(go.Scatter(x=(y + 0.5) * 10, y=(x - 2) * 10, mode="lines"))
        fig = add_field_boundaries(fig, orient="horizontal", showlegend=True)
        return fig

    def set_epv_params(self):
        data = DataFrame(index=self.states)

        if len(self.x_cols) > 1:
            drop_idxs = []
            for idx in data.index:
                if (CONFIG.score_zone_id in idx) or (CONFIG.turnover_zone_id in idx):
                    drop_idxs.append(idx)
        else:
            drop_idxs = [CONFIG.score_zone_id, CONFIG.turnover_zone_id]
        data.drop(drop_idxs, axis=0, inplace=True)
        data["zone_epv"] = data.index.map(self.epv)

        self.epv_params = data

    def predict(self, data: DataFrame) -> Series:
        if self.epv_params.empty:
            self.set_epv_params()
        y_hat = merge(
            left=data,
            right=self.epv_params["zone_epv"],
            left_on=self.x_cols,
            right_index=True,
            how="left",
        )
        return y_hat["zone_epv"]

    def plot_epv_map(self) -> Figure:

        fig = go.Figure()
        fig.add_trace(
            trace=go.Heatmap(
                z=self.epv_params.zone_epv,
                y=(self.epv_params.x_zone - 2) * 10,
                x=((self.epv_params.y_zone + 0.5) * 10),
                colorscale="RdBu",
                zmin=0,
                zmax=1,
                zsmooth="best",
            )
        )
        return add_field_boundaries(fig, orient="horizontal", showlegend=True)
