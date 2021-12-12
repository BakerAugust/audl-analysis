import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import audl_advanced_stats as audl
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from plotly.subplots import make_subplots
from clustering import add_cluster_labels
from load_data import load_data
from sklearn.model_selection import GroupKFold
from scipy.spatial.distance import euclidean
from typing import List
from plotly.graph_objs import _figure as Figure


def plot_zones(zone_df: DataFrame) -> Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=list(zone_df.index),
            y=(zone_df.x_zone - 2) * 10,
            x=((zone_df.y_zone + 0.5) * 8.33),
            colorscale="RdBu",
        )
    )
    return fig


def add_field_boundaries(fig: Figure, orient: str, showlegend: bool = False) -> Figure:
    """
    Adds ultimate field boundary to field visualization.

    Args:
        fig (figure): Plotly figure object
        orient (str): Orientation to render the field boundaries. Accepted options: 'horizontal', 'vertical'.

    """
    length_axis_args = dict(
        range=[-1, 121],
        showticklabels=False,
        ticks="",
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        fixedrange=True,
    )

    width_axis_args = dict(
        range=[-27, 35],
        showticklabels=False,
        ticks="",
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    )

    if orient == "horizontal":
        x_axis_args = length_axis_args
        y_axis_args = width_axis_args

        # Vertical lines
        fig.add_shape(type="line", y0=-25, y1=25, x0=0, x1=0, line=dict(color="black"))
        fig.add_shape(
            type="line", y0=-25, y1=25, x0=20, x1=20, line=dict(color="black")
        )
        fig.add_shape(
            type="line", y0=-25, y1=25, x0=100, x1=100, line=dict(color="black")
        )
        fig.add_shape(
            type="line", y0=-25, y1=25, x0=120, x1=120, line=dict(color="black")
        )

        # Horizontal lines
        fig.add_shape(
            type="line", y0=-25, y1=-25, x0=0, x1=120, line=dict(color="black")
        )
        fig.add_shape(type="line", y0=25, y1=25, x0=0, x1=120, line=dict(color="black"))

        # Location of attacking arrow
        arrow_args = {"x": 80, "y": 30, "ax": 60, "ay": 30}

    elif orient == "vertical":
        x_axis_args = width_axis_args
        y_axis_args = length_axis_args

        # Vertical lines
        fig.add_shape(type="line", x0=-25, x1=25, y0=0, y1=0, line=dict(color="black"))
        fig.add_shape(
            type="line", x0=-25, x1=25, y0=20, y1=20, line=dict(color="black")
        )
        fig.add_shape(
            type="line", x0=-25, x1=25, y0=100, y1=100, line=dict(color="black")
        )
        fig.add_shape(
            type="line", x0=-25, x1=25, y0=120, y1=120, line=dict(color="black")
        )

        # Horizontal lines
        fig.add_shape(
            type="line", x0=-25, x1=-25, y0=0, y1=120, line=dict(color="black")
        )
        fig.add_shape(type="line", x0=25, x1=25, y0=0, y1=120, line=dict(color="black"))

        # Location of attacking arrow
        arrow_args = {"x": 30, "y": 80, "ax": 30, "ay": 60}

    else:
        raise ValueError(
            f'Unknown value for orient "{orient}"! Options are ["horizontal","vertical"].'
        )

    # Add arrow to indicate attacking direction
    fig.add_annotation(
        xref="x",
        yref="y",
        x=arrow_args["x"],
        y=arrow_args["y"],
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        axref="x",
        ayref="y",
        ax=arrow_args["ax"],
        ay=arrow_args["ay"],
        text="Attacking",
    )

    # Set figure properties
    fig.update_layout(
        # Remove axis titles
        xaxis_title=None,
        yaxis_title=None,
        # Add tick labels to fig
        xaxis=x_axis_args,
        # Add tick labels to fig
        yaxis=y_axis_args,
        # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Change font
        font_family="TW Cen MT",
        hoverlabel_font_family="TW Cen MT",
        showlegend=showlegend,
        # Remove margins
        margin=dict(t=0, b=0, l=0, r=0),
    )

    return fig


def plot_results(data: pd.DataFrame) -> Figure:
    fig = make_subplots(rows=1, cols=2)
    for treatment in np.unique(data.treatment.values):
        fig.add_trace(
            go.Scatter(
                x=data[data.treatment == treatment]["games_train"],
                y=data[data.treatment == treatment]["maes"],
                mode="lines+markers",
                name=treatment,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data[data.treatment == treatment]["games_train"],
                y=data[data.treatment == treatment]["param_distances"],
                mode="lines+markers",
                name=treatment,
            ),
            row=1,
            col=2,
        )

    return fig


def visualize_possession(possession_data):
    fig = go.Figure()
    fig.update_layout(width=600, height=400)
    fig.add_trace(
        go.Scatter(
            x=possession_data["y"],
            y=possession_data["x"],
            mode="lines+markers",
        ),
    )

    fig = add_field_boundaries(fig, orient="horizontal")
    return fig
