import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse


def plot_grouped_metrics(
    df: pd.DataFrame,
    metric_cols: list[str],
    exp_col: str = "exp_description",
    exp_order: list[str] | None = None,
    metric_labels: dict[str, str] | None = None,
    exp_name_map: dict[str, str] | None = None,
    title: str = "Model comparison",
    ylabel: str = "Metric value",
    width: float | None = None,
    circle_best: bool = True,
    lower_is_better: list[str] | None = None,
    figsize: tuple[int, int] = (12, 5),
    rotate_value_labels: int = 90
):
    """
    Plot grouped bar chart for experiment-level metric columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing one row per experiment.
    metric_cols : list[str]
        Metric columns to plot, e.g. ["rmse_mean", "p_mean", "s_mean"].
    exp_col : str
        Column containing experiment names/descriptions.
    exp_order : list[str] | None
        Desired experiment order. If None, uses order appearing in df.
    metric_labels : dict[str, str] | None
        Mapping from metric column names to display labels.
    exp_name_map : dict[str, str] | None
        Mapping from experiment names to legend labels.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    width : float | None
        Bar width. If None, chosen automatically.
    circle_best : bool
        Whether to circle the best bar for each metric.
    lower_is_better : list[str] | None
        Metrics for which lower values are better, e.g. ["rmse_mean"].
    figsize : tuple[int, int]
        Figure size.
    rotate_value_labels : int
        Rotation angle for bar value labels.

    Returns
    -------
    fig, ax
    """
    df_plot = df.copy()

    if exp_order is None:
        exp_order = df_plot[exp_col].tolist()

    if metric_labels is None:
        metric_labels = {m: m for m in metric_cols}

    if exp_name_map is None:
        exp_name_map = {exp: exp for exp in exp_order}

    if lower_is_better is None:
        lower_is_better = []

    df_plot[exp_col] = pd.Categorical(
        df_plot[exp_col],
        categories=exp_order,
        ordered=True
    )
    df_plot = df_plot.sort_values(exp_col).reset_index(drop=True)

    n_exps = len(exp_order)
    x = np.arange(len(metric_cols))

    if width is None:
        width = min(0.8 / max(n_exps, 1), 0.25)

    fig, ax = plt.subplots(figsize=figsize)

    bar_positions = []
    bar_values = []

    center_shift = (n_exps - 1) / 2

    for i, exp in enumerate(exp_order):
        row_match = df_plot[df_plot[exp_col] == exp]
        if row_match.empty:
            continue

        row = row_match.iloc[0]
        y = [row[m] for m in metric_cols]
        positions = x + (i - center_shift) * width

        ax.bar(positions, y, width, label=exp_name_map.get(exp, exp))

        bar_positions.append(positions)
        bar_values.append(y)

        for j, val in enumerate(y):
            ax.text(
                positions[j],
                val + 0.003,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=rotate_value_labels
            )

    bar_positions = np.array(bar_positions)
    bar_values = np.array(bar_values, dtype=float)

    if circle_best and len(bar_values) > 0:
        for j, metric in enumerate(metric_cols):
            values = bar_values[:, j]

            if metric in lower_is_better:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)

            best_x = bar_positions[best_idx, j]
            best_y = values[best_idx]

            y_range = np.nanmax(values) - np.nanmin(values)
            ellipse_height = max(y_range * 0.18, 0.03)

            ellipse = Ellipse(
                (best_x, best_y + 0.03),
                width=width * 1.5,
                height=0.23,
                edgecolor="red",
                facecolor="None",
                linewidth=1.5
            )
            ax.add_patch(ellipse)

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels.get(m, m) for m in metric_cols])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    return fig, ax



def plot_horizontal_metric(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    sort_ascending: bool = True,
    label_map: dict[str, str] | None = None,
    figsize: tuple[int, int] = (12, 6),
    cmap_name: str = "viridis",
    value_fmt: str = ".4f"
):
    """
    Plot a horizontal bar chart for one metric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    label_col : str
        Column to use for bar labels.
    value_col : str
        Metric column to plot.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    sort_ascending : bool
        Whether to sort values ascending.
    label_map : dict[str, str] | None
        Optional mapping for display labels.
    figsize : tuple[int, int]
        Figure size.
    cmap_name : str
        Matplotlib colormap name.
    value_fmt : str
        Numeric format for text labels.
    """
    plot_df = df[[label_col, value_col]].copy()

    if label_map is not None:
        plot_df[label_col] = plot_df[label_col].map(lambda x: label_map.get(x, x))

    plot_df = plot_df.sort_values(value_col, ascending=sort_ascending).reset_index(drop=True)

    norm = mcolors.Normalize(vmin=plot_df[value_col].min(), vmax=plot_df[value_col].max())
    colors = cm.get_cmap(cmap_name)(norm(plot_df[value_col].values))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(plot_df[label_col], plot_df[value_col], color=colors)

    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else value_col)
    ax.set_ylabel(ylabel if ylabel else label_col)

    for bar, value in zip(bars, plot_df[value_col]):
        ax.text(
            value + 0.002,
            bar.get_y() + bar.get_height() / 2,
            format(value, value_fmt),
            va="center"
        )

    plt.tight_layout()
    return fig, ax