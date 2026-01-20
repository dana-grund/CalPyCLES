# Standard library
import os
from typing import Optional

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# First-party
from calpycles.plotting import get_name_mpl
from calpycles.plotting import make_figure
from calpycles.plotting import save_figure

# -------- Allow for crappy plotting code  -------- #
# pylint: skip-file
# mypy: ignore-errors
# noqa


def plot_heatmap(
    matrix,
    xlabels,
    ylabels,
    folder,
    name,
    plot_name,
    cbar_label="",
    vmax=None,
    fraction=1.0,
    cbar_symm=True,
):
    """Plot a heatmap of a matrix with given x and y labels."""
    extend: Optional[str] = (
        "both"  # add pointy ends since the values might exceed the cbar
    )
    cmap = "seismic"
    if vmax is None:
        vmax = np.max(np.abs(matrix))
    vmin = np.min(matrix)
    if cbar_symm:
        vmin = -vmax
        extend = None  # values guaranteed inside cbar
    else:
        cmap = "binary_r"

    fig, ax = make_figure(1, 1, ratio=1, fraction=fraction)
    ax = ax[0]

    im = _plot_heatmap(
        ax,
        matrix,
        xlabels,
        ylabels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    fig.colorbar(
        im,
        ax=ax,
        location="right",
        anchor=(0, 0.5),
        shrink=0.5,
        extend=extend,
        label=cbar_label,
    )

    plt.title(name)
    name += "_" + plot_name
    save_figure(os.path.join(folder, f"fig-heatmap_{name.replace(' ', '_')}"))


def configure_heatmap_ax(ax):
    """Configure heatmap axis with grid and ticks."""
    return


def _plot_heatmap(
    ax,
    matrix,
    xlabels=None,
    ylabels=None,
    cmap="seismic",
    vmin=None,
    vmax=None,
):
    # plot heatmap
    im = ax.imshow(matrix.T, cmap=cmap, vmin=vmin, vmax=vmax)  # "hsv"
    nx, ny = matrix.shape

    if xlabels is None:
        xlabels = [" "] * nx
    if ylabels is None:
        ylabels = [" "] * ny

    # make ticks, and labels if available
    assert len(xlabels) == nx
    xlabels = [get_name_mpl(p, add_units=False) for p in xlabels]
    ax.set_xticks(
        range(nx),
        labels=xlabels,
        rotation=90,
        ha="right",
        rotation_mode="anchor",
    )

    assert len(ylabels) == ny
    ylabels = [get_name_mpl(p, add_units=False) for p in ylabels]
    ax.set_yticks(range(ny), labels=ylabels)

    # # text annotations: add values inside pixels (too large)
    # for i in range(ny):
    #     for j in range(len(xlabels)):
    #         text = ax.text(j, i, matrix[j,i],
    #                     ha="center", va="center", color="w")

    ax.grid(False)

    # grid in between the pixels c=plt.rcParams["grid.color"]
    for i in range(nx + 1):
        ax.plot(
            [i - 0.5, i - 0.5],
            [-0.5, ny - 0.5],
            c="k",
            linewidth=plt.rcParams["grid.linewidth"],
        )
    for i in range(ny + 1):
        ax.plot(
            [-0.5, nx - 0.5],
            [i - 0.5, i - 0.5],
            c="k",
            linewidth=plt.rcParams["grid.linewidth"],
        )

    return im
