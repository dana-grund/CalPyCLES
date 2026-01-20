"""General plotting functions for PyCLES output."""

# Standard library
import os
from typing import Any
from typing import List
from typing import Union

# Third-party
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# First-party
from calpycles.case_inputs import lims_by_var
from calpycles.case_inputs import names_by_case
from calpycles.helpers import ensure_path

# -------- Allow for crappy plotting code  -------- #
# pylint: skip-file
# mypy: ignore-errors
# noqa


# -------- Custom Mpl Style  -------- #
def use_style() -> None:
    """Use custom matplotlib style from 'mplstyle' file."""
    plt.style.use("calpycles.plotting.presentation")
    return None


use_style()


# -------- Defaults  -------- #

# # always plot an axes grid
# plt.rcParams["axes.grid"] = True

# plt.style.use("tableau-colorblind10")
# COLORS = [  # plt.rcParams['axes.prop_cycle'].by_key()['color']
#     "#006BA4",  # blue
#     "#FF800E",  # orange
#     # "#ABABAB",  # light gray
#     "#595959",  # dark gray
#     "#5F9ED1",  # light blue
#     "#C85200",  # light orange
#     "#898989",  # medium gray
#     "#A2C8EC",  # light light blue
#     "#FFBC79",  # light light orange
#     "#CFCFCF",  # very light gray
# ]
COLORS = [  # custom
    "#595959",  # dark gray     ### prior
    "#1f77b4",  # blue          ### posterior (default model)
    "#d62728",  # red           ### posterior (mixed)
    "#FF800E",  # orange        ### posterior (central)
    # "#9467bd",  # purple        ### too close to nature
    "#006400",  # dark green    ### posterior (lowres)
    "#e377c2",  # pink          ### synthetic data
]
C_MEAS = "k"
C_NATURE = "#7F00FE"  # dark purple

plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLORS)
LINESTYLES = ["-", "--", ":", "-."]
NBINS = 10  # number of bins for histograms
ALPHA = 0.5  # transparency e.g. for histograms and scatters
LINEWIDTH = plt.rcParams["lines.linewidth"]  # from style file
MARKERSIZE = 10

# units
inches_per_pt = 1 / 72.27
pt_per_cm = 28.3465

# sizes
FIGWIDTH = 14  # two-column
FIGWIDTH_ONECOL = 8  # one-column
TEXTWIDTH = 13
# JAMES: https://www.ametsoc.org/ams/publications/...
# ...author-information/figure-information-for-authors/

# Golden ratio to set aesthetic figure height
# https://disq.us/p/2940ij3
GOLDEN_RATIO = (5**0.5 - 1) / 2

SUBPLOTSIZE = 3.5  # inches = 6.4 cm
# SUBPLOTSIZE = 2.5  # inches = 6.4 cm

ZORDERS = {
    # observation ensembles
    "grid": 2.5,  # default
    "ens_spread": 3,
    "data": 4,
    "ens_mean": 5,
    "single": 6,
    "legend": 10,
}

# -------- Parameter names  -------- #
names_dict = {
    # parameters
    "ug": r"$u_g$",
    "divergence": r"$D$",
    "zi": r"$z_i$",
    "tg": r"$T_g$",
    "qtg": r"$q_{tg}$",
    "sst": r"$T_{sst}$",
    "cm": r"$c_m$",
    "cs": r"$c_s$",
    "prt": r"$Pr_t$",
    # coordinates
    "x": r"$x$",
    "y": r"$y$",
    "z": r"$z$",
    "t": r"$t$",
    # observations
    "w_mean2": r"$\overline{w'w'}$",
    "w_mean3": r"$\overline{w'w'w'}$",
    "ql_mean": r"$\overline{q}_l$",
    "qt_mean": r"$\overline{q}_t$",
    "qt_mean": r"$\overline{q}_t$",
    "thetali_mean": r"$\overline{\theta}_l$",
    "boundary_layer_height": r"boundary layer height $z_i$",
    "lwp": r"liquid water path",
    "cloud_fraction": r"cloud fraction",
    "cloud_base": r"cloud base",
    "cloud_base_height": r"cloud base height",
    "cloud_base_rate": r"cloud base rate",
    "cloud_base_rate_mh": r"cloud base rate",
    "cloud_top": r"cloud top",
    "cloud_top_height": r"cloud top height",
    "cloud_top_rate": r"cloud top rate",
    "cloud_top_rate_mh": r"cloud top rate",
    "energy_spectrum": r"$E_{kin}$",
    "energy_spectrum_w": r"$E_{kin}^w$",
    "qt_spectrum": r"$E^{q_t}$",
    "s_spectrum": r"$E^s$",
    "thetali_spectrum": r"$E^{\theta_l}$",
}
units_dict = {
    # parameters
    "ug": r"$m\,s^{-1}$",
    "divergence": r"$s^{-1}$",
    "zi": r"$m$",
    "tg": r"$K$",
    "qtg": r"$kg\,kg^{-1}$",
    "sst": r"$K$",
    "cm": "",
    "cs": "",
    "prt": "",
    # coordinates
    "x": r"$m$",
    "y": r"$m$",
    "z": r"$m$",
    "t": r"$s$",
    # observations
    "w_mean2": r"$m^2\,s^{-2}$",
    "w_mean3": r"$m^3\,s^{-3}$",
    "ql_mean": r"$g\,kg^{-1}$",
    "qt_mean": r"$g\,kg^{-1}$",
    "thetali_mean": r"$K$",
    "boundary_layer_height": r"$m$",
    "lwp": r"$kg\,m^{-2}$",  # Pressel17 use micro meter?
    "cloud_fraction": r"$\%$",
    "cloud_base": r"$m$",
    "cloud_top": r"$m$",
    "cloud_base_height": r"$m$",
    "cloud_top_height": r"$m$",
    "cloud_base_rate": r"$m\,s^{-1}$",
    "cloud_top_rate": r"$m\,s^{-1}$",
    "cloud_base_rate_mh": r"$m\,h^{-1}$",
    "cloud_top_rate_mh": r"$m\,h^{-1}$",
}


def get_name_mpl(name: Union[str, List[str]], add_units: bool = True) -> str:
    """Get the matplotlib name for a variable."""

    def get_one_name(name: str) -> str:
        """Get the matplotlib name for a single variable."""
        name_mpl = name
        if name in names_dict:
            name_mpl = names_dict[name]
            if add_units:
                name_mpl += f" [{units_dict[name]}]"
        return name_mpl

    if isinstance(name, str):
        return get_one_name(name)
    return [get_one_name(n) for n in name]


def save_figure(
    save_file: str,
    dpi: int = 150,
    bbox_inches: str = "tight",
    show: bool = True,
) -> None:
    """Clever saving."""
    plt.tight_layout()
    if save_file:
        ensure_path(os.path.dirname(save_file))
        plt.savefig(save_file + ".png", dpi=dpi, bbox_inches=bbox_inches)
        plt.savefig(save_file + ".pdf", dpi=dpi, bbox_inches=bbox_inches)
        plt.savefig(save_file + ".svg", dpi=dpi, bbox_inches=bbox_inches)
    if show:
        plt.show()
    plt.close()
    print("Saved figure ", save_file)


def get_figsize(
    width: float = FIGWIDTH,
    fraction: float = 1.0,
    ratio: float = GOLDEN_RATIO,
    horizontal: bool = False,
    subplots: tuple = (1, 1),
) -> tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in cm
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    ratio: float, optional
            Ratio between figure height and width
    horizontal: bool, optional
            If True, width is larger than height by ratio.
            If False, height is larger than width by ratio.
    subplots: array-like, optional
            The number of rows and columns of subplots.

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches

    """
    # Width of figure (in pts)
    fig_width_pt = fraction * width * pt_per_cm

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt

    # Orientation
    if not horizontal:
        ratio = 1 / ratio

    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


plt.rcParams["figure.figsize"] = get_figsize()  # default


def make_figure(
    nx: int = 1,
    ny: int = 1,
    height: float = 0,  # not used (legacy)
    horizontal: bool = False,
    fraction: float = 1.0,
    ratio: float = GOLDEN_RATIO,
    width: float = FIGWIDTH,
    **kwargs: Any,
):
    """Make a figure with n subplots, such that the figure has a given width."""
    subplots = (ny, nx)  # swapped here
    figsize = get_figsize(
        subplots=subplots,
        ratio=ratio,
        horizontal=horizontal,
        fraction=fraction,
        width=width,
    )

    kwargs_default = {
        "sharex": False,
        "sharey": False,
    }
    kwargs_default.update(kwargs)

    fig, axs = plt.subplots(*subplots, figsize=figsize, **kwargs_default)
    if nx == 1 and ny == 1:
        axs = [axs]

    return fig, axs


def make_figure_profiles(profile_variables=None, case="DYCOMS_RF01", legend_ax=True):
    """Make a figure for the profile variables."""
    if profile_variables is None:
        profile_variables = names_by_case[case]["profiles"]
    n_figs = len(profile_variables)
    if legend_ax:
        n_figs += 1

    fig, axs = make_figure(n_figs, ratio=GOLDEN_RATIO, sharey=True)
    for ax in axs:
        ax.grid(True, zorder=ZORDERS["grid"])

    for i, name in enumerate(profile_variables):
        ax = axs[i]
        ax.set_xlabel(get_name_mpl(name))
        ax.set_ylim(names_by_case[case]["z_lims"])
        if i == 0:
            ax.set_ylabel(get_name_mpl("z"))

        if name in lims_by_var:
            ax.set_xlim(lims_by_var[name])

    # separate subplot for the legend
    if legend_ax:
        ax = axs[-1]
        ax.axis("off")
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

    return fig, axs


def make_figure_timeseries(timeseries_variables=None, legend_ax=True):
    """Make a figure for the timeseries variables."""
    if timeseries_variables is None:
        timeseries_variables = names_by_case["DYCOMS_RF01"]["timeseries"]
    n_figs = len(timeseries_variables)
    if legend_ax:
        n_figs += 1

    fig, axs = make_figure(n_figs, ratio=GOLDEN_RATIO)
    for ax in axs:
        ax.grid(True, zorder=ZORDERS["grid"])

    for i, name in enumerate(timeseries_variables):
        ax = axs[i]
        ax.set_xlabel(get_name_mpl("t"))
        ax.set_ylabel(get_name_mpl(name))

        if name in lims_by_var:
            ax.set_ylim(lims_by_var[name])

    # separate subplot for the legend
    if legend_ax:
        ax = axs[-1]
        ax.axis("off")
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

    return fig, axs


def color_handles(labels, colors):
    """Add patches for the C_DATA and C_OBS colors."""
    return [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=10,
        )
        for label, color in zip(labels, colors)
    ]


class ColorManager:
    """Plot several subplots with the same color scale.

    Usage:
        CM = ColorManager(vlims=(0,1))
        ax.plot(CM.color(u))
        CM.colorbar(ax)
    """

    def __init__(self, vlims=None, values=None, cmap=None, center=False, cbarlabel=""):
        """Initialize."""
        if values is not None and vlims is not None:
            print("ColorManager. Using given vlims, ignoring array input.")
        elif values is not None:
            vlims = self.get_vlims(values)
        else:
            vlims = -1, 1

        if center:
            v = max(np.abs(vlims[0]), np.abs(vlims[1]))
            vlims = -v, v

        assert vlims is not None
        self.vmin, self.vmax = vlims

        self.Normalizer = Normalize(vmin=self.vmin, vmax=self.vmax)
        if cmap is None:

            # create binary cmap
            if self.vmin < 0 and self.vmax > 0:

                # corresponding fractions of 256
                total = -self.vmin + self.vmax
                num_min = int(-self.vmin / total * 256)
                num_max = int(self.vmax / total * 256)

                # ensure we have 256 points
                while num_min + num_max < 256:
                    num_max += 1

                top = mpl.colormaps["Oranges_r"].resampled(num_max)
                bottom = mpl.colormaps["Blues"].resampled(num_min)

                newcolors = np.vstack(
                    (
                        bottom(np.linspace(0, 1, num_min)[::-1]),
                        top(np.linspace(0, 1, num_max)[::-1]),
                    )
                )
                self.cmap = mpl.colors.ListedColormap(newcolors, name="OrangeBlue")

            # create default cmap
            else:
                self.cmap = plt.get_cmap("viridis", 256)
        else:
            self.cmap = cmap

        self.cbarlabel = cbarlabel

    def color(self, x):
        """Determine the color of values to plot."""
        return self.cmap(self.Normalizer(x))

    def colorbar(self, ax, fraction=0.046, pad=0.04, **kwargs):
        """Add a colorbar to the given axes."""
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.Normalizer)
        plt.colorbar(
            sm, ax=ax, fraction=fraction, pad=pad, label=self.cbarlabel, **kwargs
        )

    def get_vlims(self, u):
        """Get the min and max of the given array."""
        return np.min(u), np.max(u)
