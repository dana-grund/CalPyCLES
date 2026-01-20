"""These functions are used to create the published Figures."""

# Standard library
import math

# First-party
from calpycles.parameters import ParametersDYCOMS_RF01 as Params
from calpycles.plotting import C_NATURE
from calpycles.plotting import COLORS
from calpycles.plotting import MARKERSIZE
from calpycles.plotting import ZORDERS
from calpycles.plotting import get_name_mpl
from calpycles.plotting import make_figure
from calpycles.plotting import save_figure
from calpycles.plotting.distributions_1d import _plot_hist_1d
from calpycles.plotting.distributions_1d import _plot_kde_1d
from calpycles.plotting.distributions_2d import _plot_samples_2d
from calpycles.plotting.distributions_2d import _plot_uniform_2d

MINS = Params.mins
MAXS = Params.maxs
NAMES = Params.names
DEFAULTS = Params.defaults

# -------- Allow for crappy plotting code  -------- #
# pylint: skip-file
# mypy: ignore-errors
# noqa


def plot_samples_1d(
    samples_list=[],
    labels_list_=[],
    save_file="fig-samples_1d",
    do_kde=True,
    do_samples=True,
    nature_params=None,
    nature_params_label="Nature",
    c_nature=C_NATURE,
    colors=None,
    alphas=None,
    linestyles=None,
    legend_title=None,
    nrows=2,
    lims=None,
):
    if colors is None:
        colors = COLORS
    if alphas is None:
        alphas = [1] * len(samples_list)
    if linestyles is None:
        linestyles = ["-"] * len(samples_list)
    labels_list = labels_list_.copy()

    n_params = 9

    # decide on format
    ncols = math.ceil(n_params / nrows)
    if nrows == 2:
        # fills a whole page, results in small font
        fig, axs = make_figure(ncols, nrows, horizontal=True, ratio=3 / 4, sharey=True)
        cax = axs[1, 4]

    elif nrows == 3:
        # smaller than a4 page
        fig, axs = make_figure(
            ncols,
            nrows,
            horizontal=True,
            ratio=3 / 4,
            sharey=True,
            width=8,  # cm
        )
        cax = None

    else:
        raise ValueError(f"nrows={nrows} not supported.")

    for i, ax_ in enumerate(axs):
        for j, ax in enumerate(ax_):
            idx = i * ncols + j
            if idx >= n_params:
                continue

            handles = []

            # true parameters
            if nature_params is not None:
                ha = ax.axvline(nature_params[idx], c=c_nature, zorder=-20)
                handles += [ha]

            for i_s, samples in enumerate(samples_list):
                c = colors[i_s]
                alpha = alphas[i_s]
                linestyle = linestyles[i_s]
                bin_values = None
                if do_samples:
                    bin_values, _, ha = _plot_hist_1d(
                        ax,
                        samples[:, idx],
                        color=c,
                        linestyle=linestyle,
                    )
                    handles.append(ha)

                if do_kde:
                    ha = _plot_kde_1d(
                        ax,
                        samples[:, idx],
                        color=c,
                        alpha=alpha,
                        linestyle=linestyle,
                        bin_values=bin_values,
                    )
                    if not do_samples:  # prefer samples label
                        handles.append(ha)

            ax.set_xlabel(get_name_mpl(Params.names[idx]))
            ax.grid(False)
            ax.set_yticklabels([])

            if lims == "constrained":
                lims_width = MAXS[idx] - MINS[idx]
                ax.set_xlim(
                    MINS[idx] - 0.05 * lims_width, MAXS[idx] + 0.05 * lims_width
                )

    # configure legend
    if cax is not None:
        if nature_params is not None:
            labels_list.insert(0, nature_params_label)

        # remove labels that are None
        handles = [handles[i] for i, la in enumerate(labels_list) if la is not None]
        labels_list = [
            labels_list[i] for i, la in enumerate(labels_list) if la is not None
        ]

        cax.legend(
            handles=handles,
            labels=labels_list,
            loc="upper left",
            title=legend_title,
        )
        cax.grid(False)
        cax.axis("off")

    save_figure(save_file)


def plot_2d_excerpt(
    samples_list=[],
    pairs=[(0, 1)],
    colors_list=COLORS,
    labels_list=[],
    is_prior_list=[],  # if True, plots uniform bounds; else, plots kde
    do_legend=False,
    do_samples=True,  # bool or List[bool]
    save_file="fig-dists_2d_corr_excerpt",
):

    if isinstance(do_samples, bool):
        do_samples = [do_samples] * len(samples_list)
    n_plots = len(pairs)
    if do_legend:
        n_plots += 1
    fig, axs = make_figure(n_plots, 1, horizontal=True, fraction=1.0, ratio=1.0)

    for i, pair in enumerate(pairs):

        ax = axs[i]

        handles = []
        labels = []
        for sample, c, is_prior, label, do_samples_ in zip(
            samples_list, colors_list, is_prior_list, labels_list, do_samples
        ):

            if is_prior:
                # only uniform bound
                _plot_uniform_2d(ax, lower=MINS[pair], upper=MAXS[pair], c=c)

            else:
                # samples and kde fit
                handles += _plot_samples_2d(
                    ax,
                    sample[pair],
                    label=label,
                    do_kde=True,
                    c=c,
                    do_samples=do_samples_,
                )[0]
                labels += [label]

        # nature values
        ax.scatter(
            DEFAULTS[pair[0]],
            DEFAULTS[pair[1]],
            s=MARKERSIZE,
            color=C_NATURE,
            zorder=ZORDERS["single"],
        )
        # handles += p[0]
        # labels += ["Nature"]

        # labels
        ax.set_xlabel(get_name_mpl(NAMES[pair[0]]))
        ax.set_ylabel(get_name_mpl(NAMES[pair[1]]))

    if do_legend:
        axs[-1].axis("off")
        axs[-1].legend(handles, labels)

    for pair in pairs:
        save_file += f"_{pair[0]}-{pair[1]}"

    save_figure(save_file)
