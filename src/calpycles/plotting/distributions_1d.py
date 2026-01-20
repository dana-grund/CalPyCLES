"""Plotting functionality for one-dinemsional (marginal) distribution plots."""

# Standard library
import os
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# Third-party
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]
from scipy.stats import gaussian_kde  # type: ignore[import-untyped]

# First-party
from calpycles.helpers import crop_nans

# from calpycles.plotting import LINESTYLES
from calpycles.plotting import ALPHA
from calpycles.plotting import C_NATURE
from calpycles.plotting import COLORS
from calpycles.plotting import NBINS
from calpycles.plotting import SUBPLOTSIZE
from calpycles.plotting import ZORDERS
from calpycles.plotting import get_name_mpl
from calpycles.plotting import make_figure
from calpycles.plotting import save_figure
from calpycles.plotting.distributions_helpers import prepare
from calpycles.random_variables import GaussRV
from calpycles.random_variables import GaussTransformedUniform

# -------- Allow for crappy plotting code  -------- #
# pylint: skip-file
# mypy: ignore-errors
# noqa


NORMALIZE = True  # scale 1d densities such that uniforms and normals have same height


# -------- Main -------- #


def plot_dists_1d(
    dists: Union[GaussRV, List[GaussRV]],
    dist_labels: Union[str, List[str]],
    samples: Optional[Union[NDArray[np.float64], List[NDArray[np.float64]]]] = None,
    true_params: Optional[NDArray[np.float64]] = None,
    true_params_label: str = "true parameters",
    params_to_plot: Optional[List[int]] = None,
    plot_dir: str = "./",
    name: str = "",
    dist_space: str = "unconstrained",
    show: bool = True,
    do_samples: Union[bool, List[bool]] = True,
    **kwargs: Any,
) -> None:
    """Plot distributions in (un)constrained 1d space.

    Distributions have to be ordered as [prior, post, post, ...] since the first is
    plotted analytically while the others are approximated by kde.

    dist_space:
        "unconstrained" (default): normal distributions, requires dist.mean and dist.var
        "constrained": uniform distributions, requires dist.upper and dist.lower
    samples:
        Passed in constrained space.
    true_params:
        Passed in unconstrained space.

    XXX to do: outsource common inputs processing of 1d and 2d
    """
    # handle inputs (shared by 1d and 2d plots)
    (
        dists,
        dist_labels,
        lims,
        add_units,
        samples_,
        true_params_,
        params_to_plot,
        do_samples,
    ) = prepare(
        dists=dists,
        dist_labels=dist_labels,
        samples=samples,
        true_params=true_params,
        params_to_plot=params_to_plot,
        name=name,
        dist_space=dist_space,
        do_samples=do_samples,
    )

    n_plots = len(params_to_plot) if params_to_plot is not None else dists[0].M
    _, axs = make_figure(
        n_plots + 1, width=n_plots * SUBPLOTSIZE, ratio=1.0
    )  # add one  for legend

    # one subplot per parameter
    for i_plot, i in enumerate(params_to_plot):
        ax = axs[i_plot]

        handles, labels = _plot_dist_1d(
            ax=ax,
            i=i,
            dists=dists,
            dist_labels=dist_labels,
            samples=samples_,
            true_params=true_params_,
            true_params_label=true_params_label,
            lims=lims,
            dist_space=dist_space,
            do_samples=do_samples,
            **kwargs,
        )

        # formatting
        ax.set_xlabel(get_name_mpl(dists[0].names[i], add_units=add_units))
        ax.set_yticklabels([])  # no y-axis labels (histograms)
        # if i == 0:
        #     ax.set_ylabel("distribution")

        axs[-1].legend(handles=handles, labels=labels, loc="upper left")
        axs[-1].axis("off")

    save_figure(os.path.join(plot_dir, f"fig-dists_1d_{name}_{dist_space}"), show=show)


# -------- Helpers -------- #


def _plot_normal_1d_i(
    ax: Axes,
    i: int,
    dist: GaussRV,
    label: str,
    c: Optional[str] = "b",
    linestyle: str = "-",
) -> Tuple[List[Any], List[str]]:
    """Plot the analytic normal distribution of the i-th parameter."""
    mu = dist.mean[i]
    var = dist.var[i]
    ha, la = _plot_normal_1d(ax, mu, var, c=c, label=label, linestyle=linestyle)
    return [ha], [la]


def _plot_uniform_1d_i(
    ax: Axes,
    i: int,
    dist: GaussTransformedUniform,
    label: str,
    c: Optional[str] = "b",
    linestyle: str = "-",
) -> Tuple[List[Any], List[str]]:
    """Plot the analytic uniform distribution of the i-th parameter."""
    ha, la = _plot_uniform_1d(
        ax, dist.lower[i], dist.upper[i], c=c, linestyle=linestyle, label=label
    )
    return [ha], [la]


def _plot_dist_1d(
    ax: Axes,
    i: int,
    dists: Union[GaussRV, List[GaussRV]],
    dist_labels: Union[str, List[str]],
    samples: Optional[Union[NDArray[np.float64], List[NDArray[np.float64]]]] = None,
    true_params: Optional[NDArray[np.float64]] = None,
    true_params_label: str = "true parameters",
    lims: Optional[Tuple[Optional[float], Optional[float]]] = None,
    dist_space: str = "unconstrained",
    do_samples: Union[bool, List[bool]] = True,
    colors: List[str] = COLORS,
    **kwargs: Any,
) -> Tuple[List[Any], List[str]]:
    """Plot the i-th subplot of plot_dists_1d()."""
    lim = lims[i] if lims is not None else None

    # keep track of legend
    handles, labels = [], []

    # prior samples
    samples_ = samples[0][:, i] if samples[0] is not None else None
    ha, la = _plot_samples_1d(
        ax,
        samples=samples_,
        label=dist_labels[0],
        c=colors[0],
        # linestyle=LINESTYLES[0],
        lims=lim,
        do_kde=False,
        do_samples=do_samples[0],
        **kwargs,
    )
    handles += ha
    labels += la

    # prior analytical
    if dist_space == "unconstrained":
        plot_i = _plot_normal_1d_i
    elif dist_space == "constrained":
        plot_i = _plot_uniform_1d_i
    else:
        raise ValueError(f"Unknown dist_space: {dist_space}")
    ha, la = plot_i(
        ax,
        i,
        dists[0],
        dist_labels[0],
        c=colors[0],
        # linestyle=LINESTYLES[0],
        **kwargs,
    )
    handles += ha
    labels += la

    # posterior(s)
    if len(dists) > 1:
        for j in range(1, len(dists)):

            # plot samples
            samples_ = samples[j][:, i] if samples[j] is not None else None
            ha, la = _plot_samples_1d(
                ax,
                samples=samples_,
                label=dist_labels[j],
                c=colors[j],
                # linestyle=LINESTYLES[j],
                lims=lim,
                do_samples=do_samples[j],
                **kwargs,
            )
            handles += ha
            labels += la

    # true parameters
    if true_params is not None:
        ha = ax.axvline(true_params[i], c=C_NATURE)  # type: ignore
        handles += [ha]
        labels += [true_params_label]

    return handles, labels


def _plot_normal_1d(
    ax: Axes,
    mu: float,
    var: float,
    label: str = "",
    normalize: bool = NORMALIZE,
    **kwargs: Any,
) -> Tuple[Any, str]:
    """Plot the density of a scalar normal distribution."""
    # # mean
    # p = ax.axvline(mu, **kwargs)

    # adjust lims for standard normal
    is_std = mu == 0 and var == 1
    if is_std:
        lims = (-3, 3)
        ax.set_xlim(lims)
    else:
        lims = ax.get_xlim()

    scale = 1.0
    if normalize:
        scale = lims[1] - lims[0]  # normalize to uniform height

    # # variance (if significant. too pointy pdfs distort ylims.)
    # xlims = ax.get_xlim()
    # xrange = xlims[1] - xlims[0]
    # if var > 0.01 * xrange:
    sigma = np.sqrt(var)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    (p,) = ax.plot(x, stats.norm.pdf(x, mu, sigma) * scale, **kwargs)
    return p, label


def _plot_uniform_1d(
    ax: Axes, lower: float, upper: float, label: str = "", **kwargs: Any
) -> Tuple[Any, str]:
    """Plot the density of a scalar uniform distribution."""
    value = 1 / (upper - lower)  # integrates to one
    if NORMALIZE:
        value = 1
    x, y = [lower, lower, upper, upper], [0, value, value, 0]
    (p,) = ax.plot(x, y, **kwargs)
    return p, label


def _plot_samples_1d(
    ax: Axes,
    samples: NDArray[np.float64],
    label: str = "",
    c: Optional[str] = "b",
    linestyle: str = "-",
    lims: Optional[Tuple[float, float]] = None,
    do_kde: bool = True,
    do_samples: bool = True,
    ylim: float = 5.0,
    **kwargs: Any,
) -> Tuple[List[Any], List[str]]:
    """Plot samples as a histogram."""
    handles, labels = [], []

    # sample histogram
    if do_samples:
        bin_values = None
        if len(samples) > NBINS:
            bin_values, _, ha = _plot_hist_1d(ax, samples, facecolor=c, **kwargs)
        else:
            for s in samples:
                ha = ax.axvline(
                    s, c=c, alpha=ALPHA, bin_values=bin_values
                )  # type: ignore[assignment]
        handles.append(ha)
        labels.append(label + " (samples)")

    # for arbitrary samples, plot a kde fit
    if do_kde:
        ha = _plot_kde_1d(
            ax, samples, color=c, lims=lims, linestyle=linestyle, **kwargs
        )
        handles.append(ha)
        labels.append(label + " (kde)")

    if NORMALIZE:
        ax.set_ylim((0, ylim))

    return handles, labels


def _plot_hist_1d(
    ax: Axes,
    samples: NDArray[np.float64],
    normalize: bool = NORMALIZE,
    **kwargs: Any,
):
    if normalize:
        # uniform corresponds to height 1, so scale to 1 sample per bin
        n_samples = samples.shape[0]
        weights = [NBINS / n_samples for _ in range(n_samples)]
        density = False
    else:
        weights = None
        density = True
    return ax.hist(
        samples,
        bins=NBINS,
        density=density,
        alpha=ALPHA,
        weights=weights,
        zorder=ZORDERS["data"],
        **kwargs,
    )


def _plot_kde_1d(
    ax: Axes,
    samples: NDArray[np.float64],
    color: str = "k",
    lims: Optional[Tuple[float, float]] = None,
    bin_values: Optional[List[float]] = None,
    normalize: bool = NORMALIZE,
    **kwargs: Any,
) -> Any:
    """Fit and plot a 1D kernel density estimate."""
    # kde cannot handle nans. Delete samples that contain nans.

    # make 2d
    if len(samples.shape) == 1:
        samples = samples[np.newaxis, :]
    ss = crop_nans(samples.T, verbose=False).T
    dens = gaussian_kde(ss)
    if lims is None:
        lims = (np.nanmin(samples), np.nanmax(samples))
    x = np.linspace(lims[0], lims[1], 100)
    y = dens.evaluate(x)

    # scale such that uniforms are at same height
    if normalize:
        if bin_values is not None:
            # such that kde sits right on top of histogram
            ymax = np.max(bin_values)
            ymax_now = np.max(y)
            y *= ymax / ymax_now * 0.95
        else:
            # normalize to uniform height given by automatic vlims
            y *= lims[1] - lims[0]

    (p,) = ax.plot(x, y, color=color, **kwargs)

    # # add vline for mode of kde
    # mode = x[np.argmax(y)]
    # ax.axvline(mode, color=color, **kwargs)

    return p
