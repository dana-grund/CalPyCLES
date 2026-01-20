"""Plotting functionality for two-dinemsional (p1 vs. p2) distribution plots."""

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
from calpycles.helpers import ensure_list
from calpycles.plotting import ALPHA
from calpycles.plotting import C_NATURE
from calpycles.plotting import COLORS
from calpycles.plotting import MARKERSIZE
from calpycles.plotting import SUBPLOTSIZE
from calpycles.plotting import get_name_mpl
from calpycles.plotting import make_figure
from calpycles.plotting import save_figure
from calpycles.plotting.distributions_1d import _plot_dist_1d
from calpycles.plotting.distributions_helpers import prepare
from calpycles.random_variables import GaussRV

# -------- Allow for crappy plotting code  -------- #
# pylint: skip-file
# mypy: ignore-errors
# noqa


# -------- Main -------- #


def plot_dists_2d(
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
    colors: List[str] = COLORS,
    **kwargs: Any,
) -> None:
    """Plot distributions in (un)constrained 2d space.

    Distributions have to be ordered as [prior, post, post, ...] since the first is
    plotted analytically while the others are approximated by kde.


    Creates a lower-triangular grid of subplots of shape (n_params, n_params),
    with histograms on the diagonal, and scattered samples and contours of
    distributions in the lower triangle.
    """
    # add parameter idcs to savename
    params_str = ""
    if params_to_plot is not None:
        params_str = "_" + "".join([str(i) for i in params_to_plot])

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

    dist_prior = dists[0] if isinstance(dists, list) else dists
    param_names = [dist_prior.names[i] for i in params_to_plot]
    fig, axs = make_figure_2d(
        param_names, vlims=[lims[i] for i in params_to_plot], add_units=add_units
    )

    # keep track of legend
    handles, labels = [], []

    # fill each pairwise suplot
    for j_plot, j in enumerate(params_to_plot):
        for i_plot, i in enumerate(params_to_plot):
            ax = axs[j_plot, i_plot]

            # --- lower triangle: scatters
            if i < j:

                # prior samples
                if do_samples[0]:
                    ha, la = _plot_samples_2d(
                        ax,
                        samples_[0][:, [i, j]].T,
                        label=dist_labels[0],
                        c=colors[0],
                        do_kde=False,
                    )
                    if i_plot == 0 and j_plot == 1:
                        handles += ha
                        labels += la

                # prior analytical
                if dist_space == "unconstrained":  # normal
                    mu, cov = dist_prior.mu, dist_prior.C.full
                    mu_ij = np.array([mu[i], mu[j]])
                    cov_ij = np.array([[cov[i, i], cov[i, j]], [cov[j, i], cov[j, j]]])
                    d = _plot_normal_2d(ax, mu_ij, cov_ij, color=colors[0])
                elif dist_space == "constrained":  # uniform
                    lower, upper = dist_prior.lower, dist_prior.upper
                    d = _plot_uniform_2d(
                        ax,
                        np.array([lower[i], lower[j]]),
                        np.array([upper[i], upper[j]]),
                        color=colors[0],
                    )
                else:
                    raise ValueError(f"Unknown dist_space: {dist_space}")
                if i_plot == 0 and j_plot == 1:
                    handles += [d]
                    labels += [dist_labels[0]]

                # posterior(s)
                if len(dists) > 1:
                    for d in range(1, len(dists)):
                        # posterior samples and kde fit
                        ha, la = _plot_samples_2d(
                            ax,
                            samples_[d][:, [i, j]].T,
                            label=dist_labels[d],
                            c=colors[d],
                            do_samples=do_samples[d],
                        )
                        if i_plot == 0 and j_plot == 1:
                            handles += ha
                            labels += la

            # --- diagonal: 1d distributions
            if i == j:
                ha, la = _plot_dist_1d(
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
                    colors=colors,
                )

    # --- true parameters
    if true_params is not None:
        true_params_ = [true_params_[i] for i in params_to_plot]
        p = _plot_true_params_2d(axs, true_params_)  # type: ignore
        handles += [p]
        labels += [true_params_label]

    # --- legend
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    save_file = os.path.join(plot_dir, f"fig-dists_2d_{name}_{dist_space}{params_str}")
    save_figure(save_file, show=show)


# -------- Helpers -------- #


def _plot_normal_2d(
    ax: Axes,
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
    color: str = "k",
    linestyle: str = "-",
) -> Any:
    """Plot contour intervals of the 2D Gaussian indicating highest density regions.

    Mathematical strategy (by github copilot):
        - pdf_normal(x) = pdf_normal(mu) * exp(-1/2 * D2(x))
        - Mahalanobis norm D2(x) = (x-mu)^T C^{-1} (x-mu)
        - The HDR (highest density region) that contains a certain portion
          of the probability in pdf_normal is an ellipse around mu
          given by P(D2(x) <= c)
        - D2(x) follows a chi squared distribution:
            - (x-mu) follows N(0,C)
            - D2(x) follows the same as y^T y, where y is N(0,I), which is chi squared
        - So from P(D2(x) <= c) = p, we get c =ppf_chi2(p) = cdf_chi2^{-1}(p)
        - To obtain the pdf_norm value to draw the contour level, we use
          the above formula, pdf_normal(mu) * exp(-1/2 * c)

    The critical values are pre-computed in c_list by
        percents = [0.99, 0.9, 0.5, 0.1, 0.01]
        c_list = [stats.chi2.ppf(alpha, df=2) for alpha in percents]

    """
    _, _, xx, yy = get_meshgrid(ax)
    rv = stats.multivariate_normal(mu, cov, allow_singular=True)
    pdf = rv.pdf(np.dstack((xx, yy)))

    # --- calculate HDR levels (highest density regions)
    c_list = [9.210340, 4.605170, 1.386294, 0.210721, 0.020101]  # pre-computed

    # --- convert critical chi squared values to levels of multivar normal pdf
    levels = [rv.pdf(mu) * np.exp(-0.5 * c) for c in c_list]
    # linestyle = "--" if color == "b" else "-"  # prior: dashed
    p, _ = ax.contour(
        xx, yy, pdf, colors=[color], levels=levels, linestyles=linestyle
    ).legend_elements()

    return p[0]


def _plot_uniform_2d(
    ax: Axes, lower: NDArray[np.float64], upper: NDArray[np.float64], **kwargs: Any
) -> Any:
    """Plot the 2D uniform distribution as a rectangle."""
    (p,) = ax.plot(
        [lower[0], upper[0], upper[0], lower[0], lower[0]],
        [lower[1], lower[1], upper[1], upper[1], lower[1]],
        **kwargs,
    )
    return p


def _plot_samples_2d(
    ax: Axes,
    samples: NDArray[np.float64],
    label: str = "",
    c: Optional[str] = None,
    do_kde: bool = True,
    do_samples: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot samples as a scatter plot in the 2D subplots."""
    handles, labels = [], []
    if do_samples:
        s = ax.scatter(samples[0], samples[1], s=MARKERSIZE, alpha=ALPHA, c=c, **kwargs)
        handles.append(s)
        labels.append(label + " (samples)")
    if do_kde:
        if c is None:
            c = s.get_edgecolors()[0]
        p = _plot_kde_2d(ax, samples, color=c, **kwargs)
        handles.append(p)
        labels.append(label + " (kde)")

    return handles, labels


def _plot_kde_2d(
    ax: Axes, samples: NDArray[np.float64], color: str = "k", linestyle: str = "-"
) -> Any:
    """Fit and plot a 2D kernel density estimate.

    Inspired from here: https://stackoverflow.com/a/79561690
    """
    x, y, xx, yy = get_meshgrid(ax)

    # kde cannot handle nans. Delete samples that contain nans.
    dens = gaussian_kde(crop_nans(samples.T, verbose=False).T)
    xy = np.vstack((xx.flatten(), yy.flatten()))
    z = dens.evaluate(xy)
    z = z.reshape(xx.shape)
    p, _ = ax.contour(
        x, y, z, colors=[color], linestyles=linestyle, levels=4
    ).legend_elements()

    return p[0]


def _plot_true_params_2d(
    axs: Any,
    true_params: NDArray[np.float64],
) -> Any:
    """Plot the true parameters as a star in the pairwise subplots."""
    p = None
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            ax = axs[j, i]
            if i < j:
                p = ax.scatter(
                    true_params[i],
                    true_params[j],
                    s=MARKERSIZE,
                    color=C_NATURE,
                    zorder=10,
                )
    assert p is not None, "No plots were created."
    return p


# -------- 2D Plots: figure -------- #


def empty_x_axis(ax: Axes) -> None:
    """Remove x-axis labels and ticks."""
    ax.set_xticklabels([])
    ax.set_xlabel("")


def empty_y_axis(ax: Axes) -> None:
    """Remove y-axis labels and ticks."""
    ax.set_yticklabels([])
    ax.set_ylabel("")


def configure_axes(
    axs: Any,
    names: List[str],
    vlims: Optional[List[List[float]]],
    samples: Optional[Any] = None,
    add_units: bool = True,
) -> None:
    """Configure the axes of a lower-triangular grid of pairwise subplots."""
    # --- defaults
    n_subplots = len(names)
    if vlims is None:
        if samples is not None:
            vlims = get_central_vlims_from_samples(ensure_list(samples))
        else:
            vlims = [[-2, 2]] * n_subplots

    n_subplots = axs.shape[0]

    for j in range(n_subplots):
        for i in range(n_subplots):
            ax = axs[j, i]

            # --- general settings
            ax.grid = True
            ax.spines[["right", "top"]].set_visible(False)

            # --- axes limits
            if i < j:
                ax.set_xlim(vlims[i])
                ax.set_ylim(vlims[j])
            elif i == j:
                ax.set_xlim(vlims[i])
                ax.set_ylim((0, 10))  # normalised pdf of standard normal
                xrange = vlims[i][1] - vlims[i][0]
                ax.set_ylim((0, 10 / xrange))
            else:  # i>j
                ax.axis("off")

            # --- labels and ticklabels only for the outer subplots
            if j == np.shape(axs)[0] - 1:
                ax.set_xlabel(get_name_mpl(names[i], add_units=add_units))
            else:
                empty_x_axis(ax)
            if i == 0 and j > 0:
                ax.set_ylabel(get_name_mpl(names[j], add_units=add_units))
            else:
                empty_y_axis(ax)


def get_central_vlims_from_samples(
    samples: List[NDArray[np.float64]],
) -> List[List[float]]:
    """Infer centered vlims per parameter that cover all samples.

    Default and minimum: vlims [[-2,2]], suitable for a standard normal distribution
    """
    n_subplots = samples[-1].shape[1]
    abit = 0.2
    max_val = 2 * np.ones(n_subplots)
    for ss in samples:
        if ss is not None:
            _max_val = np.array([max(np.abs(ss[:, i])) for i in range(n_subplots)])
            max_val = np.maximum(max_val, _max_val)
    vlims = [[-max_val[i] - abit, max_val[i] + abit] for i in range(n_subplots)]

    return vlims


def get_meshgrid(
    ax: Any,
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Create the meshgrid given by the axes limits."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)

    xx, yy = np.meshgrid(x, y)
    return x, y, xx, yy


def make_figure_2d(
    names: List[str],
    samples: Optional[Any] = None,
    vlims: Optional[List[List[float]]] = None,
    add_units: bool = True,
) -> Tuple[Any, Any]:
    """Prepare the axes for plot_dist_2d().

    If vlims is None, assumes standard normal.
    """
    n_subplots = len(names)

    fig, axs = make_figure(
        n_subplots, n_subplots, ratio=1.0, width=n_subplots * SUBPLOTSIZE
    )
    configure_axes(axs, names, vlims, samples, add_units=add_units)

    return fig, axs
