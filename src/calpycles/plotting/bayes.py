# hist over prior ens + post ens

# Third-party
import matplotlib.pyplot as plt

# First-party
from calpycles.plotting import COLORS
from calpycles.plotting import make_figure
from calpycles.plotting import save_figure

# from calpycles.plotting.distributions_1d import _plot_hist_1d
from calpycles.plotting.distributions_1d import _plot_kde_1d
from calpycles.plotting.distributions_1d import _plot_normal_1d

# -------- Allow for crappy plotting code  -------- #
# pylint: skip-file
# mypy: ignore-errors
# noqa


def plot_data_lh(ax, mu, var, c="k", **kwargs):
    handle, _ = _plot_normal_1d(ax, mu, var, c=c, normalize=False, **kwargs)
    ax.axvline(mu, c=c, **kwargs)
    return handle


def plot_bayes_cloud_ts(
    Eo=None,
    Eo_post=None,
    Eo_post_pred=None,
    y=None,
    hnoise=None,
    color=COLORS[1],
    save_file="fig-bayes-cloud-ts",
):
    """Plot the prior, posterior, and likelihood for cloud base/top height/rate."""
    # obs indices
    i_base_h = 0
    i_base_r = 1
    i_top_h = 2
    i_top_r = 3

    fig, axs = make_figure(2, 2, horizontal=True, width=8)

    def _make_subplot(ax, i, xlabel):
        ax.set_xlabel(xlabel)

        handles = []
        labels = []

        lims = (
            Eo[:, i].min(),
            Eo[:, i].max(),
        )

        if y is not None and hnoise is not None:

            p = plot_data_lh(
                ax,
                y[i],
                var=hnoise.C.diag[i],
                c="k",
                label="Meas. data likelihood",
                zorder=10,
            )
            labels += ["Data likelihood"]
            handles.append(p)

        if Eo is not None:
            # _plot_hist_1d(ax,Eo[:,i], label="Prior", normalize=False, color=COLORS[0])
            p = _plot_kde_1d(
                ax, Eo[:, i], label="Prior", normalize=False, color=COLORS[0], lims=lims
            )
            # ax.axvline(np.nanmean(Eo[:,i]), c=COLORS[0])
            labels += ["Prior"]
            handles.append(p)

        if Eo_post is not None:
            # _plot_hist_1d(
            #     ax,Eo_post[:,i], label="Posterior (model eval.)",
            #     normalize=False, color=color
            # )
            p = _plot_kde_1d(
                ax,
                Eo_post[:, i],
                label="Posterior",
                normalize=False,
                color=color,
                lims=lims,
            )
            # ax.axvline(np.nanmean(Eo_post[:,i]), c=color)
            labels += ["Posterior (model evaluation)"]
            handles.append(p)

        if Eo_post_pred is not None:
            # _plot_hist_1d(
            #     ax,Eo_post_pred[:,i], label="Posterior (linear prediction)",
            #     normalize=False, color=c_pred
            # )
            p = _plot_kde_1d(
                ax,
                Eo_post_pred[:, i],
                label="Posterior (linear prediction)",
                normalize=False,
                color=color,
                linestyle="--",
                lims=lims,
            )
            # ax.axvline(np.nanmean(Eo_post_pred[:,i]), c=color, linestyle="--")
            labels += ["Posterior (linear prediction)"]
            handles.append(p)

        return handles, labels

    handles, labels = _make_subplot(axs[0, 0], i_top_h, "Cloud top height [m]")
    _make_subplot(axs[0, 1], i_top_r, "Cloud top rate [m/h]")
    _make_subplot(axs[1, 0], i_base_h, "Cloud base height [m]")
    _make_subplot(axs[1, 1], i_base_r, "Cloud base rate [m/h]")

    fig.legend(
        handles=handles[:2],
        labels=labels[:2],
        bbox_to_anchor=(0.4, 0.0),
    )

    if len(handles) > 2:
        fig.legend(
            handles=handles[2:],
            labels=labels[2:],
            bbox_to_anchor=(1, 0.0),
        )

    for ax in axs.flatten():
        ax.set_yticklabels([])

    plt.tight_layout()
    save_figure(save_file)
