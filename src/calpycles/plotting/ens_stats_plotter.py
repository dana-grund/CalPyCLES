"""Plot ensemble statistics."""

# Standard library
import os

# Third-party
import xarray as xr

# First-party
from calpycles.plotting import ZORDERS

# -------- Allow for crappy plotting code  -------- #
# pylint: skip-file
# mypy: ignore-errors
# noqa


class EnsStatsPlotter:
    """Plot ensemble statistics."""

    def __init__(
        self,
        args=None,
        ens_stats_path=None,
        ens_stats_title=None,
        path=None,
        n_samples=None,
        color="grey",
    ):
        """Initialize."""
        # select whether to use args (from argparse) or kwargs
        if args is not None:
            self.do_plot = args.ens_stats
            self.ens_stats_path = args.ens_stats_path
            self.title = args.ens_stats_title
            self.path = args.path
        else:
            self.do_plot = True
            self.ens_stats_path = ens_stats_path
            self.title = ens_stats_title
            self.path = path

        if self.do_plot:
            if self.ens_stats_path is None:
                self.ens_stats_path = os.path.join(self.path, "ensemble_stats")

            if n_samples is None:
                # if n_samples==None, infer from available files
                n_samples = [
                    f for f in os.listdir(self.ens_stats_path) if f.startswith("Nens")
                ]
                n_samples = max([int(f.split("_")[0][4:]) for f in n_samples])
            self.n_samples = n_samples
            self.savename = f"ensemble_Nens{self.n_samples}"
        self.color = color
        self.median_kwargs = {
            "color": color,
            "linestyle": "--",
            "label": "Median",
            "zorder": ZORDERS["ens_mean"],
        }
        self.mean_kwargs = {
            "color": color,
            "label": "Mean",
            "zorder": ZORDERS["ens_mean"],
        }
        self.ci50_kwargs = {
            "color": color,
            "alpha": 0.25,
            "label": "Spread (50%, 90%)",
            "zorder": ZORDERS["ens_spread"],
        }
        self.ci90_kwargs = {
            "color": color,
            "alpha": 0.25,
            "zorder": ZORDERS["ens_spread"],
        }

    def plot_profile(self, ax, **kwargs):
        """..."""
        return self.plot(ax, stats_type="profiles", **kwargs)

    def plot_timeseries(self, ax, **kwargs):
        """..."""
        return self.plot(ax, stats_type="timeseries", **kwargs)

    def _plot_spectra(self, ax, **kwargs):
        """..."""
        return self.plot(ax, stats_type="spectra", **kwargs)

    def plot(
        self,
        ax,
        stats_type="profiles",
        name="",
        t=None,
        z=None,
        do_average_time=False,
        return_handles=False,
    ):
        """..."""
        if self.do_plot:

            # get ens ds
            profile_average = do_average_time and stats_type == "profiles"
            t_str = "_average_time" if profile_average else ""
            ds_mean = xr.open_dataset(
                os.path.join(
                    self.ens_stats_path,
                    f"Nens{self.n_samples}_{stats_type}{t_str}_mean.nc",
                )
            )
            ds_median = xr.open_dataset(
                os.path.join(
                    self.ens_stats_path,
                    f"Nens{self.n_samples}_{stats_type}{t_str}_median.nc",
                )
            )
            ds_CI = xr.open_dataset(
                os.path.join(
                    self.ens_stats_path,
                    f"Nens{self.n_samples}_{stats_type}{t_str}_CI.nc",
                )
            )
            time = "t" if "t" in ds_mean else "time"

            # plot
            if stats_type == "profiles":
                if t is None and not profile_average:
                    t = ds_mean[time].values[-1]
                if not profile_average:
                    ds_mean = ds_mean.sel(**{time: t})
                    ds_median = ds_median.sel(**{time: t})
                    ds_CI = ds_CI.sel(**{time: t})

                z = ds_mean.z.to_numpy().ravel()
                p1 = ax.plot(ds_mean[name], z, **self.mean_kwargs)[0]
                p2 = ax.plot(ds_median[name], z, **self.median_kwargs)[0]
                p3 = ax.fill_betweenx(
                    z,
                    ds_CI[name].sel(quantile=0.25),
                    ds_CI[name].sel(quantile=0.75),
                    **self.ci50_kwargs,
                )
                ax.fill_betweenx(
                    z,
                    ds_CI[name].sel(quantile=0.05),
                    ds_CI[name].sel(quantile=0.95),
                    **self.ci90_kwargs,
                )

                handles = [p1, p2, p3]

            if stats_type == "timeseries":
                # ax.plot(
                # ds[time], ds[name], label=sample_names[j],
                # linestyle=linestyles[j], c='k'
                # )
                t = ds_mean[time]

                if name in ["cloud_base", "cloud_top"]:
                    # moving averages
                    nt = len(t)
                    w = min(30, nt // 2)  # window size: w*dt
                    mean = ds_mean[name].rolling({time: w}, center=True).mean()
                    median = ds_median[name].rolling({time: w}, center=True).mean()
                    CI25 = (
                        ds_CI[name]
                        .sel(quantile=0.25)
                        .rolling({time: w}, center=True)
                        .mean()
                    )
                    CI75 = (
                        ds_CI[name]
                        .sel(quantile=0.75)
                        .rolling({time: w}, center=True)
                        .mean()
                    )
                    CI05 = (
                        ds_CI[name]
                        .sel(quantile=0.05)
                        .rolling({time: w}, center=True)
                        .mean()
                    )
                    CI95 = (
                        ds_CI[name]
                        .sel(quantile=0.95)
                        .rolling({time: w}, center=True)
                        .mean()
                    )
                else:
                    mean = ds_mean[name].sel(**{time: t})
                    median = ds_median[name].sel(**{time: t})
                    CI25 = ds_CI[name].sel(quantile=0.25).sel(**{time: t})
                    CI75 = ds_CI[name].sel(quantile=0.75).sel(**{time: t})
                    CI05 = ds_CI[name].sel(quantile=0.05).sel(**{time: t})
                    CI95 = ds_CI[name].sel(quantile=0.95).sel(**{time: t})
                p1 = ax.plot(t, mean, **self.mean_kwargs)[0]
                p2 = ax.plot(t, median, **self.median_kwargs)[0]
                p3 = ax.fill_between(t, CI25, CI75, **self.ci50_kwargs)
                ax.fill_between(t, CI05, CI95, **self.ci90_kwargs)

                handles = [p1, p2, p3]

            if stats_type == "spectra":
                if z is None:
                    z = ds_mean["z"].values[-1]
                # ax.loglog(
                # ds.wavenumber, ds[name].sel(z=z,method="nearest").sel(**{time: t}),
                # label=f"{t=}, {z=}", linestyle=linestyles[j], c='k'
                # )
                k = ds_mean.wavenumber
                ds_CI = ds_CI.sel(z=z, method="nearest").sel(**{time: t})
                p1 = ax.loglog(
                    k,
                    ds_mean[name].sel(z=z, method="nearest").sel(**{time: t}),
                    **self.mean_kwargs,
                )[0]
                p2 = ax.loglog(
                    k,
                    ds_median[name].sel(z=z, method="nearest").sel(**{time: t}),
                    **self.median_kwargs,
                )[0]
                p3 = ax.fill_between(
                    k,
                    ds_CI[name].sel(quantile=0.25),
                    ds_CI[name].sel(quantile=0.75),
                    **self.ci50_kwargs,
                )
                ax.fill_between(
                    k,
                    ds_CI[name].sel(quantile=0.05),
                    ds_CI[name].sel(quantile=0.95),
                    **self.ci90_kwargs,
                )

                handles = [p1, p2, p3]

            ds_mean.close()
            ds_median.close()
            ds_CI.close()

            if return_handles:
                return handles

    def add_to_title(self, title):
        """..."""
        if self.do_plot and self.title is not None:
            title += f" ({self.title})"
        return title

    def add_to_savename(self, savename):
        """..."""
        if self.do_plot and self.savename is not None:
            savename += f"_{self.savename}"
        return savename
