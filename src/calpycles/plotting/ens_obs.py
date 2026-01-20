# Standard library
import os
from typing import List
from typing import Optional

# Third-party
import numpy as np
import xarray as xr

# First-party
from calpycles.case_inputs import names_by_case
from calpycles.plotting import C_MEAS
from calpycles.plotting import C_NATURE
from calpycles.plotting import COLORS
from calpycles.plotting import ZORDERS
from calpycles.plotting import color_handles
from calpycles.plotting import get_name_mpl
from calpycles.plotting import make_figure
from calpycles.plotting import save_figure
from calpycles.plotting.ens_stats_plotter import EnsStatsPlotter

# -------- Allow for crappy plotting code  -------- #
# pylint: skip-file
# mypy: ignore-errors
# noqa


# --- ensemble observations

Z_MAX = 1100  # max vertical extend for profile plots


def plot_ens_obs(
    ensembles=[],
    ens_paths=[],  # only used if ensembles not given
    n_ens_list=[],  # only used if ensembles not given
    ens_labels=[],
    data=None,
    nature=None,
    nature_name=None,
    samples=[],
    samples_obs_full_list=[],  # only used if samples is not given
    sample_names=[],
    sample_colors=COLORS,
    sample_linestyles=[],
    save_file="fig",
    nrows=3,
    cloud_height_type="mean",
    add_s05_les_ensemble=False,
):
    """Create subplots of selected observations.

    cloud_height_type:
        "mean": variables cloud_base_mean, cloud_top_mean
        "minmax": variables cloud_base, cloud_top
    """
    # prepare ensembles
    if len(ensembles) > 0:
        ens_paths = [ens.path for ens in ensembles]
        n_ens_list = [ens.n_samples for ens in ensembles]
    elif len(n_ens_list) > 0:
        assert len(ens_paths) == len(
            n_ens_list
        ), "n_ens and ens_path have to be given if ensembles not given."
    ens_plotters = [
        EnsStatsPlotter(
            n_samples=n_ens,
            color=color,
            path=ens_path,
            ens_stats_title="",
        )
        for ens_path, n_ens, color in zip(ens_paths, n_ens_list, COLORS)
    ]

    # prepare samples
    profiles = None
    timeseries = None
    if len(samples) > 0:
        samples_obs_full_list = [sample.observable.obs_xr_full for sample in samples]
    if len(samples_obs_full_list) > 0:
        profiles = [obs_full[0] for obs_full in samples_obs_full_list]
        timeseries = [obs_full[1] for obs_full in samples_obs_full_list]
        if len(sample_names) == 0:
            sample_names = [f"Sample {i+1}" for i in range(len(samples_obs_full_list))]
        if len(sample_linestyles) == 0:
            sample_linestyles = ["-"] * len(samples_obs_full_list)

    profile_variables = [
        "w_mean2",
        "w_mean3",
        "thetali_mean",
        "qt_mean",
        "ql_mean",
    ]

    # make figure and decide on axes ordering
    if nrows == 3:  # assume a4 full-col paper format
        fig, axs = make_figure(3, 2, sharey="row", fraction=0.8)
        # profile_variables.append("cloud_fraction")
        axs_profiles = np.concatenate([axs[0], axs[1, :2]])
        ax_ts = axs[1, 2]
        ax_legend = axs[1, 1]  # with ql
        separate_legend = False
        legend_loc = "lower right"
        ax_ts.set_ylim(300, Z_MAX)

    elif nrows == 2:  # assume tighter ppt format
        fig, axs = make_figure(4, 2, sharey="row", fraction=0.8)
        profile_variables.append("cloud_fraction")
        axs_profiles = np.concatenate([axs[0], axs[1, :2]])
        ax_ts = axs[1, 2]
        ax_legend = axs[1, 3]
        separate_legend = True
        legend_loc = "upper left"
        ax_ts.set_ylim(300, Z_MAX)

    elif nrows == 1:  # assume wide ppt format
        fig, axs = make_figure(7, 1, sharey="row", width=20)
        # skip cloud fraction since there is no data
        axs_profiles = axs[0:5]
        ax_ts = axs[5]
        ax_legend = axs[6]
        separate_legend = True
        legend_loc = "upper left"
        axs = axs.reshape(1, len(axs))

    else:
        raise ValueError(f"nrows={nrows} not supported.")

    axs_timeseries = np.array([ax_ts, ax_ts])

    # legend subplot
    if separate_legend:
        ax_legend.axis("off")
        ax_legend.set_title("")
        ax_legend.set_xlabel("")
        ax_legend.set_ylabel("")

    # profiles
    kwargs = {
        "single": [],
        "single_labels": [],
        "single_colors": [],
        "single_linestyles": [],
    }
    if nature is not None:
        kwargs["single"].append(nature.observable.obs_xr_full[0])
        kwargs["single_labels"].append(nature_name)
        kwargs["single_colors"].append(C_NATURE)
        kwargs["single_linestyles"].append("-")
    if profiles is not None:
        for profile, name, c, ls in zip(
            profiles, sample_names, sample_colors, sample_linestyles
        ):
            kwargs["single"].append(profile)
            kwargs["single_labels"].append(name)
            kwargs["single_colors"].append(c)
            kwargs["single_linestyles"].append(ls)
    if add_s05_les_ensemble:
        for ax, var in zip(axs_profiles, profile_variables):
            _plot_S05(ax, var)
    _plot_profile_ens_DYCOMS_RF01(
        axs_profiles,
        ens_plotters,
        None,
        ens_labels,
        profile_variables,
        make_legend=False,  # needs extra axis
        **kwargs,
    )

    # measurement data
    if data is not None:
        for ax, var in zip(axs_profiles, profile_variables):
            data.plot_profile(ax, var)

    for ax, var in zip(axs_profiles, profile_variables):
        ax.set_ylim(0, Z_MAX)
        ax.set_xlabel(get_name_mpl(var))
    for ax in axs[:, 0]:
        ax.set_ylabel(get_name_mpl("z"))

    # timeseries
    timeseries_variables = timeseries_variables_s05 = [
        "cloud_base",
        "cloud_top",
    ]  # cloud_height_type == "minmax"

    if cloud_height_type == "mean":
        timeseries_variables = [
            "cloud_base_mean",
            "cloud_top_mean",
        ]

    # stevens 05 ensemble
    handles_s05 = []
    if add_s05_les_ensemble:
        for var in timeseries_variables_s05:
            handles_s05 = _plot_S05_ts(ax_ts, var)

    # choose whether to plot nature
    kwargs = {
        "single": [],
        "single_labels": [],
        "single_colors": [],
        "single_linestyles": [],
    }
    if nature is not None:
        kwargs["single"].append(nature.observable.obs_xr_full[1])
        kwargs["single_labels"].append(nature_name)
        kwargs["single_colors"].append(C_NATURE)
        kwargs["single_linestyles"].append("-")
    if timeseries is not None:
        for ts, name, c, ls in zip(
            timeseries, sample_names, sample_colors, sample_linestyles
        ):
            kwargs["single"].append(ts)
            kwargs["single_labels"].append(name)
            kwargs["single_colors"].append(c)
            kwargs["single_linestyles"].append(ls)
    # ensemble and nature
    handles = _plot_timeseries_ens_DYCOMS_RF01(
        axs_timeseries,
        ens_plotters,
        None,
        ens_labels,
        timeseries_variables,
        make_legend=False,  # needs extra axis
        legend_loc=legend_loc,
        **kwargs,
    )

    # measurement data
    if data is not None:
        for ax, var in zip(axs_timeseries, timeseries_variables):
            data.plot_timeseries(ax, var)

    # ts decoration
    ax_ts.set_xlabel(get_name_mpl("t"))
    ax_ts.text(1000, 1030, "Cloud top and base")

    # make legend
    handles_data = color_handles(["Meas. data"], [C_MEAS])
    legend = ax_legend.legend(
        handles=handles_data + handles_s05 + handles, loc=legend_loc
    )
    legend.set_zorder(ZORDERS["legend"])

    save_figure(save_file)


def plot_ens_obs_bayes(exp, **kwargs):
    """Plot simulated observations for prior and posterior ensemble."""
    ensembles = [exp.ensemble_prior]
    ens_labels = ["Prior"]
    if exp.ensemble_posterior.was_run:
        ensembles.append(exp.ensemble_posterior)
        ens_labels.append("Posterior")
    ens_labels = [
        f"{label} (N={ens.n_samples})" for label, ens in zip(ens_labels, ensembles)
    ]
    plot_ens_obs(
        ensembles,
        ens_labels,
        data=exp.data.data,
        **kwargs,
    )


# --- helpers


def _plot_profile_ens_DYCOMS_RF01(
    axs,
    ens_plotters,
    data,
    ens_labels,
    profile_variables,
    c_data=C_MEAS,
    single: Optional[List[xr.Dataset]] = None,
    single_colors: Optional[List[str]] = None,
    single_linestyles: Optional[List[str]] = None,
    single_labels: Optional[List[str]] = None,
    data_label: Optional[str] = None,
    make_legend: Optional[bool] = True,
) -> None:
    """Add plots of ensemble, data, and single samples."""
    if single_colors is None:
        single_colors = COLORS[1:]  # skip first which is grey
    if single_linestyles is None and single is not None:
        single_linestyles = ["-"] * len(single)
    colors = [e.color for e in ens_plotters]
    labels = ens_labels.copy()
    if data is not None:
        if data_label is None:
            data_label = "Meas. data"
        labels.insert(0, data_label)
        colors.insert(0, c_data)

    # plot all profiles in subplots
    for i, name in enumerate(profile_variables):
        ax = axs[i]

        he = []
        if data is not None:
            data.plot_profile(
                ax,
                name=name,
                c=c_data,
                label=None,
                zorder=ZORDERS["data"],  # data below ens
            )

        for i, ens_plotter in enumerate(ens_plotters):
            he = ens_plotter.plot_profile(
                ax, name=name, return_handles=True, do_average_time=True
            )
        hs = []
        if single is not None:
            for j, ds in enumerate(single):
                c = C_NATURE if len(single) == 1 else single_colors[j]
                hs += _plot_profile_single(
                    ax,
                    ds,
                    name,
                    label=single_labels[j],
                    c=c,
                    linestyle=single_linestyles[j],
                )

    handles = color_handles(labels, colors) + he + hs
    if make_legend:
        axs[-1].legend(handles=handles, loc="upper left")


def _plot_profile_single(ax, ds, name, **kwargs):
    """Add profile plot of a single sample, averaged over times."""
    average_times = names_by_case["DYCOMS_RF01"]["average_times"]
    t_str = "t" if "t" in ds else "time"
    return ax.plot(
        ds[name].sel({t_str: average_times}).mean(dim=t_str).values.ravel(),
        ds.z,
        zorder=ZORDERS["single"],
        **kwargs,
    )


def _plot_timeseries_ens_DYCOMS_RF01(
    axs,
    ens_plotters,
    data,
    ens_labels,
    timeseries_variables,
    c_data=C_MEAS,
    single: Optional[List[xr.Dataset]] = None,
    single_colors: Optional[List[str]] = None,
    single_linestyles: Optional[List[str]] = None,
    single_labels: Optional[List[str]] = None,
    data_label: Optional[str] = None,
    make_legend: Optional[bool] = True,
    legend_loc: str = "upper left",
) -> None:
    """Add plots of ensemble, data, and single samples."""
    if single_colors is None:
        single_colors = COLORS[1:]  # skip first which is grey
    if single_linestyles is None and single is not None:
        single_linestyles = ["-"] * len(single)
    colors = [e.color for e in ens_plotters]
    labels = ens_labels.copy()
    if data is not None:
        if data_label is None:
            data_label = "Meas. data"
        labels.insert(0, data_label)
        colors.insert(0, c_data)

    # plot all timeseries in subplots
    for i, name in enumerate(timeseries_variables):
        ax = axs[i]

        he = []
        if data is not None:
            data.plot_timeseries(
                ax,
                name=name,
                c=c_data,
                label=None,
                zorder=ZORDERS["data"],  # data below ens
            )

        for i, ens_plotter in enumerate(ens_plotters):
            he = ens_plotter.plot_timeseries(ax, name=name, return_handles=True)
        hs = []
        if single is not None:
            for j, ds in enumerate(single):
                c = C_NATURE if len(single) == 1 else single_colors[j]
                hs += _plot_timeseries_single(
                    ax,
                    ds,
                    name,
                    label=single_labels[j],
                    c=c,
                    linestyle=single_linestyles[j],
                )

    handles = color_handles(labels, colors) + he + hs
    if make_legend:
        legend = axs[-1].legend(handles=handles, loc=legend_loc)
        legend.set_zorder(ZORDERS["legend"])

    return handles


def _plot_timeseries_single(ax, ds, name, **kwargs):
    """Add timeseries plot of a single sample."""
    t_str = "t" if "t" in ds else "time"
    t = ds[t_str]
    data = ds[name]

    # apply moving average on cloud boundaries
    if name in ["cloud_base", "cloud_top"]:
        w = min(30, len(t) // 2)  # window size: w*dt
        data = data.rolling({t_str: w}, center=True).mean()

    return ax.plot(t, data.values.ravel(), zorder=ZORDERS["single"], **kwargs)


# --- plot the LES ensemble by Stevens et al. 2005 (provided by B. Stevens)
# hour refers to the hour of the simulation the ensemble was averaged over
# moment (isel): 0=avg, 1=stddev, 2=min, 3=max, 4=1st qrtl, 5=3rdqrtl


C_S05 = COLORS[0]


def open_s05():
    this_folder = os.path.dirname(__file__)
    ds = xr.open_dataset(
        os.path.join(this_folder, "Stevens_2005_LES_ensemble_gcss7.nc")
    )

    # rename variables
    ds = ds.rename(
        {
            "w_var": "w_mean2",
            "w_skw": "w_mean3",
            "thetal": "thetali_mean",
            "rt": "qt_mean",
            "rl": "ql_mean",
            "zi_bar": "cloud_top",
            "zb_bar": "cloud_base",
        }
    )

    return ds


def _plot_S05(ax, var):

    ds = open_s05()

    # get labels to make sure we don't overwrite them
    prev_label_x = ax.get_xlabel()
    prev_label_y = ax.get_ylabel()

    kwargs = dict(
        ax=ax,
        c=C_S05,
    )

    # inconsistent dimension naming??
    if "zw" in ds[var].coords:
        kwargs["y"] = "zw"
    elif "zt" in ds[var].coords:
        kwargs["y"] = "zt"
    else:
        print("No vertical coordinate found")

    ax.plot(
        ds[var].isel(hour=-1).sel(moment=1),
        ds[kwargs["y"]],
        label="S05 LES mean",
        linestyle="-",
        c=C_S05,
    )[0]

    z = ds[kwargs["y"]].to_numpy().ravel()

    ax.fill_betweenx(
        z,
        ds[var].isel(hour=-1).sel(moment=3),
        ds[var].isel(hour=-1).sel(moment=4),
        label="S05 LES min/max",
        color=C_S05,
        alpha=0.25,
    )

    ax.fill_betweenx(
        z,
        ds[var].isel(hour=-1).sel(moment=5),
        ds[var].isel(hour=-1).sel(moment=6),
        label="S05 LES quartiles",
        color=C_S05,
        alpha=0.25,
    )

    ax.set_title("")
    ax.set_xlabel(prev_label_x)
    ax.set_ylabel(prev_label_y)
    ds.close()

    # return [p1,p2,p3]
    return color_handles(["S05 LES ensemble"], [C_S05])


def _plot_S05_ts(ax, var):

    ds = open_s05()

    # get labels to make sure we don't overwrite them
    prev_label_x = ax.get_xlabel()
    prev_label_y = ax.get_ylabel()

    ax.plot(
        ds.time, ds[var].sel(moment=1), label="S05 LES mean", linestyle="-", c=C_S05
    )[0]

    time = ds["time"].to_numpy().ravel()

    ax.fill_between(
        time,
        ds[var].sel(moment=3),
        ds[var].sel(moment=4),
        label="Quartiles, min/max",
        color=C_S05,
        alpha=0.25,
    )

    ax.fill_between(
        time,
        ds[var].sel(moment=5),
        ds[var].sel(moment=6),
        color=C_S05,
        alpha=0.25,
    )

    ax.set_title("")
    ax.set_xlabel(prev_label_x)
    ax.set_ylabel(prev_label_y)
    ds.close()

    # return [p1,p2,p3]
    return color_handles(["S05 LES ensemble"], [C_S05])
