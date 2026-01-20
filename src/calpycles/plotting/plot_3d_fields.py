"""Plot 3D fields from the CalPyCLES output files.

To Do:
- Simplify get_ds and tailor this module to calpycles settings
"""

# Standard library
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# First-party
from calpycles.case_inputs import names_by_case
from calpycles.case_inputs import nan_by_var
from calpycles.plotting import ColorManager
from calpycles.plotting import save_figure

# -------- Allow for crappy plotting code  -------- #
# pylint: skip-file
# mypy: ignore-errors
# noqa


def plot_3D_fields(
    path=None,
    outdir=None,
    s=None,
    merged=False,
    case="",
    savefolder="./",
    savename="",
    times=None,
):
    """..."""
    fields_ds = get_ds(path, outdir, s, "fields", merged)

    kwargs_ = dict(
        savefolder=savefolder,
        savename=savename,
        case=case,
    )
    # plot - DYCOMS
    if case == "DYCOMS_RF01":
        for var in names_by_case["DYCOMS_RF01"]["fields_lower"]:
            plot_field_evolution(
                fields_ds,
                str(var),
                z_levels=[900, 850, 800, 500, 100],
                times=times,
                **kwargs_,
            )
        for var in names_by_case["DYCOMS_RF01"]["fields_upper"]:
            plot_field_evolution(
                fields_ds,
                str(var),
                z_levels=[900, 875, 850, 825, 800],
                times=times,
                **kwargs_,
            )

    # plot - SullivanPatton
    elif case == "SullivanPatton":
        for var in names_by_case["SullivanPatton"]["fields"]:
            plot_field_evolution(
                fields_ds, str(var), z_levels=[1000, 550, 100], times=times, **kwargs_
            )

    else:
        print(f"Can't plot fields, since the case {case} is unknown")
    fields_ds.close()


def plot_field_evolution(
    ds, variable="w", times=None, z_levels=[800], case="", savefolder="./", savename=""
):
    """..."""
    da = ds[variable]

    t_str = "t" if "t" in da.dims else "time"
    if times is None:
        times = ds[t_str].values
    nt = len(times)
    nz = len(z_levels)

    # colors
    center = False
    if variable in ["u", "v", "w"]:
        center = True
    CM = ColorManager(
        values=da.values, cmap="seismic", cbarlabel=variable, center=center
    )

    # axes
    py, px = nz + 1, nt + 1
    fig, axs_ = plt.subplots(
        py,
        px,
        figsize=(3 * nt + 1, 3 * nz + 1.5),
        height_ratios=[0.5] + [1 for _ in range(nz)],
        width_ratios=[1 for _ in range(nt)] + [0.25],
        sharex=True,
        sharey=False,
    )
    cax = fig.add_subplot(1, px, px)  # spans the whole height

    for i in range(nt):
        axs_[-1, i].set_xlabel("x [m]")
        if i > 0:
            for j in range(nz + 1):
                axs_[j, i].set_yticks([])
                axs_[j, i].set_ylabel(None)

    # colorbar axis
    cax.set_visible(False)
    for j in range(nz + 1):
        axs_[j, -1].set_visible(False)
    CM.colorbar(cax, fraction=0.2, pad=0.05, shrink=1, aspect=10)

    # xz crosssections
    axs = axs_[0]
    axs[0].set_ylabel("z [m]")
    iy = len(da.y) // 2
    for i in range(nt):
        ax = axs[i]
        _ = _plot_field_xz(ax, da.sel({t_str: times[i]}), iy, CM)
        ax.set_title(f"t={times[i]} s")
    ax_ = ax.twinx()
    ax_.set_ylabel(f"y={da.y.isel(y=iy).values:2.2f} m")
    ax_.set_yticks([])

    # xy crosssections
    def make_xy(i_ax, z):
        axs = axs_[i_ax]
        axs[0].set_ylabel("y [m]")
        for i in range(nt):
            ax = axs[i]
            _plot_field_xy(ax, da.sel({t_str: times[i]}), z, CM)
            ax.set_title(None)
        ax_ = ax.twinx()
        ax_.set_ylabel(f"{z=} m")
        ax_.set_yticks([])

    for i, z in enumerate(z_levels):
        make_xy(i + 1, z)

    # finish
    plt.suptitle(f"{case}: {variable}")
    t_str = "-".join([str(int(t)) for t in times])
    z_str = "-".join([str(int(z)) for z in z_levels])
    savename = f"fig-field_evol_{savename}_{variable}_t{t_str}_z{z_str}"
    save_figure(os.path.join(savefolder, savename))


def make_pcolormesh_coords(x, z):
    """Transform cell midpoints to edges."""
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    x = np.concatenate([x - dx / 2, [x[-1] + dx / 2]])
    z = np.concatenate([z - dz / 2, [z[-1] + dz / 2]])
    return x, z


def _plot_field_xz(ax, da, iy, CM):
    """..."""
    x, z = make_pcolormesh_coords(da.x.to_numpy(), da.z.to_numpy())
    return ax.pcolormesh(
        x, z, da.isel(y=iy).to_numpy().T, cmap=CM.cmap, norm=CM.Normalizer
    )


def _plot_field_xy(ax, da, z, CM):
    """..."""
    x, y = make_pcolormesh_coords(da.x.to_numpy(), da.y.to_numpy())
    return ax.pcolormesh(
        x,
        y,
        da.sel(z=z, method="nearest").to_numpy().T,
        cmap=CM.cmap,
        norm=CM.Normalizer,
    )


def get_ds(path=None, outdir=None, s=1, stats_type="fields", merged=False):
    """..."""
    if path != "None" and path is not None:
        # after postprocessing by the data pipeline
        if merged:
            path = os.path.join(path, "data_merged")
        else:
            path = os.path.join(path, "data")

    elif outdir != "None" and outdir is not None:
        # single sample
        path = outdir

    ds = None
    if not merged:

        # get fields file
        if stats_type == "fields":
            if os.path.exists(os.path.join(path, "Fields.nc")):
                fieldsfile = os.path.join(path, "Fields.nc")
                ds = xr.open_dataset(fieldsfile)
            elif os.path.exists(os.path.join(path, f"sample_{s}_fields.nc")):
                fieldsfile = os.path.join(path, f"sample_{s}_fields.nc")
                ds = xr.open_dataset(fieldsfile)
            else:
                print(f"No fields found for sample {s} in {path}")

        # stats structure 1: unprocessed, one file per sample
        if os.path.exists(os.path.join(path, "Stats.nc")):
            statsfile = os.path.join(path, "Stats.nc")
            condstatsfile = os.path.join(path, "CondStats.nc")
            if stats_type == "profiles":
                ds = xr.open_dataset(statsfile, group="profiles")
            elif stats_type == "timeseries":
                ds = xr.open_dataset(statsfile, group="timeseries")
            elif stats_type == "spectra":
                ds = xr.open_dataset(condstatsfile, group="spectra")

        # stats structure 2: processed, one file per sample
        elif os.path.exists(os.path.join(path, f"sample_{s}_profiles.nc")):
            if stats_type == "profiles":
                ds = xr.open_dataset(os.path.join(path, f"sample_{s}_profiles.nc"))
            elif stats_type == "timeseries":
                ds = xr.open_dataset(os.path.join(path, f"sample_{s}_timeseries.nc"))
            elif stats_type == "spectra":
                ds = xr.open_dataset(os.path.join(path, f"sample_{s}_spectra.nc"))

    else:
        # stats structure 3: processed, one file overall
        if stats_type == "profiles":
            ds = xr.open_dataset(os.path.join(path, "profiles.nc")).isel(member=s)
        elif stats_type == "timeseries":
            ds = xr.open_dataset(os.path.join(path, "timeseries.nc")).isel(member=s)
        elif stats_type == "spectra":
            ds = xr.open_dataset(os.path.join(path, "spectra.nc")).isel(member=s)
        if stats_type == "fields":
            ds = xr.open_dataset(os.path.join(path, "fields.nc")).isel(member=s)

    # no ds found
    if ds is None:
        print(f"No {stats_type} found for sample {s} in {path}")
        return ds

    # masking: set nan_by_var[var] to nan
    for var in ds.variables:
        if var in nan_by_var:
            ds[var] = ds[var].where(np.abs(ds[var] - nan_by_var[str(var)]) > 1e-10)

    return ds
