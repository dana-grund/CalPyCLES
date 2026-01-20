"""Data management for the DYCOMS_RF01 case.

The class represents measurement data ('moment': 1=mean, 2=var).
For synthetic data, the mean is taken from the simulated sample,
and the variance from the original measurements.

File structure of data.nc:

    netcdf data {
    dimensions:
            z = 21 ;
            moment = 2 ;
            fit = 2 ;
    variables:
            double w_mean2(z, moment) ;
                    w_mean2:_FillValue = NaN ;
            double w_mean3(z, moment) ;
                    w_mean3:_FillValue = NaN ;
            double ql_mean(z, moment) ;
                    ql_mean:_FillValue = NaN ;
            double qt_mean(z, moment) ;
                    qt_mean:_FillValue = NaN ;
            double thetali_mean(z, moment) ;
                    thetali_mean:_FillValue = NaN ;
            double z(z) ;
                    z:_FillValue = NaN ;
            int64 moment(moment) ;
            double cloud_top_ts_fit(fit, moment) ;
                    cloud_top_ts_fit:_FillValue = NaN ;
            double cloud_base_ts_fit(fit, moment) ;
                    cloud_base_ts_fit:_FillValue = NaN ;
            int64 fit(fit) ;
    }

To be accessed for calibration:

    Mean:   meas.observations_np
    Var:    meas.obs_error_var_np
    Dist:   meas.dist_obs_error
"""

# Standard library
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.typing import NDArray

# First-party
from calpycles.case_inputs import names_by_case
from calpycles.helpers import concat_to_1d
from calpycles.plotting import LINEWIDTH
from calpycles.plotting import MARKERSIZE
from calpycles.plotting import ZORDERS
from calpycles.plotting import get_name_mpl
from calpycles.plotting import make_figure_profiles
from calpycles.plotting import make_figure_timeseries
from calpycles.plotting import save_figure
from calpycles.random_variables import GaussRV
from calpycles.random_variables import init_rv

# pylint: disable=invalid-name
# allow for CI and DYCOMS

# pylint: disable=too-many-locals

THIS_FOLDER = os.path.dirname(__file__)


class MeasurementsDYCOMS_RF01:
    """Access to the DYCOMS RF01 data.

    Profile_heights are obtained as nearest (dz=5m) to data heights.
    Where a height would appear twice due to a distance lower than dz, we manually
    adapted them to be different consecutive grid levels.
    """

    # Model levels to be evaluated to simulate the data.
    # Used in ObsEnsembleDYCOMS_RF01._observe().
    profile_heights_insitu = [
        95.0,
        150.0,
        480.0,
        485.0,
        620.0,
        635.0,
        745.0,
        750.0,
        925.0,
        1060.0,
    ]
    profile_heights_radar = [
        685.0,
        700.0,
        715.0,
        725.0,
        740.0,
        755.0,
        775.0,
        795.0,
        810.0,
        830.0,
        850.0,
    ]
    profile_heights = profile_heights_insitu + profile_heights_radar

    # variables are sorted according to modality_idcs
    timeseries_variables = [  # as in self.data
        "cloud_base_ts_fit",
        "cloud_top_ts_fit",
    ]
    timeseries_variables_obs = [  # for plotting
        "cloud_base_height",
        "cloud_base_rate",
        "cloud_top_height",
        "cloud_top_rate",
    ]
    profile_variables = [  # as in self.data, for plotting
        "ql_mean",
        "qt_mean",
        "thetali_mean",
        "w_mean2",
        "w_mean3",
    ]
    obs_vars = timeseries_variables_obs + profile_variables  # for plotting

    n_obs = 65  # = len(self.not_nan_indices)

    # split the measurements by modality (indices in self.observations_np)
    modality_idcs = {
        "total": list(range(65)),
        "cloud_base_height": [0],
        "cloud_base_rate": [1],
        "cloud_top_height": [2],
        "cloud_top_rate": [3],
        "cloud_growth": [1, 3],
        "cloud_evolution": [0, 1, 2, 3],
        "ql_mean": list(range(4, 8)),
        "qt_mean": list(range(8, 17)),
        "thetali_mean": list(range(17, 26)),
        "w_mean2": list(range(26, 45)),
        "w_mean3": list(range(45, 65)),
        "thetali_inv+": [24],  # thetali just above the inversion
        "thetali_inv-": [23],  # thetali just below the inversion
        "thetali_BL": list(range(17, 23)),  # thetali inside BL
        "qt_inv+": [15],  # qt just above the inversion
        "qt_inv-": [14],  # qt just below the inversion
    }

    modality_groups = {
        "profiles": profile_variables,
        "timeseries": timeseries_variables,
        "total": ["total"],
    }

    _profile_mask: NDArray[np.bool_]
    _not_nan_idcs: NDArray[np.int_]
    _dist_obs_error: GaussRV

    def __init__(self) -> None:
        """Load the data."""
        self.file = os.path.join(THIS_FOLDER, "data/data.nc")
        self.data = self.load_data()

    @property
    def obs_names(self) -> List[str]:
        """Names of all observation variables, including heights for profiles."""
        return get_obs_names_with_height()

    @property
    def dist_obs_error(self) -> GaussRV:
        """Observation error distribution."""
        if not hasattr(self, "_dist_obs_error"):
            self._dist_obs_error = init_rv(
                mu=0, C=self.obs_error_var_np, M=self.n_obs, dist_type="normal"
            )
        return self._dist_obs_error

    @property
    def profile_mask(self) -> NDArray[np.bool_]:
        """Mask to select the heights with data for each profile variable.

        Used in ObsEnsembleDYCOMS_RF01._observe().
        """
        if not hasattr(self, "_profile_mask"):
            self._profile_mask = ~np.isnan(
                self.data_p.sel(moment=1).assign({"z": self.profile_heights})
            )
        return self._profile_mask

    @property
    def not_nan_indices(
        self,
    ) -> NDArray[np.int_]:
        """Indices of the variables that are not NaN in 1D array form.

        Used in ObsEnsembleDYCOMS_RF01.concat_to_1d().
        """
        if not hasattr(self, "_not_nan_indices"):
            data_np = concat_to_1d(self.data.sel(moment=1))  # only mean, not var
            self._not_nan_idcs = np.argwhere(~np.isnan(data_np)).ravel()
        return self._not_nan_idcs

    @property
    def data_p(self) -> xr.Dataset:
        """Profile data only."""
        return self.data[self.profile_variables]

    @property
    def observations_np(self) -> NDArray[np.float_]:
        """Extract measurement uncertainty to be used by data assimilation."""
        return concat_to_1d(self.data.sel(moment=1))[self.not_nan_indices]

    @property
    def obs_error_var_np(self) -> NDArray[np.float_]:
        """Extract measurement uncertainty to be used by data assimilation."""
        return concat_to_1d(self.data.sel(moment=2))[self.not_nan_indices]

    def load_data(self) -> xr.Dataset:
        """Load the data and format."""
        data = xr.open_dataset(self.file)
        data = data.transpose("z", "fit", "moment")
        return data

    def close(self) -> None:
        """Close the data."""
        self.data.close()

    def plot_profile(
        self,
        ax: Any,
        name: str,
        mean_kwargs: Optional[Dict[str, Any]] = None,
        mean_label: str = "mean",  # "meas. data",
        **kwargs: Any,
    ) -> List[Any]:
        """Plot one profile variable."""
        if name not in self.profile_variables:
            # no data available for this profile
            return []

        if "moment" in self.data.dims:
            means = self.data[name].sel(moment=1).values
            var = self.data[name].sel(moment=2).values
        else:
            means = self.data[name].values
            var = None

        heights = self.data["z"]

        handles = []

        # color only for mean (gets messy when coloring CIs as well)
        c_mean = kwargs.pop("c") if "c" in kwargs else "k"

        # confidence bounds
        if var is not None:
            CIs = np.sqrt(var)
            CI_kwargs = {
                "label": "+- 1 std",
                "c": "k",
                "marker": "|",
                "zorder": ZORDERS["data"],
                "linewidth": 0.5 * LINEWIDTH,
                "markersize": 0.25 * MARKERSIZE,
            }
            CI_kwargs.update(kwargs)
            for mean, CI, h in zip(means, CIs, heights):
                p = ax.plot([mean - CI, mean + CI], [h, h], **CI_kwargs)[0]
            handles.append(p)

        # means
        if mean_kwargs is None:
            mean_kwargs = {
                "c": c_mean,
                "label": mean_label,
                "zorder": ZORDERS["data"],
                "s": MARKERSIZE,
            }
            mean_kwargs.update(kwargs)
        p = ax.scatter(means, heights, **mean_kwargs)  # zorder=20,
        handles.append(p)
        return handles

    def plot_timeseries(self, ax: Any, name: str, **kwargs: Any) -> List[Any]:
        """Plot one timeseries variable."""
        aliases = [
            name,
            name + "_ts_fit",
            name[:-5],  # "_mean",
            name[:-5] + "_ts_fit",
        ]
        if name not in self.timeseries_variables:
            for alias in aliases:
                if alias in self.timeseries_variables:
                    name = alias
                    break
            else:
                return []

        std_m = std_b = 0.0
        if "moment" in self.data.dims:
            # measured data. moment=1: mean, moment=2: var.
            means = self.data[name].sel(moment=1)
            std = np.sqrt(self.data[name].sel(moment=2))
            std_b = float(std.sel(fit=0))
            std_m = float(std.sel(fit=1))
        else:
            # simulated data.
            means = self.data[name]
            std = None

        b = float(means.sel(fit=0))
        m = float(means.sel(fit=1))

        t0: float = names_by_case["DYCOMS_RF01"]["t0_timeseries"]  # type: ignore
        t_max: float = names_by_case["DYCOMS_RF01"]["times"][-1] - 600  # type: ignore
        # some gets lost by rolling time mean
        time = np.array([t0, t_max])
        # ax.set_xlim(0, t_max)
        # ax.set_ylim(lims_by_var[name])
        handles = []

        # confidence bounds
        if "moment" in self.data.dims and std is not None:
            CI_kwargs = {
                "label": "+- 1 std (m,b)",
                "c": "k",
                "linestyle": "--",
                "zorder": -10,
            }
            CI_kwargs.update(kwargs)
            lower = (b - std_b) + (m - std_m) * (time - t0)
            upper = (b + std_b) + (m + std_m) * (time - t0)
            p = ax.plot(time, upper, **CI_kwargs)[0]
            ax.plot(time, lower, **CI_kwargs)
            handles.append(p)

        # mean
        mean_kwargs = {"c": "k", "label": "mean (m,b)", "zorder": -10}
        mean_kwargs.update(kwargs)
        mean = b + m * (time - t0)
        p = ax.plot(time, mean, **mean_kwargs)[0]
        handles.append(p)

        return handles

    def plot(self, title_add: str = "", savefolder: str = "./") -> None:
        """Plot the data (only variables with data)."""
        # profiles
        profile_variables = self.profile_variables
        _, axs = make_figure_profiles(profile_variables)

        for i, name in enumerate(profile_variables):
            ax = axs[i]
            handles = self.plot_profile(ax, name=name, c="k")

        axs[0].legend(handles=handles, loc="upper left")
        plt.suptitle("DYCOMS_RF01: profiles " + title_add)
        save_figure(os.path.join(savefolder, "fig-profiles_measurements_time_average"))

        # timeseries
        timeseries_variables = self.timeseries_variables
        _, axs = make_figure_timeseries(timeseries_variables)

        for i, name in enumerate(timeseries_variables):
            ax = axs[i]
            handles = self.plot_timeseries(ax, name=name, c="k")

        axs[-1].legend(handles=handles, loc="lower right")
        plt.suptitle("DYCOMS_RF01: timeseries " + title_add)
        save_figure(os.path.join(savefolder, "fig-timeseries_measurements"))

    def get_meas_heights(self, var: str) -> Optional[NDArray[np.float_]]:
        """Get the heights where measurements are available for a variable."""
        # heights only for profiles
        if var not in self.modality_groups["profiles"]:
            return None

        # find the indices of the corresponding heights after flattening
        heights = self.data.z.to_numpy()
        data_np = self.data[var].isel(moment=0).to_numpy()
        not_nan_mask = np.argwhere(~np.isnan(data_np))
        result: NDArray[np.float_] = heights[not_nan_mask].flatten()

        return result

    @property
    def meas_heights(self) -> Dict[str, Optional[NDArray[np.float_]]]:
        """Measured heights for each variable."""
        heights = {}
        for var in self.profile_variables + self.timeseries_variables:
            heights[var] = self.get_meas_heights(var)
        return heights

    def modify_var_profile(self, name: str, std_factor: float = 1.0) -> None:
        """Change the variance of a profile variable by a factor."""
        modify_var_profile(self.data, name, std_factor)

    def modify_var_ts(self, std_factors: Dict[str, float]) -> None:
        """Change the variance of a timeseries variable by a factor."""
        modify_var_ts(self.data, std_factors)


class SynthMeasurementsDYCOMS_RF01(MeasurementsDYCOMS_RF01):
    """Synthetic measurements for DYCOMS_RF01.

    Same format and use, with mean from simulated sample and variance from the
    original measurements.
    """

    def __init__(
        self, file: str, parameters: Optional[NDArray[np.float64]] = None
    ) -> None:
        """Load the data.

        file should point to samples_observations.nc created by ObsEnsembleDYCOMS_RF01.
        """
        self.file = file
        self.data = self.load_data()
        self.parameters = parameters  # remember the data-generating parameters

    def load_data(self) -> xr.Dataset:
        """Load the data."""
        data = xr.open_dataset(self.file)
        if "sample" in data.dims:
            assert len(data.sample) == 1
            data = data.isel(sample=0)

        # for synthetic data, the simulation is the mean
        # add variances from original measurements
        data = add_default_variance(data)

        return data


def add_default_variance(data: xr.Dataset) -> xr.Dataset:
    """Add variances of the original masurements."""
    default_var = MeasurementsDYCOMS_RF01().data.sel(moment=2)

    # map var to synthetic heights
    default_var = default_var.assign_coords(z=data.z, moment=2)

    # add var to the data mean
    data = data.expand_dims("moment")
    data = data.assign_coords(moment=[1])
    result = xr.concat(
        [data, default_var],
        dim="moment",
    ).transpose("z", "fit", "moment")

    return result


def get_obs_names_with_height() -> List[str]:
    """Names of variables with measured heights."""
    meas = MeasurementsDYCOMS_RF01()
    obs_names_with_heights = []
    for name in meas.obs_vars:
        name_mpl = get_name_mpl(name, add_units=False)
        if name in meas.meas_heights:
            heights = meas.meas_heights[name]
            assert heights is not None
            for h in heights:
                obs_names_with_heights += [f"{name_mpl} ({int(h+0.5)}m)"]
        else:
            obs_names_with_heights += [f"{name_mpl}"]
    return obs_names_with_heights


def modify_var_profile(ds: xr.Dataset, name: str, std_factor: float = 1.0) -> None:
    """Changes the variance of a profile variable by a factor.

    ds is of the format of MeasurementsDYCOMS_RF01.data.
    """
    data_new = np.stack(
        [
            ds[name].isel(moment=0),
            ds[name].isel(moment=1) * std_factor**2,
        ],
        axis=1,
    )

    ds_new = xr.Dataset(
        {
            name: (
                ds[name].dims,
                data_new,
            ),
        },
        coords=ds[name].coords,
    )

    ds.update(ds_new)


def modify_var_ts(ds: xr.Dataset, std_factors: Dict[str, float]) -> None:
    """Changes the variance of a timeseries variable by a factor.

    std_factors = {
        "ztoph": 1,
        "ztopr": 1,
        "zbaseh": 1,
        "zbaser": 1,
    }

    ds is of the format of MeasurementsDYCOMS_RF01.data.
    """
    top = ds["cloud_top_ts_fit"]
    base = ds["cloud_base_ts_fit"]

    # same structure as in make_data.py
    ds_new = xr.Dataset(
        data_vars={
            "cloud_top_ts_fit": (
                ("fit", "moment"),
                np.array(
                    [
                        # [840, (20)**2],  # intercept
                        [
                            top.isel(fit=0).isel(moment=0),  # height mean
                            top.isel(fit=0).isel(moment=1)
                            * std_factors["ztoph"] ** 2,  # height var
                        ],
                        # [7.5/3600, (7.5/3600)**2],  # slope [m/s]]
                        [
                            top.isel(fit=1).isel(moment=0),  # rate mean
                            top.isel(fit=1).isel(moment=1)
                            * std_factors["ztopr"] ** 2,  # rate var
                        ],
                    ]
                ),
            ),
            "cloud_base_ts_fit": (
                ("fit", "moment"),
                np.array(
                    [
                        # [580, (40)**2],  # intercept
                        [
                            base.isel(fit=0).isel(moment=0),  # height mean
                            base.isel(fit=0).isel(moment=1)
                            * std_factors["zbaseh"] ** 2,  # height var
                        ],
                        # [-2/3600, (2/3600)**2],  # slope
                        [
                            base.isel(fit=1).isel(moment=0),  # rate mean
                            base.isel(fit=1).isel(moment=1)
                            * std_factors["zbaser"] ** 2,  # rate var
                        ],
                    ]
                ),
            ),
        },
        coords=top.coords,
    )

    ds.update(ds_new)
