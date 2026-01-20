"""Observables to observe a PyCLESsample."""

# Standard library
import os
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# Third-party
import numpy as np
import xarray as xr
from numpy.typing import NDArray

# First-party
from calpycles.case_inputs import names_by_case
from calpycles.case_inputs import nan_by_var
from calpycles.DYCOMS_RF01.measurements import MeasurementsDYCOMS_RF01
from calpycles.helpers import assert_shape
from calpycles.helpers import concat_to_1d
from calpycles.helpers import create_mfdataset_with_nans
from calpycles.namelist import Namelist
from calpycles.plotting.ens_obs import plot_ens_obs
from calpycles.random_variables import GaussRV

# pylint: disable=invalid-name
# mypy: disable-error-code="arg-type"

# error: Argument 1 to "sel" of "DataArray" has incompatible type "**dict[str, object]";
# expected "bool"  [arg-type]


class ObsEnsemble(ABC):
    """An ensemble of observations for PyCLES simulations.

    Instances observe and store the observations in an xr.Dataset format.
    To access the observation as a 1D numpy array, use observation.obs_as_numpy.

    The ensemble is initialized from samples, RV is only used to approximate their
    distribution, not for sampling new observations (only var, not covar used!)
    """

    # abstract properties (to be defined in child classes)
    n_obs = property(abstractmethod(lambda self: int))

    # shorthand properties
    file_xr = property(lambda self: os.path.join(self.path, "samples_observations.nc"))

    def __init__(
        self,
        namelists: List[Namelist],
        path: str,
        name: str,
        load: bool = False,
    ) -> None:
        """Initialize.

        namelist should have shared 'time_stepping' and 'stats_io' settings.
        """
        self.path = path
        self.name = name
        self.namelist = namelists[0]
        self.n_samples = len(namelists)
        self.stats_files = [namelist.files["stats_file"] for namelist in namelists]

        # properties
        self._obs_xr: xr.Dataset
        self._dist: GaussRV

        # load (or create) observations
        if load:
            self.load()

    verbose = property(lambda self: self.n_samples > 1)

    def observe(self) -> None:
        """Extract data for the ensemble."""
        if self.verbose:
            print(f"Observing observation ensemble {self.name}...")

        # load or observe
        self.get()

    @abstractmethod
    def _observe(
        self,
    ) -> xr.Dataset:
        """Assemble observations as an xarray dataset for the particular case."""
        raise NotImplementedError("This method should be implemented in child classes.")

    @property
    def obs_xr(self) -> xr.Dataset:
        """The observations."""
        if not hasattr(self, "_obs_xr"):
            self.get()
        return self._obs_xr

    @property
    def obs(self) -> NDArray[np.float64]:
        """Conversion from xr.Dataset to np.array.

        Returns a 1D array, so that multiple samples' obs can be stacked to a 2D array.
        """
        obs = self.xr_to_np(self.obs_xr)
        assert_shape((self.n_samples, self.n_obs), obs, "observation ensemble")

        return obs

    def xr_to_np(self, ds: xr.Dataset, flatten: bool = False) -> NDArray[np.float64]:
        """Convert dataset to array by concatenation.

        Return shape:
            - (n_samples, n_obs) if 'sample' in ds.dims
            - (n_obs_) if not 'sample' in ds.dims and flatten==True
            - (1, n_obs) else (single sample)
        """
        if "sample" in ds.dims:
            result = np.stack(
                [self.concat_to_1d(ds.sel(sample=s)) for s in ds.sample], axis=0
            )
            assert_shape(
                (self.n_samples, self.n_obs), result, "transformed observation"
            )
        else:
            result = np.expand_dims(self.concat_to_1d(ds), axis=0)
            if flatten:
                result = result.flatten()
                assert_shape((self.n_obs,), result, "transformed observation")
            else:
                assert_shape((1, self.n_obs), result, "transformed observation")

        return result

    def concat_to_1d(
        self,
        ds: xr.Dataset,
    ) -> NDArray[np.float64]:
        """Provide the option to overwrite in child."""
        return concat_to_1d(ds)

    @abstractmethod
    def plot(
        self,
        data: Optional[MeasurementsDYCOMS_RF01] = None,
        **kwargs: Any,
    ) -> None:
        """Plot the observations."""
        return

    @property
    def dist(self) -> GaussRV:
        """Gaussian approximation of the observation distribution.

        Takes only variance, not covariance!
        """
        assert self.n_samples > 1, "Can't take mean/var over less than two samples."

        if not hasattr(self, "_dist"):
            # ensure shape is (n_obs,) and not (1,n_obs)
            self._dist = GaussRV(mu=self.mean.ravel(), C=self.var.ravel())
        return self._dist

    @property
    def mean_xr(self) -> xr.Dataset:
        """Compute the mean of the observations as xarray dataset."""
        assert self.n_samples > 1, "Can't take the mean over less than two samples."
        return self.obs_xr.mean(dim="sample")

    @property
    def median_xr(self) -> xr.Dataset:
        """Compute the median of the observations."""
        assert self.n_samples > 1, "Can't take the median over less than two samples."
        return self.obs_xr.median(dim="sample")

    @property
    def mean(self) -> NDArray[np.float64]:
        """Compute the mean of the observations as numpy array."""
        return self.xr_to_np(self.mean_xr)

    @property
    def var_xr(self) -> xr.Dataset:
        """Compute the variance of the observations."""
        assert self.n_samples > 1, "Can't take the variance over less than two samples."
        return self.obs_xr.var(dim="sample")

    @property
    def ci_xr(self) -> xr.Dataset:
        """Compute the 68% and 95% confidence intervals of the observations."""
        assert self.n_samples > 1, "Can't take the CI over less than two samples."
        return self.obs_xr.quantile([0.025, 0.16, 0.84, 0.975], dim="sample")

    @property
    def var(self) -> NDArray[np.float64]:
        """Compute the variance of the observations as numpy array."""
        return self.xr_to_np(self.var_xr)

    def get(self) -> None:
        """Load or observe."""
        if os.path.exists(self.file_xr):
            if self.verbose:
                print(f"Loading observation ensemble {self.name}...")
            self.load()
        else:
            if self.verbose:
                print(f"Creating observation ensemble {self.name}...")

            # observe
            self._obs_xr = self._observe()

            # save
            self._obs_xr.to_netcdf(self.file_xr)
            print(f"Saved observation(s) '{self.name}' to {self.file_xr}.")

    def load(self) -> None:
        """Load (or create and save) observations in xarray format."""
        if os.path.exists(self.file_xr):
            self._obs_xr = xr.open_dataset(self.file_xr)
            if self.n_samples > 1:
                print(f"Loaded observation {self.name} from {self.file_xr}.")
        else:
            print(
                f"Observation file not found: {self.file_xr}. " "Please run observe()."
            )


class ObsEnsembleDYCOMS_RF01(ObsEnsemble):
    """Wrap a collection of observations for the DYCOMS RF01 case.

    Internal observations in self._obs_xr are sparse scalars
    (profiles at certain heights, intercept and slope of linear timeseries fit).

    For plotting, the whole profiles and timeseries are considered.
    """

    case = "DYCOMS_RF01"
    timeseries_variables: List[str] = names_by_case["DYCOMS_RF01"][
        "timeseries"
    ]  # type: ignore
    profile_variables: List[str] = names_by_case["DYCOMS_RF01"][
        "profiles"
    ]  # type: ignore
    profile_heights = MeasurementsDYCOMS_RF01.profile_heights

    def __init__(self, **kwargs: Any) -> None:
        """Initialize."""
        super().__init__(**kwargs)

        self._obs_xr_full: Tuple[xr.Dataset, xr.Dataset]

    @property
    def file_xr_full(self) -> Tuple[str, str]:
        """Where to save the full observations."""
        name = "samples_observations_full"
        return (
            os.path.join(self.path, f"{name}_profiles.nc"),
            os.path.join(self.path, f"{name}_timeseries.nc"),
        )

    @property
    def obs_xr_full(self) -> Tuple[xr.Dataset, xr.Dataset]:
        """Full profiles and timeseries ensembles."""
        if not hasattr(self, "_obs_xr_full"):
            self.get_xr_full()
        return self._obs_xr_full

    def get_xr_full(self) -> None:
        """Load or create full observations from file."""
        if os.path.exists(self.file_xr_full[0]):
            if self.n_samples > 1:
                self._obs_xr_full = (
                    xr.open_dataset(self.file_xr_full[0]),
                    xr.open_dataset(self.file_xr_full[1]),
                )
                print(f"Loaded observation {self.name} from {self.file_xr_full}.")
        else:
            self._obs_xr_full = self.observe_full()

    def observe_full(self) -> Tuple[xr.Dataset, xr.Dataset]:
        """Extract full profiles and timeseries from the ensemble."""
        print("Reading full profiles and timeseries for ObEnsembleDYCOMS_RF01...")
        profiles = create_mfdataset_with_nans(
            self.stats_files, variables=self.profile_variables, group="profiles"
        )
        timeseries = create_mfdataset_with_nans(
            self.stats_files, variables=self.timeseries_variables, group="timeseries"
        )

        # convert humidity from kg/kg to g/kg to avoid small numbers
        profiles["ql_mean"] = profiles["ql_mean"] * 1000.0
        profiles["qt_mean"] = profiles["qt_mean"] * 1000.0

        # save full profiles and timeseries
        if self.n_samples > 1:
            profiles.to_netcdf(self.file_xr_full[0])
            timeseries.to_netcdf(self.file_xr_full[1])
            print(f"Saved full obs ensemble '{self.name}' to")
            print(f"{self.file_xr_full[0]} / _timeseries.nc.")

        # save full profiles and timeseries stats for EnsStatsPlotter
        if self.n_samples > 1 and not self.stats_were_computed:
            self.compute_stats()

        return profiles, timeseries

    def compute_stats(self) -> None:
        """Compute stats of the obs ensemble."""
        if self.n_samples > 1 and not self.stats_were_computed:
            profiles, timeseries = self.obs_xr_full
            print("Computing observation ensemble profile stats...")
            compute_stats(
                self.path,
                self.n_samples,
                ds_ens=profiles,
                stats_type="profiles",
                do_average_time=True,
            )
            print("Computing observation ensemble timeseries stats...")
            compute_stats(
                self.path,
                self.n_samples,
                ds_ens=timeseries,
                stats_type="timeseries",
                do_average_time=True,
            )

    @property
    def stats_were_computed(
        self,
    ) -> bool:
        """Check if the ensemble stats were computed and saved."""
        one_file = os.path.join(
            self.path,
            f"ensemble_stats/Nens{self.n_samples}_profiles_average_time_mean.nc",
        )
        return os.path.exists(one_file)

    def _observe(self) -> xr.Dataset:
        """Extract profiles and timeseries fits that match the data."""
        profiles, timeseries = self.obs_xr_full
        time = "t" if "t" in timeseries else "time"
        # select heights at which measurements are taken (both in-situ and radar)
        # heights should be part of the grid.
        # If not, method="nearest" selects the closest grid point.
        # This may lead to duplicate heights (480, 480),
        # so manually assign the meas heights to the measured profiles.
        profiles = profiles.sel(z=self.profile_heights, method="nearest")
        profiles = profiles.assign({"z": self.profile_heights})

        # time-average (last 2h)
        profiles = profiles.sel(
            **{time: names_by_case["DYCOMS_RF01"]["average_times"]}
        ).mean(dim=time)

        # for each profile, select only heights with data
        profiles = profiles.where(MeasurementsDYCOMS_RF01().profile_mask)

        # timeseries: select times, linear fit
        t0: float = names_by_case["DYCOMS_RF01"]["t0_timeseries"]  # type: ignore
        if t0 not in timeseries[time]:  # test run with smaller t_max
            t0 = max(timeseries[time].values)

        # choose which computation of cloud heights to use
        # cloud_base and cloud_top: domain min/max height
        # cloud_base_mean and cloud_top_mean: domain average height
        base_var = "cloud_base_mean"
        top_var = "cloud_top_mean"

        # mask timeseries nan values
        for var in [base_var, top_var]:
            timeseries[var] = timeseries[var].where(timeseries[var] != nan_by_var[var])

        # do a linear fit for each sample
        # works with and without dimension 'sample'
        poly = timeseries.sel(**{time: slice(t0, None)}).polyfit(dim=time, deg=1)
        ct_s = poly[top_var + "_polyfit_coefficients"].isel(degree=0)
        cb_s = poly[base_var + "_polyfit_coefficients"].isel(degree=0)
        ct_i = poly[top_var + "_polyfit_coefficients"].isel(degree=1)
        cb_i = poly[base_var + "_polyfit_coefficients"].isel(degree=1)

        # intercepts from t0
        ct_i = ct_i + ct_s * t0
        cb_i = cb_i + cb_s * t0

        # collect timeseries observations
        dims = ("fit", "sample") if "sample" in timeseries else ("fit")
        ts_fit = xr.Dataset(
            data_vars={
                "cloud_top_ts_fit": (dims, [ct_i, ct_s]),
                "cloud_base_ts_fit": (dims, [cb_i, cb_s]),
            },
            coords={
                "fit": np.arange(2),  # 0=intercept, 1=slope
            },
        )

        # return same xr format as data is saved in
        return xr.merge([profiles, ts_fit])

    n_obs = property(lambda self: MeasurementsDYCOMS_RF01.n_obs)

    def concat_to_1d(
        self,
        ds: xr.Dataset,
    ) -> NDArray[np.float64]:
        """Remove nans when moving from xr to np representation."""
        with_nans = super().concat_to_1d(ds)

        # remove nans
        meas = MeasurementsDYCOMS_RF01()  # independent of actual measurements
        return with_nans[meas.not_nan_indices]

    def plot(
        self,
        variables: Optional[Union[list[str], str]] = None,
        data: Optional[MeasurementsDYCOMS_RF01] = None,
        **kwargs: Any,
    ) -> None:
        """Plot full ensemble stats with data."""
        # make sure the stats information was saved
        if not self.stats_were_computed:
            self.compute_stats()

        save_file = os.path.join(self.path, f"fig-ensemble_obs_{self.name}")
        plot_ens_obs(
            ens_paths=[self.path],
            n_ens_list=[self.n_samples],
            ens_labels=[self.name],
            data=data,
            save_file=save_file,
        )


class ObsDYCOMS_RF01(ObsEnsembleDYCOMS_RF01):
    """Wrap a single observable for the DYCOMS RF01 case."""

    def __init__(
        self,
        namelist: Namelist,
        **kwargs: Any,
    ) -> None:
        """Initialize."""
        super().__init__(
            namelists=[namelist],
            **kwargs,
        )

    def observe(self) -> None:
        """Read a single sample."""
        super().observe()
        self._obs_xr = self._obs_xr.isel(sample=0)

    def plot(
        self,
        data: Optional[MeasurementsDYCOMS_RF01] = None,
        **kwargs: Any,
    ) -> None:
        """Plot the observations as full profiles and timeseries."""
        save_file = os.path.join(self.path, "fig-obs")
        plot_ens_obs(
            data=data,
            samples_obs_full_list=[self.observe_full()],
            sample_names=[self.name],
            save_file=save_file,
            **kwargs,
        )


# ---


def compute_stats(
    path: str,
    n_samples: int,
    ds_ens: Optional[xr.Dataset] = None,
    stats_type: str = "profiles",
    t: Optional[float] = None,
    z: Optional[float] = None,
    case: str = "DYCOMS_RF01",
    do_average_time: bool = True,
) -> None:
    """Compute ensemble statistics for profiles and timeseries and save to netcdf."""
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    # open and select only relevant variables
    print("Reading...")
    if ds_ens is None:
        ds_list = [
            os.path.join(path, f"data/sample_{s}_{stats_type}.nc")
            for s in range(n_samples)
        ]
        ds = xr.open_mfdataset(ds_list, concat_dim="sample", combine="nested")[
            names_by_case[case][stats_type]
        ]
    else:
        ds = ds_ens[names_by_case[case][stats_type]]

    # by default, open_mfdataset uses chunks={sample:1}. CI needs no chunks.
    ds = ds.chunk({"sample": -1})

    # select times
    t_str = ""
    time = "t" if "t" in ds.dims else "time"
    if stats_type == "profiles":
        if do_average_time:
            ds = ds.sel(**{time: names_by_case[case]["average_times"]}).mean(
                dim=time, skipna=True
            )
            t_str = "_average_time"
        else:
            if t is not None:
                ds = ds.sel({time: t})
                t_str = f"_t{t}"
            else:
                t_str = ""
            if z is not None:
                ds = ds.sel(z=z, method="nearest")

    # replace nans
    for var, nan in nan_by_var.items():
        if var in ds:
            ds[var] = ds[var].where(ds[var] != nan)

    # compute
    print("Computing...")
    ds_mean = ds.mean(dim="sample", skipna=True)
    ds_median = ds.median(dim="sample", skipna=True)
    ds_CI = ds.quantile([0.05, 0.25, 0.75, 0.95], dim="sample", skipna=True)
    # ds_CI has new dimension "quantile"
    # 90% of data are in [0.05,0.95]
    # 50% of data are in [0.25,0.75]

    # save
    path = os.path.join(path, "ensemble_stats")
    if not os.path.exists(path):
        os.makedirs(path)
    names = [
        os.path.join(path, f"Nens{n_samples}_{stats_type}{t_str}_{stats}.nc")
        for stats in ["mean", "median", "CI"]
    ]
    for ds, name in zip([ds_mean, ds_median, ds_CI], names):
        print(f"Saving {name}...")
        ds.to_netcdf(name)

    print(f"Done computing {stats_type} ensemble stats.")
    ds.close()


# ---


def obs_factory(case: str = "DYCOMS_RF01", **kwargs: Any) -> ObsEnsemble:
    """Select a single observable."""
    if case == "DYCOMS_RF01":
        return ObsDYCOMS_RF01(**kwargs)

    raise ValueError(f"Unknown case to create a Namelist: {case}")


def obs_ens_factory(case: str = "DYCOMS_RF01", **kwargs: Any) -> ObsEnsemble:
    """Select an ensemble of observables."""
    if case == "DYCOMS_RF01":
        return ObsEnsembleDYCOMS_RF01(**kwargs)

    raise ValueError(f"Unknown case to create a Namelist: {case}")
