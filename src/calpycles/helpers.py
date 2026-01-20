"""Helper functionalities."""

# Standard library
import os
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# Third-party
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize as Normalize_scalar
from numpy.typing import NDArray
from scipy.optimize import minimize  # type:ignore[import-untyped]
from scipy.stats import gaussian_kde  # type:ignore[import-untyped]
from tabulate import tabulate


def ensure_path(path: str) -> None:
    """Ensure that a path exists."""
    if not os.path.exists(path) and len(path) > 0:
        os.makedirs(path)


def assert_print(
    truth: Any,
    test: Any,
    name: Optional[str] = None,
) -> None:
    """Assert the two are equal and print their values in case not."""
    msg_name = f"{name} to be " if name is not None else ""
    msg = f"Expected {msg_name}{truth}, but got {test}."
    assert truth == test, msg


def assert_shape(
    truth: Tuple[int, ...],
    test: NDArray[np.float64],
    name: Optional[str] = None,
) -> None:
    """Assert the shape of an array and print shapes if assertion fails."""
    if name is not None:
        name = "the shape of " + name
    assert_print(truth, test.shape, name)


def assert_files_in_folders(folder1: str, folder2: str, files: List[str]) -> None:
    """Assert the content of certain .npy and .nc files in two folders is equal.

    About comparing xarray datasets:
        "ds1.equals(ds2)" returned False when expecting True.
        "ds1 == ds2" returned True but it is unclear what is actually compared.
        Hence, we compare the variables one by one.
    """
    files_npy, files_nc = [], []

    # filter files by ending
    for f in files:
        if f.endswith(".npy"):
            files_npy.append(f)
        elif f.endswith(".nc"):
            files_nc.append(f)
        else:
            raise NotImplementedError(f"Cannot handle the ending of file {f}")

    # check numpy files
    for f in files_npy:
        content1 = np.load(os.path.join(folder1, f))
        content2 = np.load(os.path.join(folder2, f))
        assert np.allclose(content1, content2), (
            f"The file {f} did not turn out as expected. "
            f"Expected:\n{content1}\nGot:\n{content2}"
        )

    # check netcdf files (only xarray datasets!)
    for f in files_nc:
        content1 = xr.open_dataset(os.path.join(folder1, f))
        content2 = xr.open_dataset(os.path.join(folder2, f))
        assert content1.variables.keys() == content2.variables.keys(), (
            f"There are different variables in file {f}, "
            f"namely {content1.variables.keys()}\nand {content2.variables.keys()}."
        )
        for var in content2.variables:
            assert np.allclose(content2[var].to_numpy(), content1[var].to_numpy()), (
                f"The file {f} did not turn out as expected in variable {var}. "
                f"Expected:\n{content1[var]}\nGot:\n{content2[var]}"
            )


def ensure_list(a: Any) -> List[Any]:
    """Transform a variable into a list if it is not already."""
    if not isinstance(a, list):
        return [a]
    return a


def make_table(
    samples: NDArray[np.float64], names: List[str], do_print: bool = False
) -> str:
    """Format a numpy array as a table."""
    table = tabulate(
        samples,
        headers=names,
        tablefmt="grid",
        showindex="always",
    )

    if do_print:
        print(table)

    return table


def save_tabular_data(
    samples: NDArray[np.float64],
    names: List[str],
    save_name: str = "samples",
    save_path: str = "./",
) -> None:
    """Format a numpy array as a table and save it to npy, nc and txt files."""
    file_without_ending = os.path.join(save_path, save_name)

    # numpy
    np.save(file_without_ending + ".npy", samples)

    # netcdf
    samples_xr = xr.Dataset(
        data_vars={name: (["sample"], samples[:, i]) for i, name in enumerate(names)},
        coords={"sample": np.arange(samples.shape[0])},
    )
    samples_xr.to_netcdf(file_without_ending + ".nc")

    # human-readable text file
    with open(file_without_ending + ".txt", "w", encoding="utf-8") as f:
        f.write(make_table(samples, names))


def summarize_memory(
    n_floats: int,
    precision: str = "double",
    name: Optional[str] = None,
) -> None:
    """Print the memory usage of n_floats many floats.

    Examples
        n_times = 181
        n_meas_locs = 20
        n_vars = 3
        n_obs = n_times * n_meas_locs * n_vars
        summarize_memory(n_obs**2, name="obs_cov")
        summarize_memory(n_obs, name="obs_var")

        n_params = 6
        summarize_memory(n_params**2, name="prior_var")

    """
    one_mib = 1024**2  # bytes
    one_gib = 1024**3  # bytes

    if precision == "single":
        # n_bits_per_float = 32
        n_bytes_per_float = 4
    elif precision == "double":
        # n_bits_per_float = 64
        n_bytes_per_float = 8
    else:
        raise NotImplementedError

    # n_bits = n_floats * n_bits_per_float
    n_bytes = n_floats * n_bytes_per_float

    if name is None:
        name = ""
    else:
        name = f"of {name} "
    print(f"Memory usage {name}({n_floats} floats of type {precision}):")
    print(f"\t{n_bytes/one_mib:2.4f} Mib (mebibytes = 1024**2 bytes)")
    print(f"\t{n_bytes/one_gib:2.4f} Gib (gibibyte = 1024**3 bytes)")


def create_mfdataset_with_nans(
    nc_files: List[str],
    variables: Optional[List[str]] = None,
    group: Optional[str] = None,
) -> xr.Dataset:
    """Concatenate all Datasets along samples, inserting nans for missing samples."""
    # check which data exist
    no_data_idcs = []
    files_exist = []
    for i, sf in enumerate(nc_files):
        if os.path.exists(sf):
            files_exist.append(sf)
        else:
            no_data_idcs.append(i)

    # summarize
    print(f"\t[create_mfdataset_with_nans()] Found {len(files_exist)} files.")
    if len(no_data_idcs) > 0:
        print(
            (
                "\t[create_mfdataset_with_nans()] "
                f"Missing {len(no_data_idcs)}, namely {no_data_idcs}. "
                "Inserting nans for missing samples."
            )
        )

    # load existing data
    data_xr = xr.open_mfdataset(
        files_exist,
        combine="nested",
        concat_dim="sample",
        group=group,
        parallel=True,
    )
    if variables is not None:
        data_xr = data_xr[variables]

    # convert to numpy in order to use np.insert
    ds_vars = data_xr.data_vars
    data_np = np.stack([data_xr[var].values for var in ds_vars], axis=-1)

    # insert nans for samples that miss data
    # fixed ordering of dimensions: "sample" is first
    for i in no_data_idcs:
        data_np = np.insert(data_np, i, np.nan, axis=0)

    # convert back to xarray dataset
    dim_names = list(data_xr.dims.keys())
    dim_names = ["sample"] + [d for d in dim_names if d != "sample"]
    coords = {dim: data_xr[dim] for dim in dim_names}
    coords["sample"] = np.arange(len(nc_files))  # type: ignore[assignment]

    return xr.Dataset(
        data_vars={var: (dim_names, data_np[..., i]) for i, var in enumerate(ds_vars)},
        coords=coords,
    )


class NanCropper:
    """Wrap around the analysis to ensure failed samples are not considered."""

    def __init__(self, a: NDArray[np.float64], verbose: bool = True) -> None:
        """Initialize the NanCropper by detecting NaNs and saving their locations.

        The sample dimension is assumed to be dimension 0,
        e.g. a.shape = (n_samples, n_vars).
        """
        if not np.all(np.isfinite(a)):
            self.do_crop = True
            self.nan_samples = np.where(np.any(np.isnan(a), axis=1))[0]
            self.not_nan_samples = np.where(~np.any(np.isnan(a), axis=1))[0]

            if verbose:
                print(
                    (
                        "WARNING: Ensemble not finite. "
                        f"Cropping {len(self.nan_samples)} samples: {self.nan_samples}"
                    )
                )
        else:
            self.do_crop = False

    def crop(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        """Crop an ensemble by removing the samples that had NaNs at init."""
        if self.do_crop:
            if len(a.shape) > 1:
                return a[self.not_nan_samples, :]
            return a[self.not_nan_samples]
        return a

    def uncrop(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        """Uncrop an ensemble by reinserting NaNs at their original locations."""
        a_ = a
        if self.do_crop:
            if len(a.shape) > 1:
                # e.g., an ensemble
                nans = np.empty((a.shape[1],))
                nans[:] = np.nan
            else:
                # e.g., a scalar statistic per sample
                nans = np.nan  # type: ignore
            for i in self.nan_samples:
                a_ = np.insert(a_, i, nans, axis=0)
        return a_


def crop_nans(a: NDArray[np.float64], **kwargs: Any) -> NDArray[np.float64]:
    """Single-use cropping."""
    nan_cropper = NanCropper(a, **kwargs)
    return nan_cropper.crop(a)


def concat_to_1d(ds: xr.Dataset) -> NDArray[np.float64]:
    """Transform an xarray dataset to a 1D numpy array.

    Using numpy's ravel to view instead of copy (default: flatten before concat).
    Return shape (n_obs,).
    """
    variables = list(ds.data_vars.keys())
    variables.sort()
    return np.concatenate([ds[var].to_numpy().ravel() for var in variables])


class Normalizer:
    """Normalize data towards standard normal, e.g. given by a data mean and std."""

    def __init__(
        self,
        mean: Union[int, float, NDArray[np.float64]],
        var: Union[int, float, NDArray[np.float64]],
    ) -> None:
        """Prepare to normalize such that (mean,var) is mapped to (0,1)."""
        std = np.sqrt(var)
        if isinstance(mean, int):
            mean = float(mean)
        if isinstance(std, int):
            std = float(std)
        if isinstance(mean, float):
            mean = np.array([mean])
        if isinstance(std, float):
            std = np.array([std])
        self.n = len(mean)
        assert mean.shape == std.shape, f"Shapes {mean.shape=} and {std.shape=} differ."

        self.normalizers = [
            Normalize_scalar(vmin=m, vmax=m + s) for m, s in zip(mean, std)
        ]
        self.mean = mean
        self.std = std

    def __call__(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize samples of shape (n_obs, n_samples) or (n_samples,)."""
        if self.n == 1:
            data = np.expand_dims(data, axis=0)

        assert (
            data.shape[0] == self.n
        ), f"Data shape {data.shape} should be (n_obs={self.n}, n_samles)."

        data_normalized = np.zeros_like(data)
        for i in range(self.n):
            data_normalized[i] = self.normalizers[i](data[i])

        if self.n == 1:
            data_normalized = np.squeeze(data_normalized, axis=0)

        return data_normalized

    def print(
        self,
    ) -> None:
        """Print the normalizing properties."""
        print(f"Normalizing with mean: {self.mean}")
        print(f"Normalizing with std: {self.std}")


def process_camlab_style(
    fields_file: str,
    stats_file: str,
    cond_stats_file: str,
) -> None:
    """Rename time, sort coordinates, and convert to float32."""
    # mypy: ignore-errors

    # check if a group exists
    def group_exists(files: List[str], group: Optional[str]) -> bool:
        try:
            ds = xr.open_dataset(files, group=group)
            exists = True
            ds.close()
        except:  # pylint: disable=broad-except
            exists = False
        return exists

    # process all groups in a file
    def process_groups(func, file: str, groups: Optional[list[str]] = None) -> None:
        if not os.path.exists(file):
            return

        tmp_file = file[:-3] + "_clean.nc"

        if groups is not None:
            for i, group in enumerate(groups):
                if not group_exists(file, group):
                    pass
                mode = "a" if i > 0 else "w"
                success = func(file, tmp_file, group=group, mode=mode)
        else:
            success = func(file, tmp_file)

        if success:
            os.remove(file)
            os.rename(tmp_file, file)

    # loop over all files group-by-group and apply the same cleaning function
    def process(func) -> None:  # type: ignore
        process_groups(func, fields_file)
        process_groups(func, stats_file, ["timeseries", "profiles", "reference"])
        process_groups(func, cond_stats_file, ["spectra"])

    # rename "t" to "time" and sort time to the front
    def rename(
        file: str, tmp_file: str, group: Optional[str] = None, mode: str = "w"
    ) -> bool:
        ds = xr.open_dataset(file, group=group)
        if "t" not in ds:
            ds.close()
            return False

        ds = ds.rename({"t": "time"})
        ds = ds.transpose("time", ...)

        ds.to_netcdf(tmp_file, group=group, mode=mode)
        ds.close()
        return True

    # convert dtype from double to float
    def convert(
        file: str, tmp_file: str, group: Optional[str] = None, mode: str = "w"
    ) -> bool:
        ds = xr.open_dataset(file, group=group)

        # convert data
        ds = ds.astype(np.float32)

        # convert coordinates
        coord_dict = {
            coord_name: ds[coord_name].astype(np.float32) for coord_name in ds.coords
        }
        ds = ds.assign_coords(coord_dict)

        ds.to_netcdf(tmp_file, group=group, mode=mode)  # type: ignore
        ds.close()
        return True

    # execute processing steps
    process(rename)
    process(convert)


def get_nd_mode_kde(
    samples: NDArray[np.float64],
    bandwidth: str = "scott",
) -> NDArray[np.float64]:
    """
    Find the n-dimensional mode using kernel density estimation."""
    # Fit KDE to samples
    kde = gaussian_kde(samples.T, bw_method=bandwidth)

    # Define negative density for minimization (avoid log issues)
    def neg_density(x: NDArray[np.float64]) -> float:
        density = kde(x.reshape(-1, 1))[0]
        return -density

    # Start from sample mean for optimization
    x0 = np.mean(samples, axis=0)

    # Find the mode by minimizing negative log-density
    result = minimize(neg_density, x0, method="BFGS")

    mode = result.x
    # mode_density = kde(mode.reshape(-1, 1))[0]

    return mode
