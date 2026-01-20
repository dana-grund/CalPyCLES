"""Parameters to be used by PyCLESsample.

TO DO
    - implement abstract Params and ParamEnsemble classes
    - implement parameter classes for other test cases
"""

# Standard library
import os
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# Third-party
import numpy as np
from numpy.typing import NDArray

# First-party
from calpycles.helpers import assert_print
from calpycles.helpers import assert_shape
from calpycles.helpers import make_table
from calpycles.helpers import save_tabular_data
from calpycles.namelist import Namelist
from calpycles.random_variables import GaussRV
from calpycles.random_variables import can_load_rv
from calpycles.random_variables import draw_uniform
from calpycles.random_variables import init_rv
from calpycles.random_variables import load_rv


class ParamEnsemble:
    """An ensemble of parameters to be used by PyCLESensemble."""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods

    def __init__(
        self,
        case: str,
        name: str,
        n_samples: Optional[int] = None,
        path: str = "./",
        parameter_ranges: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize.

        parameter_ranges = {
            "names": List[str],
            "defaults": NDArray[np.float64],
            "mins": NDArray[np.float64],
            "maxs": NDArray[np.float64],
        }

        """
        self.case = case
        self.name = name
        self.path = path
        self.verbose = verbose
        self.dist_name = "dist_parameters"

        # parameter properties
        if parameter_ranges is not None:
            self.parameter_ranges = parameter_ranges
        self.n_params = len(self.parameter_ranges["names"])

        # number of samples
        self.n_samples: int
        if n_samples is not None:
            self.n_samples = n_samples

        # properties
        self._dist: GaussRV
        self._samples: NDArray[np.float64]

    @property
    def parameter_ranges(self) -> Dict[str, Any]:
        """Assemble parameter properties."""
        if not hasattr(self, "_parameter_ranges"):
            if self.can_load_parameter_ranges:
                self.load_parameter_ranges()
            else:
                self._parameter_ranges = parameters_factory(self.case).parameter_ranges
                print("Using default parameter properties in parameter ensemble init.")
        return self._parameter_ranges

    @parameter_ranges.setter
    def parameter_ranges(
        self,
        parameter_ranges: Dict[str, Any],
    ) -> None:
        """Set parameter properties."""
        for key in ["defaults", "mins", "maxs"]:
            if not isinstance(parameter_ranges[key], np.ndarray):
                parameter_ranges[key] = np.array(parameter_ranges[key])
        self._parameter_ranges = parameter_ranges
        self.write_parameter_ranges()

    def write_parameter_ranges(self) -> None:
        """Write param props to file."""
        write_parameter_ranges_to_file(**self._parameter_ranges, path=self.path)

    def load_parameter_ranges(self) -> None:
        """Load parameter properties from file."""
        if self.can_load_parameter_ranges:
            self._parameter_ranges = read_parameter_ranges_from_file(self.path)

    @property
    def can_load_parameter_ranges(self) -> bool:
        """Check whether the parameter properties can be loaded."""
        return os.path.exists(os.path.join(self.path, "parameter_ranges.txt"))

    @property
    def dist_kwargs(self) -> Dict[str, Any]:
        """Assemble the kwargs passed to GaussRV."""
        return get_dist_kwargs(
            case=self.case,
            parameter_ranges=self.parameter_ranges,
        )

    @property
    def dist(self) -> GaussRV:
        """The Gaussian parameter ensemble distribution.

        self.dist_kwargs["dist_type"]:
            "uniform": a uniform parameter distribution on
                [self.mins, self.maxs],
                transformed to a normal distribuiton by GaussTransformedUniform().
                self.samples will hold the normal samples used for DA etc.
            "normal": a normal distribution with mean self.defaults
                and variance 0.1 * mean

        Saves the normal samples, also for dist="uniform"!
        """
        if not hasattr(self, "_dist"):
            self.init_dist()
        return self._dist

    @dist.setter
    def dist(self, d: GaussRV) -> None:
        """Set the distribution manually."""
        self.init_dist(d, load=False)

    def init_dist(
        self,
        d: Optional[GaussRV] = None,
        load: bool = False,
        from_samples: bool = False,
        **kwargs: Any,
    ) -> None:
        """Set or load the distribution.

        Default: Use the case-specific kwargs given by get_dist_kwargs(case).

        Optional key words (see documentation of init_rv):
            dist_type: str = 'uniform',
            mu: Optional[Union[NDArray[np.float64], float]] = None,
            var: Optional[Union[NDArray[np.float64], float]] = None,
            lower: Optional[Union[NDArray[np.float64], float]] = None,
            upper: Optional[Union[NDArray[np.float64], float]] = None,
            dist:
                'uniform': a uniform parameter distribution, transformed to a normal
                distribuiton by GaussTransformedUniform().
                'normal': a normal distribution with GaussRV()
            lower, upper:
                optional bounds for dist='uniform', default: U([0, 1])
            mu,var:
                optional mean and variance for dist='normal', default: N(0,1)
        """
        if load:
            # load distribution
            assert (
                self.can_load
            ), f"Can't load parameter distribution {self.dist_name} from {self.path}"
            self._dist = load_rv(
                path=self.path, name=self.dist_name, **self.dist_kwargs
            )
        else:
            if from_samples:
                # infer distribution from samples
                if self.has_samples:
                    # samples are available
                    self.dist = init_rv(
                        samples=self.samples, **self.dist_kwargs, **kwargs
                    )
                elif self.can_load:
                    # samples can be loaded
                    self.load()
                    self.dist = init_rv(
                        samples=self.samples, **self.dist_kwargs, **kwargs
                    )
                else:
                    raise ValueError(
                        "Missing param_ens._samples to initialize a distribution."
                    )
            elif d is None:
                # default distribution
                self._dist = init_rv(**self.dist_kwargs, **kwargs)
            else:
                # custom distribution
                self._dist = d

            # save newly set distribution
            self.save_dist()

    def save_dist(
        self,
    ) -> None:
        """Save the mean and covariance of the distribution."""
        self._dist.save(path=self.path, name=self.dist_name)

    def sample(self) -> None:
        """Force the sampling (done by dapper)."""
        _ = self.samples

    @property
    def samples(self) -> NDArray[np.float64]:
        """Draws samples and save self.n_samples from self.dist."""
        if not self.has_samples:
            assert hasattr(
                self, "n_samples"
            ), "Don't know how many parameter samples to draw."

            # draw samples, and call the setter to save them
            print(f"Drawing parameter samples for {self.name}...")
            self.samples = self.dist.sample(self.n_samples)
            assert_shape(
                (self.n_samples, self.n_params), self.samples, "parameter samples"
            )

            # save the samples
            if not os.path.exists(self.file_unconstrained):
                self.save()

        return self._samples

    @samples.setter
    def samples(
        self,
        values: NDArray[np.float64],
    ) -> None:
        """Sample and save the parameters (in unconstrained space).

        Usage:
            parameters.samples = E # where E is returned by dapper
        """  # noqa: D402
        if hasattr(self, "n_samples"):
            assert_shape((self.n_samples, self.n_params), values)
        else:
            self.n_samples = values.shape[0]
            assert_print(self.n_params, values.shape[1])
            assert (
                self.n_samples > 1
            ), "Can't take distribution over less than two samples!"

        # set the samples
        self._samples = values

        # save the samples
        if not os.path.exists(self.file_unconstrained):
            self.save()

    def set(
        self,
        values: NDArray[np.float64],
        constrained: bool = False,
    ) -> None:
        """Set parameter values in constrained or unconstrained space.

        Constr. can be used to manually choose the parameter values in physical space.
        Unconstr. is used by DAPPER to perform DA in normalized space.
        """
        if constrained:
            # approximation at the uniform limits
            self.samples = self.dist.to_unconstrained(values)
        else:
            self.samples = values

    @property
    def has_samples(
        self,
    ) -> bool:
        """Check whether samples were set or generated."""
        return hasattr(self, "_samples")

    def draw_uniform(self, which: str = "latin_hypercube") -> None:
        """Draw uniform samples in physical space."""
        self.set(
            draw_uniform(
                names=self.names,
                mins=self.parameter_ranges["mins"],
                maxs=self.parameter_ranges["maxs"],
                n_samples=self.n_samples,
                which=which,
            ),
            constrained=True,
        )

    @property
    def save_name(
        self,
    ) -> str:
        """Name of the saved samples."""
        return "samples_parameters"

    @property
    def names(self) -> List[str]:
        """The names of the parameters."""
        return self.parameter_ranges["names"]  # type: ignore

    def save(self) -> None:
        """Save the samples in numpy and netcdf formats."""
        # unconstrained space
        save_tabular_data(
            samples=self.samples,
            names=self.names,
            save_name=f"{self.save_name}_unconstrained",
            save_path=self.path,
        )

        # constrained space
        if self.dist.dist_type == "uniform":
            samples_c = self.dist.to_constrained(self.samples)
            save_tabular_data(
                samples=samples_c,
                names=self.names,
                save_name=f"{self.save_name}_constrained",
                save_path=self.path,
            )

        if self.verbose:
            print("Saved parameter samples (.npy, .nc, .txt) to ", self.path)

    @property
    def samples_constrained(self) -> NDArray[np.float64]:
        """Give samples in physical constrained space."""
        if self.dist_kwargs["dist_type"] == "uniform":
            result: NDArray[np.float64] = self.dist.to_constrained(self.samples)
        else:
            result = self.samples
        return result

    def plot(self, **kwargs: Any) -> None:
        """Plot the parameter distribution and samples."""
        assert self.has_samples, "No samples to plot."
        self.dist.plot(
            samples=self.samples_constrained,
            plot_dir=self.path,
            plot_2d=False,
            **kwargs,
        )

    @property
    def can_load(self) -> bool:
        """Check whether the distribution and samples can be loaded."""
        return can_load_rv(self.path, self.dist_name) and (
            os.path.exists(self.file_unconstrained)
            or os.path.exists(self.file_constrained)
        )

    @property
    def file_unconstrained(self) -> str:
        """Where the data is saved in unconstrained format."""
        return os.path.join(self.path, f"{self.save_name}_unconstrained.npy")

    @property
    def file_constrained(self) -> str:
        """Where the data is saved in constrained format."""
        return os.path.join(self.path, f"{self.save_name}_constrained.npy")

    @property
    def file(self) -> str:
        """Where the data is saved in constrained format (old)."""
        return os.path.join(self.path, f"{self.save_name}.npy")

    def load(self) -> None:
        """Load the samples and re-creates the distribution."""
        # initialize dist as usual
        self.init_dist(load=True)

        # parameter properties
        self.load_parameter_ranges()

        # samples
        if os.path.exists(self.file_unconstrained):
            self.samples = np.load(self.file_unconstrained)
            if self.verbose:
                print("Loaded parameter samples from ", self.file_unconstrained)
            self.n_samples = self.samples.shape[0]
            return

        if os.path.exists(self.file_constrained):
            file = self.file_constrained
        elif os.path.exists(self.file):
            file = self.file
        else:
            print(
                (
                    "WARNING: Could not load parameter samples from "
                    f"{self.file} nor _constrained nor _unconstrained"
                )
            )
            return

        samples = np.load(file)
        self.samples = self.dist.to_unconstrained(samples)
        if self.verbose:
            print("Loaded parameter samples from ", file)
        self.n_samples = self.samples.shape[0]

    def print(self) -> None:
        """Print and save a table of the samples."""
        print(f"\nParameter samples ({self.name}):")
        make_table(
            samples=self.samples_constrained, names=self.dist.names, do_print=True
        )


class Parameters(ABC):
    """Class managing general parameters.

    Needs to specify: n_params, names, defaults, mins, maxs, dist_type, namelist_keys.
    """

    def __init__(
        self,
        namelist: Optional[Namelist] = None,
    ) -> None:
        """Initialize parameters from namelist."""
        if namelist is not None:
            self.namelist = namelist

    case: str
    names: list[str]
    defaults: NDArray[np.float64]
    mins: NDArray[np.float64]
    maxs: NDArray[np.float64]
    dist_type: str
    n_params: int

    namelist_keys = property(abstractmethod(lambda self: list[list[str]]))

    @property
    def parameters_dict(
        self,
    ) -> Dict[str, float]:
        """Infer parameter values from namelist.

        Checks the namelist at every call to not miss potential updates.
        """
        d = {}
        for name, keys in zip(self.names, self.namelist_keys):
            if len(keys) == 2:
                d[name] = self.namelist.namelist[keys[0]][keys[1]][name]
            elif len(keys) == 1:
                d[name] = self.namelist.namelist[keys[0]][name]
            else:
                raise ValueError(f"Wrong number of keys for {name}: {keys}")
        return d

    @property
    def parameters(self) -> NDArray[np.float64]:
        """Access the parameter values as numpy array."""
        params = self.parameters_dict
        return np.array([params[p] for p in self.names])

    def set(
        self,
        values: Union[NDArray[np.float64], Dict[str, float]],
    ) -> None:
        """Set the parameters in the namelist from a numpy array or dict.

        Here, parameters are set in physical space (as written in namelist).
        """
        if isinstance(values, dict):
            self.set_from_dict(values)
        elif isinstance(values, np.ndarray):
            self.set_from_np(values)
        else:
            raise ValueError("Values must be given as dict or 1D array.")

    def set_from_dict(
        self,
        values_dict: Dict[str, float],
    ) -> None:
        """Set the parameters in the namelist from a dict (non-nested: {p:v}).

        Here, parameters are set in physical space (as written in namelist).
        """
        values = []
        for p in self.names:
            if p in values_dict:
                values.append(values_dict[p])
            else:
                values.append(self.parameters_dict[p])
        self.set_from_np(np.array(values))

    def set_from_np(
        self,
        values: NDArray[np.float64],
    ) -> None:
        """Set the parameters in the namelist from a numpy array.

        Here, parameters are set in physical space (as written in namelist).
        """
        assert len(np.shape(values)) == 1, "Values must be given as 1D array."
        assert len(values) == self.n_params, "Wrong number of parameters."

        # reconstruct namelist dict for each parameter
        d = {}  # type: ignore
        for i, v in enumerate(values):
            name = self.names[i]
            k = self.namelist_keys[i]
            assert isinstance(k, list) and len(k) > 0, f"No namelist keys for {name}."

            if k[0] not in d:
                d[k[0]] = {}
            if len(k) > 1:
                # e.g. ["sgs", "UniformViscosity"] -> diffusivity
                if k[1] not in d[k[0]]:
                    d[k[0]][k[1]] = {}
                d[k[0]][k[1]][name] = v
            else:
                # e.g. ["surface"] -> cm
                d[k[0]][name] = v

        # update the namelist
        self.namelist.update(d)

    @property
    def dist_kwargs(
        self,
    ) -> Dict[str, Any]:
        """Assemble the kwargs passed to GaussRV."""
        return get_dist_kwargs(parameter_ranges=self.parameter_ranges)

    @property
    def parameter_ranges(self) -> Dict[str, Any]:
        """Assemble parameter properties."""
        return {
            "names": self.names,
            "defaults": self.defaults,
            "mins": self.mins,
            "maxs": self.maxs,
        }

    @property
    def is_nan(self) -> bool:
        """Check if any parameter is NaN."""
        return bool(np.isnan(self.parameters).any())


class ParametersDYCOMS_RF01(Parameters):
    """Class managing the parameters for a DYCOMS RF01 test case sample."""

    # pylint: disable=invalid-name

    case = "DYCOMS_RF01"
    names = ["ug", "divergence", "zi", "tg", "qtg", "sst", "cm", "cs", "prt"]

    defaults = np.array(  # = nature (corresponds to Stevens05)
        [
            7.35,
            3.75e-06,
            # 3.87e-06, # mean between two estimates
            840.0,
            290.4619859957094,
            # 290.87, # corrected to yield exact fluxes
            0.009,
            292.5,
            # 292.37, # corrected to yield exact fluxes
            0.0011,
            0.0,
            0.3333333333333333,
        ]
    )
    mins = np.array([0.0, 0.0, 756.0, 289.6, 0.0081, 291.0, 0.00099, 0.0, 0.25])
    maxs = np.array([14.7, 7.74e-06, 924.0, 290.8, 0.0099, 293.0, 0.00121, 0.2, 1.0])
    dist_type = "uniform"
    n_params = len(names)
    namelist_keys = property(
        lambda self: [
            ["forcing"],  # ug
            ["forcing"],  # divergence
            ["initial"],  # zi
            ["initial"],  # tg
            ["initial"],  # qtg
            ["surface"],  # sst
            ["surface"],  # cm
            ["sgs", "Smagorinsky"],  # cs
            ["sgs", "Smagorinsky"],  # prt
        ]
    )


# ---


def parameters_factory(case: str = "DYCOMS_RF01", **kwargs: Any) -> Parameters:
    """Return the corresponding parameter class, like a factory."""
    if case == "DYCOMS_RF01":
        return ParametersDYCOMS_RF01(**kwargs)

    raise NotImplementedError(
        f"The case '{case}' is not implemented as a Parameters child!"
    )


def parameter_classes(case: str = "DYCOMS_RF01") -> Type[Parameters]:
    """Return the corresponding parameter class."""
    if case == "DYCOMS_RF01":
        return ParametersDYCOMS_RF01

    raise NotImplementedError(
        f"The case '{case}' is not implemented as a Parameters child!"
    )


def get_dist_kwargs(
    case: str = "DYCOMS_RF01",
    path: Optional[str] = None,
    parameter_ranges: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Format parameter ranges to be passed to random variables like GaussRV.

    Transform passed parameter ranges, read them from a file,
    or return the class default.
    """
    class_ = parameter_classes(case)

    if path is not None:
        param_file = os.path.join(path, "parameter_ranges.txt")
        if os.path.exists(param_file):
            print(f"Reading parameter properties from {param_file}.")
            parameter_ranges = read_parameter_ranges_from_file(path)

    if parameter_ranges is not None:
        print("Using specified parameter ranges to assemble dist_kwargs.")
        return {
            "M": len(parameter_ranges["names"]),
            "defaults": parameter_ranges["defaults"],
            "names": parameter_ranges["names"],
            "lower": parameter_ranges["mins"],
            "upper": parameter_ranges["maxs"],
            "dist_type": class_.dist_type,
        }

    print("Using class default for parameter properties.")
    return {
        "M": class_.n_params,
        "names": class_.names,
        "defaults": class_.defaults,
        "lower": class_.mins,
        "upper": class_.maxs,
        "dist_type": class_.dist_type,
    }


def read_parameter_ranges_from_file(
    path: str, file: str = "parameter_ranges.txt"
) -> Dict[str, Any]:
    """Read parameter properties from file."""
    names, defaults, mins, maxs = None, None, None, None
    with open(os.path.join(path, file), "r", encoding="utf-8") as f:
        for _ in range(4):
            line = f.readline()

            def prep_line(
                line: str, as_array: bool = True
            ) -> Union[List[str], NDArray[np.float64]]:
                """Read a line from the parameter_ranges.txt.

                File content:
                    names = ['divergence', ...]
                    defaults = [3.87e-06, ...]
                """
                line = line.split("=")[1].strip()
                line = line[1:-1]  # remove []
                values = line.split(", ")
                if as_array:
                    return np.array([float(n) for n in values])
                return values

            if line.startswith("names"):
                names = [n[1:-1] for n in prep_line(line, as_array=False)]

            if line.startswith("defaults"):
                defaults = prep_line(line)

            if line.startswith("mins"):
                mins = prep_line(line)

            if line.startswith("maxs"):
                maxs = prep_line(line)

    assert names is not None, "No names found in parameter_ranges.txt."
    assert defaults is not None, "No defaults found in parameter_ranges.txt."
    assert mins is not None, "No mins found in parameter_ranges.txt."
    assert maxs is not None, "No maxs found in parameter_ranges.txt."

    return {
        "names": names,
        "mins": mins,
        "defaults": defaults,
        "maxs": maxs,
    }


def write_parameter_ranges_to_file(
    names: List[str],
    defaults: NDArray[np.float64],
    mins: NDArray[np.float64],
    maxs: NDArray[np.float64],
    path: str,
    file: str = "parameter_ranges.txt",
) -> None:
    """Write the parameter properties to file."""
    file_ = os.path.join(path, file)
    print(f"Writing parameter properties to file {file_}.")
    with open(file_, "w", encoding="utf-8") as f:
        f.write("names = " + str(list(names)) + "\n")
        f.write("defaults = " + str(list(defaults)) + "\n")
        f.write("mins = " + str(list(mins)) + "\n")
        f.write("maxs = " + str(list(maxs)) + "\n")
