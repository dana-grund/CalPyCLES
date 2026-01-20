"""Interface to dapper.tools.randvars.

TO DO
    - Implement a subclass of Cov with sparse kind='diag'
    - Avoid doubling memory in init_rv(). These two both allocate the same amount
      of memory, even though once should be enough:
        - covar = np.load(file_covar)
        - return GaussRV(C=covar)
"""

# Standard library
import os
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# Third-party
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm  # type: ignore[import-untyped]
from scipy.stats import qmc
from scipy.stats import uniform

# First-party
from calpycles.dapper.matrices import CovMat
from calpycles.dapper.randvars import GaussRV as GaussRV_
from calpycles.dapper.seeding import set_seed

# pylint: disable=import-outside-toplevel  # avoid circular imports for plotting

SEED = 5
set_seed(sd=SEED)
np.random.seed(SEED)


class GaussRV(GaussRV_):
    """A Gaussian random variable that can be stored, plotted, init from an ensemble.

    type: ignore[misc] silences "Class cannot subclass "GaussRV_" (has type "Any")"
    """

    dist_type = "normal"

    def __init__(
        self,
        mu: Optional[Union[NDArray[np.float64], float]] = None,
        C: Optional[Union[NDArray[np.float64], float]] = None,
        samples: Optional[NDArray[np.float64]] = None,
        names: Optional[List[str]] = None,
        defaults: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize.

        If samples are not given:
            Behaves like a normal Gaussian with mu and C.
            This could, for example, be the prior in Gaussian DA, given by
            mean and (co)variance.

        If samples are given:
            Approximates mu, C from the ensemble (in case not given).
            This could, for example, be the posterior in Gaussian DA, given by
            posterior samples. The mean and covariance can either also be handed
            over (in case given by the DA as well), or are inferred from the samples.
            The samples are not saved within RV!

        In contrast to GaussRV_, mu and C have to be passed as kwargs.

        The CovMat implementation in DAPPER is not sparse, so it does not matter
        if we use CovMat(cov, kind='full') or CovMat(samples, kind='E')
        that infers the full covariance from the ensemble.

        However, the initialization is faster for kind="diag" than for
        kind="full", as it avoids an eigenvalue coputation (d, V = sla.eigh(C)).

        TO DO:
            - For N<M, ensmeble DA profits from that the ensemble matrix is smaller
              than the covariance matrix. Hence, implement a GaussRV that stores
              the ensemble instead of the covariance!

        Parameters
        ----------
        M: int
            The number of components.
        mu: NDArray[np.float64], float
            float: constant mean.
            array: mean per component.
        C: NDArray[np.float64], float
            float: constant variance.
            array (M,): variance, i.e. diag(covariance).
            array (M,M): covariance.
        samples: NDArray[np.float64]
            array (N,M): samples used to infer mean and covariance.
        names: List[str]
            Names of the components.
        defaults: [NDArray[np.float64]
            Default values used for plotting.

        """
        # pylint: disable=invalid-name
        # allow C as a name

        # init from ensemble
        if samples is not None:
            # if gaussian approx not given, infer from samples
            if mu is None:
                # disregard nan samples
                mu = np.nanmean(samples, axis=0)
            if C is None:
                C = CovMat(samples, kind="E")

        # init from mean and (co)variance
        else:
            # default: standard normal
            if mu is None:
                mu = 0.0
            if C is None:
                C = 1.0

        super().__init__(mu=mu, C=C, **kwargs)

        # add names for random variable components
        self._names: List[str]
        if names is not None:
            assert len(names) == self.M, (
                f"The number of names {len(names)} does not match "
                f"the number of components {self.M}"
            )
            self._names = names

        # add defaults (not used, but maybe useful?)
        self.defaults = defaults

    @property
    def names(self) -> List[str]:
        """Get the names of the components."""
        if not hasattr(self, "_names"):
            self._names = [f"var {i}" for i in range(self.M)]
        return self._names

    @property
    def mean(self) -> NDArray[np.float64]:
        """Alias."""
        mean = self.mu
        assert isinstance(mean, np.ndarray)
        return mean

    @property
    def var(self) -> NDArray[np.float64]:
        """Alias."""
        var = self.C.diag
        assert isinstance(var, np.ndarray)
        return var

    def save(
        self,
        name: str = "dist",
        path: str = "./",
    ) -> None:
        """Save mean and covar that define this Gaussian, to be read by init_rv()."""
        file_mean = os.path.join(path, name + "_mean.npy")
        file_covar = os.path.join(path, name + "_covar.npy")
        if os.path.exists(file_mean):
            print("WARNING: distribution file already exists. Overwriting:", file_mean)

        np.save(file_mean, self.mu)
        if self.C.kind == "diag":
            np.save(file_covar, self.C.diag)
        else:
            np.save(file_covar, self.C.full)
        print(f"Saved GaussRV mean and var to {file_mean}; _covar.npy")

    def plot(
        self,
        samples: Optional[NDArray[np.float64]] = None,
        plot_2d: bool = True,
        **kwargs: Any,
    ) -> None:
        """Plot a Gaussian distribution, possibly with samples."""
        # First-party
        from calpycles.plotting.distributions_1d import plot_dists_1d
        from calpycles.plotting.distributions_2d import plot_dists_2d

        plot_dists_1d(
            dists=[self],
            dist_labels=[self.dist_type],
            samples=samples,
            name=self.dist_type,
            dist_space="unconstrained",
            **kwargs,
        )
        if plot_2d:
            plot_dists_2d(
                dists=[self],
                dist_labels=[self.dist_type],
                samples=samples,
                name=self.dist_type,
                dist_space="unconstrained",
                **kwargs,
            )


class GaussTransformedUniform(GaussRV):
    """A transform mapping between a uniform and unit normal distribuiton."""

    dist_type = "uniform"

    def __init__(
        self,
        *args: Any,
        M: int = 1,
        lower: Optional[Union[NDArray[np.float64], float]] = None,
        upper: Optional[Union[NDArray[np.float64], float]] = None,
        std_scaling: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize.

        Samples from the uniform distribution in (lower,upper) are mapped to a standard
        normal by to_unconstrained(), and to_constrained() performs the inverse.

        If non-default mu and C are specified, these are used for sampling,
        while the transforms targeting the standard normal stay valid.

        If initialized from samples (see GaussRV.__init__()), the samples must be
        given in unconstrined space.

        As an example, when a uniform distribution is to be specified as a Bayesian
        prior, it is implemented as GaussTransformedUniform(mu=0, C=1).
        The inversion is performed in the unconstrained space with prior N(0,1)
        and returns a Gaussian approximation N(mu_, C_). This is translated
        back to constrained space with GaussTransformedUniform(mu=mu_, C=C_).

        TO DO:
            - Kill non-default mu, C. Always project to standard normal and
              use to_constrained to project non-standard normal samples to
              constrained space.
        """
        # choose limits for the uniform
        if lower is None:
            lower = np.zeros(M)
        if upper is None:
            upper = np.ones(M)
        if isinstance(upper, float):
            upper = np.array([upper], dtype=np.float64)
        if isinstance(lower, float):
            lower = np.array([lower], dtype=np.float64)
        self.lower = lower
        self.upper = upper

        # internal representation: unconstrained normal distribution
        super().__init__(*args, M=M, **kwargs)

        # transform: maps the uniform to standard normal, irrespective of mu, C
        # scaling: match the std_scaling-th quantile
        # e.g.: weight = 0.997 = 99.7%, std_scaling = 3
        self.normalization = np.ones(M)  # init

        weight = norm.cdf(std_scaling)  # one-sided
        weight = weight - (1 - weight)  # zero-sided

        c_mean = (self.upper + self.lower) / 2
        c_std_x = weight * (self.upper - self.lower) / 2 + c_mean
        self.normalization = (self.to_unconstrained(c_std_x) - 0) / std_scaling

    def to_unconstrained(self, xi_: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform a uniform sample to normal.

        xi_ must be of shape (n_samples, n_params)
        """
        if len(xi_.shape) == 0:
            xi_ = np.array([xi_])
        assert (
            xi_.shape[-1] == self.lower.shape[0]
        ), f"Expected n_params={self.lower.shape[0]} by dist, but got {xi_.shape=}."

        # make resilient for xi=lower or xi=upper for some component: +- epsilon
        epsilon = 1e-4 * (self.upper - self.lower)
        xi_ = np.clip(xi_, self.lower + epsilon, self.upper - epsilon)

        result: NDArray[np.float64] = (
            np.log((xi_ - self.lower) / (self.upper - xi_)) / self.normalization
        )
        return result

    def to_constrained(self, xi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform a normal sample to uniform.

        xi must be of shape (n_samples, n_params)
        """
        if len(xi.shape) == 0:
            xi = np.array([xi])
        assert xi.shape[-1] == self.lower.shape[0]
        xi_norm = xi * self.normalization
        result: NDArray[np.float64] = (self.upper * np.exp(xi_norm) + self.lower) / (
            np.exp(xi_norm) + 1.0
        )
        return result

    @property
    def mean_constrained(self) -> NDArray[np.float64]:
        """Transform the mean to physical constrained space.

        This gives the median of the constrained distribution, not the mean!
        This transform does not make sense for the variance.
        """
        return self.to_constrained(self.mean)

    def plot(
        self,
        samples: Optional[NDArray[np.float64]] = None,
        plot_2d: bool = True,
        **kwargs: Any,
    ) -> None:
        """Plot the pdf and (if given) the samples pairwise in 2D, in both spaces.

        Assumes the samples passed are in unconstrained (normal) space,
        as output by GaussTransformedUniform.sample()!
        """
        super().plot(samples=samples, plot_2d=plot_2d, **kwargs)
        # First-party
        from calpycles.plotting.distributions_1d import plot_dists_1d
        from calpycles.plotting.distributions_2d import plot_dists_2d

        plot_dists_1d(
            dists=[self],
            dist_labels=[self.dist_type],
            samples=samples,
            name=self.dist_type,
            dist_space="constrained",
            **kwargs,
        )
        if plot_2d:
            plot_dists_2d(
                dists=[self],
                dist_labels=[self.dist_type],
                samples=samples,
                name=self.dist_type,
                dist_space="constrained",
                **kwargs,
            )


def init_standard_gaussian(n: int) -> GaussRV:
    """Initialize a standard Gaussian random variable."""
    return init_rv(dist_type="normal", mu=0, C=1, M=n)


def can_load_rv(path: str, name: str) -> bool:
    """Check whether the mean and covar files exist."""
    file_mean = os.path.join(path, name + "_mean.npy")
    file_covar = os.path.join(path, name + "_covar.npy")
    return os.path.exists(file_mean) and os.path.exists(file_covar)


def load_rv(
    path: str,
    name: str,
    **kwargs: Any,
) -> Union[GaussRV, GaussTransformedUniform]:
    """Load a random variable from mean and cov np files."""
    assert can_load_rv(
        path, name
    ), f"Can't load distribution named {name} from {path}. "

    file_mean = os.path.join(path, name + "_mean.npy")
    file_covar = os.path.join(path, name + "_covar.npy")

    mean = np.load(file_mean)
    covar = np.load(file_covar)  # may be scalar var, vector var, or matrix cov!
    kwargs.update(
        {
            "mu": mean,
            "C": covar,
        }
    )
    return init_rv(**kwargs)


def init_rv(
    dist_type: str = "uniform",
    **kwargs: Any,
) -> Union[GaussRV, GaussTransformedUniform]:
    """Initialize a random variable from keyword arguments.

    kwargs:
        mu: Optional[Union[NDArray[np.float64], float]] = None,
        C: Optional[Union[NDArray[np.float64], float]] = None,
        lower: Optional[Union[NDArray[np.float64], float]] = None,
        upper: Optional[Union[NDArray[np.float64], float]] = None,
        samples: Optional[NDArray[np.float64]] = None,

    dist_type:
        'uniform': a uniform parameter distribution, transformed to a normal
            distribuiton by GaussTransformedUniform().
        'normal': a normal distribution with GaussRV()
    lower, upper: optional bounds for dist='uniform', default: U([0, 1])
    mu,var: optional mean and variance for dist='normal', default: N(0,1)
    M: number of parameters
    names: names of parameters
    """
    # choose the corresponding distribution class
    if dist_type == "uniform":
        # takes lower, upper, mu, and C as kwargs
        return GaussTransformedUniform(**kwargs)

    if dist_type == "normal":
        # takes only mu and C as kwargs
        for kwarg in ["lower", "upper"]:
            if kwarg in kwargs:
                del kwargs[kwarg]
        return GaussRV(**kwargs)

    raise RuntimeError(f"Unknown distribution '{dist_type}'!")


def draw_uniform(
    names: List[str],
    mins: List[float],
    maxs: List[float],
    n_samples: int,
    which: str = "latin_hypercube",
) -> NDArray[np.float64]:
    """Draw samples from a uniform distribution."""
    n_params = len(names)

    # sample
    samples = np.zeros((n_samples, n_params))
    if which == "uniform":
        samples = uniform.rvs(size=(n_samples, n_params))
    elif which == "sobol":
        sampler = qmc.Sobol(d=n_params)
        samples = sampler.random(n=n_samples)
    elif which == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=n_params)
        samples = sampler.random(n=n_samples)
    else:
        raise ValueError(f"Unknown sampling method '{which}'!")

    # adapt ranges
    ranges = np.array(maxs) - np.array(mins)
    samples = samples * ranges + np.array(mins)

    return samples
