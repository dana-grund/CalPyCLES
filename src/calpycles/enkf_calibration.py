"""Single-step EnKF assimilation for small matrices."""

# Standard library
from typing import Any
from typing import Optional
from typing import Tuple

# Third-party
import numpy as np
from numpy.typing import NDArray

# First-party
from calpycles.dapper.linalg import mrdiv
from calpycles.dapper.linalg import pad0
from calpycles.dapper.linalg import svd0
from calpycles.dapper.seeding import set_seed
from calpycles.dapper.stats import mean0
from calpycles.DYCOMS_RF01.measurements import MeasurementsDYCOMS_RF01
from calpycles.DYCOMS_RF01.measurements import get_obs_names_with_height
from calpycles.helpers import NanCropper
from calpycles.helpers import get_nd_mode_kde
from calpycles.parameters import ParametersDYCOMS_RF01
from calpycles.plotting.distributions import plot_samples_1d
from calpycles.plotting.heatmap import plot_heatmap
from calpycles.pycles_ensemble import PyCLESensemble
from calpycles.random_variables import GaussRV

# pylint: disable=invalid-name
# allow for upper-case matrix names

# defaults
MEAS = MeasurementsDYCOMS_RF01()
PARAMS = ParametersDYCOMS_RF01()


# selection of observation used in paper plotting
# z=480m as a mid-BL height with rather large updates
# z=747m since larger updates by higher ql measurements
OBS_IDCS = [
    MEAS.obs_names.index(obs_name)
    for obs_name in [
        "cloud base height",
        "cloud base rate",
        "cloud top height",
        "cloud top rate",
        "$\\overline{q}_l$ (747m)",
        "$\\overline{q}_t$ (480m)",
        "$\\overline{\\theta}_l$ (480m)",
        "$\\overline{w'w'}$ (480m)",
        "$\\overline{w'w'w'}$ (480m)",
    ]
]


def get_ens(ens: PyCLESensemble) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Assemble prior and observed matrices."""
    try:
        E = np.load(ens.path + "/samples_parameters.npy")
    except FileNotFoundError:
        E = np.load(ens.path + "/samples_parameters_unconstrained.npy")
    Eo = ens.obs_ens.obs
    return E, Eo


def get_data(
    y: Optional[NDArray[np.float64]] = None,
    hnoise: Optional[GaussRV] = None,
    meas: Optional[MeasurementsDYCOMS_RF01] = None,
) -> Tuple[NDArray[np.float64], GaussRV]:
    """Assemble data vector and observation error distribution."""
    if meas is None:
        meas = MEAS
    if y is None:
        y = meas.observations_np
    if hnoise is None:
        hnoise = meas.dist_obs_error
    return y, hnoise


class EnKFcalibration:
    """A class to perform the EnKF calibration."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        ens: PyCLESensemble,  # prior ensemble
        y: Optional[NDArray[np.float64]] = None,  # default: measured
        hnoise: Optional[GaussRV] = None,
        L: Optional[NDArray[np.float64]] = None,  # localization
        plot_folder: str = "./",
        name: str = "",
        skip_obs: Optional[list[str]] = None,
        seed: int = 574123,  # for measurement perturbation
    ) -> None:
        """Assemble the matrices."""

        self.plot_folder = plot_folder
        self.name = name
        self.seed = seed

        # get observation names
        self.obs_names = get_obs_names_with_height()

        # get ensembles
        self.E, self.Eo = get_ens(ens)

        # remove nans in observations
        self.crop_nans()

        # get data
        self.y, self.hnoise = get_data(y, hnoise)

        # change unit of cloud rates from m/s to m/h
        self.change_rate_units()

        # skip some observations with too large model-data mismatch
        self.skip_obs(skip_obs)

        # assemble other matrices
        self.assemble_matrices()

        # save localization
        if L is None:
            L = np.ones(self.C_XY.shape)
        self.L = L

        # remember how to map back between spaces
        self.to_constrained = ens.param_ens.dist.to_constrained  # (E)
        self.to_unconstrained = ens.param_ens.dist.to_unconstrained  # (E)

    def change_rate_units(self) -> None:
        """Change units of cloud rates from m/s to m/h.

        hnoise.C.full diagonal entries seem to be mapped to 0 if too small.
        However, calling change_rate_units() doesn't seem to change the result.
        """
        self.Eo, self.y, self.hnoise = change_rate_units(self.Eo, self.y, self.hnoise)

    def skip_obs(self, obs_names: Optional[list[str]] = None) -> None:
        """Remove selected observations from the assimilation.

        obs_names are as in self.get_obs_names_with_height().
        """
        if obs_names is None or len(obs_names) == 0:
            return

        idcs = [self.obs_names.index(name) for name in obs_names]

        self.Eo = np.delete(self.Eo, idcs, axis=1)
        self.y = np.delete(self.y, idcs, axis=0)

        diag = self.hnoise.C.diag.copy()
        diag = np.delete(diag, idcs, axis=0)
        mean = self.hnoise.mean.copy()
        mean = np.delete(mean, idcs, axis=0)
        self.hnoise = GaussRV(mean, diag * np.eye(len(diag)))

        self.obs_names = [
            name for i, name in enumerate(self.obs_names) if i not in idcs
        ]

    def crop_nans(self) -> None:
        """Crop samples with NaNs in the observations."""
        self.nan_cropper = NanCropper(self.Eo)  # diagnose nans in observations
        if self.nan_cropper.do_crop:
            self.E = self.nan_cropper.crop(self.E)  # crop state
            self.Eo = self.nan_cropper.crop(self.Eo)  # crop observations

    @property
    def sample_names(self) -> list[str]:
        """Assemble names for the samples."""
        if self.nan_cropper.do_crop:
            return [str(i) for i in self.nan_cropper.not_nan_samples]

        return [str(i) for i in range(self.N)]

    @property
    def param_names(self) -> list[str]:
        """Assemble names for the parameters."""
        return PARAMS.names

    def assemble_matrices(self) -> None:
        """Assemble matrices from the parameter and observation ensembles."""
        self.N, self.Nx = self.E.shape  # Dimensionality
        self.Ny = self.Eo.shape[1]

        self.mu = np.mean(self.E, 0)  # Ens mean
        self.A = self.E - self.mu  # Ens anomalies

        self.xo = np.mean(self.Eo, 0)  # Obs ens mean
        self.Y = self.Eo - self.xo  # Obs ens anomalies
        self.misfit = self.y - self.Eo

        self.CX = self.A.T @ self.A / (self.N - 1)  # Param covariance
        self.CY = self.Y.T @ self.Y / (self.N - 1)  # Data covariance
        self.C_XY = self.A.T @ self.Y / (self.N - 1)  # Param-data covariances

    @property
    def Corr_XY(self) -> NDArray[np.float64]:
        """Cross-correlation matrix between parameters and observations."""
        Std_X = np.std(self.E, axis=0)
        Std_Y = np.std(self.Eo, axis=0)

        Std_X = np.expand_dims(Std_X, axis=1)
        Std_Y = np.expand_dims(Std_Y, axis=0)

        # correlations
        return self.C_XY / Std_X / Std_Y

    @property
    def Corr_Y(self) -> NDArray[np.float64]:
        """Cross-correlation matrix between observations."""
        Std_Y = np.std(self.Eo, axis=0)
        Std_Y = np.expand_dims(Std_Y, axis=0)

        # correlations
        result = self.CY / Std_Y / Std_Y.T
        return result

    @property
    def KG(
        self,
    ) -> NDArray[np.float64]:
        """Compute the (localized) Kalman gain matrix."""
        R = self.hnoise.C.diag * np.eye(self.Ny)

        # KG = (self.L * self.C_XY) @ np.linalg.inv(self.CY + R)

        # Equivalent alternative (dapper):
        C = self.Y.T @ self.Y + R * (self.N - 1)
        YC = mrdiv(self.Y, C)
        KG = self.A.T @ YC

        return KG

    def measurement_perturbation(
        self, seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """Generate measurement perturbations, known as what contributes to D or R."""
        # if given, fix random seed for measurement perturbations
        if seed is not None:
            print(f"Setting measurement perturbation seed to {seed}.")
            set_seed(seed)
        else:
            set_seed(self.seed)
        D = mean0(self.hnoise.sample(self.N))
        return D

    def analysis(
        self, constrained: bool = False, seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """Compute the analysis with the default stochastic EnKF (Pertobs)."""
        # if given, fix random seed for measurement perturbations
        D = self.measurement_perturbation(seed)

        KG = self.KG
        dE = (KG @ (self.misfit - D).T).T
        E_new = self.E + dE

        # return in normalized space
        if not constrained:
            return E_new

        # return in physical space
        return self.to_constrained(E_new)

    def analysis_observations(self, seed: Optional[int] = None) -> NDArray[np.float64]:
        """Compute the linear prediction of the measurements following Evensen 2019."""
        D = self.measurement_perturbation(seed)
        R = self.hnoise.C.diag * np.eye(self.Ny)

        dY = self.CY @ np.linalg.inv(self.CY + R) @ (self.misfit - D).T
        Y_new = self.Eo + dY.T

        return Y_new

    def analysis_dapper(self, da_type: str = "PertObs") -> NDArray[np.float64]:
        """Compute the analysis using different flavors of the EnKF."""
        # pylint: disable=too-many-locals

        E = self.E
        Eo = self.Eo
        y = self.y
        hnoise = self.hnoise

        R = hnoise.C  # Obs noise cov
        N, Nx = E.shape  # Dimensionality
        N1 = N - 1  # Ens size - 1

        mu = np.mean(E, 0)  # Ens mean
        A = E - mu  # Ens anomalies

        xo = np.mean(Eo, 0)  # Obs ens mean
        Y = Eo - xo  # Obs ens anomalies
        dy = y - xo  # Mean "innovation"

        misfit = y - Eo

        if da_type == "PertObs":
            # Uses classic, perturbed observations (Burgers'98)
            C = Y.T @ Y + R.full * N1
            D = mean0(hnoise.sample(N))
            YC = mrdiv(Y, C)
            KG = A.T @ YC
            # HK = Y.T @ YC

            dE = (KG @ (misfit - D).T).T
            E_new = E + dE

        elif da_type == "Sqrt":
            # Uses a symmetric square root (ETKF)
            # to deterministically transform the ensemble.
            assert N > Nx, "For N<=Nx, see dapper."

            # elif 'svd' in upd_a:
            # Implementation using svd of Y R^{-1/2}.
            V, s, _ = svd0(Y @ R.sym_sqrt_inv.T)
            d = pad0(s**2, N) + N1

            T = (V * d ** (-0.5)) @ V.T * np.sqrt(N1)

            Pw = (V * d ** (-1.0)) @ V.T
            w = dy @ R.inv @ Y.T @ Pw

            # docs/snippets/trHK.jpg
            # trHK    = np.sum((s**2+N1)**(-1.0) * s**2)

            E_new = mu + w @ A + T @ A

            # KG = R.inv @ Y.T @ Pw @ A  # see dapper code comment?

        else:
            raise ValueError(f"Unknown da_type {da_type}.")

        return E_new

    def post_corr_transform(self) -> NDArray[np.float64]:
        """Compute the posterior correlation matrix between parameters.

        When plotted, the lower triangle shows corr in normalized space,
        the upper triangle in physical space.
        """
        E_new = self.analysis()

        corr = corr_matrix(E_new)
        corr_c = corr_matrix(self.to_unconstrained(E_new))

        corr_both = corr.copy()
        Nx = corr.shape[0]
        for i in range(1, Nx):
            corr_both[i, :i] = corr_c.T[i, :i]

        return corr_both

    @property
    def partial_updates(self) -> NDArray[np.float64]:
        """Partial updates for each parameter and observation."""
        KG = self.KG
        misfit_mean = np.mean(
            self.misfit, axis=0
        )  # mean misfit per obs over the ensemble

        Nx, Ny = KG.shape
        update = np.zeros((Nx, Ny))

        for ix in range(Nx):
            for iy in range(Ny):
                update[ix, iy] = KG[ix, iy] * misfit_mean[iy]

        return update

    def get_posterior_mode(self, constrained: bool = False) -> NDArray[np.float64]:
        """Compute the posterior mode (maximum a posteriori)."""
        E_new = self.analysis(constrained=constrained)
        return get_nd_mode_kde(E_new)

    def plot_matrices(
        self, excerpt: bool = True, max_samples: int = 10, **kwargs: Any
    ) -> None:
        """Plot Kalman gain and partial updates matrices."""

        if excerpt:
            # plot only part of the observations
            mask = OBS_IDCS
            obs_names = [self.obs_names[i] for i in mask]
            fraction = 0.5  # plot size
            folder = self.plot_folder + "/matrices_excerpts/"
        else:
            mask = slice(None)  # type:ignore
            obs_names = self.obs_names
            fraction = 1  # plot size
            folder = self.plot_folder + "/matrices_full/"

        kwargs.update(
            {
                "name": self.name,
                "folder": folder,
                "fraction": fraction,
            }
        )

        kwargs_xy = kwargs.copy()
        kwargs_xy.update(
            {
                "xlabels": self.param_names,
                "ylabels": obs_names,
            }
        )

        plot_heatmap(
            self.KG[:, mask],
            plot_name="K",
            cbar_label=r"Kalman gain matrix $K$",
            vmax=5,
            **kwargs_xy,
        )

        # plot_heatmap(
        #     self.C_XY[:, mask],
        #     plot_name="Cxy",
        #     cbar_label=r"Covariance $C_{\theta y}$",
        #     vmax=150,
        #     **kwargs_xy,
        # )

        plot_heatmap(
            self.Corr_XY[:, mask],
            plot_name="Corr_xy",
            cbar_label=r"Correlation $C_{\theta y}/(\sigma_\theta \sigma_y)$",
            vmax=1,
            **kwargs_xy,
        )

        plot_heatmap(
            self.partial_updates[:, mask],
            plot_name="partial_updates",
            cbar_label=r"Update $(\Delta\theta)_{ij}$ to $\theta_i$ by $y_j$",
            vmax=0.7,
            **kwargs_xy,
        )

        plot_heatmap(
            self.misfit[:max_samples, mask],
            self.sample_names[:max_samples],
            obs_names,
            plot_name="misfit",
            cbar_label=r"Misfit $(d-\mathcal{G}(\theta)$",
            **kwargs,
        )

        plot_heatmap(
            self.Corr_Y[mask][:, mask],
            obs_names,
            obs_names,
            vmax=1,
            plot_name="Corr_y",
            cbar_label=r"Correlation $C_Y/(\sigma_Y^2)$",
            **kwargs,
        )

        # plot_heatmap(
        #     self.CY[mask][:,mask],
        #     obs_names,
        #     obs_names,
        #     vmax=1,
        #     plot_name="Cy",
        #     cbar_label=r"Covariance $C_Y$",
        #     **kwargs,
        # )

        plot_heatmap(
            self.post_corr_transform(),
            self.param_names,
            self.param_names,
            vmax=1,
            plot_name="post_corr_transform",
            cbar_label="Post. corr. (l: normal, u: uniform)",
            folder=folder,
            name=self.name,
            fraction=0.5,
        )

    def plot_1d_dists(
        self, nature_params_c: Optional[NDArray[np.float64]] = None, nrows: int = 2
    ) -> None:
        """Plot prior and posterior, in constrained and unconstrained space."""

        E = self.E
        E_new = self.analysis()
        E_c = self.to_constrained(self.E)
        E_new_c = self.to_constrained(E_new)

        if nature_params_c is None:
            nature_params_c = PARAMS.defaults
        nature_params = self.to_unconstrained(nature_params_c)

        kwargs = {
            "do_kde": True,
            "do_samples": True,
            "nature_params_label": "Nature",
            "colors": None,
            "alphas": None,
            "linestyles": None,
            "legend_title": None,
            "nrows": nrows,
        }
        save_file = f"{self.plot_folder}/fig-dist1d_{self.name}"

        plot_samples_1d(
            [E, E_new],
            ["Prior", "Posterior"],
            nature_params=nature_params,
            save_file=save_file + "_unconstrained",
            **kwargs,
        )
        plot_samples_1d(
            [E_c, E_new_c],
            ["Prior", "Posterior"],
            nature_params=nature_params_c,
            save_file=save_file + "_constrained",
            lims="constrained",
            **kwargs,
        )


def corr_matrix(
    E: NDArray[np.float64],
    lower_triag: bool = False,
) -> NDArray[np.float64]:
    """Compute the correlation matrix of the ensemble E."""
    N, Nx = E.shape  # Dimensionality

    mu = np.mean(E, 0)  # Ens mean
    A = E - mu  # Ens anomalies

    Std = np.std(E, axis=0)
    Std = np.expand_dims(Std, axis=0)

    # covariance
    Cov = A.T @ A / (N - 1)

    # correlations
    Corr = Cov / Std / Std.T

    # blank out upper triangle for plotting
    if lower_triag:
        for i in range(1, Nx):
            Corr[i, :i] = np.nan

    return Corr


def change_rate_units(
    Eo: Optional[NDArray[np.float64]] = None,
    y: Optional[NDArray[np.float64]] = None,
    hnoise: Optional[GaussRV] = None,
) -> Tuple[
    Optional[NDArray[np.float64]],
    Optional[NDArray[np.float64]],
    Optional[GaussRV],
]:
    """Convert rate observations from m/s to m/h."""
    obs_names = get_obs_names_with_height()
    idcs = [
        obs_names.index("cloud base rate"),
        obs_names.index("cloud top rate"),
    ]
    if Eo is not None:
        Eo[:, idcs] *= 3600.0

    if y is not None:
        y[idcs] *= 3600.0

    if hnoise is not None:
        diag = hnoise.C.diag.copy()
        diag[idcs] *= 3600.0**2
        hnoise = GaussRV(hnoise.mean.copy(), diag * np.eye(len(diag)))

    return Eo, y, hnoise
