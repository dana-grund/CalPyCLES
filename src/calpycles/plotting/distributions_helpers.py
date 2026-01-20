"""Plotting functionality for two-dinemsional (p1 vs. p2) distribution plots."""

# Standard library
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# Third-party
import numpy as np
from numpy.typing import NDArray

# First-party
from calpycles.random_variables import GaussRV

# -------- Allow for crappy plotting code  -------- #
# pylint: skip-file
# mypy: ignore-errors
# noqa


# -------- Helpers -------- #


def prepare(
    dists: Union[GaussRV, List[GaussRV]],
    dist_labels: Union[str, List[str]],
    samples: Optional[Union[NDArray[np.float64], List[NDArray[np.float64]]]] = None,
    true_params: Optional[NDArray[np.float64]] = None,
    params_to_plot: Optional[List[int]] = None,
    name: str = "",
    dist_space: str = "unconstrained",
    do_samples: Union[bool, List[bool]] = True,
) -> Any:
    """Prepare the inputs for 1d and 2d plotting."""
    # handle prior-only case
    if isinstance(dists, GaussRV):
        dists = [dists]
    if isinstance(dist_labels, str):
        dist_labels = [dist_labels]
    if isinstance(samples, np.ndarray):
        samples = [samples]

    # ensure same lengths
    assert len(dist_labels) == len(dists)
    if samples is None:
        samples = []
        do_samples = False
    if len(samples) < len(dists):
        samples += [None] * (len(dists) - len(samples))
    if isinstance(do_samples, bool):
        do_samples = [do_samples] * len(dists)

    add_units = dist_space == "constrained"
    if dist_space == "constrained":
        name += "_constrained"

    # select parameters to plot
    dist_prior = dists[0]
    n_params = dist_prior.M
    if params_to_plot is not None:
        n_plots = len(params_to_plot)
    else:
        n_plots = n_params
        params_to_plot = list(range(n_plots))

    # make sure all quantities are in the right space
    true_params_ = true_params
    samples_ = samples
    dist_prior = dists[0]
    if dist_prior.dist_type == "uniform":
        if dist_space == "unconstrained":
            lims = [(-3, 3) for _ in range(n_params)]

            # samples are passed in constrained space, so convert
            if true_params is not None:
                true_params_ = dist_prior.to_unconstrained(true_params)

            samples_ = []
            for ss in samples:
                if ss is not None:
                    samples_.append(dist_prior.to_unconstrained(ss))
                else:
                    samples_.append(None)

        elif dist_space == "constrained":
            lims = []
            for i in range(n_params):
                lower, upper = dists[0].lower[i], dists[0].upper[i]
                abit = 0.1 * (upper - lower)
                lims.append([lower - abit, upper + abit])

    return (
        dists,
        dist_labels,
        lims,
        add_units,
        samples_,
        true_params_,
        params_to_plot,
        do_samples,
    )
