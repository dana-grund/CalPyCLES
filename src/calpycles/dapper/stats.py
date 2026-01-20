# Third-party
import numpy as np

# -------- Dont check this imported code  -------- #
# pylint: disable=invalid-name
# pylint: skip-file
# mypy: ignore-errors
# noqa


def center(E, axis=0, rescale=False):
    r"""Center ensemble.

    Makes use of `np` features: keepdims and broadcasting.

    Parameters
    ----------
    E: ndarray
        Ensemble which going to be inflated

    axis: int, optional
        The axis to be centered. Default: 0

    rescale: bool, optional
        If True, inflate to compensate for reduction in the expected variance.
        The inflation factor is \(\sqrt{\frac{N}{N - 1}}\)
        where N is the ensemble size. Default: False

    Returns
    -------
    X: ndarray
        Ensemble anomaly

    x: ndarray
        Mean of the ensemble
    """
    x = np.mean(E, axis=axis, keepdims=True)
    X = E - x

    if rescale:
        N = E.shape[axis]
        X *= np.sqrt(N / (N - 1))

    x = x.squeeze(axis=axis)

    return X, x


def mean0(E, axis=0, rescale=True):
    """Like `center`, but only return the anomalies (not the mean).

    Uses `rescale=True` by default, which is beneficial
    when used to center observation perturbations.
    """
    return center(E, axis=axis, rescale=rescale)[0]
