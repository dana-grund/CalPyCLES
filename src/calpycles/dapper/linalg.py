"""Linear algebra."""

# Third-party
import numpy as np
import scipy.linalg as sla

# -------- Dont check this imported code  -------- #
# pylint: disable=invalid-name
# pylint: skip-file
# mypy: ignore-errors
# noqa


def mrdiv(b, A):
    """b/A."""
    return sla.solve(A.T, b.T).T


def truncate_rank(s, threshold, avoid_pathological):
    """Find `r` such that `s[:r]` contains the threshold proportion of `s`."""
    assert isinstance(threshold, float)
    if threshold == 1.0:
        r = len(s)
    elif threshold < 1.0:
        r = np.sum(np.cumsum(s) / np.sum(s) < threshold)
        r += 1  # Hence the strict inequality above
        if avoid_pathological:
            # If not avoid_pathological, then the last 4 diag. entries of
            # svdi( *tsvd(np.eye(400),0.99) )
            # will be zero. This is probably not intended.
            r += np.sum(np.isclose(s[r - 1], s[r:]))
    else:
        raise ValueError
    return r


def svd0(A):
    """Similar to Matlab's `svd(A,0)`.

    Compute the

    - full    svd if `nrows > ncols`
    - reduced svd otherwise.

    As in Matlab: `svd(A,0)`,
    except that the input and output are transposed, in keeping with DAPPER convention.
    It contrasts with `scipy.linalg.svd(full_matrice=False)`
    and Matlab's `svd(A,'econ')`, both of which always compute the reduced svd.


    See Also
    --------
    tsvd : rank (and threshold) truncation.
    """
    M, N = A.shape
    if M > N:
        return sla.svd(A, full_matrices=True)
    return sla.svd(A, full_matrices=False)


def pad0(x, N):
    """Pad `x` with zeros so that `len(x)==N`."""
    out = np.zeros(N)
    out[: len(x)] = x
    return out
