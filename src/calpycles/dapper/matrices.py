"""Covariance matrix tools."""

# Third-party
import numpy as np
import scipy.linalg as sla
from numpy import ones
from numpy import sqrt
from numpy import zeros

# First-party
from calpycles.dapper.linalg import svd0
from calpycles.dapper.linalg import truncate_rank
from calpycles.helpers import crop_nans

# -------- Dont check this imported code  -------- #
# pylint: disable=invalid-name
# pylint: skip-file
# mypy: ignore-errors
# noqa


class lazy_property:
    """Lazy evaluation of property.

    Should represent non-mutable data,
    as it replaces itself.

    From https://stackoverflow.com/q/3012421
    """

    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value


class CovMat:
    """Covariance matrix class.

    Main tasks:

    - Unify the covariance representations: full, diagonal, reduced-rank sqrt.
    - Streamline init. and printing.
    - Convenience transformations with caching/memoization.
      This (hiding it internally) would be particularly useful
      if the covariance matrix changes with time (but repeat).
    """

    ##################################
    # Init
    ##################################
    def __init__(self, data, kind="full_or_diag", trunc=1.0):
        """Construct object.

        The covariance (say P) can be input (specified in the following ways):

            kind    | data
            --------|-------------
            'full'  | full M-by-M array (P)
            'diag'  | diagonal of P (assumed diagonal)
            'E'     | ensemble (N-by-M) with sample cov P
            'A'     | as 'E', but pre-centred by mean(E,axis=0)
            'Right' | any R such that P = R.T@R (e.g. weighted form of 'A')
            'Left'  | any L such that P = L@L.T
        """
        # Cascade if's down to 'Right'
        if kind == "E":
            data = crop_nans(data)
            mu = np.mean(data, 0)
            data = data - mu
            kind = "A"
        if kind == "A":
            N = len(data)
            data = data / sqrt(N - 1)
            kind = "Right"
        if kind == "Left":
            data = data.T
            kind = "Right"
        if kind == "Right":
            # If a cholesky factor has been input, we will not
            # automatically go for the EVD, seeing as e.g. the
            # diagonal can be computed without it.
            R = np.atleast_2d(data)
            assert R.ndim == 2
            self._R = R
            self._m = R.shape[1]
        else:
            if kind == "full_or_diag":
                data = np.atleast_1d(data)
                if data.ndim == 1 and len(data) > 1:
                    kind = "diag"
                else:
                    kind = "full"
            if kind == "full":
                # If full has been input, then we have memory for an EVD,
                # which will probably be put to use in the DA.
                C = np.atleast_2d(data)
                assert C.ndim == 2
                self._C = C
                M = len(C)
                d, V = sla.eigh(C)
                d = CovMat._clip(d)
                rk = (d > 0).sum()
                d = d[-rk:][::-1]
                V = (V.T[-rk:][::-1]).T
                self._assign_EVD(M, rk, d, V)
            elif kind == "diag":
                # With diagonal input, it would be great to use a sparse
                # (or non-existent) representation of V,
                # but that would require so much other adaption of other code.
                d = np.atleast_1d(data)
                assert d.ndim == 1
                self.diag = d
                M = len(d)
                if np.all(d == d[0]):
                    V = np.eye(M)
                    rk = M
                else:
                    # Clip and sort diag
                    d = CovMat._clip(d)
                    idx = np.argsort(d)[::-1]
                    rk = (d > 0).sum()
                    # Sort d
                    d = d[idx][:rk]
                    # Make rectangular V that un-sorts d
                    V = zeros((M, rk))
                    V[idx[:rk], np.arange(rk)] = 1
                self._assign_EVD(M, rk, d, V)
            else:
                raise KeyError

        self._kind = kind
        self._trunc = trunc

    ##################################
    # Protected
    ##################################

    @property
    def M(self):
        """`ndims`"""
        return self._m

    @property
    def kind(self):
        """Form in which matrix was specified."""
        return self._kind

    @property
    def trunc(self):
        """Truncation threshold."""
        return self._trunc

    ##################################
    # "Non-EVD" stuff
    ##################################
    @property
    def full(self):
        """Full covariance matrix"""
        if hasattr(self, "_C"):
            return self._C
        else:
            C = self.Left @ self.Left.T
        self._C = C
        return C

    @lazy_property
    def diag(self):
        """Diagonal of covariance matrix"""
        if hasattr(self, "_C"):
            return np.diag(self._C)
        else:
            return (self.Left**2).sum(axis=1)

    @property
    def Left(self):
        """Left sqrt.

        `L` such that $$ C = L L^T .$$

        Note that `L` is typically rectangular, but not triangular,
        and that its width is somewhere between the rank and `M`.
        """
        if hasattr(self, "_R"):
            return self._R.T
        else:
            return self.V * sqrt(self.ews)

    @property
    def Right(self):
        """Right sqrt. Ref `CovMat.Left`."""
        if hasattr(self, "_R"):
            return self._R
        else:
            return self.Left.T

    ##################################
    # EVD stuff
    ##################################
    def _assign_EVD(self, M, rk, d, V):
        self._m = M
        self._d = d
        self._V = V
        self._rk = rk

    @staticmethod
    def _clip(d):
        return np.where(d < 1e-8 * d.max(), 0, d)

    def _do_EVD(self):
        if not self.has_done_EVD():
            V, s, UT = svd0(self._R)
            M = UT.shape[1]
            d = s**2
            d = CovMat._clip(d)
            rk = (d > 0).sum()
            d = d[:rk]
            V = UT[:rk].T
            self._assign_EVD(M, rk, d, V)

    def has_done_EVD(self):
        """Whether or not eigenvalue decomposition has been done for matrix."""
        return all(key in vars(self) for key in ["_V", "_d", "_rk"])

    @property
    def ews(self):
        """Eigenvalues. Only outputs the positive values (i.e. len(ews)==rk)."""
        self._do_EVD()
        return self._d

    @property
    def V(self):
        """Eigenvectors, output corresponding to ews."""
        self._do_EVD()
        return self._V

    @property
    def rk(self):
        """Rank, i.e. the number of positive eigenvalues."""
        self._do_EVD()
        return self._rk

    ##################################
    # transform_by properties
    ##################################
    def transform_by(self, fun):
        """Generalize scalar functions to covariance matrices (via Taylor expansion)."""
        r = truncate_rank(self.ews, self.trunc, True)
        V = self.V[:, :r]
        w = self.ews[:r]

        return (V * fun(w)) @ V.T

    @lazy_property
    def sym_sqrt(self):
        """S such that C = S@S (and i.e. S is square). Uses trunc-level."""
        return self.transform_by(sqrt)

    @lazy_property
    def sym_sqrt_inv(self):
        """S such that C^{-1} = S@S (and i.e. S is square). Uses trunc-level."""
        return self.transform_by(lambda x: 1 / sqrt(x))

    @lazy_property
    def pinv(self):
        """Pseudo-inverse. Uses trunc-level."""
        return self.transform_by(lambda x: 1 / x)

    @lazy_property
    def inv(self):
        if self.M != self.rk:
            raise RuntimeError(
                "Matrix is rank deficient, "
                "and cannot be inverted. "
                "Use .tinv() instead?"
            )
        # Temporarily remove any truncation
        tmp = self.trunc
        self._trunc = 1.0
        # Compute and restore truncation level
        Inv = self.pinv
        self._trunc = tmp
        return Inv

    ##################################
    # __repr__
    ##################################
    def __repr__(self):
        s = "\n    M: " + str(self.M)
        s += "\n kind: " + repr(self.kind)
        s += "\ntrunc: " + str(self.trunc)

        # Rank
        s += "\n   rk: "
        if self.has_done_EVD():
            s += str(self.rk)
        else:
            s += "<=" + str(self.Right.shape[0])

        # Full (as affordable)
        s += "\n full:"
        if hasattr(self, "_C") or np.get_printoptions()["threshold"] > self.M**2:
            # We can afford to compute full matrix
            t = "\n" + str(self.full)
        else:
            # Only compute corners of full matrix
            K = np.get_printoptions()["edgeitems"]
            s += " (only computing/printing corners)"
            if hasattr(self, "_R"):
                U = self.Left[:K, :]  # Upper
                L = self.Left[-K:, :]  # Lower
            else:
                U = self.V[:K, :] * sqrt(self.ews)
                L = self.V[-K:, :] * sqrt(self.ews)

            # Corners
            NW = U @ U.T
            NE = U @ L.T
            SW = L @ U.T
            SE = L @ L.T

            # Concatenate corners. Fill "cross" between them with nan's
            N = np.hstack([NW, np.nan * ones((K, 1)), NE])
            S = np.hstack([SW, np.nan * ones((K, 1)), SE])
            All = np.vstack([N, np.nan * ones(2 * K + 1), S])

            with np.printoptions(threshold=0):
                t = "\n" + str(All)

        # Indent all of cov array, and add to s
        s += t.replace("\n", "\n   ")

        # Add diag. Indent array +1 vs cov array
        with np.printoptions(threshold=0):
            s += "\n diag:\n   " + " " + str(self.diag)

        s = "<" + type(self).__name__ + ">" + s.replace("\n", "\n  ")
        return s


# Note: The diagonal representation is NOT memory-efficient.
#
# But there's no simple way of making so, especially since the sparse class
# (which would hold the eigenvectors) is a subclass of the matrix class,
# which interprets * as @, and so, when using this class,
# one would have to be always careful about it
#
# One could try to overload +-@/ (for CovMat),
# but unfortunately there's no right/post-application version of @ and /
# (indeed, how could there be for binary operators?)
# which makes this less interesting.
# Hopefully this restriction is not an issue,
# as diagonal matrices are mainly used for observation error covariance,
# which are usually not infeasibly large.
#
# Another potential solution is to subclass the sparse matrix,
# and revert its operator definitions to that of ndarray.
# and use it for the V (eigenvector) matrix that gets output
# by various fields of CovMat.
