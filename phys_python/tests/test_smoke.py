# -*- coding: utf-8 -*-
"""
Smoke tests for phys_python.  Run before committing a change to the package:

    python tests/test_smoke.py          # no pytest needed
    pytest tests/test_smoke.py          # also works if pytest is installed

They are deliberately cheap (a couple of seconds) and pin down the *contracts*
that project code relies on -- conventions, not implementation details -- so
that a refactor which silently changes one of them is caught here instead of
six months later in an old script.  The reference numbers were produced by the
code itself and independently cross-checked; if a test starts failing, decide
whether the change was intended before touching the number.
"""
import io
import contextlib

import numpy as np
from scipy import linalg as alg

from eig.decomp import decomp_schur_, sort_ortho
from hamiltonian.ferm_tool import Ferm_hamiltonian
from entangle.ent_ferm import GetEntFerm


@contextlib.contextmanager
def quiet():
    """The library modules print progress; keep the test output readable."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def h_xx(N, P='OBC', J=1.0):
    """XX / free-fermion hopping matrix, N x N.  Kept local on purpose: the
    package test must not depend on any project folder."""
    h = np.diag(np.tile(-J/2, N-1), k=1)
    if P == '+':
        h[N-1, 0] = -J/2
    elif P == '-':
        h[N-1, 0] = J/2
    return h + h.T


def test_sort_ortho():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((10, 10)) + 1j*rng.standard_normal((10, 10))
    h = A + A.conj().T
    with quiet():
        eigval, V = sort_ortho(h)

    assert abs(V.conj().T @ V - np.eye(10)).max() < 1E-10, "not orthonormal"
    assert abs(V.conj().T @ h @ V - np.diag(eigval)).max() < 1E-10, "not diagonalized"
    assert np.all(np.diff(eigval) > -1E-12), "eigenvalues not ascending"


def test_decomp_schur_convention():
    """decomp_schur_ returns K = Q.T @ T @ Q  (note: NOT Q @ T @ Q.T).
    Project code builds covariance matrices as Q.T @ Gamma_mode @ Q, so this
    convention must not drift."""
    rng = np.random.default_rng(1234)
    M = rng.standard_normal((8, 8))
    K = M - M.T
    with quiet():
        Q, T, Lam = decomp_schur_(K, is_reverse=1, is_sort=-1)

    assert abs(K - Q.T @ T @ Q).max() < 1E-10, "K = Q.T T Q broken"
    assert abs(Q @ Q.T - np.eye(8)).max() < 1E-10, "Q not orthogonal"

    # T is block diagonal with 2x2 blocks [0,-lam; lam,0] for is_reverse=1
    off = T.copy()
    for i in range(4):
        off[2*i:2*i+2, 2*i:2*i+2] = 0
    assert abs(off).max() < 1E-10, "T not block diagonal"
    assert np.all(np.diag(T, 1)[::2] < 0), "is_reverse=1 sign convention broken"

    lam = Lam[::2]
    assert np.all(lam >= 0) and np.all(np.diff(lam) > -1E-12), \
        "is_sort=-1 should give non-negative, ascending Lambda"
    assert abs(Lam[::2] - Lam[1::2]).max() < 1E-12, "Lambda not pairwise repeated"
    assert abs(np.sort(lam) - np.sort(abs(alg.eigvals(K).imag))[::2]).max() < 1E-10, \
        "Lambda are not the |eigenvalues| of K"


def test_xx_obc_tower():
    """OBC XX chain: the c=1 tower, and the U(1) charge label combine[:,2].
    First gap normalized to 1/2 must give 0, 1/2, 1/2, 1 -- the two states at
    1/2 are the m = -1 and m = +1 primaries."""
    N = 20
    with quiet():
        Hp = Ferm_hamiltonian(H=h_xx(N), P='OBC')
        combine = Hp.many_eig(cutoff=16, is_fold=0)[2]

    E = combine[:4, 0]
    E_scl = (E - E[0])/(E[1] - E[0])*0.5
    assert abs(E_scl - np.array([0, 0.5, 0.5, 1.0])).max() < 1E-10, \
        "OBC tower is not 0, 1/2, 1/2, 1"
    assert abs(combine[:4, 2] - np.array([0, -1, 1, 0])).max() < 1E-10, \
        "charge labels of the first four states changed"
    assert abs(E[0] - (-6.190744999827)) < 1E-10, "ground-state energy changed"


def test_xx_pbc_momenta():
    """The P in ['+','-'] branch goes through simult_diag: check it returns the
    free-fermion band and integer crystal momenta."""
    N = 12
    with quiet():
        Hk = Ferm_hamiltonian(H=h_xx(N, P='+'), P='+')
        s_eig = Hk.many_eig(cutoff=16, is_fold=0)[0]

    band = -np.cos(2*np.pi*np.arange(N)/N)
    assert abs(np.sort(s_eig) - np.sort(band)).max() < 1E-10, "band is not -cos(k)"
    P_eig = np.sort(Hk.P_eig)
    assert abs(P_eig - np.round(P_eig)).max() < 1E-10, "momenta are not integers"


def test_xx_ground_state_entropy():
    """Half-chain von Neumann entropy of the XX ground state, from the
    covariance matrix produced by Ferm_hamiltonian.get_gamma."""
    N = 20
    occ = np.zeros(N)
    occ[:N//2] = 1
    with quiet():
        Hp = Ferm_hamiltonian(H=h_xx(N), P='OBC')
        Hp.many_eig(cutoff=4, is_fold=0)          # fills r_eigvec / l_eigvec
        Corr, Gamma = Hp.get_gamma(occ)
        S = GetEntFerm(GammaR=Gamma[:N, :N], ent_type=[1]).S

    assert abs(S.real - 1.458553440961) < 1E-8, "half-chain entropy changed"
    assert abs(S.imag) < 1E-10, "entropy should be real"
    assert abs(np.trace(Corr).real - N/2) < 1E-10, "half filling violated"


if __name__ == "__main__":
    import sys
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as err:
            n_fail += 1
            print("  FAIL  %s : %s" % (name, err))
        except Exception as err:
            n_fail += 1
            print("  ERROR %s : %s: %s" % (name, type(err).__name__, err))
    print("\n%d/%d passed" % (len(tests) - n_fail, len(tests)))
    sys.exit(1 if n_fail else 0)
