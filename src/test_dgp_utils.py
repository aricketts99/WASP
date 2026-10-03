#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:50:56 2026

@author: andrew
"""


"""
Basic tests for all data-generating processes in src.utils.

Tests:
    - output types
    - output shapes
    - finite values
    - number of active coefficients
    - mixture assignments
    - reproducibility with a fixed seed
    - basic parameter validation

This is intended as a simple sanity-check script rather than
a formal pytest test suite.
"""

import numpy as np

from src.utils import (
    iid_gaussian,
    ar1,
    equicorrelated,
    block_correlated,
    decoy,
    latent_factor,
    heavy_tailed,
    ar1_heavy_tailed,
    mixture_disjoint,
    mixture_overlapping,
    mixture_disjoint_unbalanced,
    mixture_overlapping_unbalanced,
)


# ================================================================
# Global test settings
# ================================================================

N = 100
P = 200
ACTIVE_P = 10
N_SUBMODELS = 3
SEED = 12345

SNR = 2.0
SIGMA2 = 1.0


# ================================================================
# Helper functions
# ================================================================

def check_array(
    name,
    x,
    expected_shape,
    expected_dtype=None,
):
    """Check basic properties of an ndarray."""

    assert isinstance(x, np.ndarray), (
        f"{name}: expected np.ndarray, "
        f"got {type(x)}"
    )

    assert x.shape == expected_shape, (
        f"{name}: expected shape {expected_shape}, "
        f"got {x.shape}"
    )

    assert np.all(np.isfinite(x)), (
        f"{name}: contains non-finite values"
    )

    if expected_dtype is not None:
        assert x.dtype == expected_dtype, (
            f"{name}: expected dtype {expected_dtype}, "
            f"got {x.dtype}"
        )


def check_standard_output(
    beta,
    y,
    X,
    n,
    p,
    active_p,
):
    """Check output from an ordinary DGP."""

    # ------------------------------------------------------------
    # Types
    # ------------------------------------------------------------

    check_array(
        "beta",
        beta,
        (p, 1),
    )

    check_array(
        "y",
        y,
        (n, 1),
    )

    check_array(
        "X",
        X,
        (n, p),
    )

    # ------------------------------------------------------------
    # Shapes
    # ------------------------------------------------------------

    assert beta.shape == (p, 1)
    assert y.shape == (n, 1)
    assert X.shape == (n, p)

    # ------------------------------------------------------------
    # Number of active coefficients
    # ------------------------------------------------------------

    n_active = np.count_nonzero(beta)

    assert n_active == active_p, (
        f"Expected {active_p} active coefficients, "
        f"got {n_active}"
    )

    # ------------------------------------------------------------
    # Active variables should be the first active_p
    # ------------------------------------------------------------

    assert np.all(beta[:active_p] != 0), (
        "First active_p coefficients are not all non-zero."
    )

    if active_p < p:
        assert np.all(beta[active_p:] == 0), (
            "Coefficients after active_p are not all zero."
        )


def check_mixture_output(
    beta,
    y,
    X,
    z,
    n,
    p,
    active_p,
    n_submodels,
):
    """Check output from a mixture DGP."""

    # ------------------------------------------------------------
    # Types
    # ------------------------------------------------------------

    check_array(
        "beta",
        beta,
        (n_submodels, p),
    )

    check_array(
        "y",
        y,
        (n, 1),
    )

    check_array(
        "X",
        X,
        (n, p),
    )

    assert isinstance(z, np.ndarray), (
        f"z: expected np.ndarray, got {type(z)}"
    )

    assert z.shape == (n,), (
        f"z: expected shape {(n,)}, got {z.shape}"
    )

    # z is integer-valued
    assert np.issubdtype(z.dtype, np.integer), (
        f"z: expected integer dtype, got {z.dtype}"
    )

    # ------------------------------------------------------------
    # Finite values
    # ------------------------------------------------------------

    assert np.all(np.isfinite(z)), (
        "z contains non-finite values."
    )

    # ------------------------------------------------------------
    # Valid cluster labels
    # ------------------------------------------------------------

    unique_z = np.unique(z)

    assert np.all(
        (unique_z >= 0)
        & (unique_z < n_submodels)
    ), (
        f"Invalid submodel labels: {unique_z}"
    )

    # ------------------------------------------------------------
    # Every submodel should actually occur
    # ------------------------------------------------------------

    for k in range(n_submodels):

        count = np.sum(z == k)

        assert count > 0, (
            f"Submodel {k} has no observations."
        )

    # ------------------------------------------------------------
    # Each submodel should have active_p coefficients
    # ------------------------------------------------------------

    for k in range(n_submodels):

        n_active = np.count_nonzero(beta[k])

        assert n_active == active_p, (
            f"Submodel {k}: expected {active_p} "
            f"active coefficients, got {n_active}"
        )


def print_standard_result(
    name,
    beta,
    y,
    X,
):
    """Pretty-print standard DGP result."""

    print(f"\n{name}")
    print("-" * len(name))

    print(f"beta : shape={beta.shape}, dtype={beta.dtype}")
    print(f"y    : shape={y.shape}, dtype={y.dtype}")
    print(f"X    : shape={X.shape}, dtype={X.dtype}")

    print(
        f"active coefficients: "
        f"{np.count_nonzero(beta)}"
    )

    print("PASS")


def print_mixture_result(
    name,
    beta,
    y,
    X,
    z,
):
    """Pretty-print mixture DGP result."""

    print(f"\n{name}")
    print("-" * len(name))

    print(f"beta : shape={beta.shape}, dtype={beta.dtype}")
    print(f"y    : shape={y.shape}, dtype={y.dtype}")
    print(f"X    : shape={X.shape}, dtype={X.dtype}")
    print(f"z    : shape={z.shape}, dtype={z.dtype}")

    print(
        "observations per submodel:",
        np.bincount(z),
    )

    print("active coefficients per submodel:")

    for k in range(beta.shape[0]):
        print(
            f"    model {k}: "
            f"{np.count_nonzero(beta[k])}"
        )

    print("PASS")


# ================================================================
# 1. IID Gaussian
# ================================================================

print("\n" + "=" * 70)
print("STANDARD DGPs")
print("=" * 70)

beta, y, X = iid_gaussian(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

check_standard_output(
    beta,
    y,
    X,
    N,
    P,
    ACTIVE_P,
)

print_standard_result(
    "iid_gaussian",
    beta,
    y,
    X,
)


# ================================================================
# 2. AR(1)
# ================================================================

beta, y, X = ar1(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    rho=0.5,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

check_standard_output(
    beta,
    y,
    X,
    N,
    P,
    ACTIVE_P,
)

print_standard_result(
    "ar1",
    beta,
    y,
    X,
)


# ================================================================
# 3. Equicorrelated
# ================================================================

beta, y, X = equicorrelated(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    rho=0.5,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

check_standard_output(
    beta,
    y,
    X,
    N,
    P,
    ACTIVE_P,
)

print_standard_result(
    "equicorrelated",
    beta,
    y,
    X,
)


# ================================================================
# 4. Block correlated
# ================================================================

beta, y, X = block_correlated(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    rho=0.5,
    block_size=20,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

check_standard_output(
    beta,
    y,
    X,
    N,
    P,
    ACTIVE_P,
)

print_standard_result(
    "block_correlated",
    beta,
    y,
    X,
)


# ================================================================
# 5. Decoy
# ================================================================

beta, y, X = decoy(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    rho=0.9,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

check_standard_output(
    beta,
    y,
    X,
    N,
    P,
    ACTIVE_P,
)

print_standard_result(
    "decoy",
    beta,
    y,
    X,
)


# ================================================================
# 6. Latent factor
# ================================================================

beta, y, X = latent_factor(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    rho=0.8,
    n_factors=5,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

check_standard_output(
    beta,
    y,
    X,
    N,
    P,
    ACTIVE_P,
)

print_standard_result(
    "latent_factor",
    beta,
    y,
    X,
)


# ================================================================
# 7. Heavy-tailed errors
# ================================================================

beta, y, X = heavy_tailed(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    nu=3,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

check_standard_output(
    beta,
    y,
    X,
    N,
    P,
    ACTIVE_P,
)

print_standard_result(
    "heavy_tailed",
    beta,
    y,
    X,
)


# ================================================================
# 8. AR(1) + heavy-tailed errors
# ================================================================

beta, y, X = ar1_heavy_tailed(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    rho=0.5,
    nu=3,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

check_standard_output(
    beta,
    y,
    X,
    N,
    P,
    ACTIVE_P,
)

print_standard_result(
    "ar1_heavy_tailed",
    beta,
    y,
    X,
)


# ================================================================
# MIXTURE DGPs
# ================================================================

print("\n" + "=" * 70)
print("MIXTURE DGPs")
print("=" * 70)


# ================================================================
# 9. Disjoint balanced
# ================================================================

beta, y, X, z = mixture_disjoint(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    n_submodels=N_SUBMODELS,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

check_mixture_output(
    beta,
    y,
    X,
    z,
    N,
    P,
    ACTIVE_P,
    N_SUBMODELS,
)

print_mixture_result(
    "mixture_disjoint",
    beta,
    y,
    X,
    z,
)


# ================================================================
# 10. Overlapping balanced
# ================================================================

beta, y, X, z = mixture_overlapping(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    n_submodels=N_SUBMODELS,
    overlap=0.5,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

check_mixture_output(
    beta,
    y,
    X,
    z,
    N,
    P,
    ACTIVE_P,
    N_SUBMODELS,
)

print_mixture_result(
    "mixture_overlapping",
    beta,
    y,
    X,
    z,
)


# ================================================================
# 11. Disjoint unbalanced
# ================================================================

weights = [0.6, 0.3, 0.1]

beta, y, X, z = mixture_disjoint_unbalanced(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    n_submodels=N_SUBMODELS,
    weights=weights,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

check_mixture_output(
    beta,
    y,
    X,
    z,
    N,
    P,
    ACTIVE_P,
    N_SUBMODELS,
)

print_mixture_result(
    "mixture_disjoint_unbalanced",
    beta,
    y,
    X,
    z,
)


# ================================================================
# 12. Overlapping unbalanced
# ================================================================

beta, y, X, z = mixture_overlapping_unbalanced(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    n_submodels=N_SUBMODELS,
    overlap=0.5,
    weights=weights,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

check_mixture_output(
    beta,
    y,
    X,
    z,
    N,
    P,
    ACTIVE_P,
    N_SUBMODELS,
)

print_mixture_result(
    "mixture_overlapping_unbalanced",
    beta,
    y,
    X,
    z,
)


# ================================================================
# Reproducibility tests
# ================================================================

print("\n" + "=" * 70)
print("REPRODUCIBILITY TESTS")
print("=" * 70)


# Standard DGP
beta1, y1, X1 = ar1(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    rho=0.5,
    SNR=SNR,
    seed=SEED,
)

beta2, y2, X2 = ar1(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    rho=0.5,
    SNR=SNR,
    seed=SEED,
)

assert np.array_equal(beta1, beta2)
assert np.array_equal(y1, y2)
assert np.array_equal(X1, X2)

print("ar1 reproducibility: PASS")


# Mixture DGP
beta1, y1, X1, z1 = mixture_disjoint(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    n_submodels=N_SUBMODELS,
    seed=SEED,
)

beta2, y2, X2, z2 = mixture_disjoint(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    n_submodels=N_SUBMODELS,
    seed=SEED,
)

assert np.array_equal(beta1, beta2)
assert np.array_equal(y1, y2)
assert np.array_equal(X1, X2)
assert np.array_equal(z1, z2)

print("mixture_disjoint reproducibility: PASS")


# ================================================================
# Check that changing the seed changes the data
# ================================================================

print("\n" + "=" * 70)
print("SEED DIFFERENCE TEST")
print("=" * 70)

_, y1, X1 = iid_gaussian(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    seed=1,
)

_, y2, X2 = iid_gaussian(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    seed=2,
)

assert not np.array_equal(X1, X2)
assert not np.array_equal(y1, y2)

print("different seeds produce different data: PASS")


# ================================================================
# Check coefficient scaling
# ================================================================

print("\n" + "=" * 70)
print("COEFFICIENT SCALING TEST")
print("=" * 70)

beta, _, _ = iid_gaussian(
    n=N,
    p=P,
    active_p=ACTIVE_P,
    SNR=SNR,
    sigma2=SIGMA2,
    seed=SEED,
)

expected_value = (
    SNR
    * np.sqrt(
        SIGMA2 * np.log(P) / N
    )
)

assert np.allclose(
    beta[:ACTIVE_P, 0],
    expected_value,
)

assert np.all(
    beta[ACTIVE_P:, 0] == 0
)

print(
    "expected active coefficient:",
    expected_value,
)

print(
    "actual active coefficient:",
    beta[0, 0],
)

print("coefficient scaling: PASS")


# ================================================================
# Final result
# ================================================================

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)