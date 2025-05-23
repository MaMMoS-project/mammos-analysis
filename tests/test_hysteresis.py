"""Tests for hysteresis functions."""

from __future__ import annotations

import numpy as np
import pytest

import mammos_entity as me
import mammos_units as u
from mammos_analysis.hysteresis import (
    extract_coercive_field,
    extract_remanent_magnetization,
    _check_monotonicity,
    extract_B_curve,
    extract_maximum_energy_product,
    extrinsic_properties,
)


def linear_hysteresis_data(m, b):
    """Generate linear hysteresis data for testing.

    Args:
        m: Slope of the linear hysteresis.
        b: Intercept of the linear hysteresis.

    Returns:
        H: External magnetic field.
        M: Spontaneous magnetisation.
        expected: Expected values for coercive field and remanence.
    """
    # Create a simple linear hysteresis with known intercepts
    h_values = np.linspace(-100, 100, 101)
    m_values = m * h_values + b

    H = me.H(h_values * u.A / u.m)
    M = me.Ms(m_values * u.A / u.m)

    # Expected values for testing
    expected = {
        "Mr": abs(b),  # y-intercept
        "Hc": abs(
            np.divide(-b, m, where=m != 0, out=np.zeros_like(m, dtype=np.float64))
        ),  # x-intercept
    }

    return H, M, expected


def test_check_monotonicity():
    """Test the check_monotonicity function."""
    # Test with a monotonic increasing array
    arr = np.array([1, 2, 3, 4, 5])
    _check_monotonicity(arr)
    _check_monotonicity(arr, direction="increasing")
    with pytest.raises(ValueError, match="Array is not monotonically decreasing."):
        _check_monotonicity(arr, direction="decreasing")

    # Test with a monotonic decreasing array
    arr = np.array([5, 4, 3, 2, 1])
    _check_monotonicity(arr)
    _check_monotonicity(arr, direction="decreasing")
    with pytest.raises(ValueError, match="Array is not monotonically increasing."):
        _check_monotonicity(arr, direction="increasing")

    # Test with a non-monotonic array
    arr = np.array([1, 2, 3, 2, 5])
    with pytest.raises(ValueError, match="Array is not monotonic."):
        _check_monotonicity(arr)
    with pytest.raises(ValueError, match="Array is not monotonically increasing."):
        _check_monotonicity(arr, direction="increasing")
    with pytest.raises(ValueError, match="Array is not monotonically decreasing."):
        _check_monotonicity(arr, direction="decreasing")

    # Test with constant array (should pass as monotonic)
    arr = np.array([3, 3, 3, 3])
    _check_monotonicity(arr)
    _check_monotonicity(arr, direction="increasing")
    _check_monotonicity(arr, direction="decreasing")

    # Test with single element array (should pass as monotonic)
    arr = np.array([42])
    _check_monotonicity(arr)
    _check_monotonicity(arr, direction="increasing")
    _check_monotonicity(arr, direction="decreasing")

    # Test with array containing NaN (should raise ValueError)
    arr = np.array([1, 2, np.nan, 4])
    with pytest.raises(ValueError):
        _check_monotonicity(arr)


@pytest.mark.parametrize(
    "m, b",
    [
        (0.5, 10),  # +ve slope, +ve y-intercept
        (0.5, -10),  # +ve slope, -ve y-intercept
        (-0.5, 10),  # -ve slope, +ve y-intercept
        (-0.5, -10),  # -ve slope, -ve y-intercept
        (0.5, 0),  # +ve slope, 0 y-intercept
        (-0.5, 0),  # -ve slope, 0 y-intercept
    ],
)
def test_linear_Hc_properties(m, b):
    """Test the coercive field extraction from linear hysteresis data."""
    H, M, expected = linear_hysteresis_data(m, b)

    # Test Entity
    Hc = extract_coercive_field(H, M)
    assert isinstance(Hc, me.Entity)
    assert u.isclose(Hc, expected["Hc"] * u.A / u.m)

    # Test Quantity
    Hc = extract_coercive_field(H.quantity, M.quantity)
    assert isinstance(Hc, u.Quantity)
    assert u.isclose(Hc, expected["Hc"] * u.A / u.m)

    # Test Numpy Array
    Hc = extract_coercive_field(H.value, M.value)
    assert isinstance(Hc, np.ndarray)
    assert np.isclose(Hc, expected["Hc"])


@pytest.mark.parametrize(
    "m, b",
    [
        (0, 10),  # 0 slope, +ve y-intercept
        (0, -10),  # 0 slope, -ve y-intercept
    ],
)
def test_linear_Hc_errors(m, b):
    """Test coercive field extraction errors for linear hysteresis data."""
    H, M, _ = linear_hysteresis_data(m, b)

    with pytest.raises(ValueError):
        extract_coercive_field(H, M)


def test_partial_Hc_errors():
    """Test coercive field extraction errors for partial hysteresis data."""
    # Create a partial hysteresis loop
    h_values = np.linspace(-100, 100, 21)
    m_values = np.linspace(80, 100, 21)

    H = me.H(h_values * u.A / u.m)
    M = me.Ms(m_values * u.A / u.m)

    with pytest.raises(ValueError):
        extract_coercive_field(H, M)


@pytest.mark.parametrize(
    "m, b",
    [
        (0.5, 10),  # +ve slope, +ve y-intercept
        (0.5, -10),  # +ve slope, -ve y-intercept
        (-0.5, 10),  # -ve slope, +ve y-intercept
        (-0.5, -10),  # -ve slope, -ve y-intercept
        (0.5, 0),  # +ve slope, 0 y-intercept
        (-0.5, 0),  # -ve slope, 0 y-intercept
    ],
)
def test_linear_Mr_properties(m, b):
    """Test the remanent magnetization extraction from linear hysteresis data."""
    H, M, expected = linear_hysteresis_data(m, b)

    # Test Entity
    Mr = extract_remanent_magnetization(H, M)
    assert isinstance(Mr, me.Entity)
    assert u.isclose(Mr, expected["Mr"] * u.A / u.m)

    # Test Quantity
    Mr = extract_remanent_magnetization(H.quantity, M.quantity)
    assert isinstance(Mr, u.Quantity)
    assert u.isclose(Mr, expected["Mr"] * u.A / u.m)

    # Test Numpy Array
    Mr = extract_remanent_magnetization(H.value, M.value)
    assert isinstance(Mr, np.ndarray)
    assert np.isclose(Mr, expected["Mr"])


def test_partial_Mr_errors():
    """Test remanent magnetization extraction errors where field doesn't cross axis."""
    # Create a partial hysteresis loop where field doesn't cross zero
    h_values = np.linspace(1, 100, 21)  # All positive field values
    m_values = np.linspace(80, 100, 21)  # Magnetization crosses zero but field doesn't

    H = me.H(h_values * u.A / u.m)
    M = me.Ms(m_values * u.A / u.m)

    with pytest.raises(ValueError):
        extract_remanent_magnetization(H, M)


def test_B_curve():
    """Test the extraction of the B curve from hysteresis data."""
    # Create a simple linear hysteresis loop
    h_values = np.linspace(-100, 100, 101)
    m_values = 0.5 * h_values + 10

    H = me.H(h_values * u.A / u.m)
    M = me.Ms(m_values * u.A / u.m)

    # Extract the B curve
    B_curve = extract_B_curve(H, M, demagnetisation_coefficient=1 / 3)

    # Check if the B curve is an Entity
    assert isinstance(B_curve, me.Entity)

    # Check if the B curve has the expected shape
    assert B_curve.shape == (101,)


def test_B_curve_errors():
    """Test the extraction of the B curve from hysteresis data."""
    # Create a simple linear hysteresis loop
    h_values = np.linspace(-100, 100, 101)
    m_values = 0.5 * h_values + 10

    H = me.H(h_values * u.A / u.m)
    M = me.Ms(m_values * u.A / u.m)

    # Test with invalid demagnetisation coefficient
    with pytest.raises(ValueError):
        extract_B_curve(H, M, demagnetisation_coefficient=None)
    with pytest.raises(ValueError):
        extract_B_curve(H, M, demagnetisation_coefficient=1.5)
    with pytest.raises(ValueError):
        extract_B_curve(H, M, demagnetisation_coefficient=-1)


@pytest.mark.parametrize(
    "m, c",
    [
        (2.0, 5.0),  # positive slope, positive intercept
        (1.5, -2.0),  # positive slope, negative intercept
    ],
)
def test_extract_maximum_energy_product_linear(m, c):
    """Tests the maximum energy product for a linear B(H) = m*H + c.

    This uses the analytic derivation:
        BH = H * (m*H + c) = mH^2 + cH
        d(BH)/dH = 2mH + c = 0  ->  H_opt = -c/(2m)
        BH_max = |m H_opt^2 + c H_opt| = |(-c^2)/(4m)| = c^2/(4|m|)

    Args:
        m (float): slope of the linear B(H) relationship.
        c (float): intercept of the linear B(H) relationship.

    Raises:
        AssertionError: if the computed BHmax deviates from the analytic result.
    """
    H_opt = -c / (2 * m)
    H = np.linspace(H_opt - 1.0, H_opt + 1.0, 500) * u.A / u.m
    dh = H[1] - H[0]
    B = (m * H.value + c) * u.T

    # Analytic expected maximum energy product
    expected_val = (c**2 / (4 * abs(m))) * (u.A / u.m * u.T)

    h_opt_calc, bh = extract_maximum_energy_product(H, B)

    assert isinstance(bh, me.Entity)

    assert u.isclose(h_opt_calc, H_opt * u.A / u.m, atol=dh)
    assert u.isclose(bh, expected_val)


@pytest.mark.parametrize(
    "m, c",
    [
        (-1.5, 2.0),  # negative slope, positive intercept
        (-2.0, -5.0),  # negative slope, negative intercept
    ],
)
def test_extract_maximum_energy_product_linear_error(m, c):
    """Tests the maximum energy product for a linear B(H) = m*H + c.

    This uses the analytic derivation:
        BH = H * (m*H + c) = mH^2 + cH
        d(BH)/dH = 2mH + c = 0  ->  H_opt = -c/(2m)
        BH_max = |m H_opt^2 + c H_opt| = |(-c^2)/(4m)| = c^2/(4|m|)

    Args:
        m (float): slope of the linear B(H) relationship.
        c (float): intercept of the linear B(H) relationship.

    Raises:
        AssertionError: if the computed BHmax deviates from the analytic result.
    """
    H_opt = -c / (2 * m)
    H = np.linspace(H_opt - 1.0, H_opt + 1.0, 500) * u.A / u.m
    B = (m * H.value + c) * u.T

    with pytest.raises(ValueError):
        extract_maximum_energy_product(H, B)


def test_extract_maximum_energy_product_non_monotonic():
    """Test the maximum energy product extraction from non-monotonic data."""
    # Create a non-monotonic B(H) curve
    h_values = np.linspace(-100, 100, 101)
    b_values = np.concatenate((np.linspace(0, 50, 51), np.linspace(50, 0, 51)))

    # Test with non-monotonic data
    with pytest.raises(ValueError):
        extract_maximum_energy_product(h_values, b_values)


def test_extrinsic_properties():
    """Test the extraction of extrinsic properties from hysteresis data."""
    # Create a simple linear hysteresis loop
    h_values = np.linspace(-100, 100, 101)
    m_values = 0.5 * h_values + 10

    H = me.H(h_values * u.A / u.m)
    M = me.Ms(m_values * u.A / u.m)

    # Extract the extrinsic properties
    ep = extrinsic_properties(H, M, demagnetisation_coefficient=1 / 3)

    # Check if the extracted properties are correct
    assert isinstance(ep.Hc, me.Entity)
    assert isinstance(ep.Mr, me.Entity)
    assert isinstance(ep.BHmax, me.Entity)

    ep = extrinsic_properties(H.quantity, M.quantity, demagnetisation_coefficient=1 / 3)
    assert isinstance(ep.Hc, me.Entity)
    assert isinstance(ep.Mr, me.Entity)
    assert isinstance(ep.BHmax, me.Entity)

    ep = extrinsic_properties(H.value, M.value, demagnetisation_coefficient=1 / 3)
    assert isinstance(ep.Hc, np.ndarray)
    assert isinstance(ep.Mr, np.ndarray)
    assert isinstance(ep.BHmax, np.ndarray)

    ep = extrinsic_properties(H, M, demagnetisation_coefficient=None)
    assert isinstance(ep.Hc, me.Entity)
    assert isinstance(ep.Mr, me.Entity)
    assert isinstance(ep.BHmax, me.Entity)
    assert np.isnan(ep.BHmax.value)
