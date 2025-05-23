"""Tests for hysteresis functions."""

from __future__ import annotations

import numpy as np
import pytest

import mammos_entity as me
import mammos_units as u
from mammos_analysis.hysteresis import extract_coercive_field, _check_monotonicity


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

    # Test with a monotonic decreasing array
    arr = np.array([5, 4, 3, 2, 1])
    _check_monotonicity(arr)

    # Test with a non-monotonic array
    arr = np.array([1, 2, 3, 2, 5])
    with pytest.raises(ValueError, match="Array is not monotonic."):
        _check_monotonicity(arr)

    # Test with constant array (should pass as monotonic)
    arr = np.array([3, 3, 3, 3])
    _check_monotonicity(arr)

    # Test with single element array (should pass as monotonic)
    arr = np.array([42])
    _check_monotonicity(arr)

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
