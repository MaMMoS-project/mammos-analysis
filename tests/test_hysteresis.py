"""Tests for hysteresis functions."""

from __future__ import annotations

import mammos_entity as me
import mammos_units as u
import numpy as np
import pytest

from mammos_analysis.hysteresis import (
    LinearSegmentProperties,
    MaximumEnergyProductProperties,
    _check_monotonicity,
    _unit_processing,
    extract_B_curve,
    extract_coercive_field,
    extract_maximum_energy_product,
    extract_remanent_magnetization,
    extrinsic_properties,
    find_linear_segment,
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

    arr = np.array([1, 2, float("nan"), 4])
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
    assert isinstance(Hc, me.Entity)
    assert u.isclose(Hc, expected["Hc"] * u.A / u.m)

    # Test Numpy Array
    Hc = extract_coercive_field(H.value, M.value)
    assert isinstance(Hc, me.Entity)
    assert u.isclose(Hc, expected["Hc"] * u.A / u.m)


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
    assert isinstance(Mr, me.Entity)
    assert u.isclose(Mr, expected["Mr"] * u.A / u.m)

    # Test Numpy Array
    Mr = extract_remanent_magnetization(H.value, M.value)
    assert isinstance(Mr, me.Entity)
    assert u.isclose(Mr, expected["Mr"] * u.A / u.m)


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

    def linear_B(H):
        """Linear B(H) function."""
        return m * H + c

    H_opt = -c / (2 * m)
    H = np.linspace(H_opt - 1.0, H_opt + 1.0, 500) * u.A / u.m
    dh = H[1] - H[0]
    B = linear_B(H.value) * u.T

    # Analytic expected maximum energy product
    expected_val_BHmax = (c**2 / (4 * abs(m))) * (u.A / u.m * u.T)
    expected_val_Bd = linear_B(H_opt) * u.T

    result = extract_maximum_energy_product(H, B)

    assert isinstance(result, MaximumEnergyProductProperties)
    assert isinstance(result.Hd, me.Entity)
    assert isinstance(result.Bd, me.Entity)
    assert isinstance(result.BHmax, me.Entity)

    assert u.isclose(result.Hd, H_opt * u.A / u.m, atol=dh)
    assert u.isclose(
        result.Bd, expected_val_Bd, atol=(m * dh.value) * u.T
    )  # B tolerance related to H discretization
    assert u.isclose(
        result.BHmax,
        expected_val_BHmax,
        atol=(2 * m * H_opt + c) * dh.value * (u.A / u.m * u.T),
    )  # BHmax tolerance related to discretization


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
    assert isinstance(ep.Hc, me.Entity)
    assert isinstance(ep.Mr, me.Entity)
    assert isinstance(ep.BHmax, me.Entity)

    ep = extrinsic_properties(H, M, demagnetisation_coefficient=None)
    assert isinstance(ep.Hc, me.Entity)
    assert isinstance(ep.Mr, me.Entity)
    assert isinstance(ep.BHmax, me.Entity)
    assert np.isnan(ep.BHmax.value)


def test_unit_processing():
    """Test the unit processing."""
    # Test correct unit processing with Entity
    assert np.isclose(
        _unit_processing(me.H(1 * u.A / u.m), u.A / u.m, return_quantity=False), 1
    )
    assert np.isclose(
        _unit_processing(me.H(1 * u.kA / u.m), u.A / u.m, return_quantity=False), 1000
    )
    assert u.isclose(
        _unit_processing(me.H(1 * u.A / u.m), u.A / u.m, return_quantity=True),
        1 * u.A / u.m,
    )
    assert u.isclose(
        _unit_processing(me.H(1 * u.kA / u.m), u.A / u.m, return_quantity=True),
        1000 * u.A / u.m,
    )

    # Test correct unit processing with Quantity
    assert np.isclose(
        _unit_processing(1 * u.A / u.m, u.A / u.m, return_quantity=False), 1
    )
    assert np.isclose(
        _unit_processing(1 * u.kA / u.m, u.A / u.m, return_quantity=False), 1000
    )
    assert u.isclose(
        _unit_processing(1 * u.A / u.m, u.A / u.m, return_quantity=True), 1 * u.A / u.m
    )
    assert u.isclose(
        _unit_processing(1 * u.kA / u.m, u.A / u.m, return_quantity=True),
        1000 * u.A / u.m,
    )

    # Test correct unit processing with Numpy Array
    assert np.isclose(_unit_processing(1, u.A / u.m, return_quantity=False), 1)
    assert np.isclose(_unit_processing(1000, u.A / u.m, return_quantity=False), 1000)
    assert u.isclose(
        _unit_processing(1 * u.A / u.m, u.A / u.m, return_quantity=True), 1 * u.A / u.m
    )
    assert u.isclose(
        _unit_processing(1000 * u.A / u.m, u.A / u.m, return_quantity=True),
        1000 * u.A / u.m,
    )

    # Test with arrays of each type
    assert np.allclose(
        _unit_processing(
            me.H(np.array([1, 2, 3]) * u.A / u.m), u.A / u.m, return_quantity=False
        ),
        [1, 2, 3],
    )
    assert np.allclose(
        _unit_processing(
            me.H(np.array([1, 2, 3]) * u.kA / u.m), u.A / u.m, return_quantity=False
        ),
        [1000, 2000, 3000],
    )
    assert np.allclose(
        _unit_processing(
            np.array([1, 2, 3]) * u.A / u.m, u.A / u.m, return_quantity=False
        ),
        [1, 2, 3],
    )
    assert np.allclose(
        _unit_processing(
            np.array([1, 2, 3]) * u.kA / u.m, u.A / u.m, return_quantity=False
        ),
        [1000, 2000, 3000],
    )

    assert u.allclose(
        _unit_processing(
            me.H(np.array([1, 2, 3]) * u.A / u.m), u.A / u.m, return_quantity=True
        ),
        np.array([1, 2, 3]) * u.A / u.m,
    )
    assert u.allclose(
        _unit_processing(
            me.H(np.array([1, 2, 3]) * u.kA / u.m), u.A / u.m, return_quantity=True
        ),
        np.array([1000, 2000, 3000]) * u.A / u.m,
    )
    assert u.allclose(
        _unit_processing(
            np.array([1, 2, 3]) * u.A / u.m, u.A / u.m, return_quantity=True
        ),
        np.array([1, 2, 3]) * u.A / u.m,
    )
    assert u.allclose(
        _unit_processing(
            np.array([1, 2, 3]) * u.kA / u.m, u.A / u.m, return_quantity=True
        ),
        np.array([1000, 2000, 3000]) * u.A / u.m,
    )

    # Test with invalid inputs
    with pytest.raises(TypeError):
        _unit_processing("invalid", u.A / u.m)
    with pytest.raises(ValueError):
        _unit_processing(1 * u.T, u.A / u.m)
    with pytest.raises(ValueError):
        _unit_processing(np.array([1, 2, 3]) * u.m, u.A / u.m)


@pytest.mark.parametrize("method", ["maxdev", "rms"])
def test_find_linear_segment_line(method):
    """Test finding linear segment in a near linear loop."""
    # Perfect linear M = 2*H gives slope 2, intercept 0
    H = np.linspace(0, 20, 101) * u.kA / u.m
    transition = 10 * u.kA / u.m
    M = np.where(H.value <= transition.value, 2 * H, 20 * u.kA / u.m)
    results = find_linear_segment(
        H, M, margin=1 * u.A / u.m, min_points=3, method=method
    )
    assert isinstance(results, LinearSegmentProperties)
    assert isinstance(results.Mr, me.Entity)
    assert isinstance(results.Hmax, me.Entity)
    assert isinstance(results.gradient, u.Quantity)

    assert u.isclose(results.Mr, 0 * u.kA / u.m, atol=1 * u.A / u.m)
    assert u.isclose(results.Hmax, 10 * u.kA / u.m)
    assert u.isclose(results.gradient, 2 * u.dimensionless_unscaled)

    # Too few points (<10) should raise ValueError
    H = np.linspace(0, 5, 5) * u.A / u.m
    M = H
    with pytest.raises(ValueError):
        find_linear_segment(H, M, margin=1 * u.A / u.m, min_points=10)
    H = np.linspace(0, 10, 11) * u.m
    M = np.linspace(0, 10, 11) * u.A / u.m
    with pytest.raises(ValueError):
        find_linear_segment(H, M, margin=1 * u.A / u.m)
    with pytest.raises(ValueError):
        find_linear_segment(H, M, margin=1 * u.A / u.m, method="invalid_method")


@pytest.mark.parametrize("method", ["maxdev", "rms"])
def test_find_linear_segment_reversed(method):
    """Test finding linear segment in a reversed loop."""
    # Perfect linear M = 2*H gives slope 2, intercept 0
    H = np.linspace(0, 20, 101) * u.kA / u.m
    transition = 10 * u.kA / u.m
    M = np.where(H.value <= transition.value, 2 * H, 20 * u.kA / u.m)
    H = H[::-1]  # Reverse the H array
    M = M[::-1]  # Reverse the M array
    results = find_linear_segment(
        H, M, margin=1 * u.A / u.m, min_points=3, method=method
    )
    assert isinstance(results, LinearSegmentProperties)
    assert isinstance(results.Mr, me.Entity)
    assert isinstance(results.Hmax, me.Entity)
    assert isinstance(results.gradient, u.Quantity)

    assert u.isclose(results.Mr, 0 * u.kA / u.m, atol=1 * u.A / u.m)
    assert u.isclose(results.Hmax, 10 * u.kA / u.m)
    assert u.isclose(results.gradient, 2 * u.dimensionless_unscaled)
