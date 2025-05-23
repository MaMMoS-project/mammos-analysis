"""Tests for hysteresis functions."""
from __future__ import annotations

import numpy as np
import pytest

import mammos_entity as me
import mammos_units as u
from mammos_analysis.hysteresis import extrinsic_properties, ExtrinsicProperties, extract_coercive_field, _check_monotonicity


def linear_hysteresis_data(m, b):
    # Create a simple linear hysteresis with known intercepts
    h_values = np.linspace(-100, 100, 101)
    m_values = m * h_values + b
    
    H = me.H(h_values * u.A / u.m)
    M = me.Ms(m_values * u.A / u.m)
    
    # Expected values for testing
    expected = {
        "Mr": abs(b),         # y-intercept
        "Hc": abs(-b / m),    # x-intercept 
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
        (0.5, 10),   # +ve slope, +ve y-intercept
        (0.5, -10),  # +ve slope, -ve y-intercept
        (-0.5, 10),  # -ve slope, +ve y-intercept
        (-0.5, -10), # -ve slope, -ve y-intercept
        (0.5, 0),    # +ve slope, 0 y-intercept
        (-0.5, 0),   # -ve slope, 0 y-intercept
    ]
)
def test_linear_Hc_properties(m, b):
    
    H, M, expected = linear_hysteresis_data(m, b)

    Hc = extract_coercive_field(H, M)
    
    assert isinstance(Hc, me.Entity)
    assert u.isclose(Hc, expected["Hc"] * u.A / u.m)



@pytest.mark.parametrize(
    "m, b",
    [
        (0.5, 10),  # slope, y-intercept
    ]
)
def test_linear_extrinsic_properties(m, b):
    """Test calculation of extrinsic properties with a linear hysteresis loop.
    
    This test verifies that the extrinsic_properties function correctly calculates
    the x-intercept (Hc) and y-intercept (Mr) for a linear hysteresis loop.
    """
    H, M, expected = linear_hysteresis_data(m, b)

    # Calculate extrinsic properties
    props = extrinsic_properties(H, M)
    
    # Verify properties are calculated with correct types
    assert isinstance(props, ExtrinsicProperties)
    assert isinstance(props.Hc, me.Entity)
    assert isinstance(props.Mr, me.Entity)
    assert isinstance(props.BHmax, me.Entity)
    
    # Verify that calculated values match expected intercepts
    # Note: the extrinsic_properties function returns absolute values
    assert pytest.approx(props.Hc.to(u.A / u.m).value) == abs(expected["Hc"])
    assert pytest.approx(props.Mr.to(u.A / u.m).value) == abs(expected["Mr"])
