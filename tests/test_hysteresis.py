"""Tests for hysteresis functions."""

from __future__ import annotations

import io

import mammos_entity as me
import mammos_units as u
import numpy as np
import pytest

from mammos_analysis.hysteresis import (
    LinearSegmentProperties,
    _check_monotonicity,
    extract_B_curve,
    extract_BHmax,
    extract_coercive_field,
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


def hysteresis_data_loop():
    """Hysteresis data from loop computed with mammos-mumag (0.10).

    Returns:
        H: External magnetic field. (numpy array)
        M: Spontaneous magnetisation. (numpy array)
        expected: Expected values for coercive field and remanence. (dict)

    Rough extrinsic parameters:
       "Hc": 3049705.665855338,
       "Mr": 1159972.916462917,
       "BHmax": 416124.72892616026

    # -------------
    import json
    import pathlib

    import mammos_analysis
    import mammos_entity as me
    import mammos_mumag
    import mammos_units as u
    import pandas as pd

    HERE = pathlib.Path(__file__).parent.resolve()
    u.set_enabled_equivalencies(u.magnetic_flux_field())

    with open(HERE / "inp_parameters.json") as f:
        parameters = json.load(f)

    H_max = (5 * u.T).to("A/m")

    results_hysteresis = mammos_mumag.hysteresis.run(
        mesh="mesh.fly",  # this is cube50_singlegrain_msize2
        Ms=me.Ms(parameters["Ms"]),
        A=me.A(parameters["A"]),
        K1=me.Ku(parameters["K1"]),
        theta=0,
        phi=0,
        h_start=H_max,
        h_final=-H_max,
        h_n_steps=30,
    # with "inp_parameters.json":
    {
    "T": 0.0,
    "Ms": 1160762.1515272781,
    "A": 6.26240767831441e-12,
    "K1": 2810000.0,
    }
    .
    """
    data = """
    3978873.5751313814	1160559.4819430034
    3713615.3367892895	1160545.5026512272
    3448357.0984471976	1160530.0222534367
    3183098.8601051057	1160512.8175065364
    2917840.621763014	1160493.621826069
    2652582.3834209214	1160472.1149034314
    2387324.145078829	1160447.9091288103
    2122065.906736737	1160420.5318234407
    1856807.6683946445	1160389.4017226764
    1591549.4300525526	1160353.7974835385
    1326291.1917104605	1160312.8149333182
    1061032.9533683686	1160265.3082222815
    795774.7150262764	1160209.807434096
    530516.4766841844	1160144.4013702823
    265258.2383420923	1160066.5670153636
    2.6504622331100836e-10	1159972.916462917
    -265258.2383420918	1159858.8108498764
    -530516.4766841839	1159717.7547178767
    -795774.715026276	1159540.4119885948
    -1061032.9533683679	1159312.941958013
    -1326291.19171046	1159014.0429553061
    -1591549.4300525521	1158609.3692674541
    -1856807.668394644	1158040.1533276264
    -2122065.906736736	1157197.223335705
    -2387324.145078829	1155852.1045097725
    -2652582.383420921	1153416.390926262
    -2917840.621763013	1147218.002471127
    -3183098.860105105	-1160512.8175084556
    -3448357.098447197	-1160530.0221632489
    -3713615.336789289	-1160545.502608302
    -3978873.575131381	-1160559.4820289502
    """
    arr = np.loadtxt(io.StringIO(data))  # shape (N, 2)
    H = arr[:, 0]
    M = arr[:, 1]

    expected = {
        "Mr": 1159972.916462917,
        "Hc": 3049705.665855338,
        "BHmax": 416124.72892616026,
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
    assert u.isclose(Hc.q, expected["Hc"] * u.A / u.m)

    # Test Quantity
    Hc = extract_coercive_field(H.quantity, M.quantity)
    assert isinstance(Hc, me.Entity)
    assert u.isclose(Hc.q, expected["Hc"] * u.A / u.m)

    # Test Numpy Array
    Hc = extract_coercive_field(H.value, M.value)
    assert isinstance(Hc, me.Entity)
    assert u.isclose(Hc.q, expected["Hc"] * u.A / u.m)


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
    assert u.isclose(Mr.q, expected["Mr"] * u.A / u.m)

    # Test Quantity
    Mr = extract_remanent_magnetization(H.quantity, M.quantity)
    assert isinstance(Mr, me.Entity)
    assert u.isclose(Mr.q, expected["Mr"] * u.A / u.m)

    # Test Numpy Array
    Mr = extract_remanent_magnetization(H.value, M.value)
    assert isinstance(Mr, me.Entity)
    assert u.isclose(Mr.q, expected["Mr"] * u.A / u.m)


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
    B_curve = extract_B_curve(H, M, demagnetization_coefficient=1 / 3)

    # Check if the B curve is an Entity
    assert isinstance(B_curve, me.Entity)

    # Check if the B curve has the expected shape
    assert B_curve.value.shape == (101,)


def test_B_curve_errors():
    """Test the extraction of the B curve from hysteresis data."""
    # Create a simple linear hysteresis loop
    h_values = np.linspace(-100, 100, 101)
    m_values = 0.5 * h_values + 10

    H = me.H(h_values * u.A / u.m)
    M = me.Ms(m_values * u.A / u.m)

    # Test with invalid demagnetizing factor
    with pytest.raises(ValueError):
        extract_B_curve(H, M, demagnetization_coefficient=None)
    with pytest.raises(ValueError):
        extract_B_curve(H, M, demagnetization_coefficient=1.5)
    with pytest.raises(ValueError):
        extract_B_curve(H, M, demagnetization_coefficient=-1)


def test_extract_BHmax_non_monotonic():
    """Test the maximum energy product extraction from non-monotonic data."""
    # Create a non-monotonic B(H) curve
    h_values = np.linspace(-100, 100, 101)
    m_values = np.concatenate(
        (np.linspace(0, 500000, 510000), np.linspace(500000, 0, 510000))
    )

    # Test with non-monotonic data
    with pytest.raises(ValueError):
        extract_BHmax(h_values, m_values, 1 / 3)


def test_extract_BHmax_square_loop():
    """Test the maximum energy product extraction from a square loop.

    For a square loop (constant M = M₀), the analytical result is:
        BHmax = μ₀ * M₀² / 4
    This is independent of the demagnetization coefficient N because
    the optimal internal field point shifts to compensate.
    """
    mu0 = u.constants.mu0

    Ms = me.Ms(me.J(1.61).quantity / mu0)
    Ms = me.Ms(1_281_197, "A/m")

    # Create square loop
    H = np.linspace(10.0 / mu0.value, -10.0 / mu0.value, 1000)
    M = np.ones(shape=1000) * Ms.value
    M[-1] = -M[0]

    N = 1 / 3
    BHmax = extract_BHmax(H=H, M=M, demagnetization_coefficient=N)

    BHmax_analytic = mu0 * Ms.quantity**2 / 4

    assert np.isclose(BHmax.quantity, BHmax_analytic, atol=1000, rtol=1e-6)

    # Test fourth quadrant
    H_inv = -H
    M_inv = -np.ones(shape=1000) * Ms.value
    M_inv[-1] = -M_inv[0]

    BHmax_inv = extract_BHmax(H_inv, M_inv, demagnetization_coefficient=N)

    assert np.isclose(BHmax_inv.q, BHmax.q)
    assert np.isclose(BHmax_inv.q, BHmax_analytic, atol=1000, rtol=1e-6)


@pytest.mark.parametrize(
    "N, M0",
    [
        (0, 1_000_000),
        (1 / 3, 1_000_000),
        (0.5, 1_000_000),
    ],
)
def test_extract_BHmax_analytical_constant_M(N, M0):
    """Test BHmax for a square loop with known analytical result.

    For a square loop where M = M₀ (constant), the internal fields are:

        H_int = H - N·M₀
        B_int = μ₀(H_int + M₀) = μ₀(H - N·M₀ + M₀) = μ₀(H + (1 - N)M₀)

    The energy product in the second quadrant is:

        P = -B_int · H_int = -μ₀(H + (1 - N)M₀)(H - N·M₀)

    Substituting H_int = H - N·M₀ (so H = H_int + N·M₀):

        B_int = μ₀(H_int + M₀)
        P = -μ₀(H_int + M₀)·H_int = -μ₀(H_int² + M₀·H_int)

    Maximizing with respect to H_int:

        dP/dH_int = -μ₀(2·H_int + M₀) = 0  →  H_int_opt = -M₀/2

    Substituting back:

        BHmax = -μ₀((-M₀/2)² + M₀·(-M₀/2))
              = -μ₀(M₀²/4 - M₀²/2)
              = μ₀·M₀²/4

    Remarkably, this is independent of the demagnetization coefficient N
    because the optimal internal field point shifts to compensate.
    """
    mu0 = u.constants.mu0

    H = np.linspace(2e6, -2e6, 2000)
    M = np.full(2000, M0)
    M[-1] = -M0

    BHmax = extract_BHmax(H=H, M=M, demagnetization_coefficient=N)

    expected = mu0 * M0**2 / 4 * (u.A / u.m) ** 2

    assert np.isclose(BHmax.q, expected, atol=500, rtol=1e-4)


@pytest.mark.parametrize(
    "alpha, beta",
    [
        (0, 1_000_000),
        (0.5, 1_000_000),
    ],
)
def test_extract_BHmax_analytical_linear_M(alpha, beta):
    """Test BHmax for linear M(H) = α·H + β with N = 0.

    With N = 0, the internal field equals the external field:

        H_int = H
        M = α·H + β
        B_int = μ₀(H_int + M) = μ₀(H + α·H + β) = μ₀((1 + α)H + β)

    The energy product in the second quadrant (H < 0, B > 0) is:

        P = -B_int · H_int = -μ₀((1 + α)H + β)·H
          = -μ₀((1 + α)H² + β·H)

    Maximizing with respect to H:

        dP/dH = -μ₀(2(1 + α)H + β) = 0
          →  H_opt = -β / (2(1 + α))

    Substituting H_opt back into P:

        BHmax = -μ₀((1 + α)·(-β/(2(1+α)))² + β·(-β/(2(1+α))))
              = -μ₀((1 + α)·β²/(4(1+α)²) - β²/(2(1+α)))
              = -μ₀(β²/(4(1+α)) - 2β²/(4(1+α)))
              = -μ₀(-β²/(4(1+α)))
              = μ₀·β² / (4(1 + α))

    This reduces to the constant-M result (μ₀·β²/4) when α = 0.

    Note: alpha must be >= 0 to satisfy the monotonicity requirement
    that H and M change in the same direction.
    """
    mu0 = u.constants.mu0

    H = np.linspace(2e6, -2e6, 2000)
    M = alpha * H + beta

    BHmax = extract_BHmax(H=H, M=M, demagnetization_coefficient=0)

    expected = mu0 * beta**2 / (4 * (1 + alpha)) * (u.A / u.m) ** 2

    assert np.isclose(BHmax.q, expected, atol=500, rtol=1e-4)


def test_extract_BHmax_few_values():
    """Test warnings and failure of maximum energy product extraction."""
    mu0 = u.constants.mu0
    Ms = me.Ms(1_281_197, "A/m")

    # Create square loop with no values in second quadrant:
    H = np.linspace(10.0 / mu0.value, -10.0 / mu0.value, 10)
    M = np.ones(shape=10) * Ms.value  # 1000 values of H
    M[-5:] = -M[0]  # magnetisation switches while H>0

    with pytest.raises(ValueError):
        # use demagnetization_coefficient = 0 to avoid complication
        # that H != H_internal
        _ = extract_BHmax(H=H, M=M, demagnetization_coefficient=0)

    # Create square loop with 1 values in 2nd quadrant:
    H = np.linspace(10.0 / mu0.value, -10.0 / mu0.value, 10)
    M = np.ones(shape=10) * Ms.value  # 1000 values of H
    M[-4:] = -M[0]  # magnetisation switches while H>0

    with pytest.warns(UserWarning) as rec:
        # use demagnetization_coefficient = 0 to avoid complication
        # that H != H_internal
        _ = extract_BHmax(H=H, M=M, demagnetization_coefficient=0)
    w = rec[0]
    assert "Only" in str(w.message)
    assert "1" in str(w.message)
    assert "values" in str(w.message)
    assert "inaccurate" in str(w.message)


def test_extrinsic_properties():
    """Test the extraction of extrinsic properties from linear hysteresis data."""
    # Create a simple linear hysteresis loop
    h_values = np.linspace(-100, 100, 101)
    m_values = 0.5 * h_values + 10

    H = me.H(h_values * u.A / u.m)
    M = me.Ms(m_values * u.A / u.m)

    # Extract the extrinsic properties
    ep = extrinsic_properties(H, M, demagnetization_coefficient=1 / 3)

    # Check if the extracted properties are correct
    assert isinstance(ep.Hc, me.Entity)
    assert isinstance(ep.Mr, me.Entity)
    assert isinstance(ep.BHmax, me.Entity)

    ep = extrinsic_properties(H.quantity, M.quantity, demagnetization_coefficient=1 / 3)
    assert isinstance(ep.Hc, me.Entity)
    assert isinstance(ep.Mr, me.Entity)
    assert isinstance(ep.BHmax, me.Entity)

    ep = extrinsic_properties(H.value, M.value, demagnetization_coefficient=1 / 3)
    assert isinstance(ep.Hc, me.Entity)
    assert isinstance(ep.Mr, me.Entity)
    assert isinstance(ep.BHmax, me.Entity)

    ep = extrinsic_properties(H, M, demagnetization_coefficient=None)
    assert isinstance(ep.Hc, me.Entity)
    assert isinstance(ep.Mr, me.Entity)
    assert isinstance(ep.BHmax, me.Entity)
    assert np.isnan(ep.BHmax.value)


def test_extrinsic_properties2():
    """Test the extraction of extrinsic properties from simulated hysteresis data."""
    H, M, expected = hysteresis_data_loop()
    result = extrinsic_properties(me.H(H), me.M(M), demagnetization_coefficient=1 / 3)
    assert np.isclose(
        result.Hc.value, expected["Hc"], atol=0.1, rtol=1e-8
    )  # "Hc": 3049705.665855338,
    assert np.isclose(
        result.Mr.value, expected["Mr"], atol=0.1, rtol=1e-8
    )  # "Mr": 3049705.665855338
    assert np.isclose(
        result.BHmax.value, expected["BHmax"], atol=0.1, rtol=1e-8
    )  # "BHmax": 416124.72892616026


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

    assert u.isclose(results.Mr.q, 0 * u.kA / u.m, atol=1 * u.A / u.m)
    assert u.isclose(results.Hmax.q, 10 * u.kA / u.m)
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

    assert u.isclose(results.Mr.q, 0 * u.kA / u.m, atol=1 * u.A / u.m)
    assert u.isclose(results.Hmax.q, 10 * u.kA / u.m)
    assert u.isclose(results.gradient, 2 * u.dimensionless_unscaled)
