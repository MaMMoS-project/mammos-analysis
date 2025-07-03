"""Tests for Kuzmin functions."""

import math

import mammos_entity as me
import mammos_units as u
import numpy as np
import pytest

from mammos_analysis.kuzmin import (
    KuzminResult,
    _A_function_of_temperature,
    _K1_function_of_temperature,
    _Ms_function_of_temperature,
    kuzmin_formula,
    kuzmin_properties,
)


def test_kuzmin_formula_below_Tc():
    """Test Kuzmin formula for temperatures below Tc."""
    Ms0 = 100.0
    Tc = 300.0
    s = 0.5
    T = np.array([0.0, 100.0, 200.0])
    expected = Ms0 * (
        (1 - s * (T / Tc) ** 1.5 - (1 - s) * (T / Tc) ** 2.5) ** (1.0 / 3)
    )
    result = kuzmin_formula(Ms0, Tc, s, T)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected)


def test_kuzmin_formula_above_Tc():
    """Test Kuzmin formula for temperatures above Tc."""
    Ms0 = 100.0
    Tc = 300.0
    s = 0.5
    T = np.array([300.0, 400.0])
    result = kuzmin_formula(Ms0, Tc, s, T)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, 0.0)


def test_kuzmin_formula_full_range():
    """Test Kuzmin formula for a full range of temperatures."""
    Ms0 = 100.0
    Tc = 300.0
    s = 0.5
    T = np.array([0.0, 150.0, 300.0, 450.0])
    result = kuzmin_formula(Ms0, Tc, s, T)
    assert isinstance(result, np.ndarray)


def test_kuzmin_formula_ints():
    """Test Kuzmin formula with integer inputs."""
    Ms0 = 100
    Tc = 300
    s = 0.5
    T = np.array([0, 150, 300, 450])
    result = kuzmin_formula(Ms0, Tc, s, T)
    assert isinstance(result, np.ndarray)


def test_Ms_function_of_temperature():
    """Test the Ms function of temperature."""
    T = me.Entity("ThermodynamicTemperature", value=[0, 100, 200], unit="K")
    Ms0 = 100.0
    Tc = 300.0
    s = 0.5
    ms_func = _Ms_function_of_temperature(Ms0, Tc, s, T)
    # repr
    assert repr(ms_func) == "Ms(T)"
    # numeric input
    m = ms_func(100.0)
    assert isinstance(m, me.Entity)
    assert u.allclose(m.q, kuzmin_formula(Ms0, Tc, s, 100.0) * u.A / u.m)
    # quantity input
    Tq = 100.0 * u.K
    m_q = ms_func(Tq)
    assert m_q == m
    # entity input
    T_entity = me.Entity("ThermodynamicTemperature", value=100, unit="K")
    m_entity = ms_func(T_entity)
    assert m_entity == m


def test_A_function_of_temperature():
    """Test the A function of temperature."""
    T = me.Entity("ThermodynamicTemperature", value=[0, 100, 200], unit="K")
    A0 = me.A(2.0, unit=u.J / u.m)
    Ms0 = 100.0
    Tc = 300.0
    s = 0.5
    a_func = _A_function_of_temperature(A0, Ms0, Tc, s, T)
    # repr
    assert repr(a_func) == "A(T)"
    # numeric input
    a = a_func(100.0)
    assert isinstance(a, me.Entity)
    expected_a = me.A(A0.q * (kuzmin_formula(Ms0, Tc, s, 100.0) / Ms0) ** 2)
    assert a == expected_a
    # quantity input
    Tq = 100.0 * u.K
    a_q = a_func(Tq)
    assert isinstance(a_q, me.Entity)
    assert a_q == a
    # entity input
    T_entity = me.Entity("ThermodynamicTemperature", value=100, unit="K")
    a_entity = a_func(T_entity)
    assert isinstance(a_entity, me.Entity)
    assert a_entity == a


def test_K1_function_of_temperature():
    """Test the K1 function of temperature."""
    T = me.Entity("ThermodynamicTemperature", value=[0, 100, 200], unit="K")
    K1_0 = me.Ku(1e5, unit=u.J / u.m**3)
    Ms_0 = 100.0
    T_c = 300.0
    s = 0.5
    k1_func = _K1_function_of_temperature(K1_0, Ms_0, T_c, s, T)
    # repr
    assert repr(k1_func) == "K1(T)"
    # numeric input
    k1 = k1_func(100.0)
    assert isinstance(k1, me.Entity)
    expected_k1 = me.Ku(K1_0.q * (kuzmin_formula(Ms_0, T_c, s, 100.0) / Ms_0) ** 3)
    assert k1 == expected_k1
    # quantity input
    Tq = 100.0 * u.K
    k1_q = k1_func(Tq)
    assert isinstance(k1_q, me.Entity)
    assert k1_q == k1
    # entity input
    T_entity = me.Entity("ThermodynamicTemperature", value=100, unit="K")
    k1_entity = k1_func(T_entity)
    assert isinstance(k1_entity, me.Entity)
    assert k1_entity == k1


def test_kuzmin_properties_all_info():
    """Test the kuzmin_properties function with all information.

    We create virtual data with some fixed values of Tc and s in order to
    anticipate the results of the optimization.
    """
    s = 0.75
    Tc = me.Tc(value=500, unit="K")
    K1_0 = me.Ku(1e5, unit=u.J / u.m**3)
    T_data = me.Entity("ThermodynamicTemperature", value=[0, 100, 200, 300, 400, 500])
    Ms_0 = me.Ms(100)
    Ms_data = me.Ms(kuzmin_formula(Ms_0=Ms_0, T_c=Tc, s=s, T=T_data))
    result = kuzmin_properties(Ms=Ms_data, T=T_data, Tc=Tc, Ms_0=Ms_0, K1_0=K1_0)
    assert isinstance(result, KuzminResult)
    assert isinstance(result.Ms, _Ms_function_of_temperature)
    assert isinstance(result.A, _A_function_of_temperature)
    assert isinstance(result.K1, _K1_function_of_temperature)
    assert isinstance(result.Tc, me.Entity)
    assert isinstance(result.s, u.Quantity)
    assert result.Tc == Tc
    assert result.K1(0) == K1_0
    assert math.isclose(result.s, 0.75, rel_tol=1e-02)
    assert result.Ms(T_data) == Ms_data
    assert result.Ms(0) == Ms_0
    A_0 = me.A(
        Ms_0.q
        * 0.1509
        * ((6 * u.constants.muB) / (s * Ms_0.q)) ** (2.0 / 3)
        * u.constants.k_B
        * Tc.q
        / (4 * u.constants.muB)
    )
    assert result.A(0) == A_0
    Tc = me.Tc(value=[500], unit="K")
    K1_0 = me.Ku([1e5], unit=u.J / u.m**3)
    Ms_0 = me.Ms([100])
    Ms_data = me.Ms(kuzmin_formula(Ms_0=Ms_0, T_c=Tc, s=0.75, T=T_data))
    result = kuzmin_properties(Ms=Ms_data, T=T_data, Tc=Tc, Ms_0=Ms_0, K1_0=K1_0)
    # result.Tc is a 0-d vector even though Tc was a 1-d vector.
    assert result.Tc == me.Tc(500)
    assert result.K1(0) == K1_0
    assert result.Ms(0) == Ms_0
    Tc = me.Tc(value=[[500]], unit="K")
    K1_0 = me.Ku([[1e5]], unit=u.J / u.m**3)
    # Ms_0 = me.Ms([[100]]) # TODO: fix in future PR
    Ms_data = me.Ms(kuzmin_formula(Ms_0=Ms_0, T_c=Tc, s=0.75, T=T_data))
    result = kuzmin_properties(Ms=Ms_data, T=T_data, Tc=Tc, Ms_0=Ms_0, K1_0=K1_0)
    # result.Tc is a 0-d vector even though Tc was a 2-d vector.
    assert result.Tc == me.Tc(500)
    assert result.K1(0) == K1_0
    assert result.Ms(0) == Ms_0


def test_kuzmin_properties_no_K1_0():
    """Test the kuzmin_properties function without K1_0."""
    Tc = me.Tc(value=500, unit="K")
    T_data = me.Entity("ThermodynamicTemperature", value=[0, 100])
    Ms_0 = me.Ms(100)
    Ms_data = me.Ms([100, 90])
    result = kuzmin_properties(Ms=Ms_data, T=T_data, Tc=Tc, Ms_0=Ms_0)
    assert isinstance(result, KuzminResult)
    assert isinstance(result.Ms, _Ms_function_of_temperature)
    assert isinstance(result.A, _A_function_of_temperature)
    assert result.K1 is None
    assert isinstance(result.Tc, me.Entity)
    assert isinstance(result.s, u.Quantity)
    assert result.Ms(0) == Ms_0


def test_kuzmin_properties_no_Tc():
    """Test the kuzmin properties function without Tc.

    We create virtual data with some fixed value of s in order to
    anticipate the results of the optimization.
    """
    Tc = me.Tc(value=500, unit="K")
    K1_0 = me.Ku(1e5, unit=u.J / u.m**3)
    T_data = me.Entity("ThermodynamicTemperature", value=[0, 100, 200, 300, 400, 500])
    Ms_0 = me.Ms(100)
    Ms_data = me.Ms(kuzmin_formula(Ms_0=Ms_0, T_c=Tc, s=0.75, T=T_data))
    result = kuzmin_properties(Ms=Ms_data, T=T_data, Ms_0=Ms_0, K1_0=K1_0)
    assert isinstance(result, KuzminResult)
    assert isinstance(result.Ms, _Ms_function_of_temperature)
    assert isinstance(result.A, _A_function_of_temperature)
    assert isinstance(result.K1, _K1_function_of_temperature)
    assert isinstance(result.Tc, me.Entity)
    assert isinstance(result.s, u.Quantity)
    assert result.Tc == Tc
    assert result.K1(0) == K1_0
    assert math.isclose(result.s, 0.75, rel_tol=1e-02)
    assert result.Ms(T_data) == Ms_data
    assert result.Ms(0) == Ms_0


def test_kuzmin_properties_no_Ms_0():
    """Test the kuzmin_properties function without Ms_0."""
    Tc = me.Tc(value=500, unit="K")
    Ms_data = me.Ms([200, 100.0], unit=u.A / u.m)
    T_data = me.Entity("ThermodynamicTemperature", value=[0, 100], unit="K")
    K1_0 = me.Ku(1e5, unit=u.J / u.m**3)
    result = kuzmin_properties(Ms=Ms_data, T=T_data, K1_0=K1_0, Tc=Tc)
    assert isinstance(result, KuzminResult)
    assert isinstance(result.Ms, _Ms_function_of_temperature)
    assert isinstance(result.A, _A_function_of_temperature)
    assert isinstance(result.K1, _K1_function_of_temperature)
    assert isinstance(result.Tc, me.Entity)
    assert isinstance(result.s, u.Quantity)
    assert result.Tc == Tc
    assert result.K1(0) == K1_0
    assert result.Ms(T_data) == Ms_data
    assert result.Ms(0) == me.Ms(200)
    T_data = me.Entity("ThermodynamicTemperature", value=[50, 100], unit="K")
    with pytest.raises(ValueError):
        # This test will fail as there is no data on Ms_0
        kuzmin_properties(Ms=Ms_data, T=T_data, K1_0=K1_0, Tc=Tc)
