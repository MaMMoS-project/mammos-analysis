"""Tests for Kuzmin functions."""

import numpy as np

import mammos_entity as me
import mammos_units as u

from mammos_analysis.kuzmin import (
    kuzmin_formula,
    _Ms_function_of_temperature,
    _A_function_of_temperature,
    _K1_function_of_temperature,
    kuzmin_properties,
    KuzminResult,
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


def test_Ms_function_of_temperature():
    """Test the Ms function of temperature."""
    Ms0 = 100.0
    Tc = 300.0
    s = 0.5
    ms_func = _Ms_function_of_temperature(Ms0, Tc, s)
    # repr
    assert repr(ms_func) == "Ms(T)"
    # numeric input
    m = ms_func(100.0)
    assert isinstance(m, me.Entity)
    assert u.allclose(m, kuzmin_formula(Ms0, Tc, s, 100.0) * u.A / u.m)
    # quantity input
    Tq = 100.0 * u.K
    m_q = ms_func(Tq)
    assert u.allclose(m_q, m)
    # enitity input
    T_entity = me.Entity("ThermodynamicTemperature", value=100, unit="K")
    m_entity = ms_func(T_entity)
    assert u.allclose(m_entity, m)


def test_A_function_of_temperature():
    """Test the A function of temperature."""
    A0 = me.A(2.0, unit=u.J / u.m)
    Ms0 = 100.0
    Tc = 300.0
    s = 0.5
    a_func = _A_function_of_temperature(A0, Ms0, Tc, s)
    # repr
    assert repr(a_func) == "A(T)"
    # numeric input
    a = a_func(100.0)
    assert isinstance(a, me.Entity)
    expected_a = me.A(A0 * (kuzmin_formula(Ms0, Tc, s, 100.0) / Ms0) ** 2)
    assert u.allclose(a, expected_a)
    # quantity input
    Tq = 100.0 * u.K
    a_q = a_func(Tq)
    assert isinstance(a_q, me.Entity)
    assert u.allclose(a_q, a)
    # entity input
    T_entity = me.Entity("ThermodynamicTemperature", value=100, unit="K")
    a_entity = a_func(T_entity)
    assert isinstance(a_entity, me.Entity)
    assert u.allclose(a_entity, a)


def test_K1_function_of_temperature():
    """Test the K1 function of temperature."""
    K1_0 = me.Ku(1e5, unit=u.J / u.m**3)
    Ms_0 = 100.0
    T_c = 300.0
    s = 0.5
    k1_func = _K1_function_of_temperature(K1_0, Ms_0, T_c, s)
    # repr
    assert repr(k1_func) == "K1(T)"
    # numeric input
    k1 = k1_func(100.0)
    assert isinstance(k1, me.Entity)
    expected_k1 = me.Ku(K1_0 * (kuzmin_formula(Ms_0, T_c, s, 100.0) / Ms_0) ** 3)
    assert u.allclose(k1, expected_k1)
    # quantity input
    Tq = 100.0 * u.K
    k1_q = k1_func(Tq)
    assert isinstance(k1_q, me.Entity)
    assert u.allclose(k1_q, k1)
    # entity input
    T_entity = me.Entity("ThermodynamicTemperature", value=100, unit="K")
    k1_entity = k1_func(T_entity)
    assert isinstance(k1_entity, me.Entity)
    assert u.allclose(k1_entity, k1)


def test_kuzmin_properties():
    """Test the kuzmin_properties function."""
    Ms = me.Ms([200, 100.0], unit=u.A / u.m)
    Tc = me.Entity("ThermodynamicTemperature", value=[0, 100], unit="K")
    K1_0 = me.Ku(1e5, unit=u.J / u.m**3)
    result = kuzmin_properties(Ms, Tc, K1_0)
    assert isinstance(result, KuzminResult)
    assert isinstance(result.Ms, _Ms_function_of_temperature)
    assert isinstance(result.A, _A_function_of_temperature)
    assert isinstance(result.K1, _K1_function_of_temperature)
    assert isinstance(result.Tc, me.Entity)
    assert isinstance(result.s, u.Quantity)
