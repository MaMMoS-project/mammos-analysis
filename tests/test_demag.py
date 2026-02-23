"""Tests for demagnetization factor calculation."""

import astropy.units as u
import mammos_entity as me
import numpy as np
import pytest

from mammos_analysis.demag import rectangular_prism


def test_demag_entities():
    """Test rectangular_prism with entities."""
    a = me.Entity("Length", 2.1, "mm")
    b = me.Entity("Length", 0.0022, "m")
    c = me.Entity("Length", 2.2e3, "um")
    expected = np.array([0.34374604, 0.32812698, 0.32812698])
    assert isinstance(rectangular_prism(a, b, c), me.Entity)
    assert np.all(np.isclose(rectangular_prism(a, b, c).value, expected))


def test_demag_quantities():
    """Test rectangular_prism with quantities."""
    a = u.Quantity(2.1, "mm")
    b = u.Quantity(0.0022, "m")
    c = u.Quantity(2.2e3, "um")
    expected = np.array([0.34374604, 0.32812698, 0.32812698])
    assert isinstance(rectangular_prism(a, b, c), me.Entity)
    assert np.all(np.isclose(rectangular_prism(a, b, c).value, expected))


def test_demag_float():
    """Test rectangular_prism with floating point numbers."""
    a = 2.1
    b = 2.2
    c = 2.2
    expected = np.array([0.34374604, 0.32812698, 0.32812698])
    assert isinstance(rectangular_prism(a, b, c), me.Entity)
    assert np.all(np.isclose(rectangular_prism(a, b, c).value, expected))


def test_demag_entitie_quantity():
    """Test rectangular_prism with mix of entities and quantities."""
    a = me.Entity("Length", 2.1, "mm")
    b = u.Quantity(0.0022, "m")
    c = me.Entity("Length", 2.2e3, "um")
    expected = np.array([0.34374604, 0.32812698, 0.32812698])
    assert isinstance(rectangular_prism(a, b, c), me.Entity)
    assert np.all(np.isclose(rectangular_prism(a, b, c).value, expected))


def test_demag_array():
    """Test rectangular_prism with objects containing array of elements."""
    a = me.Entity("Length", [2.1] * 3, "mm")
    b = u.Quantity([0.0022] * 3, "m")
    c = me.Entity("Length", [2.2e3] * 3, "um")
    expected = np.array([[0.34374604] * 3, [0.32812698] * 3, [0.32812698] * 3])
    assert isinstance(rectangular_prism(a, b, c), me.Entity)
    assert np.all(np.isclose(rectangular_prism(a, b, c).value, expected))


def test_demag_Exceptions():
    """Test whether expected exceptions correctly occur."""
    a = me.Entity("Length", 2.1, "mm")
    b = u.Quantity(0.0022, "m")
    c = 2.2
    with pytest.raises(ValueError):
        rectangular_prism(a, b, c)
